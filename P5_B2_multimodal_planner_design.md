# P5 B2 设计:planner 输出 K 条多模轨迹(novelty 本体)

> 前置已就绪:P6 三相机 intent 点亮侧方臂;骨架端点 anchor 提取覆盖全方向(路口 K=4-5,直路 K=1)。
> 铁律(P5b 教训):planner 绝不随机重训 —— 从 B1/StepA 初始化 + 冻结 backbone + 小 lr + 最小随机增量。

---

## 0. 一句话

在 B1(30m 单条 route + P6 blob intent)基础上,把 **route 头从出 1 条改成出 K 条臂**,K 个 mode query 各注入 blob 骨架 anchor 破缺对称,WTA loss 只训 winner,置信头选臂,**闭环由碰撞代价在 K 条 valid 里晚期消解**(区别于 DiffusionDrive 的置信度选取)。

---

## 1. 数据流

```
3相机图 → Qwen → [P6 intent头(冻结)] → blob intent (B,1,320,384)
                                              │
                          [骨架端点 anchor 提取] → K_MAX 个 anchor (valid, angle, reach, tip)
                                              │ anchor embedding 注入
   backbone(3相机+LiDAR,冻结) → BEV feature ─┤
                                              │ intent_adapter(1→D) 不变
   planner query: [route K×30 | wp 8 | speed 1]  ← route 段加模态维
       每个 mode 的 30 个 route query += 该 mode 的 anchor embedding
                                              │
   TransformerDecoder → route (B,K,30,2) + conf (B,K) ; wp (B,8,2) 跟 winner
                                              │
   训练: WTA(valid槽 winner L1) + conf BCE + intent一致性 + 碰撞代价(all valid)
   闭环: K条各算碰撞代价 → 选无碰撞且最优 → PID
```

---

## 2. 结构改动(planner)

当前 query 布局(planning_decoder.py:36-50):`[route(30) | wp(8) | speed(1)]` = 39 个,一整块 `nn.Parameter(1,39,D)`。

**B2 改动:**
- route 段 `30` → `K_MAX×30 = 180`(K_MAX=6)。总 query = `180+8+1 = 189`。
- forward 里 route 段 reshape:`route_queries (B,180,D)` → decode → `(B,180,2)` → cumsum(per-mode,沿 30 维) → `(B,K_MAX,30,2)`。
- **新增置信头** `conf_decoder = Linear(D,1)`:对每个 mode 的 route query 池化(如取该 mode 30 个 query 的均值)→ `(B,K_MAX)` logit。
- wp/speed 段不变(8+1),wp 出单条(跟 winner 臂,见 §4)。

**anchor 注入(破缺对称的命门):**
- K_MAX 个 mode,每个 mode 一个 anchor `(valid, angle, reach)`。
- `anchor_embed = MLP([sin θ, cos θ, reach/30, valid])` → `(B,K_MAX,D)`。
- 加到该 mode 的 30 个 route query 上:`route_q[:,k,:,:] += anchor_embed[:,k,None,:]`。
- padding 槽(valid=0)的 anchor_embed 也编码了 valid=0 → planner 学会该槽是 no-object。

---

## 3. 变长 K 进 batch(DETR no-object)

- 固定 K_MAX=6 槽。场景 K_scene 条真 anchor 填前 K_scene 槽,其余 padding + valid mask=0。
- loss / 匹配 / 闭环只在 valid 槽算。padding 槽的 conf 监督为 0(no-object 类)。

---

## 4. Loss(决定不塌成单模)

只在 **valid 槽** 算:
1. **WTA 回归**:K_scene 条预测里,找离专家 GT route 最近的一条(winner)吃 L1(近处加权,同 StepA)。只 winner 回传回归梯度 → 其余 mode 不被专家单条拉扯 → 保住多模。
2. **置信 BCE**:winner 槽 conf=1,其余 valid 槽 conf=0,padding 槽 conf=0。
3. **intent 一致性**:每条 valid route 落在 blob intent 高密度区(拉预测臂贴合 blob,防漂移/防塌)。
4. **碰撞代价**(已有 `differentiable_collision`):作用于所有 valid 条(每条都要避障)。
- **wp**:跟 winner 臂的方向,单模 L1(§0 决策4)。

**WTA 的对称破缺**靠 §2 anchor 注入 —— 否则 K 个同质 query 在 WTA 下 winner 通吃、其余饿死(经典 WTA 死模态)。anchor 让每个 mode 有不同初始朝向,WTA 才有意义。

---

## 5. 闭环消费(novelty 兑现,区别 EvaDrive/DiffusionDrive)

- **不用 argmax 置信选条**(那是 DiffusionDrive,退化成分类)。
- **碰撞代价晚期消解**:K 条 valid route 各转成 BEV 占据,与实时障碍(bev_semantic/bboxes)算碰撞代价 → 选**无碰撞 + 置信最高 + 命令一致**的一条 → PID。
- 这才是"模糊意图走廊 + 反应控制晚期消解"的本体。
- **baseline 对照**:同一模型用 argmax 置信选条(伪多模)vs 碰撞消解,对比闭环 —— 证明晚期消解的价值。

---

## 6. 初始化(铁律)

- `load_file = B1`(30m 单条 + P6 intent 已适配);intent 头换 P6(`vlm_intent_p6_3cam`,冻结,用 training_utils 的重载修复)。
- route query 从 30 → 180:**每个 mode 的 30 query 都从 B1 的 30 query 复制初始化**(K 个 mode 起点相同,靠 anchor 注入分化)。wp/speed query 继承 B1。→ 扩 `_migrate_planner_query` 支持"route 段复制成 K 份"。
- backbone 冻结,小 lr(3e-5),VLM 头冻结。
- 唯一随机增量:conf_decoder + anchor_embed MLP(小)。

---

## 7. 评测(B1 已把总 DS 推到 94.94,天花板效应)

- **不能只看总 DS** —— 挑**多模场景**专门对比:路口选择(命令切换/多出口)、避障绕行(主路被挡切备选臂)。
- 指标:这些场景子集的 DS/success、碰撞率、off-road 率。
- 对照:B1(单条) vs B2(K条+碰撞消解) vs B2-argmax(伪多模)。

---

## 8. 分步落地(继续消变量)

- **B2a** ✅ 已完成:anchor 提取(骨架端点),可视化验证覆盖全方向。
- **B2b**:planner 出 K 条 + anchor 注入 + WTA + conf,开环验证——**不塌单模**(K 条 route 在路口真分叉)、winner ADE ≈ B1。
- **B2c**:闭环碰撞消解 + 多模场景评测。

**B2b go/no-go**:开环路口场景 K 条 route 可视化真分叉(不重合)、winner ADE 不掉点。塌了查:anchor 注入是否生效 / WTA 是否 permutation-invariant / intent 一致性权重。

---

## 9. 待拍板的细节

1. **K_MAX=6** 确认?(anchor 已按 6 提)
2. **anchor 注入方式**:加到 query(§2)还是 concat 后投影?建议加(简单,同 pos embedding 机制)。
3. **wp 跟 winner**:训练时 winner 已知(按 GT 匹配);闭环时 winner=碰撞消解选中的臂。一致?
4. **intent 一致性 loss 权重**:防塌关键,但太大会把所有 mode 拉成 blob 形状(又塌)。需调。
