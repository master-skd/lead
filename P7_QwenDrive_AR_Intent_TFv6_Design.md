# P7-A 技术设计：Frozen Qwen-Drive 自回归驾驶意图条件化 LEAD TFv6 Planner（无 Scorer）

> 状态：**设计方案，尚未实施或验证闭环性能**
> 基线：[`master-skd/lead`](https://github.com/master-skd/lead) 的 `visual-intent-p2` 分支（2026-10-26 检查），而非默认 `main` 分支。
> 目标：保留 LEAD posttrain Planner 以及 `route + target speed → controller` 的执行接口；冻结 Qwen-Drive-1.0，用它自回归生成的**结构化驾驶语义**初始化 K 组 TFv6 路径 Query；**不增加学习型 K Intent Queries、不训练独立 Intent Mask Decoder、不启用 Joint Scorer**。

> **当前实现决定（2026-10-27）**：主实验固定为 **K intent token → K route + K target speed**，以 route confidence head 做有效分支选择，训练使用 masked WTA；不接 scorer。以第 2、3、5、6 节为主要实现契约。现有 `PlanningDecoder` 明确把 `multimodal_target_speed` 绑定到 joint scorer，因此接入 P7 时必须先解开这个限制，并保证选中的 route、speed 使用同一个索引。

## 人工审核后的安全修订（2026-10-27）

第二组 CARLA 图片人工审核发现，VLM 曾把对向车道、黄线另一侧和道路边缘误报成可变道候选；即使强化提示词，两步 VQA 在事故路段仍输出了错误的左变道，且分析仅生成 111 token，未触及预算上限。因此 VLM 的 `maneuver + obligations + scene_rules` 用作**规划条件**，不能作为道路合法性的真值或闭环控制指令。第 2、3、5、6 节描述当前主实验；本节的独立规则诊断作为评估/安全消融。

### A. Intent 的语义：路径与必须履行的交通义务

每个 Intent 是一条**遵守道路方向、车道边界、导航约束的候选路径**，附带该路径当前必须履行的交通义务。停车、让行、跟车会改变时间与速度，不凭空增加另一条几何路径；等红灯后可继续的同一条路线应保留，但其义务必须是停车等待。越过双黄线进入对向车道、驶向不存在的相邻车道、使用转弯专用道直行等行为，不能作为有效 Intent。

建议的 VLM 输出契约为：

```json
{
  "scene_rules": {
    "signal": "red",
    "stop_sign": "no",
    "pedestrian_conflict": "uncertain",
    "vehicle_conflict": "yes"
  },
  "intents": [
    {
      "maneuver": "keep_lane",
      "obligations": ["stop_for_red", "yield_vehicle"],
      "confidence": 0.8,
      "evidence": "forward lane and red light visible"
    }
  ]
}
```

`maneuver` 沿用原五类几何行为。`obligations` 是可多选的固定词表，例如 `stop_for_red`、`stop_at_sign`、`yield_pedestrian`、`yield_vehicle`、`follow_lead`、`slow_for_obstacle`；空数组表示没有看见这些约束。`scene_rules` 各项允许 `uncertain`，不得把看不清当成“无约束”。`evidence` 仅用于审核日志，不直接编码进 Planner。VLM 的 `confidence` 不是合法性证明。

### B. Query 输入与独立诊断

VLM 输出的 intent 是 P7 要检验的语义条件。进入 Query 前仅做 JSON/schema、去重、容量和**明显导航矛盾**检查；保留 VLM 给出的各分支 `maneuver + obligations`，不要求另一个模型先证明每条车道合法。见 [`lead/tfv6/semantic_intent_slots.py`](lead/tfv6/semantic_intent_slots.py)。其 `valid[B,K]` 只表示语义槽位可用，**不表示道路合法性已获证明**。红灯/停牌等 VLM 自己报告的全局规则要与各分支义务保持内部一致；VLM confidence 不作为真实性证明。

独立的规则/几何检查作为候选 route 生成后的**诊断和可选安全约束**，使用当前帧 BEV、交通控制和可用拓扑证据；不能读取未来真值。若实验启用安全否决，缺失的独立证据必须被记录为 `unknown`，不能默默把所有 intent 在 Query 前清空。闭环若所有候选经实际安全检查均不可执行，应使用经验证的保守 fallback。训练时先比较不加独立拓扑门控的 P7 基线，再单独评估安全门控收益与误杀率。

Planner 的每个有效 Query 接收 `maneuver embedding + obligation multi-hot embedding + scene rule context`，而不只接收导航类别。为使不同路线/义务能产生不同纵向行为，规则约束主实验采用 **K route + K target speed**；单个 temporal-waypoint head 继续作为辅助监督。若仅有共享速度，`yield` 与 `proceed` 分支可能被迫使用相同速度，无法兑现此处的语义。闭环基础实验在语义有效分支中按 route confidence 选择；安全约束作为单独消融，选中路线和速度必须保持配对。

训练标签不能直接把 VLM 的文字当作真值：利用 CARLA 当前帧可用的交通灯、地图车道/标线、行人/车辆状态和专家控制构造场景规则监督；专家轨迹与速度监督 Planner；碰撞、逆行、闯灯等训练期信号用于合法性损失。需要单独统计 VLM 错误候选被过滤的比例、遗漏合法候选的比例、规则状态误判、停车/让行是否兑现，以及 fallback 率。审核集和训练集须分开，避免把本次人工纠错写进推理输入。

### C. 进入 Planner 实验的门槛

先在新的人工审核帧上分别验证：道路方向/黄线/边缘判断，红灯与停牌义务，行人和车辆冲突，JSON/schema 稳定性，分析生成是否触及 token 上限。P7 基线允许通过 schema 的 VLM intent 进入 Query；独立校验器的拒绝结果单独记录，不作为启动训练的前置条件。随后比较无 VLM 的 Planner 基线、P7 语义条件化和可选安全门控。任何单帧相机无法判断的法规或遮挡状态都必须标成不确定；**不能声称仅凭一个 VQA prompt 保证交通合法。**

规则感知输出的首个离线探针是 [`scripts/p7/probe_qwendrive_rule_intents_cpu.py`](scripts/p7/probe_qwendrive_rule_intents_cpu.py)。它只检验 VLM 能否表达 `scene_rules + intents[].obligations`、记录 token 上限和可由输出内部发现的规则矛盾；探针结果一律标记 `planner_usable=false`，因为它并未运行 Planner 或闭环。这一标记不等于 P7 语义训练必须等待独立几何校验器。

独立诊断器的纯逻辑核心现见 [`lead/tfv6/semantic_intent_validator.py`](lead/tfv6/semantic_intent_validator.py)。它要求**独立于 VLM 输出**的当前帧证据：每种 maneuver 的可达车道是否存在、方向是否一致、边界是否合法、是否可通行、是否符合导航，以及该路线上的行人/车辆/前车/障碍冲突；另需独立信号灯与停牌状态。证据不足会 fail-closed，这适合诊断严格可证实性，**不适合直接充当 P7 的 Query 输入 mask**。它尚未接入 Planner；不能把训练集未来标签、专家轨迹、VLM 自述或 confidence 填作独立证据。

当前帧 BEV 证据适配器见 [`lead/tfv6/semantic_intent_bev_evidence.py`](lead/tfv6/semantic_intent_bev_evidence.py)：对 Planner 产生的每条候选 route，从 LEAD **预测**的 BEV 语义图取道路、车辆、行人和障碍物证据，再送入独立诊断器。BEV 类别只有通用 `LANE_MARKERS`，没有黄线/虚实线与行车方向，因此它只能覆盖部分事实；交通灯与停牌的“没有”也不能由 BEV 未检出直接推断。适配器已用合成 BEV 验证，但**尚未接入在线 TFv6 forward**，不作为 P7 基线 Query 的门控。

## 0. 设计决定与边界

本版只验证一个假设：**预训练驾驶 VLM 自回归输出的分支级语义，能否在保留 LEAD 后训练规划能力的条件下，给多路线生成提供比旧 `corridor → skeleton → anchor` 更有效的条件？**

- **冻结**：Qwen-Drive-1.0 VLM 全部参数，使用 `eval()`、推理模式、无反向梯度。
- **训练**：TFv6 Planner（含已有 Query、Transformer Decoder、Route/Speed Heads）；一个小型 **Semantic Intent Encoder/Projector**（类别 embedding + MLP/线性层，非 K 个随机 Cross-Attention Queries）；LEAD backbone 是否解冻作为独立选项。
- **移除/旁路**：P6 `VLMIntentDecoder` 的栅格 corridor、骨架提取、从预测 blob 生成 anchors 的在线主路径；七维 `JointTrajectorySceneScorer`、在线 scorer 标签与 scorer 损失。
- **保留**：LEAD backbone/BEV、导航信息、原有规划和感知辅助损失、posttrain 参数初始化、Route + Target Speed 控制接口。
- **默认候选形式**：`K intent token → K route + K target speed`，每个语义分支都产生自己的几何路径和纵向目标；单个 temporal-waypoint head 保留为辅助监督。
- **绝不使用**：Qwen-Drive 生成的文字作为已验证的交通真值；不存在于路面的分支不能因为 VLM 提到就被当作可行。

## 1. 为什么不直接拿自回归 token IDs 做 Query？

Qwen-Drive 的自回归输出是**词法序列**；单个 Token ID 通常不对应一条驾驶分支。我们复用的是 **Qwen-Drive 生成的驾驶行为内容**，但在 Planner 前需要一个可解析、定长的**分支级数据契约**。

> 重要区分：以下分支级 JSON 是**新增的约束提示与解析协议**，不是 Qwen-Drive 官方保证的原生输出接口。冻结权重不代表它无需提示调试或格式校验。

建议输出：

```json
{
  "intents": [
    {"maneuver": "keep_lane", "interaction": "follow", "confidence": 0.8},
    {"maneuver": "turn_left", "interaction": "yield", "confidence": 0.6},
    {"maneuver": "turn_right", "interaction": "caution", "confidence": 0.5}
  ]
}
```

枚举建议（初版）：
- `maneuver`: `keep_lane`, `turn_left`, `turn_right`, `change_lane_left`, `change_lane_right`, `stop_wait`；
- `interaction`: `none`, `follow`, `yield`, `blocked`, `caution`；
- `confidence`: **不当作已校准的概率**；仅作日志诊断，初版不据此给训练权重；
- 可选 `reason`: 仅审查日志使用，不送入 Planner。

**两级验证**：JSON schema 有效性；与可见车道/LEAD BEV、导航指令的一致性。禁止将 VLM 随口列出的“其他道路”直接当作真实道路 GT。

### 1.1 输入提示示例

```text
You are a driving-scene assistant. Based only on the supplied current multi-camera images and navigation command, enumerate distinct plausible near-term ego driving maneuvers. Output ONLY JSON matching the provided schema. Do not invent unseen lanes or actors. Distinguish a maneuver's geometric possibility from whether it is currently safe. Use the fixed maneuver and interaction vocabularies. Prefer fewer valid maneuvers to unsupported claims. No chain-of-thought is required.

Navigation: {command}
Ego speed: {speed_summary}
Images: {multi_camera_images}
```

要求：导航信息进入同一次 VLM 输入；相机拼接与 Qwen-Drive 官方 processor 兼容性需实测；不依赖旧 Qwen3-VL 隐状态 `[B,12,36,2560]` 的任何形状假设。

### 1.2 语义和几何的分工

- Qwen-Drive：建议语义候选（`keep_lane/turn_left/...`）及行为提示（`yield/follow/...`）。
- LEAD BEV：判断障碍、路线几何和可行性，继续承担空间表征主职责。
- Planner：从分支级条件生成可执行 Route、Speed。
- 控制器：直接执行选中的 Route + Speed。

## 2. 数据流与张量契约

```text
RGB (3 cameras) + navigation/ego summary
     │
     ▼
Frozen Qwen-Drive VLM  ── autoregressive generation (inference only)
     │
     ▼
Strict JSON parser + dedup + coarse nav consistency
     │
     ▼
K slots: maneuver_id [B,K], obligations [B,K,O],
         scene_rules [B,S], semantic_valid [B,K]
     │
     ▼
Small SemanticIntentEncoder (TRAINABLE; no random K intent queries)
     │
     ▼
intent_emb [B,K,D=TFv6 token dim]
     │
     ▼
LEAD posttrain route queries + per-intent gated residual
     │
     ▼
TFv6 Decoder  ← LEAD BEV / radar / ego / navigation scene memory
     │
     ├── routes [B,K,N_route,2]
     ├── paired target-speed logits [B,K,C_speed]
     └── temporal waypoints [B,N_wp,2] (existing auxiliary head)
     │
     ▼
Masked route confidence selection (no scorer)
     │
     ▼
route[k*] + speed[k*] → existing controller
```

> K 是槽位容量，不是每帧必须生成的分支数。`semantic_valid` 仅标记格式/导航有效的 VLM intent，必须贯穿 Query、WTA 和 confidence 选择；不得对 padding 做 WTA 或选择无效路线。初版 K=6，与仓库已有多路线容量对齐。每个槽位对应同一 intent 的一条 route 和一个 speed。独立 BEV/规则诊断有自己的有效性结果，不能混同为 Query 输入 mask。

## 3. Intent 的无 Query 编码方式

对第 k 个已解析分支：

\[
 e_k = E_m(m_k) + W_o b_k + W_s s + E_{nav}(c) ;\quad
 z_k = W_2\,\mathrm{GELU}(W_1 e_k),\quad z_k\in\mathbb R^D.
\]

其中 `m_k` 是 maneuver，`b_k` 是该分支的 obligation multi-hot，`s` 是同帧 `scene_rules`，`c` 是导航命令。可用 `nn.Embedding` + 线性投影 + 2 层 MLP。**不能只编码 maneuver**，否则路况、停车/让行/跟车信息到 Query 前就丢失了。Embedding 是小型语义映射表，不是新引入的 K 组学习型查询。槽位通过语义内容而非列表索引定义；相同 maneuver 重复出现须去重或以更明确的语义标识区分。

初始化 TFv6 的第 k 组路径 Query：

\[
Q^{route}_{k,j}=Q^{LEAD}_{j}+\alpha\,\mathrm{Proj}(z_k),\qquad j=1,\dots,N_{route}.
\]

- `Q_j^LEAD` 来自 posttrain checkpoint 中已有路线查询，复制到各模式；**不要直接丢弃**其空间先验。
- `alpha` 可采用较小正值常数开局（如 0.05–0.1，待验证），或参数化小幅可学习 gate；不要无约束放大。
- 不再使用 `[sin(angle),cos(angle),reach/30,valid]` 的 `anchor_embed` 作为在线条件。
- 全局 `Target Speed Query` 可保留为场景/自车速度上下文，但每个 `speed_k` 必须从**该分支 route features + 全局 speed context**预测；选中的 `route_k` 与 `speed_k` 使用相同 `k`。`Temporal Waypoint Queries` 保留一组作为辅助头。
- 有效分支排序必须确定性，例如按固定 maneuver vocabulary 顺序填槽；但这只是工程稳定性，不代表模型自动学到了完整的跨帧身份一致性。

### 3.1 初版并不要求精确目标点

本实验刻意**不**强迫冻结 VLM 回归 `x/y`：它提供的是语义意图，TFv6 BEV 与导航来恢复路径几何。若结果显示转弯细节、路口拓扑缺失，可在下一版加 coarse goal/path 监督；不在 P7-A 同时引入。

## 4. 无 Scorer 时如何选择最终 Route？

这是本版必须明确解决的接口问题。原 P6 的 `joint_scorer_replaces_conf=true` 时，多路径选择依赖 scorer；去掉 scorer **不能使用 slot 0，也不能随便 argmax 未训练置信度**。

**推荐基础实验用已存在的 route-confidence 分支**：

1. 使 `joint_scorer_replaces_conf=false`，创建/加载 route confidence head（若 posttrain 不具备对应参数，则新增头初始化并训练）。
2. 每帧训练时仅对有效分支做专家路径 WTA；匹配专家的分支为 confidence 正样本，其余有效分支为负样本，mask 掉 padding；可以进行类别平衡。
3. 推理时在 schema/导航有效的 intent 分支中选择最高 route confidence；独立道路安全约束另作对照实验，不在 Query 前清空未证实的 VLM intent。
4. 对无有效候选、生成失败、明显越界的情况，**进入已验证的单路线 LEAD fallback**；不得把任意第一个候选直接执行。

置信度选择仅是本版的低成本基线，并不声称等价于安全 scorer；它主要学习“接近训练专家”，可能不充分奖励另一条安全路线。将来可与几何规则评分、七维 scorer 对比。

同时建议记录两个对照：**(a)** route confidence；**(b)** 导航一致性 + 几何可行性规则选择（无学习 scorer）。二者都不允许读取未来真值。

## 5. 训练目标与 Planner 初始化

### 5.1 从 Posttrain 精确初始化

- 使用当前仓库里的 `posttrain/model_0030.pth` 作为**候选来源示例**，但实际运行必须以目标服务器 checkpoint 的真实路径及其 config 为准。
- 初始化 LEAD Backbone、已有 TFv6 Decoder、Route/Speed/Temporal Heads；检查 `load_state_dict` 的 matched/missing/unexpected 列表，输出逐模块加载率。
- 多路线 Query 新增的 K 倍部分，按照单模式 route query 复制（或从当前多模 posttrain 若存在则显式映射）；新 SemanticIntentEncoder 与置信头独立初始化。
- **新训练不能把 P6 的 optimizer/scheduler resume 当作结构相同**；需要启动新优化器/调度器。不要覆盖原 P6 checkpoint。
- 建议 Phase 0 冻结 LEAD Backbone，只训练 Intent Encoder、Query Adapter、Planner Head/Decoder；稳定后 Phase 1 解冻适当 backbone 层作为对照。Qwen-Drive 全程冻结。

### 5.2 损失

专家单轨迹监督：

\[
k^*=\arg\min_{k\in\mathcal K_{valid}} D(R_k,R_{GT}),\qquad
L_{route}=L_{WTA}(R_{k^*},R_{GT}).
\]

第 `k*` 条 route **配对的速度**接受已有 two-hot 分类监督：

\[
L_{speed}=CE_{twohot}(v_{k^*},v_{expert}),\qquad
L_{confidence}=BCE_{masked}(c_k,\mathbf 1[k=k^*]).
\]

再加：

\[
L=L_{route}+\lambda_vL_{speed}+\lambda_cL_{confidence}
+\lambda_fL_{feasibility}+\lambda_aL_{auxiliary}.
\]

- WTA 的 route、speed、confidence 必须共用同一个 `k*` 和同一个 `semantic_valid[B,K]` mask。训练阶段只有专家实际走过的一条路线/速度有直接模仿标签；其余分支的速度不能伪称有反事实 GT。针对红灯、停牌、让行等 VLM 义务，可另加可观测的停车/碰撞约束与诊断，逐项验证其是否真的影响纵向行为。
- `L_feasibility` 可利用原有碰撞、road-compliance 等**训练期**信息构造。**不能直接原封不动复用 anchor-angle/reach 或 GT corridor loss**，它们依赖已经移除的几何 anchor / lane-graph label；先在代码中梳理哪些损失可独立成立。
- 纯几何 WTA 可能把专家路线分配给语义不相符的 intent，造成 Query 语义退化；如果能从专家当前路线可靠推导 maneuver，应在 winner 匹配时增加语义兼容约束，并记录无法匹配/语义冲突比例。不能仅用导航命令替代这项检查，否则模型仍只学导航。
- 本版**不设计新 K Intent GT**。有监督的是 Planner 的实际行为和有效性，而非每个语义 token 的数值。
- 若非 winner 路线只接受弱约束，可能坍缩；须记录候选间距离、相同 maneuver 的重复率、真实多分支场景覆盖率。不要仅对距离施加无约束的 repulsion。
- 当语义候选没覆盖专家路线时，保留 teacher-forcing 或导航命令定义的合法 expert-consistent branch 进行训练，并把该事件单独计数；训练期补充分支不应伪装成推理时 VLM 能发现它。

### 5.3 检验 Planner 是否真的使用路况语义

只看 loss 下降无法证明模型使用了 visual intent。固定导航命令与 ego 状态，在诊断集中比较不同 `scene_rules`/义务（红灯、停牌、跟车、让行、障碍）的 `intent_emb`、route、`speed_k` 与选中分支；还要做移除 obligation/scene-rule 输入的消融。若相同导航下这些输入改变却几乎不影响输出，说明模型仍主要依赖导航和 LEAD 场景记忆，需检查编码器梯度、残差门控大小、训练匹配和标签覆盖。按路况类别分别报告速度误差、停车/让行兑现率和闭环表现，不能仅以总体 ADE 或 Driving Score 判断语义是否被学到。

## 6. 代码修改清单（基于 `visual-intent-p2`）

| 位置 | 新增/修改 | 注意事项 |
|---|---|---|
| `lead/tfv6/tfv6.py` | 增加 AR intent 输入与解析结果接入；禁用 joint scorer、旧 blob→anchor 主路径；按 confidence 选中同一个 `k` 的 route/speed | 确保 `Prediction.pred_route` 和 `pred_target_speed_scalar` 仍供现有控制器读取 |
| `lead/tfv6/planning_decoder.py` | `anchor_embed` 条件替换为 `intent_emb [B,K,D]` 残差；每个 intent 条件化一组 route queries、输出对应的 `speed_k`；masked WTA/confidence | 解开当前 `multimodal_target_speed` 对 joint scorer 的硬依赖；保留 posttrain Query / Transformer 参数 |
| `lead/tfv6/semantic_intent_encoder.py`（已实现，未训练） | 将 `maneuver + obligation multi-hot + scene_rules + navigation` 融合成 `intent_emb [B,K,D]`，小门控残差注入 | 每个语义 token 对应一条 route 和一个 speed；不能仅编码导航或 maneuver |
| `lead/tfv6/semantic_intent_slots.py`（已有初版） | JSON 后的去重、固定词表、padding、语义有效 mask、fallback | `valid` 只表示语义槽位可用，不表示独立道路合法性已验证 |
| `lead/tfv6/semantic_intent_validator.py`、`semantic_intent_bev_evidence.py`（已有诊断初版） | 独立 BEV/规则诊断与可选安全消融 | 不作为主实验 Query 前的硬门控；证据缺失必须记录为 unknown |
| `lead/inference/` 与 Qwen 服务代码 | 已新增独立的 Qwen-Drive HF AR Unix socket 服务，传原始 JPEG 路径、导航与速度，返回结构化文本 | 与现有特征服务分离，通信有超时和限长；单样本错误会记录并走训练 fallback |
| `lead/training/config_training.py` | 已加入 `use_semantic_intent_planner=false`、`semantic_intent_query_gate_init=0.1`；P7 同时配置 `multimodal_planner=true`、`multimodal_target_speed=true`、`joint_scorer_replaces_conf=false` | 旧 P6 配置默认行为不变；P7 的 HF 训练入口见 `scripts/p7/train_p7_hf_online.sh` |
| `scripts/p4/train_p6_joint_online.sh` | **不直接修改**旧训练入口；为 P7 增加独立启动脚本 | 确保旧 P6 仍可复现 |
| `scripts/p7/`（新） | Prompt 校准、AR 离线缓存、训练、B2D 评测脚本 | 评测使用相同 prompt/processor/schema |
| `tests/` | 解析、空候选、mask、Query 形状、每槽位 route/speed 对齐、WTA 同索引、checkpoint mapping | 禁止选择 padding 分支；测试 VLM 义务变化能改变对应 Query |

当前 Planner 数据字典新增四个张量：`semantic_intent_maneuver_ids [B,K]`、`semantic_intent_obligations [B,K,6]`、`semantic_intent_scene_rules [B,4]`、`semantic_intent_valid [B,K]`；仍使用已有导航 `command [B,6]`。`semantic_intent_slots.service_results_to_tensors` 将在线回复转为这些张量，并增加 `semantic_intent_fallback [B]`。当前已接通 HF 在线服务和训练入口，能够在有数据与 GPU 的机器上启动训练；CPU 测试通过，但尚未完成 GPU 端到端 smoke、posttrain 加载审计与闭环验证。运行说明见 `docs/p7_hf_training.md`。

代码参考：
- [TFv6 主模型](https://github.com/master-skd/lead/blob/visual-intent-p2/lead/tfv6/tfv6.py)
- [Planning Decoder](https://github.com/master-skd/lead/blob/visual-intent-p2/lead/tfv6/planning_decoder.py)
- [P6 训练入口](https://github.com/master-skd/lead/blob/visual-intent-p2/scripts/p4/train_p6_joint_online.sh)
- [现有 K 组速度扩展](https://github.com/master-skd/lead/blob/visual-intent-p2/docs/p6_multimodal_target_speed.md)

### 6.1 数据服务与训练吞吐

AR 生成的耗时高于单次视觉 prefill，**不能假设实时每帧调用一定满足 B2D latency 预算**。建议先离线产生 `scene/frame → intents` 缓存，用于新训练的固定数据；B2D 闭环时再验证在线调用与缓存策略（后者不能使用未来帧或偷看 route GT）。

协议建议：`request_id`, `frame_id`, `camera_payload`, `command`, `ego_speed`, `prompt_version`; 返回 `raw_text`, `parsed_intents`, `parse_status`, `generation_latency_ms`, `model_version`。生成上限、temperature、seed 等参数固定并入实验配置；对自主驾驶验证建议先使用温度 0 或可复现的低温解码。

### 6.2 推理失败安全回退

- JSON 解析失败、候选全被过滤、Qwen-Drive 超时：走独立验证的 LEAD 单路线 fallback。
- Qwen-Drive 提出相互矛盾分支：去重、标记 invalid，保留日志；不要直接作为控制指令。
- 车道不存在、目标不符合导航：由 LEAD 感知/路线条件执行过滤；地图只是调试监督时，不可在正式无图推理中悄悄使用 GT OpenDRIVE。
- 较低调用频率的 AR Intent 需要防止导航变化时旧 Intent 继续控制（结合时间戳、有效期与命令变化即时刷新）。

## 7. 实验计划与验收

### 7.1 最小闭环消融

| ID | 模型/条件 | 要回答的问题 |
|---|---|---|
| B0 | 原始 LEAD posttrain，单路径 | 可信基线 |
| B1 | 无 VLM，复制 K 个 LEAD 路径 Query + route confidence | 多路径结构本身的收益 |
| B2 | 冻结 Qwen-Drive AR → 离散语义 → Intent Encoder + TFv6，多路线，无 scorer | 本设计核心假设 |
| B3 | 与 B2 同结构，用 Qwen3-VL 生成同一语义 schema | 驾驶专用预训练是否有额外收益 |
| B4 | B2 + K 组配对速度（独立追加） | 解除共享速度是否改进闭环 |
| B5（后续） | B2 + 学习型 K Intent Queries（独立追加） | 比较自回归语义 vs 连续隐式 Intent |

建议固定：同一 LEAD posttrain 权重、分割方式、seed、episode 集、导航命令、控制器设置、相机布局以及预算；Prompt 版本和生成策略必须版本化。

### 7.2 Intent / Planner / Closed-loop 三层指标

**Intent 层：** JSON 解析成功率、有效分支数、导航一致率、重复率、VLM 幻觉分支率、缺失专家相容分支率、AR p50/p95 延时。

**Planner 层：** Best-of-K ADE/FDE（仅作候选覆盖诊断）、selected ADE/FDE、候选多样性、合法道路占比、mode collapse、选择置信与专家一致性；额外记录 `N_valid >= 2` 占比。

**Closed-loop 层：** B2D Driving Score、Success Rate、Route Completion、碰撞/闯灯/逆行/卡死、各场景能力组表现；并记录 VLM 调用率、推理延迟和 fallback 比例。Best-of-K 使用 GT 只可作诊断，不可用于最终在线选路。

### 7.3 Go / No-Go

进入下一阶段的最低要求：
1. 推理协议稳定，无因解析/延迟造成不可控行动；
2. B2 的多分支预测在有效多分支场景中明显不是复制同一路线；
3. 候选能够通过导航/几何约束检查，选择路径不会持续偏离路面；
4. 完成完整 B2D 闭环对照，且能拆分「分支发现错误、路径生成错误、选择错误、控制执行错误」；
5. 若 B2 不优于 B1，应优先检查语义是否进入 Query、Prompt 的分支覆盖率及 posttrain 权重载入率，而非直接增加 Scorer。

## 8. 分阶段实现顺序

1. **接口验证**：Qwen-Drive 官方环境、三相机输入、文本生成稳定性、结构化解析测试；不修改 Planner。
2. **离线标签/缓存**：固定 prompt，输出结构化 AR Intent，统计 schema pass rate、场景分支覆盖率、导航一致性；抽样人工/规则审核。
3. **小型 Query Adapter**：在 `PlanningDecoder` 中以 `intent_emb` 替换 `anchor` 条件，保持 LEAD Posttrain Query；测试 K、mask、梯度和 checkpoint loading。
4. **无 Scorer 选路**：训练 masked route confidence；确保闭环 `selected route + speed` 走原控制器。
5. **训练 smoke**：先冻 LEAD backbone，训练 Intent Encoder + Planner；对照解冻 backbone，记录质量与回归风险。
6. **开环诊断与 B2D**：先 fixed routes，后全套 B2D；保存逐帧 Intent、route、置信、控制日志。
7. **再讨论扩展**：语义输出不足则研究几何 goal/path 弱监督；选择能力不足再引入 scorer；速度不足再开 K-speed，不同时多改。

## 9. 现存风险与明确不做的事

- **冻结 Qwen-Drive ≠ 永远准确，也不保证多分支**：它可以输出错误、缺漏或互相矛盾的语义候选。
- **无学习型 K Intent Queries ≠ 无额外训练**：Semantic Encoder/Projector 与 Planner 仍需要训练以对齐表征空间。
- **语义 Intent ≠ 精确几何 Intent**：转弯和变道的幅度、空间边界由 LEAD BEV + Planner 决定；如果缺乏足够空间控制，后续再做 coarse path。
- **去掉 scorer 后必须有明确的分支选择逻辑**：训练 route confidence 或可检验的规则选择，而非 slot 0。
- **只能证明相对于受控基线的增益**：不能把 B2D 的提升直接归因于“世界知识”而未控制模型容量、训练预算和 AR Prompt。
- 本版**不**实现 Qwen-Drive Planning Expert、BEV head 融合、任意轨迹排序、闭环 counterfactual scorer、非条件 K query 训练、K temporal waypoint 全量输出。

## 10. 文档与代码依据

- LEAD 分支及现有路径：[master-skd/lead (`visual-intent-p2`)](https://github.com/master-skd/lead/tree/visual-intent-p2)。
- [`planning_decoder.py`](https://github.com/master-skd/lead/blob/visual-intent-p2/lead/tfv6/planning_decoder.py)：现有 anchor-conditioned K 路径 Query、可选 K route-speed、masked WTA/选择。
- [`tfv6.py`](https://github.com/master-skd/lead/blob/visual-intent-p2/lead/tfv6/tfv6.py)：现有条件接口、Joint Scorer、Prediction 输出；P7 需显式旁路旧 VLM corridor 和 scorer。
- [`docs/p6_multimodal_target_speed.md`](https://github.com/master-skd/lead/blob/visual-intent-p2/docs/p6_multimodal_target_speed.md)：K route-speed 可选实现，不是本版默认。
- [Qwen-Drive-1.0 官方仓库](https://github.com/QwenLM/Qwen-Drive-1.0) 与 [模型说明](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/model.md)：区分 VLM 自回归输出和 Planning Expert 读取 KV Cache 的机制。

---

**一句话定义 P7-A：以冻结 Qwen-Drive 自回归生成的结构化驾驶语义作为多分支先验，通过轻量语义编码器对 LEAD Posttrain Route Queries 做残差条件化，使用原有规划/感知监督继续训练，不引入新的学习型 Intent Queries 或独立 Scorer，并在 Bench2Drive 闭环验证实际价值。**
