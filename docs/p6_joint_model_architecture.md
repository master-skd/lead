# P6：Visual Intent 条件化多模态 Planner + Joint Trajectory Scorer

本文对应 `scripts/p4/train_p6_joint_online.sh` 启动的**当前 P6 实现**（2026-10-08），以代码行为为准。它不是之前的“选定一条 path × K 个速度词表”的 B3a 方案，也不是已经验证过闭环收益的模型。核心变化是：LEAD planner 在视觉 intent 条件下直接产生最多 6 条空间路径；scorer 对**当次生成的每条路径及其时空展开**与 scene tokens 做交互，输出七项质量分数，代替旧的 route confidence。

## 1. 总体数据流

```text
三相机 RGB ──> 冻结 Qwen3-VL-4B ──> hidden [B,12,36,2560]
                                      │
                                      v
                          冻结 VLMIntentDecoder
                                      │
                              intent logits [B,1,320,384]
                              ┌───────┴───────────┐
                              v                   v
                     几何提取最多 6 个 anchor    intent scene tokens
                              │                   │
相机/LiDAR/Radar/自车状态 ──> LEAD backbone ──> planner scene memory
                              │                   │
                              └──────┬────────────┘
                                     v
                        多模态 PlanningDecoder
                           ├─ K 条 route [B,6,30,2]
                           ├─ 一组目标速度分布/标量
                           └─ 一组 temporal waypoints
                                     │
                 route + 共享目标速度 -> 2 秒候选轨迹 [B,6,8,6]
                                     │
                   JointTrajectorySceneScorer <── scene memory
                                     │
                     7 个维度的逐候选 logits [B,6]
                                     │
                      推理时聚合分数并选一条 route
```

这里的两个“6”都是容量上限，不代表每帧一定有 6 条有效分支。直行路通常只有少量有效 anchor；其余位置由 `valid` mask 排除。图中用于 scorer 的 2 秒轨迹是根据路径和速度做的**运动学近似展开**，不是 CARLA/PID 的真实滚动执行。

## 2. LEAD 原模型及本轮初始化

LEAD 的主模型是 [`TFv6`](../lead/tfv6/tfv6.py)：`TransfuserBackbone` 融合相机与 LiDAR 特征；RadarDetector 提供可选 radar tokens；模型还保留图像语义、深度、BEV 语义、检测等感知辅助头。PlanningDecoder 从 BEV 特征、radar、当前速度、离散 command、target point 等状态构造 scene memory，再预测规划输出。感知辅助损失仍参与当前联合训练；这不是只训练一个独立 scorer。

本轮 `load_file` 为 `outputs/local_training/pretrain/model_0030.pth`，**不是**此前约 94 分的 corridor planner 权重。该 pretrain 的 `config.json` 中 `use_planning_decoder=false`；已匹配的感知预训练参数按 `strict=False` 加载，规划分支和联合 scorer 从新初始化，而不继承已训好的 corridor planner。`freeze_backbone=false`（默认），所以 backbone、planner 和 scorer 均可训练；Qwen 与 VLM intent decoder 冻结。具体加载/冻结逻辑见 [`training_utils.py`](../lead/training/training_utils.py)。

## 3. Visual intent：如何产生、如何监督

### 3.1 离线生成监督标签

标签不是“唯一专家轨迹的窄高斯线”。[`precompute_lanegraph_intent.py`](../scripts/p4/precompute_lanegraph_intent.py) 在每帧的 CARLA 地图与自车位姿下，沿可行的左/直/右分支构造约 30 m 的车道中心线，将各分支画成约 3.5 m 宽的带状区域，做高斯平滑与归一化，得到 `[1,320,384]` 的**多分支软 corridor**。标签写在 `data/p6/lanegraph_label/.../*.npy`。这个 GT 地图标签只用于训练监督及训练期的路径/评分标签构造，不是闭环推理输入。

### 3.2 Intent decoder 的预训练

[`train_vlm_intent_online.py`](../scripts/p4/train_vlm_intent_online.py) 直接读取 RGB 与上述标签，冻结 Qwen3-VL-4B，仅训练 `VLMIntentDecoder`，不保存约 2 TB 的逐帧 VLM hidden cache。当前 3-camera strip 为 `384×1152`，Qwen hidden 为 `[B,12,36,2560]`。decoder 用卷积、BN、ReLU 和多级双线性上采样，最后双线性缩放至 `320×384`，输出**logits**，不是二值 mask。代码目前没有显式的深度几何投影；[`vlm_intent_decoder.py`](../lead/tfv6/vlm_intent_decoder.py) 顶部关于 `12×12`、缓存 hidden、专家单轨迹标签的旧说明不适用于本轮 3cam 训练。

单独训练 intent 时的损失为：

\[
L_{intent}=\operatorname{BCEWithLogits}(\hat I,I;\;pos\_weight=10)
+0.75\,(1-\operatorname{Tversky}(\sigma(\hat I),I)),
\]

其中 Tversky 的假阳性系数 `α=0.3`、假阴性系数 `β=0.7`，偏重召回可能遗漏的分支；优化器为 AdamW，默认学习率 `6e-4`、weight decay `1e-4`，按验证 loss 保存 `model_best.pth`。使用 route 互斥的密集帧划分：训练 `872,137` 帧、held-out `96,573` 帧；这里的“全量”指不再按旧版约 1/10 的稀疏帧抽样，不表示将 held-out 也用于 intent 参数训练。实际启动参数可覆盖默认 batch、epoch、学习率。

### 3.3 联合训练中的 intent 用法

当前联合训练每个 GPU 启动一个冻结的 Qwen 服务，按帧在线前向；intent decoder 加载上述最佳权重并保持 `eval()`、`requires_grad_(False)`。同一张预测 intent 图有两条通路：

1. 经 `sigmoid`/阈值、连通域与骨架端点提取，得到最多 6 个方向性 anchor；
2. 作为软图，经 1×1 `intent_adapter` 转成 planner scene memory 中的 intent tokens。

Anchor 提取见 [`anchor_extraction.py`](../lead/tfv6/anchor_extraction.py)：阈值 `0.5`、优先取 ego 所在连通域、骨架端点距 ego 至少 `6 m`、方向差小于 `20°` 的端点合并；按 reach 保留最多 6 个。传给 planner 的每个 anchor 是 `[sin(angle), cos(angle), reach/30, valid]`。**联合训练与推理均从预测 intent 提取 anchor**，不把 GT lane-graph anchor 喂给 planner；若一帧没有有效 anchor，代码令第 0 槽有效作为兜底。

## 4. Planner：为什么能输出 K 条路径

[`PlanningContextEncoder`](../lead/tfv6/planning_decoder.py) 把 backbone BEV 网格映射到 `D=256` 的 tokens，并拼接 intent tokens、自车状态/导航点 tokens、可用的 radar tokens。`PlanningDecoder` 有 `K×30` 个 route queries（此处 `K=6`），另有 8 个 temporal waypoint queries 和 1 个 target-speed query。每组 route query 都加上本分支 anchor 的 MLP embedding，避免 6 组 query 完全对称；随后 TransformerDecoder 对同一组 scene memory 做交叉注意力。每个 route query 输出二维增量，沿 30 点累加得到 `[B,6,30,2]` 的 ego 坐标空间路径。

速度**没有**为每条路径各自枚举 16/32/64 个词表项：target-speed query 只输出一组速度类别分布，解码为一个标量目标速度；8 个 temporal waypoints 也是单组输出。当前联合 scorer 消费的是 `K` 条路径各自结合这**同一个** planner 目标速度得到的轨迹，故主要学习“在给定速度下选哪条完整轨迹”，尚非路径×速度联合搜索。

旧 B2 版本有 route confidence head，采用 winner=1、其余有效分支=0 的 BCE。本轮 `joint_scorer_replaces_conf=true`，**不创建** `conf_decoder`，也不计算 `loss_route_conf` 和 padding-confidence loss；scorer 才承担候选排序。直行场景候选少由 `valid` mask 处理，而不是强迫模型发明 6 条不同的路。

## 5. Planner 的监督与损失

每帧专家导航 route 是 `[30,2]` 的监督目标。先在**有效**分支中计算每条预测路径与 GT route 的平均点位 L2，选误差最小的一条为 WTA winner；仅 winner 承担逐点 L1 与终点 FDE L1。逐点项中前 10 点权重 `1`，后 20 点权重 `0.3`。其余有效分支不被硬拉到同一条专家路线，而是由各自 anchor 的方向容差 hinge（默认 `25°`）、最小 reach hinge（默认 anchor reach 的 `0.5`）、所有有效路径上的障碍碰撞 cost 和 lane-graph corridor cost 约束，减少模态坍缩与偏离可行道路。另有 temporal waypoints L1、target-speed two-hot 分布交叉熵，以及 LEAD 原有的感知辅助损失。实现见 [`planning_decoder.py`](../lead/tfv6/planning_decoder.py) 和 [`config_training.py`](../lead/training/config_training.py)。

注意两个不同的路线监督：`data['route']` 是 planner 的导航 route GT；`joint_ego_future` 是 scorer imitation 维度的实际自车未来位置。前者确定 WTA winner，后者用于衡量候选 2 秒轨迹与专家未来的距离。把两者混成一个标签会误解 scorer 目标。

## 6. Joint scorer：输入、结构、七个维度

### 6.1 输入与 scene 交互

[`compose_route_states`](../lead/tfv6/joint_online_training.py) 将当前车速到 planner 目标速度在 8 个 `0.25 s` 时间点做线性插值，沿各条空间路径按累计距离插值得到 2 秒轨迹：

\[
S\in\mathbb R^{B\times K\times8\times6},\qquad
s_t=[x_t,y_t,\sin\psi_t,\cos\psi_t,v_t,a_t].
\]

路径超过末点时会按末段方向外推。这个展开没有模拟 PID、制动延迟或 CARLA 的闭环动力学。scene 输入复用 planner 的 cross-attention memory：BEV 与 intent 各自从空间网格池化至 `4×5` tokens，状态/radar token 原样保留。候选状态和 scene memory 都 `.detach()` 后送进 scorer：**scorer loss 不反传给 planner、backbone、intent**；planner/backbone 仍由自身规划及辅助损失更新。这就是当前“同一训练过程、阻断 scorer 梯度牵引 planner”的含义，而非 scorer 与 planner 完全分开训练。

[`JointTrajectorySceneScorer`](../lead/tfv6/joint_trajectory_scorer.py) 对每个候选 8 帧状态做线性投影与 2 层 temporal Transformer（hidden 128、8 heads），时间池化为候选 query；再用 2 层 TransformerDecoder 与压缩后的 scene tokens 交叉注意力，最后通过 7 个独立 MLP head 各输出一个 logit `[B,K]`。`sigmoid(logit)` 在推理时用于聚合；这些分数不是 PDMS，也不是 route confidence。

### 6.2 七个监督目标

标签由**当前前向生成的候选轨迹**即时构造，见 [`build_same_pass_labels`](../lead/tfv6/joint_online_training.py)。未来 actor 真值、专家未来、GT corridor 和导航 route 只用于构造标签，不作为 scorer 推理输入。无效分支不参加 loss。各目标均在 `[0,1]`：

| Head | 目标定义 | 含义与边界 |
| --- | --- | --- |
| `collision_free` | 2 秒内与 GT 未来 actor 无碰撞为 1，否则 0；碰撞 margin `0.2 m` | 几何代理安全性，不等于真实闭环碰撞率 |
| `ttc` | `min(TTC / 2 s, 1)` | 越接近 1 表示候选在窗口内越晚碰撞/未碰撞 |
| `drivable` | 8 个轨迹位置在 **GT visual-intent corridor** 图上的采样均值 | 实际是 lane-graph 分支走廊贴合度，不是独立的全路面可行驶概率 |
| `task` | 末端行进方向相对 GT 导航 route 方向误差的 `exp[-0.5(Δθ/35°)^2]` | 方向/任务一致性，并非逐条指令的离散分类 |
| `progress` | 沿 GT 导航方向的终点投影距离，除以 `max(平均速度×2 s,0.5 m)` 后裁剪 | 归一化任务方向进展 |
| `comfort` | `exp[-平均|加速度|/4 - 平均|yaw rate|/1.5]` | 用 2 秒代理轨迹计算的平顺性 |
| `imitation` | `exp[-候选与 GT 自车未来的 ADE / 2 m]` | 模仿专家；不等同于安全性 |

上述碰撞/TTC 来自 GT future actors，因此训练标签带有**特权未来信息**；推理时仅预测其分数，不读取未来真值。

### 6.3 Scorer loss 与总训练目标

七个 head 都使用有效候选上的 `BCEWithLogits`；连续软标签也按 soft BCE 处理。head 系数依次为：

| collision_free | ttc | drivable | task | progress | comfort | imitation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 1.0 | 1.5 | 1.0 | 0.5 | 0.25 | 0.5 |

碰撞 head 的 unsafe 样本额外乘 `8`，再按有效样本权重归一化。若同一帧同时有安全与不安全候选，另加 `0.5 × softplus(-(logit_safe-logit_unsafe))` 的成对排序损失。各 head 及排序项加和为 `loss_joint_scorer`。训练器再把它与 planner、感知辅助损失一起按 [`detailed_loss_weights`](../lead/training/config_training.py) 加权，并将启用的顶层权重归一化到总和 1；因此上表系数是 scorer **内部**权重，并非相对整个 LEAD loss 的最终比例。

## 7. 当前训练与推理状态

目前正在运行的 8 卡配置使用 `P6_JOINT_FULL_DATA=1`，把 route-split 的 train 与 held-out manifest 合并成 `968,710` 帧训练集；global batch `128`，workers `16`，计划 `20` epochs。这里不再有独立的 held-out 选模数据，训练程序保存每个 epoch checkpoint；不能仅凭训练 loss 选择“闭环最佳”模型。intent 预训练仍按上节 route-level train/held-out 划分。需要区分两阶段的数据口径。

代码中的推理选择为：各 head logit 过 `sigmoid` 后，计算

\[
U=2p_{collision\_free}+p_{ttc}+p_{drivable}+p_{task}
+0.5p_{progress}+0.25p_{comfort}+0.5p_{imitation},
\]

对有效分支取 `argmax U`，将对应 30 点 route 交给后续原有控制逻辑。此聚合权重是**当前代码中的试验性规则**，尚未经过独立 held-out 校准；没有硬碰撞阈值、风险 veto，也没有证明可超过 corridor baseline 的闭环结果。`target_speed`、temporal waypoints 仍由原 planner 的单组输出提供，并非 scorer 对速度词表选优。

下一步应先检查各 head 的校准、候选覆盖率、有效分支数量、选中路线相对专家/原 baseline 的差异，再在固定 ckpt 和固定 routes 上做闭环对照；尤其要验证“2 秒展开标签中的安全/舒适”与实际 PID 执行之间的偏差。当前结构提供了实验基础，**不是**已经成立的闭环性能结论。

## 8. 关键实现索引

- P6 入口与参数：[`train_p6_joint_online.sh`](../scripts/p4/train_p6_joint_online.sh)
- 主模型与选路：[`tfv6.py`](../lead/tfv6/tfv6.py)
- Qwen intent decoder / 在线单独训练：[`vlm_intent_decoder.py`](../lead/tfv6/vlm_intent_decoder.py)、[`train_vlm_intent_online.py`](../scripts/p4/train_vlm_intent_online.py)
- Lane-graph 标签 / 预测 blob anchor：[`precompute_lanegraph_intent.py`](../scripts/p4/precompute_lanegraph_intent.py)、[`anchor_extraction.py`](../lead/tfv6/anchor_extraction.py)
- Planner 与规划损失：[`planning_decoder.py`](../lead/tfv6/planning_decoder.py)
- Scorer 与 same-pass 标签：[`joint_trajectory_scorer.py`](../lead/tfv6/joint_trajectory_scorer.py)、[`joint_online_training.py`](../lead/tfv6/joint_online_training.py)、[`trajectory_scene_scorer.py`](../lead/tfv6/trajectory_scene_scorer.py)
