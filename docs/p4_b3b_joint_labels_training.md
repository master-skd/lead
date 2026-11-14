# B3b joint Path×可执行速度：GT 标签与 scorer 训练

输入是冻结 `p5_stepB2_corridor/model_0019.pth` 的同次前向 scene cache，不能与旧 B3a 特征缓存拼接。scene shard 的历史字段 `expert_route` 实际来自 `data["route"]`，是**导航路线**；标签生成器仅用它做任务方向参照。真正的 ego 未来从同帧 `metas/*.pkl` 的 `future_positions` 按 20Hz→4Hz 采样，不能把导航路线误当作模仿目标。

每帧最多 10 条 Path（6 原始 + 4 个 winner 周围局部变体），每条枚举 6 个**目标速度动作**，重复动作被 mask。纵向用现有 LEAD PID 调用与显式车辆动力学近似得到 2 秒速度/位移，横向用限 30°/s 的路径跟随代理得到 8 个姿态。训练候选的 `[x,y,sin(yaw),cos(yaw),v,a]` 与未来动态车辆/行人的 GT 包围盒做逐时刻 SAT 碰撞（沿用此前 B3b 审计的 class 1/2 和 0.2m margin）；同时从原始 GT HD-map 取得 8 帧、三横向点的道路占用软标签。静态障碍物尚未包含在碰撞头里。此处是 GT 特权监督及代理动力学，不等同于闭环 CARLA 行为。

7 个独立 0–1 监督头：`collision_free`、`ttc`（首碰撞时间/2s）、`drivable`、`task`（与导航路线末端方向一致程度）、`progress`（相对于原速度执行进度）、`comfort`（PID 加减速及路径跟踪误差）、`imitation`（同时间戳候选轨迹与 GT ego 未来的 2s ADE 指数映射）。完整轨迹与 65 个冻结 scene tokens 先分别编码，再 cross-attention；多头加权 BCE，稀有碰撞负例提高权重，另加帧内安全候选高于碰撞候选的排序项。训练输入**不含** GT future actor、HD-map 或 ego future；这些仅用于标签。

当前机器已验证 32 train + 32 held-out 帧的标签构造及 CPU 前反向冒烟；尚未完成全量训练或闭环验证。全量标签可在 CPU 上续跑（默认 8 进程，已完成分片会核对 key/来源并跳过）：

```bash
cd /mmu_mllm_hdd_3/liuzihan08/vla/lead
B3B_LABEL_WORKERS=8 bash scripts/p4/build_p5_stepB3b_joint_labels.sh both
```

输出在 `outputs/local_training/p5_stepB3b_joint_scorer/labels/{train,heldout}`，各分片 `.log` 在同目录。中断后直接重跑命令。只有看到最后的 `B3b joint GT labels ready` 且无 FAILED 才进入训练。全量标签约 966k 帧、每帧约 20–30 条有效候选；pilot 单进程约 14 帧/秒，实际全量时间取决于共享盘与 CPU 竞争。

训练用冻结 scene tokens，不更新原 LEAD 权重。默认 8 卡 DDP、8 epoch，训练/held-out 按 route 隔离：

```bash
cd /mmu_mllm_hdd_3/liuzihan08/vla/lead
B3B_TRAIN_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/p4/train_p5_stepB3b_joint_scorer.sh
```

中断后可把 `--resume outputs/local_training/p5_stepB3b_joint_scorer/training/joint_scorer_last.pth` 加在命令末尾。权重在 `.../training/joint_scorer_best.pth`，每 epoch 打印完整 held-out loss、原始/诊断选择的 GT 碰撞率、切换率、救回率及引入碰撞率。这些选择指标只用于审计，尚未把 scorer 接进闭环，也没有把最优 checkpoint 自动宣称为可部署。应先比较 held-out 的引入碰撞与切换，再设计 baseline-preserving shadow/active 评测。
