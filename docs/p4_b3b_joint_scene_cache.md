# B3b joint scorer：第一阶段同次前向缓存

目标是为后续 DrivoR 式的轨迹–场景交互、多维度 joint scorer 准备**同次前向**数据，而不是现在就训练或闭环评测。冻结 `p5_stepB2_corridor/model_0019.pth`，关闭所有后加的速度/安全 gate，保持原有 confidence 路径选择。一次模型前向内同时保存：规划器的压缩 scene tokens、六条原始 Path、当前 confidence winner 的四条平滑局部偏移 Path（-1.5/-0.75/+0.75/+1.5 m）、原始 route confidence、当前车速、原始目标速度、专家 Path 与命令。每个分片带 checkpoint 路径/文件元数据、manifest 路径、索引范围和 schema 版本；分片先写临时文件，校验后原子重命名。

本阶段**没有**构造 GT 碰撞/道路/模仿标签，也**没有**训练 scorer。下一阶段应直接基于这些分片生成可执行的 Path×目标速度时空候选，结合 future-actor cache、GT HD-map 制作多维标签。不要把旧 B3a scene 缓存和独立 B3b oracle 按帧 key 拼接。

先在 GPU 机器上做 512 帧 pilot（独立输出目录，避免与全量分片混用）：

```bash
cd /mmu_mllm_hdd_3/liuzihan08/vla/lead
B3B_JOINT_GPUS=0 \
B3B_JOINT_START=0 B3B_JOINT_LIMIT=512 B3B_JOINT_CHUNK_SIZE=512 \
B3B_JOINT_OUTPUT_ROOT="$PWD/outputs/local_training/p5_stepB3b_joint_scorer/scene_cache_pilot" \
bash scripts/p4/build_p5_stepB3b_joint_scene_cache.sh heldout
```

pilot 校验通过后，跑全量训练/held-out（默认 8 卡；按机器实际可用卡号修改）：

```bash
cd /mmu_mllm_hdd_3/liuzihan08/vla/lead
B3B_JOINT_GPUS=0,1,2,3,4,5,6,7 \
bash scripts/p4/build_p5_stepB3b_joint_scene_cache.sh both
```

默认批量 64、DataLoader workers=0，避免此前 8 卡并发引起 `/dev/shm` Bus error。默认 12000 帧/分片，当前 VLM sparse-cache 可覆盖的 route 上约有 train 869958 帧、held-out 96406 帧，即 73+9 个分片。训练/held-out 严格按 route split，不能把 held-out 的 5000 帧 oracle 用作训练。每张 GPU 只启动一个持久 Python 进程：模型与密集 CARLA 索引仅加载一次，该进程连续写自己负责的多个原格式分片。日志为 `outputs/local_training/p5_stepB3b_joint_scorer/scene_cache/logs/{train,heldout}/gpu*_persistent.log`，分片仍在 `.../scene_cache/{train,heldout}`。脚本检测已有分片并续跑，末尾核对数目、key 唯一性、来源和覆盖；若已有分片来源不符，校验会报错，不会悄悄混用。没有 gpu-burn。

若显存或 I/O 紧张，可用 `B3B_JOINT_BATCH_SIZE=32` 或减少 `B3B_JOINT_GPUS`；保持输出目录及分片大小不变即可续跑。`B3B_JOINT_NUM_WORKERS` 默认仍为 0，勿直接在 8 卡同时设为 4。若 pilot 失败，先查看相应 `.log`，不要直接启动全量。旧版单分片产生的 `.npz` 与新版持久 worker 的格式相同，可以保留并续跑；不要删除已完成分片。

每个完成分片的日志末尾会报告 `data_wait`、`forward_pack`、`write_verify` 的耗时，便于判断后续瓶颈。在另一个终端查看 GPU 0 的实时日志：

```bash
tail -f outputs/local_training/p5_stepB3b_joint_scorer/scene_cache/logs/train/gpu0_persistent.log
```

如果旧版提取仍在运行，新代码不会改变那个已启动进程；不要同时启动两套任务。确认旧任务停止后重新运行上述全量命令，已完成且文件名匹配的分片会保留并跳过。
