# P5b 闭环评测(IPC 版:Qwen-VL 服务 + lead agent)

## 为什么要这样
P5b 模型推理需要 `vlm_hidden`(冻结 VLM intent 头的输入)。训练时它来自离线缓存,但**闭环时车实时开、没有缓存**,必须实时跑 Qwen-VL。

问题:**Qwen3-VL 无法在 lead 环境加载**(lead 的 transformers 4.46 不认 `qwen3_vl`,需要 qwenvl 的 5.x)。而闭环 agent 跑在 lead 环境。

解法:**两进程 IPC** —— Qwen 单独跑在 qwenvl 环境的常驻服务里,lead 环境的 agent 通过 Unix socket 每帧发前视图、收 vlm_hidden。互不干扰,不动任何环境。

```
[lead env]  sensor_agent ──前视RGB──> [qwenvl env] vlm_service (常驻Qwen)
             CARLA+TFv6    <─vlm_hidden──
```

已验证:跨环境往返 OK,稳态延迟 ~85ms/帧,输出 (12,12,2560) fp16 确定性一致。

---

## 前置

1. **代码同步**:把本次改动 pull 到 5090。关键文件:
   - `lead/inference/vlm_service.py`（服务，qwenvl 环境跑）
   - `lead/inference/vlm_client.py`（客户端，lead 环境）
   - `lead/inference/vlm_feature_extractor.py`（共享抽取逻辑）
   - `lead/inference/sensor_agent.py`（改：连服务注入 vlm_hidden）
   - `lead/training/config_training.py`（新增 `vlm_service_socket` / `vlm_front_frac`）

2. **两个环境都要在 5090 上**：`lead`（跑 CARLA+agent）和 `qwenvl`（跑 Qwen 服务）。

3. **Qwen 模型 + P5b 权重就位**：
   - Qwen3-VL-4B-Instruct 目录（记下路径，如 `/root/autodl-tmp/models/Qwen3-VL-4B-Instruct`）
   - `outputs/local_training/p5b_M/`（config.json 里 `use_vlm_intent=true` 已设）
   - `vlm_intent_p5a_tversky/model_0014.pth`（p5b config 指向它）

---

## 步骤 1：启动 Qwen 服务（qwenvl 环境，非 root 用户下）

服务只跑 Qwen、不碰 CARLA，**不需要非 root**（不启动 UE）。但如果整套评测在 carla 用户下，服务也建议同用户跑，避免 socket 权限问题。

```bash
conda activate qwenvl
cd /root/autodl-tmp/lead
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python -m lead.inference.vlm_service \
    --model /root/autodl-tmp/models/Qwen3-VL-4B-Instruct \
    --socket /tmp/vlm_service.sock \
    --prompt-mode drivable \
    > /tmp/vlm_service.log 2>&1 &
```
等日志出现 `listening on /tmp/vlm_service.sock` 再进行下一步。

- **M（多模）用 `--prompt-mode drivable`**（去命令，对应 P5a-Tversky）。
- **S（单模）用 `--prompt-mode command`**（命令条件，对应 P4a）—— 评 S 时改这个。
- socket 路径要和 config 的 `vlm_service_socket` 一致（默认 `/tmp/vlm_service.sock`）。

---

## 步骤 2：跑闭环评测（lead 环境，非 root 用户）

和普通闭环评测完全一样（见 `RUN_CLOSED_LOOP_AS_NONROOT.md`），只是 checkpoint 换成 p5b：

```bash
# 单条 route 先验证
runuser -u carla -- bash -c '
  export HOME=/home/carla CARLA_ROOT=/root/autodl-tmp/lead/3rd_party/CARLA_0915 LEAD_PROJECT_ROOT=/root/autodl-tmp/lead
  source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh && conda activate lead
  cd /root/autodl-tmp/lead
  python -m lead --checkpoint outputs/local_training/p5b_M \
    --routes data/benchmark_routes/bench2drive/23687.xml --bench2drive
'
```
agent 会自动连 `/tmp/vlm_service.sock`（config 里的 `vlm_service_socket`），每帧发前视图取 vlm_hidden。

单条通了再跑全量 220（`bash scripts/eval_bench2drive_local.sh outputs/local_training/p5b_M "0" p5b_M`）。

---

## 步骤 3：GPU 显存共享

Qwen 服务(~9G)和 CARLA+TFv6 都在同一张 5090（32G）。显存够，但两者都 `CUDA_VISIBLE_DEVICES=0` 指同一卡即可。TFv6+planner 很小，CARLA 渲染占一部分，Qwen ~9G，总量在 32G 内。

---

## 验证对齐（保险，可选）
相机布局已确认：3×384 干净横条、前视在中间、无 fov_crop / 无相机子选，`front_frac=[1/3,2/3]` 正确。理论上不用再验。若不放心，跑一条 route 时在 sensor_agent 注入处 dump 一帧 `front_rgb` 存 png，肉眼看是不是正前方视野。

---

## 排错

| 现象 | 原因 / 处理 |
| :-- | :-- |
| agent 启动即 `ConnectionRefusedError` / socket 不存在 | 服务没起或 socket 路径不符。先确认服务日志 `listening`，且 `--socket` == config `vlm_service_socket`。 |
| `KeyError: 'vlm_hidden'` | agent 没走 IPC 分支 → config `use_vlm_intent` 没 true（检查 p5b config.json）。 |
| 服务 `KeyError: 'qwen3_vl'` | 服务跑在了 lead 环境（老 transformers）。必须 qwenvl 环境。 |
| 每帧太慢 → watchdog 超时 | 单帧 ~85ms 一般 OK。若超时，考虑降频（隔 N 帧抽一次、中间复用上次 intent）—— 需改 sensor_agent，先看实测。 |
| socket 权限 denied | 服务和 agent 不同用户。让两者同用户（都 carla），或 socket 放两者都可写的目录。 |
| M/S 用错 prompt | M=drivable，S=command。评哪个就用哪个 `--prompt-mode` 重启服务。 |

## 注意
- 服务是**长驻**的，评完一个模型不用重启（除非换 M↔S 要改 prompt-mode）。
- 评完记得 `pkill -f vlm_service` 释放显存。
