# 在 root 容器里以非 root 用户跑闭环评测(autodl / Bench2Drive)

## 背景
CARLA 的 UE4 二进制**硬拒 root 启动**,`--allow-root` 对这个 build 无效(它不是 UE 官方那种可绕过的检查)。所以在 root 容器(如 autodl)里跑闭环,必须建一个非 root 用户,并让 **CARLA 和 python agent 都在该用户下运行**。

`scripts/eval_bench2drive_local.sh` 内部自己管 CARLA 的起停(每条 route 起一个、跑完杀),所以正确做法是**整个脚本用 `runuser` 降权跑**,而不是只降权 CARLA。

> 路径按本机改。下面用 autodl 的实际路径:
> - 项目 `LEAD_PROJECT_ROOT=/root/autodl-tmp/lead`
> - conda `/root/autodl-tmp/miniconda3`
> - CARLA `CARLA_ROOT=/root/autodl-tmp/lead/3rd_party/CARLA_0915`

---

## 一次性准备(root 下执行)

### 1. 建用户
```bash
useradd -m -s /bin/bash carla || true
```

### 2. 放权限(autodl 的东西都在 /root 下,非 root 用户默认进不去)
```bash
# 只给"可进入",不给读，够用
chmod o+x /root /root/autodl-tmp
# 项目 + conda 可读可执行
chmod -R o+rX /root/autodl-tmp/lead
chmod -R o+rX /root/autodl-tmp/miniconda3
# 评测输出目录可写
chmod -R o+rwX /root/autodl-tmp/lead/outputs
```

### 3. 解决 HF 缓存(resnet34)—— 二选一

CARLA 用户 `HOME=/home/carla`,读不到 root 之前下在 `/root/.cache` 的 resnet34 缓存。两条路：

**(A) 关掉 image encoder 的 pretrained(推荐,彻底不碰网络)**
timm 那份 ImageNet 权重反正会被 checkpoint 覆盖,评测时设 False 零损失。给要评的模型 config 补一个 key：
```bash
python -c "import json; p='outputs/closed_loop_models/posttrain/config.json'; d=json.load(open(p)); d['image_encoder_pretrained']=False; json.dump(d,open(p,'w')); print('done')"
```
> P5b 的 `p5b_M` / `p5b_S` config 里**已经**有 `image_encoder_pretrained=false`(训练脚本设的),不用改；只有 posttrain/LEAD 这种老 config 需要补。

**(B) 把 root 的 HF 缓存拷给 carla 用户**
```bash
mkdir -p /home/carla/.cache
cp -r /root/.cache/huggingface /home/carla/.cache/
chown -R carla:carla /home/carla/.cache
```

---

## 跑单条 route(先验证非 root 链路通)

```bash
export LEAD_PROJECT_ROOT=/root/autodl-tmp/lead
export CARLA_ROOT=/root/autodl-tmp/lead/3rd_party/CARLA_0915

runuser -u carla -- bash -c '
  export HOME=/home/carla
  export CARLA_ROOT='"$CARLA_ROOT"' LEAD_PROJECT_ROOT='"$LEAD_PROJECT_ROOT"'
  source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
  conda activate lead
  cd '"$LEAD_PROJECT_ROOT"'
  python -m lead \
    --checkpoint outputs/closed_loop_models/posttrain \
    --routes data/benchmark_routes/bench2drive/23687.xml \
    --bench2drive
'
```
能起 CARLA、agent 连上、跑出结果就说明非 root 链路通。

---

## 跑批量 220 条(整脚本降权)

```bash
export LEAD_PROJECT_ROOT=/root/autodl-tmp/lead
export CARLA_ROOT=/root/autodl-tmp/lead/3rd_party/CARLA_0915

runuser -u carla -- bash -c '
  export HOME=/home/carla
  export CARLA_ROOT='"$CARLA_ROOT"' LEAD_PROJECT_ROOT='"$LEAD_PROJECT_ROOT"'
  source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
  conda activate lead
  cd '"$LEAD_PROJECT_ROOT"'
  bash scripts/eval_bench2drive_local.sh outputs/closed_loop_models/posttrain "0" posttrain_test
'
```
参数：`eval_bench2drive_local.sh <ckpt_dir> "<gpu列表>" <tag>`。单卡写 `"0"`；多卡写 `"0 1 2 3"`。

**建议后台跑**(220 条很久，SSH 断了不受影响)：
```bash
runuser -u carla -- bash -c '... 同上 ...' >/tmp/b2d_run.log 2>&1 &
```

---

## 收集 + 合并结果
脚本跑完会打印这两行，照着执行（把 `posttrain_test` 换成你的 tag）：
```bash
mkdir -p outputs/b2d_posttrain_test
for f in outputs/local_evaluation_posttrain_test/*/checkpoint_endpoint.json; do
  cp "$f" outputs/b2d_posttrain_test/$(basename $(dirname "$f")).json
done
python slurm/evaluation/merge_route_json.py -f outputs/b2d_posttrain_test
```

---

## 评 P5b（S vs M 对比）
posttrain 跑通后，换 checkpoint 和 tag 即可，其余不变：
```bash
# 多模 M
bash scripts/eval_bench2drive_local.sh outputs/local_training/p5b_M "0" p5b_M
# 单模 S
bash scripts/eval_bench2drive_local.sh outputs/local_training/p5b_S "0" p5b_S
```
两个用**不同 tag**（输出到 `outputs/local_evaluation_p5b_M/` 和 `_p5b_S/`），可同时跑不冲突。各自合并出 Driving Score 后对比。

---

## 排错

| 现象 | 原因 / 处理 |
| :-- | :-- |
| `Refusing to run with root` | 没降权。必须 `runuser -u carla`，`--allow-root` 无效。 |
| `Permission denied` 读/写某目录 | carla 用户对该路径无权限。补 `chmod o+rX`（读）或 `o+rwX`（写）。`outputs/`、`/tmp/b2d_*` 必须可写。 |
| timm 去 HF 下 resnet34 / offline 报错 | 走上面「HF 缓存二选一」的 (A) 关 pretrained，或 (B) 拷缓存到 `/home/carla`。 |
| CARLA 起不来（`carla_gpu0.log`） | 看日志。可能要调 CARLA flag（画质/`-graphicsadapter`/`-RenderOffScreen`）。把手动起成功的命令对齐到脚本的 `start_carla()`。 |
| CARLA/UE 卡死或写 shader 失败 | `HOME=/home/carla` 没设对。UE 要写 `~/.config`、`~/.cache`。 |
| TM 端口 bind 失败 | 脚本已按 route 轮换 TM 端口规避 TIME_WAIT；若仍撞，多半是上一次的孤儿进程没清，`pkill -9 -f graphicsadapter=<gpu>`。 |

## 注意
- 这些是**机器相关**的运行步骤，别 commit 到代码里（用户/权限/路径都是 autodl 特有）。
- CARLA 二进制拒 root 是这个 build 的硬限制，换支持 `--allow-root` 的 build 才能免降权。
