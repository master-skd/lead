<p align="center">
  <img src="https://avatars.githubusercontent.com/u/269669339?s=200&v=4" alt="KE:SAI" height="60">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://www.iapb.org/wp-content/uploads/2020/09/The-Eberhard-Karls-University-of-Tubingen.png" alt="University of Tübingen" height="60">&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://research.nvidia.com/labs/gear/images/media/nvidia_hu_9c090f760c3ff52d.png" alt="NVIDIA" height="60">
</p>

<h2 align="center">
  <b>LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving</b>
</h2>

<p align="center">
  <a href="https://ln2697.github.io/lead">Website</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/lead/docs">Docs</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/datasets/ln2697/lead">Dataset</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/ln2697/tfv6">Model</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://huggingface.co/ln2697/ltfv6-navsim">NAVSIM Model</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://ln2697.github.io/assets/pdf/Nguyen2026LEADSUPP.pdf">Supplementary</a>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="https://arxiv.org/abs/2512.20563">Paper</a>
</p>

<p align="center">
  An open-source end-to-end driving stack for CARLA.</br>
  ▶ State-of-the-art performance on all major Leaderboard 2.0 benchmarks ◀
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Bench2Drive-95 DS 🏆-00484B?style=for-the-badge&labelColor=006064" alt="Bench2Drive">
  <img src="https://img.shields.io/badge/Longest6_V2-62 DS 🏆-141A5F?style=for-the-badge&labelColor=1A237E" alt="Longest6 V2">
  <img src="https://img.shields.io/badge/Town13-10 DS 🏆-BF8E1E?style=for-the-badge&labelColor=FFBE28" alt="Town13">
  <img src="https://img.shields.io/badge/Fail2Drive-75 HM 🏆-4A148C?style=for-the-badge&labelColor=6A1B9A" alt="Fail2Drive">
</p>


<p align="center">
  <a href="https://ln2697.github.io">Long Nguyen</a><sup>1,3</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://de.linkedin.com/in/micha-fauth-b4492a22b">Micha Fauth</a><sup>1</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://kait0.github.io">Bernhard Jaeger</a><sup>1,3</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://danieldauner.github.io">Daniel Dauner</a><sup>1,3</sup>
  <br>
  <a href="https://maximilianigl.com">Maximilian Igl</a><sup>2</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="http://www.cvlibs.net">Andreas Geiger</a><sup>1,3</sup>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://kashyap7x.github.io">Kashyap Chitta</a><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>University of Tübingen, Tübingen AI Center&nbsp;&nbsp;·&nbsp;&nbsp;<sup>2</sup>NVIDIA Research&nbsp;&nbsp;·&nbsp;&nbsp;<sup>3</sup>KE:SAI
</p>

## Table of Contents

- [Updates](#updates)
- [1. Quick Start](#1-quick-start)
  - [1.1. Environment initialization](#11-environment-initialization)
  - [1.2. Install dependencies](#12-install-dependencies)
  - [1.3. Download Qwen-Drive](#13-download-qwen-drive)
  - [1.4. Download CARLA training data](#14-download-carla-training-data)
  - [1.5. Train P7 on eight GPUs](#15-train-p7-on-eight-gpus)

## Updates

<div align="center">

| Date         | Content                                                                                                                                                                                                               |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **26.04.29** | [PlanT 2.0](https://github.com/autonomousvision/plant2) privileged planner integration added. |
| **26.04.25** | [LEAD360](https://huggingface.co/datasets/ln2697/lead360) dataset published, features six cameras and metas to train PlanT 2.0.                                                                                                |
| **26.04.19** | [LEAD123D](https://huggingface.co/datasets/ln2697/lead123d) dataset published: five hours driving logs in 123D format.                                                                                                             |
| **26.04.17** | [NAVSIM LTFv6](https://huggingface.co/ln2697/ltfv6-navsim/) checkpoint predicts in CARLA's left-handed frame. When using inference code in HuggingFace, convert waypoints/headings back to ISO 8855 before evaluation. |
| **26.04.14** | Bug: [transfuser_token_dim](lead/training/config_training.py)'s default value is `256`.                                                                                                                               |
| **26.04.13** | Training instability reported; see [known issues](docs/source/known_issues.md). |
| **26.04.11** | [Fail2Drive](https://github.com/autonomousvision/fail2drive) benchmark support added. |
| **26.03.21** | [CaRL](https://github.com/autonomousvision/CaRL) reinforcement-learning planner evaluation support added. |
| **26.03.18** | Removed Kalman Filter and post-processing heuristics. |
| **26.02.25** | 🎉🎉 LEAD is accepted to **CVPR 2026** 🎉🎉                                                                                                                                                                                   |
| **26.02.25** | [NAVSIM](https://github.com/autonomousvision/navsim) extension released. Supplementary data coming soon. |
| **26.02.02** | [123D](https://github.com/autonomousvision/py123d) data collection format support added. |
| **26.01.13** | [LEAD](https://huggingface.co/ln2697/lead) dataset and training documentation released.                                                                                                                               |
| **25.12.24** | Initial release - paper, checkpoints, expert driver, and inference code.                                                                                                                                              |

</div>

## 1. Quick Start

Set up the repository and download Qwen-Drive for the P7 intent-conditioned planner. The original LEAD release reported the following hardware compatibility:

<div align="center">

| OS           | GPU        | CUDA | Driver | Inference | Training |
| ------------ | ---------- | ---- | :----: | :-------: | :------: |
| Ubuntu 22.04 | L40S       | 13.0 |  580   |     ✓     |    ✓     |
| Ubuntu 22.04 | A100       | 13.0 |  580   |     ✗     |    ✓     |
| Ubuntu 24.04 | RTX 5090   | 13.1 |  590   |     ✓     |    ✗     |
| Ubuntu 22.04 | RTX A4000  | 13.0 |  580   |     ✓     |    ✗     |
| Ubuntu 22.04 | GTX 1080ti | 13.0 |  580   |     ✓     |    ✗     |

</div>

### 1.1. Environment initialization

Clone the repository and register the project root:

```bash
git clone https://github.com/kesai-labs/lead.git
cd lead

# Set project's environment variables
echo -e "export LEAD_PROJECT_ROOT=$(pwd)" >> ~/.bashrc
echo "source $(pwd)/scripts/main.sh" >> ~/.bashrc

# Reload shell config
source ~/.bashrc
```

Verify that `~/.bashrc` reflects these paths correctly.

### 1.2. Install dependencies

Install the CARLA simulator **if you plan to collect data or run closed-loop evaluation**. If you already have CARLA 0.9.15, link that installation instead of downloading it again:

```bash
# Download and setup CARLA at 3rd_party/CARLA_0915
bash scripts/setup_carla.sh

# Or symlink your pre-installed CARLA
ln -s /your/carla/path 3rd_party/CARLA_0915
```

Training on an existing CARLA dataset, including P7 Qwen-Drive planner training, does not start the simulator and does **not** require `bash scripts/setup_carla.sh`. The training Python environment still needs the `carla` Python package, which is declared in `pyproject.toml`. The full simulator and its maps are needed when you collect new demonstrations or evaluate routes in closed loop.

We use Miniconda as container and uv for Python dependencies. Runtime and dev dependencies are declared entirely in [pyproject.toml](pyproject.toml).

```bash
# (Optional, needed in some cases) Accept terms and services
conda tos accept --override-channels --channel \
  https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel \
  https://repo.anaconda.com/pkgs/r

# Create a conda environment
conda create -n lead python=3.10 -y
conda activate lead

# Install system-level tools
conda install -c conda-forge ffmpeg parallel tree gcc zip unzip git-lfs uv aria2 zstd

# Tell uv to use conda environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export VIRTUAL_ENV=$CONDA_PREFIX' > $CONDA_PREFIX/etc/conda/activate.d/uv.sh
echo 'unset VIRTUAL_ENV' > $CONDA_PREFIX/etc/conda/deactivate.d/uv.sh
conda activate lead

# Install dependencies
uv sync --active --extra dev

# Optional: activate git hooks
pre-commit install
```

> [!TIP]
> 1. Blackwell and newer GPUs (RTX 5090, etc.) require PyTorch 2.7+ with CUDA 12.8. This setup is not tested for training and might lead to instability.
> ```bash
> pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
> ```
>
> 2. uv cheatsheet:
> ```bash
> # Add dependency
> uv add --active <pkg>
>
> # Add dev dependency
> uv add --active --optional dev <pkg>
>
> # Update uv.lock
> uv lock
> ```

### 1.3. Download Qwen-Drive

Download the [Qwen/Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) repository with [hfd](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f). Run these commands from the LEAD project root:

```bash
curl -fL https://gist.githubusercontent.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f/raw/hfd.sh -o /tmp/hfd.sh
mkdir -p models
bash /tmp/hfd.sh Qwen/Qwen-Drive-1.0-4B \
  --local-dir models/Qwen-Drive-1.0-4B --tool aria2c -x 8
```

The download includes the root VLM weights and the separately released planner heads. P7's HF intent service reads only the root VLM checkpoint; set `P7_QWEN_MODEL="$PWD/models/Qwen-Drive-1.0-4B"` when using the [P7 training script](scripts/p7/train_p7_hf_online.sh). See the [P7 training guide](docs/p7_hf_training.md) for the remaining data, environment, and posttrain checkpoint requirements.

P7 also needs the separate [Qwen-Drive source repository](https://github.com/QwenLM/Qwen-Drive-1.0) for its `qwen_drive` Python module. Install it in a dedicated GPU environment; the `lead` environment above runs the planner and does not need these Qwen-Drive dependencies:

```bash
# From the LEAD project root, on the GPU training machine.
git clone https://github.com/QwenLM/Qwen-Drive-1.0.git models/Qwen-Drive-1.0-code
conda create -n qwendrive_official python=3.12 -y
conda activate qwendrive_official
cd models/Qwen-Drive-1.0-code
python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install -e . --no-deps --no-build-isolation
python -c 'import torch, qwen_drive; print(torch.__version__, torch.cuda.is_available())'
cd ../..
conda activate lead
```

The check must print `True` for CUDA availability. The pinned Qwen-Drive requirements target Python 3.12 and CUDA 12.8; compiling `flash-attn` and related packages needs a suitable CUDA toolkit on the machine. A CPU-only PyTorch installation cannot serve P7 training on GPUs.

### 1.4. Download CARLA training data

From the LEAD project root, download the route archives and buckets from `master-skd/visual-intent-training-data`, then unpack the routes:

```bash
hf download master-skd/visual-intent-training-data \
  --repo-type dataset --include '*/*.zip' \
  --local-dir data/carla_leaderboard2/zip
hf download master-skd/visual-intent-training-data \
  --repo-type dataset --include 'buckets/*' \
  --local-dir data/carla_leaderboard2
python scripts/unpack_carla_training_zips.py
```

This restores the existing `data/carla_leaderboard2/data` and `data/carla_leaderboard2/buckets` layout. The old cache is not needed.

### 1.5. Train P7 on eight GPUs

Download the LEAD posttrain checkpoint from its separate private repository (run `hf auth login` first):

```bash
hf download master-skd/lead-posttrain posttrain.zip --local-dir .
unzip posttrain.zip
```

This restores `outputs/local_training/posttrain/model_0030.pth` and `config.json`. Run the following two stages from the LEAD project root.

#### 1.5.1. Extract the Qwen-Drive intent cache

Generate each training frame's structured intent JSON once using eight GPUs:

```bash
conda activate lead
CONDA_BASE=$(conda info --base)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P7_LEAD_PYTHON="$CONDA_BASE/envs/lead/bin/python" \
P7_QWEN_PYTHON="$CONDA_BASE/envs/qwendrive_official/bin/python" \
P7_QWEN_MODEL="$PWD/models/Qwen-Drive-1.0-4B" \
P7_POSTTRAIN=outputs/local_training/posttrain/model_0030.pth \
P7_MANIFEST=data/p7/intent_manifest.jsonl \
P7_INTENT_DIR=data/p7/intent_cache \
P7_GENERATION_BATCH_SIZE=8 P7_PREPROCESS_WORKERS=8 \
  bash scripts/p7/generate_p7_intents.sh
```

The launcher exports all posttrain bucket frames to the manifest, runs one generator per GPU, and saves results in this layout:

```text
data/p7/
├── intent_manifest.jsonl          # Frame keys, image paths, navigation and speed
└── intent_cache/
    ├── shard-00000.jsonl          # Per-frame analysis and structured intent JSON
    ├── ...
    ├── shard-00007.jsonl
    ├── generate_rank0.log         # One progress log per GPU
    ├── ...
    ├── generate_rank7.log
    ├── intents.sqlite            # Indexed JSON records consumed by training
    └── intents.summary.json      # Real branch counts, fallback and rejections
```

Watch progress with `tail -f data/p7/intent_cache/generate_rank0.log`. Re-run the same command with the same GPU count to resume; completed frames are skipped. Keep the manifest, model and prompt/token settings fixed when resuming. Set `P7_RETRY_FAILED=1` to regenerate unsuccessful or empty frames. A pre-existing manifest is reused; if it was created for a small smoke run, choose new manifest/cache paths for full generation.

`P7_GENERATION_BATCH_SIZE` is the inference batch **per GPU**, separate from the planner's `P7_BATCH_SIZE`. The default is 8 independent three-camera scenes, batched for analysis and then for JSON generation, with left padding for different prompt lengths. The example uses 8 CPU preprocessing workers per GPU (64 workers over eight GPUs); use 4 if fewer CPU cores are allocated. `P7_CPU_THREADS` defaults to 1 to limit nested CPU thread pools. GPU OOM halves the inference batch and retries the unwritten frames, logging the effective batch size. Progress logs include `batch_seconds`, `frames_per_second` and `peak_allocated_gib`. Existing serial JSONL shards can be resumed; batch/worker settings may change without discarding completed frames. Stop existing generators before restarting on the same shard files.

To tune extraction, change these environment variables in the generation command above:

| Variable | Default | Meaning |
| --- | --- | --- |
| `P7_GENERATION_BATCH_SIZE` | `8` | Independent frames generated together on each GPU; each frame has three camera images. |
| `P7_PREPROCESS_WORKERS` | `8` | CPU threads per GPU for image loading and official input preprocessing. |
| `P7_CPU_THREADS` | `1` | Threads in each nested Torch/BLAS CPU pool. |

For example, after measuring batch 8, try batch 16 with the same 8 workers per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P7_GENERATION_BATCH_SIZE=16 P7_PREPROCESS_WORKERS=8 \
  bash scripts/p7/generate_p7_intents.sh
```

This example assumes the Python/model path variables from the extraction command are exported in your shell; otherwise repeat them in the command. Compare `frames_per_second` and `peak_allocated_gib` in `generate_rankN.log` before increasing the batch again. If CPU allocation is limited, lower `P7_PREPROCESS_WORKERS` to 4. `P7_WORKERS` and `P7_BATCH_SIZE` below apply to planner training, not cache extraction.

Wait until all generators finish and `intents.sqlite` is built successfully. Check `intents.summary.json` for the actual 0/1/2/3-branch distribution before fallback and for generation failures. The JSONL files are retained for inspection; the indexed SQLite cache supplies the same raw answers to the existing intent parser during training.

#### 1.5.2. Train with the saved intent cache

Train SemanticIntentEncoder and the planner on the cached manifest's frames for 20 epochs:

```bash
conda activate lead
CONDA_BASE=$(conda info --base)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
P7_LEAD_PYTHON="$CONDA_BASE/envs/lead/bin/python" \
P7_POSTTRAIN=outputs/local_training/posttrain/model_0030.pth \
P7_MANIFEST=data/p7/intent_manifest.jsonl \
P7_INTENT_CACHE=data/p7/intent_cache/intents.sqlite \
P7_LOGDIR=outputs/local_training/p7_cached_full \
P7_BATCH_SIZE=64 P7_WORKERS=4 P7_EPOCHS=20 \
  bash scripts/p7/train_p7_cached.sh
```

`P7_BATCH_SIZE` is global (64 means 8 samples per GPU). Cached training loads no Qwen model or service and reuses the same JSON throughout all epochs. It still trains K paired routes/speeds with the same semantic encoder, masked WTA and confidence losses. Keep the manifest and SQLite cache together when moving machines; frame keys are independent of the original absolute paths. See the [P7 training guide](docs/p7_hf_training.md) for a small smoke run and the optional online baseline.
