from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import random
import typing

import diskcache
import numpy as np
import torch
import torch.multiprocessing as mp
from beartype import beartype
from diskcache import Cache
from torch import optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
)
from torch.utils.data import DataLoader

from lead.data_loader.carla_dataset import CARLAData
from lead.data_loader.navsim_dataset import NavsimData
from lead.data_loader.waymo_e2e_dataset import WODE2EData
from lead.tfv6 import transfuser_utils as fn
from lead.training import mixed_training_utils
from lead.training.config_training import TrainingConfig

LOG = logging.getLogger(__name__)


@beartype
def increase_limit_file_descriptors(n: int = 4096):
    # On some systems it is necessary to increase the limit on open file descriptors.
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (n, rlimit[1]))
    except (ModuleNotFoundError, ImportError) as e:
        LOG.error(str(e))


@beartype
def initialize_config() -> TrainingConfig:
    config = TrainingConfig()
    if config.model_type == "plant":
        from lead.plant.plant_config import PlantConfig

        config = PlantConfig()
    if config.load_file is not None:
        with open(
            os.path.join("/".join(config.load_file.split("/")[:-1]), "config.json"),
        ) as f:
            loaded_config = json.load(f)
        config_cls = type(config)
        config = config_cls(loaded_config, raise_error_on_missing_key=False)
    return config


@beartype
def create_model(config: TrainingConfig) -> torch.nn.Module:
    """Factory: instantiate model based on ``config.model_type``."""
    if config.model_type == "plant":
        from lead.plant.plant_model import PlantModel

        return PlantModel(config.device, config)
    from lead.tfv6.tfv6 import TFv6

    return TFv6(config.device, config)


@beartype
def initialize_training_session_cache(config: TrainingConfig) -> Cache | None:
    training_session_cache = None
    if config.use_training_session_cache:
        LOG.info(
            "Initializing training session cache at %s",
            config.training_session_cache_path,
        )
        training_session_cache = Cache(
            directory=config.training_session_cache_path,
            size_limit=int(2048 * 1024**3),
        )
    return training_session_cache


@beartype
def initialize_torch(config: TrainingConfig) -> int:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    ngpus_per_node = torch.cuda.device_count()
    ncpus_per_node = config.assigned_cpu_cores
    num_workers = int(ncpus_per_node / ngpus_per_node) * config.workers_per_cpu_cores

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=config.world_size,
            rank=config.rank,
            timeout=datetime.timedelta(minutes=120),
        )

    torch.cuda.device(config.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    return num_workers


@beartype
def _migrate_planner_query(
    loaded_state: dict,
    model: torch.nn.Module,
    config: TrainingConfig,
) -> dict:
    """Reconcile a checkpoint's planner `query` Parameter with a resized route head.

    The planning decoder concatenates its learnable queries as
    ``[route (num_route_points_prediction), waypoints (num_way_points_prediction),
    target_speed (1)]`` in that fixed order (see PlanningDecoder.forward). P5 Step A
    grows only the route segment (10 -> 30). A plain ``strict=False`` load would drop
    the whole `query` tensor on the shape mismatch, re-initializing EVERY query from
    scratch -- exactly the "random planner" collapse that sank P5b.

    Instead we copy the checkpoint's queries segment-by-segment into a fresh tensor
    pre-seeded with the current model's initialization, so the freshly-added far-range
    route queries keep their init while route[:old]/waypoint/speed inherit P2 verbatim.
    Only the route segment is assumed to differ in length; waypoint/speed counts must
    match (asserted). No-op when shapes already agree or the key is absent.
    """
    key = "planning_decoder.query"
    if key not in loaded_state:
        return loaded_state
    core = model.module if hasattr(model, "module") else model
    cur = core.state_dict()[key]  # (1, new_total, D)
    old = loaded_state[key]       # (1, old_total, D)
    if tuple(cur.shape) == tuple(old.shape):
        return loaded_state  # nothing to migrate

    n_wp = config.num_way_points_prediction if config.predict_temporal_spatial_waypoints else 0
    n_sp = 1 if config.predict_target_speed else 0
    new_route = config.num_route_points_prediction if config.predict_spatial_path else 0
    old_route = old.shape[1] - n_wp - n_sp
    # Sanity: only the route segment may resize; tail (wp+speed) length is invariant.
    assert new_route == cur.shape[1] - n_wp - n_sp, (
        f"query tail mismatch: cur={cur.shape[1]} new_route={new_route} n_wp={n_wp} n_sp={n_sp}"
    )
    assert old_route > 0 and new_route >= old_route, (
        f"unexpected route resize old_route={old_route} new_route={new_route}"
    )

    migrated = cur.clone()  # keep current init for the new far-range route queries
    # route[:old_route] <- checkpoint route queries (near segment inherits P2)
    migrated[:, :old_route] = old[:, :old_route]
    # waypoint + speed queries: copy the checkpoint tail onto the new tail
    if n_wp + n_sp > 0:
        migrated[:, new_route:] = old[:, old_route:]
    loaded_state = dict(loaded_state)
    loaded_state[key] = migrated
    LOG.info(
        f"Migrated planner query: route {old_route}->{new_route} "
        f"(inherited near {old_route}, fresh far {new_route - old_route}), "
        f"wp+speed {n_wp + n_sp} inherited.",
    )
    return loaded_state


@beartype
def initialize_model(
    config: TrainingConfig,
) -> tuple[typing.Any | torch.nn.parallel.distributed.DistributedDataParallel, int]:
    model = create_model(config)

    model.cuda(device=config.device)
    if config.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        LOG.info("Using sync_batch_norm")

    # Convert all norm layers to use fp32
    fn.patch_norm_fp32(model)

    start_epoch = 0  # Epoch to continue training from
    if config.load_file is not None:
        LOG.info(f"Loading model from {config.load_file}")
        # Add +1 because the epoch before that was already trained
        load_name = str(pathlib.Path(config.load_file).stem)
        if config.continue_failed_training:
            start_epoch = int("".join(filter(str.isdigit, load_name))) + 1
            LOG.info(f"Continuing training from epoch {start_epoch}")
        loaded_state = torch.load(
            config.load_file, map_location=config.device, weights_only=True,
        )
        # P5 Step A: route head extended from 10 to 30 points. The planner's `query`
        # Parameter (route|waypoint|speed queries, in that order) changes shape, so a
        # naive strict=False load would silently DROP it -> all queries re-init from
        # scratch (the P5b failure mode). Instead migrate per-segment: inherit P2's
        # route/waypoint/speed queries verbatim; only the newly-added far-range route
        # queries stay fresh. See planning_decoder query layout.
        loaded_state = _migrate_planner_query(loaded_state, model, config)
        model.load_state_dict(
            loaded_state,
            strict=config.continue_failed_training,
        )
        # B1: the frozen VLM intent head must stay whatever config.vlm_intent_ckpt says
        # (e.g. the P5a multimodal head), but load_file (e.g. a Step A checkpoint) carries
        # its OWN vlm_intent_decoder weights (the P4a single-mode head) and load_state_dict
        # just overwrote the head TFv6.__init__ loaded. Re-load + re-freeze from the config
        # ckpt so the intent source is unambiguous regardless of load_file's contents.
        if config.use_vlm_intent and config.vlm_intent_ckpt is not None:
            core = model.module if hasattr(model, "module") else model
            vlm_ckpt = torch.load(
                config.vlm_intent_ckpt, map_location=config.device, weights_only=True,
            )
            core.vlm_intent_decoder.load_state_dict(vlm_ckpt["model"])
            core.vlm_intent_decoder.eval().requires_grad_(False)
            LOG.info(
                f"Re-loaded frozen VLM intent head from {config.vlm_intent_ckpt} "
                f"(overrides any vlm_intent_decoder weights in load_file).",
            )

    model.backbone.requires_grad_(not config.freeze_backbone)
    LOG.info(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters",
    )
    if config.channel_last:
        model = model.to(memory_format=torch.channels_last)
        LOG.info("Using channel last memory format")
    if torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
        )
    else:
        model_wrapper = model
    if config.compile:
        model = torch.compile(
            model,
            fullgraph=True,  # require entire model to be compiled, fail if not
            dynamic=False,  # aggressively specialize to current input shapes
            backend="inductor",
            mode="max-autotune",  # highest autotune + CUDA graph
        )
    return model_wrapper, start_epoch


@beartype
def initialize_optimizer(
    model_wrapper: typing.Any | torch.nn.parallel.DistributedDataParallel,
    model: torch.nn.Module,
    config: TrainingConfig,
    gradient_steps_per_epoch: int,
) -> tuple[
    ZeroRedundancyOptimizer | torch.optim.AdamW,
    CosineAnnealingWarmRestarts | LambdaLR | CosineAnnealingLR,
    torch.amp.GradScaler,
    int,
]:
    params = model_wrapper.parameters()
    if config.use_zero_redundancy and torch.cuda.device_count() > 1:
        optimizer = ZeroRedundancyOptimizer(
            params,
            optimizer_class=torch.optim.AdamW,
            lr=config.lr,
            amsgrad=True,
            weight_decay=config.weight_decay,
            fused=True,
        )
    else:
        optimizer = optim.AdamW(
            params,
            lr=config.lr,
            amsgrad=True,
            weight_decay=config.weight_decay,
            fused=True,
        )

    if config.use_cosine_annealing_with_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=gradient_steps_per_epoch,
            T_mult=2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=gradient_steps_per_epoch * config.epochs,
        )

    if config.load_file is not None and config.continue_failed_training:
        scheduler.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "scheduler_"),
                map_location=config.device,
                weights_only=True,
            ),
        )

    if config.load_file is not None and config.continue_failed_training:
        optimizer.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "optimizer_"),
                map_location=config.device,
                weights_only=True,
            ),
        )

    scaler = torch.amp.GradScaler(
        init_scale=config.grad_scaler_init_scale,
        growth_factor=config.grad_scaler_growth_factor,
        backoff_factor=config.grad_scaler_backoff_factor,
        growth_interval=config.grad_scaler_growth_interval,
        enabled=config.need_grad_scaler,
    )
    if config.load_file is not None and config.continue_failed_training:
        scaler.load_state_dict(
            torch.load(
                config.load_file.replace("model_", "scaler_"),
                map_location=config.device,
                weights_only=True,
            ),
        )

    gradient_steps_skipped = 0
    if config.load_file is not None and config.continue_failed_training:
        gradient_steps_skipped_path = config.load_file.replace(
            "model_",
            "gradient_steps_skipped_",
        ).replace(".pth", ".txt")
        if os.path.exists(gradient_steps_skipped_path):
            with open(gradient_steps_skipped_path) as f:
                gradient_steps_skipped = int(f.read().strip())

    return optimizer, scheduler, scaler, gradient_steps_skipped


@beartype
def initialize_dataloader(
    config: TrainingConfig,
    ssd_cache: dict | diskcache.core.Cache | None,
    num_workers: int,
):
    g_cuda = torch.Generator(device="cpu")
    g_cuda.manual_seed(config.seed)

    datasets, samplers = [], []
    if config.use_carla_data:
        if config.model_type == "plant":
            from lead.plant.plant_dataset import PlantCARLAData

            carla_dataset_cls = PlantCARLAData
        else:
            carla_dataset_cls = CARLAData
        datasets.append(
            carla_dataset_cls(
                root=config.carla_data,
                config=config,
                training_session_cache=ssd_cache,
            ),
        )
        assert not datasets[-1].build_cache and not datasets[-1].build_buckets
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1],
                shuffle=True,
                num_replicas=config.world_size,
                rank=config.rank,
                drop_last=True,
            ),
        )
    if config.use_navsim_data:
        datasets.append(
            NavsimData(
                root=config.navsim_data_root,
                config=config,
                training_session_cache=ssd_cache,
            ),
        )
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1],
                shuffle=True,
                num_replicas=config.world_size,
                rank=config.rank,
                drop_last=True,
            ),
        )
    if config.use_waymo_e2e_data:
        datasets.append(
            WODE2EData(
                root=config.waymo_e2e_training_data_root,
                config=config,
                training_session_cache=ssd_cache,
                training=True,
            ),
        )
        samplers.append(
            torch.utils.data.DistributedSampler(
                datasets[-1],
                shuffle=True,
                num_replicas=config.world_size,
                rank=config.rank,
                drop_last=True,
            ),
        )

    assert len(datasets) > 0, "No datasets selected for training!"

    for ds in datasets:
        LOG.info(f"Dataset size: {len(ds)} samples")

    if config.schedule_carla_num_samples:
        assert config.use_carla_data and config.mixed_data_training
        sample_scheduler = mixed_training_utils.Sim2RealSampleScheduler(
            config,
            datasets,
        )
    else:
        sample_scheduler = mixed_training_utils.UniformSampleScheduler(config, datasets)

    train_dataset = mixed_training_utils.MixedDataset(
        config=config,
        datasets=datasets,
    )

    mixed_sampler = mixed_training_utils.MixedSampler(
        samplers=samplers,
        sample_scheduler=sample_scheduler,
        config=config,
    )

    if config.model_type == "plant":
        from lead.plant.plant_dataset import plant_collate_fn

        collate_fn = plant_collate_fn
    else:
        collate_fn = mixed_training_utils.mixed_data_collate_fn

    dataloader_train = DataLoader(
        train_dataset,
        batch_sampler=mixed_sampler,
        worker_init_fn=seed_worker,
        generator=g_cuda,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    return dataloader_train, mixed_sampler


@beartype
def save_config(config: TrainingConfig, rank: int):
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except (TypeError, OverflowError):
            return False

    if rank == 0 and config.logdir is not None:
        os.makedirs(config.logdir, exist_ok=True)
        json_config = {
            k: v
            for k, v in config.training_dict().items()
            if is_json_serializable(v)
            and not k.startswith("_")
            and not k.endswith("__")
        }
        json_config = json.dumps(json_config, indent=4)
        # LOG.info(json_config)
        with open(os.path.join(config.logdir, "config.json"), "w") as f2:
            f2.write(json_config)


def seed_worker(_):
    # We need to seed the workers individually otherwise random processes in the
    # dataloader return the same values across workers!
    worker_seed = (
        torch.initial_seed()
    ) % 2**32  # this is different across workers, but not gpus when setting config.seed
    rank = int(os.environ.get("RANK", "0"))
    worker_seed = worker_seed + rank * 1000
    # if config.seed is not None, torch.initial_seed is the same across different gpus,
    # so we need to combine it with the rank to get different rng seeds on different gpus.
    # multiply with 1000 because the last digit is already incremented across workers
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def set_start_method():
    # Select how the threads in the data loader are spawned
    # See this: https://stackoverflow.com/a/66113051
    # To edit code while processes run, we generally prefer fork.
    available_start_methods = mp.get_all_start_methods()
    if "fork" in available_start_methods:
        mp.set_start_method("fork", force=True)
    # Available on all OS.
    elif "spawn" in available_start_methods:
        mp.set_start_method("spawn", force=True)
    elif "forkserver" in available_start_methods:
        mp.set_start_method("forkserver", force=True)
