"""Differentiable collision cost for the reactive control head (P2.2).

Penalizes predicted waypoints that pass through occupied BEV cells (vehicles,
pedestrians, obstacles, ...). The occupancy is blurred into a soft "danger
field" so ``grid_sample`` yields a smooth gradient that pushes waypoints away
from obstacles (a binary map would only have gradient exactly at the edge).

This is an OPEN-LOOP soft prior (obstacles come from the recorded BEV semantic),
NOT true reactive safety -- see P2 doc §5. Self-contained; run
``python -m lead.tfv6.collision_cost`` for a gradient smoke test (no data).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from beartype import beartype

from lead.common.constants import TransfuserBEVSemanticClass as C
from lead.training.config_training import TrainingConfig

# BEV-semantic class ids that count as physical obstacles for path collision.
OBSTACLE_CLASSES: tuple[int, ...] = (
    int(C.VEHICLE),
    int(C.WALKER),
    int(C.OBSTACLE),
    int(C.PARKING_VEHICLE),
    int(C.SPECIAL_VEHICLE),
    int(C.BIKER),
)


@beartype
def occupancy_from_bev_semantic(bev_semantic: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` class ids -> ``(B, 1, H, W)`` binary obstacle occupancy."""
    if bev_semantic.dim() == 4:  # (B,1,H,W) -> (B,H,W)
        bev_semantic = bev_semantic[:, 0]
    occ = torch.zeros_like(bev_semantic, dtype=torch.float32)
    for cls in OBSTACLE_CLASSES:
        occ = occ + (bev_semantic == cls).float()
    return occ.clamp(0.0, 1.0).unsqueeze(1)  # (B, 1, H, W)


def _gaussian_kernel(sigma_px: float, device, dtype) -> torch.Tensor:
    radius = max(int(3 * sigma_px), 1)
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k1 = torch.exp(-(xs**2) / (2 * sigma_px**2))
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1)  # (K, K)
    return k2.to(dtype).view(1, 1, *k2.shape)


@beartype
def soft_danger_field(
    occ: torch.Tensor,
    config: TrainingConfig,
    sigma_m: float = 2.0,
) -> torch.Tensor:
    """Blur binary occupancy into a smooth danger field in ``[0, 1]``.

    High near obstacles, decaying outward -> gives waypoints a gradient ramp.
    """
    sigma_px = max(sigma_m * config.pixels_per_meter, 1.0)
    kernel = _gaussian_kernel(sigma_px, occ.device, occ.dtype)
    pad = kernel.shape[-1] // 2
    danger = F.conv2d(occ, kernel, padding=pad)
    # normalize so a lone obstacle cell peaks near 1
    peak = danger.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return (danger / peak).clamp(0.0, 1.0)


@beartype
def waypoints_to_grid(waypoints: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """``(B, N, 2)`` ego metres (x=long, y=lat) -> ``(B, N, 1, 2)`` grid_sample coords in [-1,1].

    Uses the same BEV convention as the intent rasterizer: x->width, y->height.
    """
    ppm = config.pixels_per_meter
    w = config.lidar_width_pixel
    h = config.lidar_height_pixel
    col = (waypoints[..., 0] - config.min_x_meter) * ppm  # [0, W] along width
    row = (waypoints[..., 1] - config.min_y_meter) * ppm  # [0, H] along height
    gx = col / (w - 1) * 2.0 - 1.0
    gy = row / (h - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (B, N, 2)
    return grid.unsqueeze(2)  # (B, N, 1, 2)


@beartype
def differentiable_collision(
    waypoints: torch.Tensor,  # (B, N, 2) ego metres, requires_grad for training
    bev_semantic: torch.Tensor,  # (B, H, W) or (B,1,H,W) class ids
    config: TrainingConfig,
    sigma_m: float = 2.0,
) -> torch.Tensor:
    """Mean danger-field value sampled at the predicted waypoints (scalar loss).

    Differentiable w.r.t. ``waypoints`` -> gradient pushes the path off obstacles.
    """
    occ = occupancy_from_bev_semantic(bev_semantic)
    danger = soft_danger_field(occ, config, sigma_m=sigma_m).to(waypoints.dtype)
    grid = waypoints_to_grid(waypoints, config).to(waypoints.dtype)
    sampled = F.grid_sample(
        danger, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (B, 1, N, 1)
    return sampled.mean()


@beartype
def collision_cost_per_mode(
    routes: torch.Tensor,  # (B, K, N, 2) ego metres -- K route arms per sample
    bev_semantic: torch.Tensor,  # (B, H, W) or (B,1,H,W) class ids
    config: TrainingConfig,
    sigma_m: float = 2.0,
) -> torch.Tensor:
    """Per-arm mean danger value -> ``(B, K)``, differentiable w.r.t. ``routes``.

    Same danger field as :func:`differentiable_collision` but WITHOUT reducing over the
    mode axis, so the caller can mask out padded arms before averaging (B2: only valid
    arms should contribute) and can rank arms by cost at inference for late resolution.
    The danger field is built once per sample and shared across that sample's K arms.
    """
    b, k, n, _ = routes.shape
    occ = occupancy_from_bev_semantic(bev_semantic)
    danger = soft_danger_field(occ, config, sigma_m=sigma_m).to(routes.dtype)  # (B,1,H,W)
    # flatten the mode axis into the batch axis, repeating each sample's field K times
    grid = waypoints_to_grid(routes.reshape(b * k, n, 2), config).to(routes.dtype)
    danger_rep = danger.repeat_interleave(k, dim=0)  # (B*K,1,H,W)
    sampled = F.grid_sample(
        danger_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (B*K, 1, N, 1)
    return sampled.reshape(b, k, n).mean(dim=2)  # (B,K)


@beartype
def _clipped_distance_field(
    corridor: torch.Tensor,  # (B,1,H,W) in [0,1]
    reach_px: int,
    stride: int = 4,
) -> torch.Tensor:
    """``(B,1,H,W)`` normalised distance OUTSIDE the corridor: 0 inside, 1 at ``reach_px``.

    Why not a gaussian blur of ``1 - corridor`` (the obvious choice, and what this function
    replaces): no single sigma works. Measured on 200 junction frames, sigma=1.5 m charged the
    GT EXPERT ROUTE 0.349 -- the term would have fought expert fidelity -- while shrinking it
    to keep GT free puts any arm more than a metre off-road back on a flat plateau with zero
    gradient, which is exactly the arm the term exists to pull back.

    A distance ramp has no such tradeoff: flat zero inside the corridor (GT pays nothing,
    however the label wobbles), and a constant-slope ramp outside. ``reach_px`` must be long
    enough to still be ramping where the bad arms actually are -- B2-final's non-winner arms
    sit >10 m off-road, so an 8 m reach saturates and gives them nothing.

    Built by iterated dilation (a clipped Chebyshev distance) at ``1/stride`` resolution, then
    upsampled: a 20 m reach is 80 px at 4 px/m, and 40 full-res dilations of a (B,1,320,384)
    tensor every step is far too slow. Downsampling first costs ~1 m of ramp precision, which
    is irrelevant for a constraint this deliberately loose. The label needs no gradient, so a
    non-differentiable transform is fine; ``grid_sample`` supplies the gradient w.r.t. routes.
    """
    inside = (corridor.clamp(0.0, 1.0) > 0.5).to(corridor.dtype)
    h, w = inside.shape[-2:]
    # max_pool for the downsample: it GROWS the corridor slightly, so the field errs toward
    # charging less -- the safe direction for a term that must never fight the expert.
    lr = F.max_pool2d(inside, kernel_size=stride, stride=stride)
    # kernel 3 (radius 1), NOT 5: a radius-2 dilation reaches low-res distances 1 and 2 on the
    # same iteration, so they share a value and the field gets plateaus `2*stride` px wide --
    # every route point landing in one receives exactly zero gradient (verified: a point 15 m
    # off-road read the correct 0.5 but had no gradient). Radius 1 advances one low-res cell
    # per iteration, giving a strictly monotone ramp.
    n_iter = max(reach_px // stride, 1)
    dist = torch.zeros_like(lr)
    cur = lr
    for _ in range(n_iter):
        cur = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
        dist = dist + (1.0 - cur)  # cells still outside after this dilation
    dist = (dist / n_iter).clamp(0.0, 1.0)
    return F.interpolate(dist, size=(h, w), mode="bilinear", align_corners=True)


@beartype
def corridor_cost_per_mode(
    routes: torch.Tensor,  # (B, K, N, 2) ego metres -- K route arms per sample
    corridor: torch.Tensor,  # (B, 1, H, W) or (B, H, W) lane-graph corridor in [0,1]
    config: TrainingConfig,
    reach_m: float = 20.0,
) -> torch.Tensor:
    """Per-arm mean OFF-corridor distance -> ``(B, K)``, differentiable w.r.t. ``routes``.

    Complements :func:`collision_cost_per_mode`, which cannot express this: its danger
    field is built from ``OBSTACLE_CLASSES`` (vehicles / walkers / ...) only, so an arm that
    leaves the road entirely -- onto grass, across oncoming lanes, through a curb -- costs
    exactly zero. Measured on B2-final's own arms, obstacle cost was actually INVERTED
    (non-winner 0.014 vs winner 0.067): it rates off-road arms as the safer ones. That is how
    B2-final's non-winner arms went off-road with every loss still falling -- the anchor
    hinges constrain BEARING and REACH, and nothing constrained "stay on a drivable branch".

    The lane-graph corridor label supplies the missing shape. It is deliberately a WEAK
    constraint -- the union of all L/S/R arms dilated to lane width, not a single trajectory --
    so an arm may curve freely, pick any branch, and overshoot; it just cannot leave the road.
    That keeps intent a fuzzy "which way" and leaves the actual path for the planner to resolve
    late, which an L1-to-centreline would destroy.

    Cost is the clipped distance outside the corridor, averaged along each arm: exactly 0 while
    the arm stays on any branch, ramping to 1 at ``reach_m`` metres off-road. See
    :func:`_clipped_distance_field` for why this is a ramp and not a blur.
    """
    b, k, n, _ = routes.shape
    if corridor.dim() == 3:  # (B,H,W) -> (B,1,H,W)
        corridor = corridor.unsqueeze(1)
    reach_px = max(int(reach_m * config.pixels_per_meter), 1)
    field = _clipped_distance_field(corridor.to(routes.dtype), reach_px)  # (B,1,H,W)
    grid = waypoints_to_grid(routes.reshape(b * k, n, 2), config).to(routes.dtype)
    # padding_mode="border": a point sampled outside the BEV window keeps the edge value
    # instead of reading 0, which would otherwise reward arms for leaving the grid entirely.
    sampled = F.grid_sample(
        field.repeat_interleave(k, dim=0), grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )  # (B*K, 1, N, 1)
    return sampled.reshape(b, k, n).mean(dim=2)  # (B,K)


def _smoke_test() -> None:
    torch.manual_seed(0)
    config = TrainingConfig()
    h, w = config.lidar_height_pixel, config.lidar_width_pixel
    b = 2
    # fake BEV semantic with a VEHICLE block ahead of ego
    sem = torch.zeros(b, h, w, dtype=torch.long)
    sem[:, 150:180, 150:190] = int(C.VEHICLE)
    # waypoints that walk forward through the block (ego metres)
    wp = torch.zeros(b, 8, 2)
    wp[:, :, 0] = torch.linspace(2.0, 12.0, 8)  # x forward
    wp.requires_grad_(True)

    loss = differentiable_collision(wp, sem, config)
    loss.backward()
    print("collision loss:", float(loss))
    print("waypoints grad norm:", float(wp.grad.norm()))
    print("occ obstacle cells:", int(occupancy_from_bev_semantic(sem).sum()))
    assert wp.grad is not None and wp.grad.norm() > 0, "no gradient to waypoints!"
    print("COLLISION COST OK (gradient flows to waypoints)")

    # per-mode (B2): arm 0 drives through the block, arm 1 detours around it -> arm 1
    # must score strictly lower, and each arm must get its own gradient.
    routes = torch.zeros(b, 2, 8, 2)
    routes[:, :, :, 0] = torch.linspace(2.0, 12.0, 8)
    routes[:, 1, :, 1] = 8.0  # arm 1 offset laterally, clear of the block
    routes.requires_grad_(True)
    per_mode = collision_cost_per_mode(routes, sem, config)
    per_mode.sum().backward()
    print("per-mode cost:", per_mode[0].tolist())
    assert per_mode.shape == (b, 2), f"expected (B,K), got {tuple(per_mode.shape)}"
    assert per_mode[0, 1] < per_mode[0, 0], "detour arm should be cheaper!"
    assert routes.grad[0, 0].norm() > 0, "no gradient to the colliding arm!"
    print("PER-MODE COLLISION OK (arms scored independently, gradient per arm)")

    # corridor (B2 term 5): a straight-ahead corridor at CARLA lane width (~3.5 m half-width
    # once the lane-graph dilation is applied). Arm 0 follows it, arm 1 veers off-road. The
    # obstacle-based collision cost CANNOT tell these apart (no vehicle out there), which is
    # exactly why this term exists -- so assert both.
    h, w = config.lidar_height_pixel, config.lidar_width_pixel
    ppm = config.pixels_per_meter
    corridor = torch.zeros(b, 1, h, w)
    row_ego = int((0 - config.min_y_meter) * ppm)
    col_ego = int((0 - config.min_x_meter) * ppm)
    half = int(3.5 * ppm)
    corridor[:, :, row_ego - half : row_ego + half, col_ego:] = 1.0

    arms = torch.zeros(b, 2, 8, 2)
    arms[:, :, :, 0] = torch.linspace(2.0, 20.0, 8)  # both drive forward
    arms[:, 1, :, 1] = torch.linspace(0.0, 12.0, 8)  # arm 1 drifts laterally off-corridor
    arms.requires_grad_(True)
    corr = corridor_cost_per_mode(arms, corridor, config)
    corr.sum().backward()
    print("corridor cost:", [round(v, 4) for v in corr[0].tolist()])
    assert corr.shape == (b, 2), f"expected (B,K), got {tuple(corr.shape)}"
    # the on-corridor arm must be EXACTLY free: measured on real data, a gaussian-blurred
    # field charged the GT expert route 0.349, i.e. the term would fight expert fidelity.
    assert corr[0, 0] < 1e-6, f"on-corridor arm must be free, got {corr[0,0]:.4f}"
    # Values are naturally SMALLER than the old saturating blur field: this is a linear ramp
    # over reach_m, so an arm averaging d metres outside the corridor reads ~d/reach_m. This
    # test arm averages ~3 m out over a 20 m ramp -> ~0.15, not the ~0.9 a blur gave.
    # Deliberate: cost now scales with HOW FAR off-road rather than "off-road at all", which
    # is also why the loss weight is not comparable to the blur version's.
    assert corr[0, 1] > 0.05, f"off-road arm must cost more, got {corr[0,1]:.4f}"
    assert arms.grad[0, 1].norm() > 0, "no gradient to the off-road arm!"

    # the property a blur cannot provide: an arm ALREADY FAR off-road (well past the
    # corridor edge, where 1-corridor is a flat plateau) must still receive gradient,
    # since those are precisely the arms this term must pull back.
    far = torch.zeros(b, 2, 8, 2)
    far[:, :, :, 0] = torch.linspace(2.0, 20.0, 8)
    far[:, :, :, 1] = 15.0  # ~11 m clear of the 3.5 m corridor edge
    far.requires_grad_(True)
    corridor_cost_per_mode(far, corridor, config).sum().backward()
    print("far-off-road cost:", round(float(corridor_cost_per_mode(far.detach(), corridor, config)[0, 0]), 4),
          " grad norm:", round(float(far.grad[0, 0].norm()), 6))
    assert far.grad[0, 0].norm() > 0, "far off-road arm has NO gradient -- cannot be pulled back!"

    # the blind spot this term covers: obstacle collision is ~equal for both arms
    coll = collision_cost_per_mode(arms.detach(), sem, config)
    print("obstacle-only cost for the same arms:", [round(v, 4) for v in coll[0].tolist()],
          "<- cannot distinguish off-road, hence the corridor term")
    print("PER-MODE CORRIDOR OK (off-road arms penalised, gradient reaches far arms)")


if __name__ == "__main__":
    _smoke_test()
