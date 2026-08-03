"""P5 B2a: extract K directional arms (anchors) from a (predicted) drivable-blob intent.

The P5a/P6 intent head emits a single-channel drivable-support blob. To let the planner
emit K distinct route arms (B2), we derive K anchors -- one per feasible direction --
geometrically from the blob (no lane graph; works on the predicted blob at closed-loop
time, including P6's side arms).

Method B (skeleton/endpoint), replaces the earlier ego-ray fan which was fragile when the
blob's main road is lateral or the ego isn't on the blob's centreline (e.g. a left-turn
junction whose through-road runs left-right):
  1. Binarize the blob (BIN_THRESH); keep the connected component containing ego.
  2. Skeletonize the component; find skeleton ENDPOINTS (degree-1 pixels) -- the tips of
     each protruding arm, in any direction.
  3. Keep endpoints far enough from ego (geodesic/euclidean > REACH_MIN_M). Merge endpoints
     whose direction-from-ego is within MERGE_ANGLE_DEG (same arm split by skeleton noise).
  4. One anchor per surviving endpoint: angle = atan2(dir from ego), reach = distance.
     Keep the K_MAX arms with largest reach.

Ego/orientation fixed by the BEV convention (matches rasterize_reachable_support):
  ego at (row=160, col=128); forward = +col; lateral y = +row (right). 0 rad = straight
  forward; +angle -> +row (right), -angle -> -row (left). Angles span full [-pi, pi]
  (side/rear arms allowed -- P6 lights them up).
"""

from __future__ import annotations

import numpy as np

K_MAX = 6
BIN_THRESH = 0.5
REACH_MIN_M = 6.0        # endpoint must be at least this far from ego (metres)
MERGE_ANGLE_DEG = 20.0   # endpoints within this angular spread (from ego) merge to one arm


def _skeleton_endpoints(mask):
    """Return (row, col) of skeleton endpoints of a binary mask."""
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    # endpoint = skeleton pixel with exactly 1 skeleton neighbour (8-conn)
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    resp = convolve(sk.astype(int), kernel, mode="constant")
    # skeleton pixel (center contributes 10) with exactly 1 neighbour -> resp == 11
    eps = np.argwhere((resp == 11))
    return eps, sk


def extract_anchors_from_blob(
    blob: np.ndarray,
    ppm: float,
    row_ego: int,
    col_ego: int,
    horizon_m: float = 30.0,
    k_max: int = K_MAX,
    bin_thresh: float = BIN_THRESH,
    reach_min_m: float = REACH_MIN_M,
    merge_angle_deg: float = MERGE_ANGLE_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract up to k_max arms via skeleton-endpoint detection.

    Returns:
        anchors:   (k_max, 3) float32 -> [valid(0/1), angle_rad, reach_m]
        endpoints: (k_max, 2) float32 -> [row, col] pixel of the arm tip
    """
    from scipy.ndimage import label as cc_label

    H, W = blob.shape
    mask = blob > bin_thresh

    # keep the connected component containing ego (fall back to nearest lit pixel)
    lab, n = cc_label(mask, structure=np.ones((3, 3)))
    ego_lab = lab[row_ego, col_ego]
    if ego_lab == 0:
        win = lab[max(0, row_ego - 8): row_ego + 9, max(0, col_ego - 8): col_ego + 9]
        nz = win[win > 0]
        ego_lab = int(nz[0]) if nz.size else 0
    comp = (lab == ego_lab) if ego_lab > 0 else mask

    anchors = np.zeros((k_max, 3), dtype=np.float32)
    endpoints = np.zeros((k_max, 2), dtype=np.float32)
    if comp.sum() < 5:
        return anchors, endpoints

    eps, _ = _skeleton_endpoints(comp)
    if len(eps) == 0:
        return anchors, endpoints

    # candidate arms from endpoints: (reach_m, angle, (row,col))
    cands = []
    for (r, c) in eps:
        dr, dc = r - row_ego, c - col_ego
        reach_m = np.hypot(dr, dc) / ppm
        if reach_m < reach_min_m:
            continue
        ang = np.arctan2(dr, dc)  # +col forward=0; +row (right)=+; -row (left)=-
        cands.append((reach_m, float(ang), (int(r), int(c))))
    if not cands:
        return anchors, endpoints

    # merge candidates whose angle-from-ego is within merge_angle_deg (same arm); keep farthest
    cands.sort(key=lambda x: -x[0])  # by reach desc
    merge_rad = np.deg2rad(merge_angle_deg)
    arms = []
    for rm, ang, tip in cands:
        if any(abs(np.angle(np.exp(1j * (ang - a)))) < merge_rad for _, a, _ in arms):
            continue
        arms.append((rm, ang, tip))
        if len(arms) >= k_max:
            break

    for i, (rm, ang, tip) in enumerate(arms):
        anchors[i] = [1.0, ang, rm]
        endpoints[i] = tip
    return anchors, endpoints
