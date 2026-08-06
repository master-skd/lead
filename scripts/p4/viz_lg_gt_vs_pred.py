"""Compare the lane-graph GT label vs the retrained intent decoder's prediction, side by
side on junction frames. GT = data/p6/lanegraph_label cache (what the decoder was trained
to distill); pred = vlm_intent_p6_lanegraph decoder on the 3-cam VLM features.
"""
from __future__ import annotations
import json, os, glob
import numpy as np


def _heat(x):
    h = np.clip(x, 0, 1)
    return (np.stack([np.clip(h*2,0,1), np.where(h<.5,h*2,1-(h-.5)*2), np.zeros_like(h)],-1)*255).astype(np.uint8)


def main():
    import cv2, torch
    from lead.data_loader.carla_dataset import CARLAData
    from lead.data_loader.vlm_intent_dataset import VLMIntentDataset
    from lead.tfv6.vlm_intent_decoder import VLMIntentDecoder
    from lead.training.config_training import TrainingConfig
    import sys; sys.path.insert(0, "scripts/p4")
    from viz_b2_arms import _hdmap_bev

    dev = torch.device("cuda:0")
    cfg = TrainingConfig({"use_multimodal_intent": True}, raise_error_on_missing_key=False)
    dec = VLMIntentDecoder(cfg).to(dev)
    dec.load_state_dict(torch.load("outputs/local_training/vlm_intent_p6_lanegraph/model_0014.pth",
                                   map_location=dev, weights_only=True)["model"])
    dec.eval().requires_grad_(False)

    cds = CARLAData(root="data/carla_leaderboard2/data", config=cfg)
    vds = VLMIntentDataset(cds, vlm_cache_dir="data/p6/vlm_cache_3cam", manifest_path="data/p4/manifest.jsonl")
    cmds = {(json.loads(l)["scenario"], json.loads(l)["route"], json.loads(l)["frame"]): json.loads(l).get("command")
            for l in open("data/p4/manifest.jsonl")}
    os.makedirs("outputs/viz_lg_gt_vs_pred", exist_ok=True)

    def srf(cidx):
        p = str(cds.images[cidx], encoding="utf-8").split("/"); return (p[-4], p[-3], p[-1].split(".")[0])

    saved = 0
    for idx in range(len(vds)):
        if saved >= 10:
            break
        cidx = vds.valid_indices[idx]
        s = srf(cidx)
        if cmds.get(s) in (None, "LANEFOLLOW"):
            continue
        lg_path = os.path.join("data/p6/lanegraph_label", s[0], s[1], s[2] + ".npy")
        if not os.path.exists(lg_path):
            continue
        gt = np.load(lg_path)[0].astype(np.float32)  # (H,W) lane-graph label
        data = vds[idx]
        vh = data["vlm_hidden"].to(dev).float().unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(dec(vh)).cpu().numpy()[0, 0]

        # hdmap background; overlay GT / pred as semi-transparent red so we can judge
        # whether the corridor stays on the actual road.
        hd = _hdmap_bev(str(cds.bev_semantics[cidx], encoding="utf-8"), cfg)
        def overlay(field):
            o = hd.copy().astype(np.float32)
            a = np.clip(field, 0, 1)[..., None]
            o = o * (1 - a) + np.array([0, 0, 255], np.float32) * a  # red overlay
            return o.clip(0, 255).astype(np.uint8)
        # Rotate each tile to forward-UP FIRST, then stack top-to-bottom. Rotating an
        # already-concatenated strip reverses the reading order (CCW sends the rightmost
        # tile to the top), which made the empty hdmap tile look like a dead prediction.
        tiles = [cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE)
                 for t in (hd, overlay(gt), overlay(pred))]
        sep = np.full((2, tiles[0].shape[1], 3), 255, np.uint8)
        panel = np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], axis=0)

        gb, pb = gt > 0.5, pred > 0.5
        iou = (gb & pb).sum() / max((gb | pb).sum(), 1)
        h_t = tiles[0].shape[0]
        for row, tag in ((0, "hdmap"), (h_t + 2, "GT"), (2 * h_t + 4, f"pred IoU={iou:.3f}")):
            cv2.putText(panel, tag, (5, row + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
        cv2.putText(panel, f"idx{idx} {s[0][:16]} cmd={cmds.get(s)} (fwdUP)",
                    (5, panel.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        outp = f"outputs/viz_lg_gt_vs_pred/{saved:02d}_idx{idx}.png"
        cv2.imwrite(outp, panel[..., ::-1])
        saved += 1
        print(f"{outp}  {s[0]}  gt_mean={gt.mean():.3f} pred_mean={pred.mean():.3f} IoU={iou:.3f}")
    print(f"done, {saved} panels in outputs/viz_lg_gt_vs_pred")


if __name__ == "__main__":
    main()
