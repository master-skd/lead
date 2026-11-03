"""Live same-frame Qwen hidden-state conditioning, without JSON generation."""
import time
import numpy as np
import torch
from lead.inference.qwendrive_intent_client import NAV_COMMANDS
from lead.inference.vlm_image_preprocess import carla_bgr_to_vlm_rgb


class P8ClosedLoopIntent:
    def __init__(self, client, jpeg_quality=30):
        self.client = client
        self.jpeg_quality = jpeg_quality
        self.last_trace = None

    def prepare(self, image_bgr, command, speed, frame_id, timestamp=None):
        command = np.asarray(command)
        if command.shape != (6,) or not np.isin(command, [0, 1]).all() or command.sum() != 1:
            raise ValueError('hidden planner requires one-hot navigation')
        rgb = carla_bgr_to_vlm_rgb(np.asarray(image_bgr, dtype=np.uint8), self.jpeg_quality)
        start = time.monotonic()
        arrays = self.client.extract_rgb(rgb, NAV_COMMANDS[int(command.argmax())], speed, frame_id, timestamp=timestamp)
        self.last_trace = dict(frame_id=str(frame_id), roundtrip_seconds=time.monotonic()-start,
                               **self.client.last_metrics)
        return {key: torch.from_numpy(value) for key, value in arrays.items()}

    def validate_prediction(self, prediction):
        """Fail explicitly if a future selector unpairs a route and its speed."""
        routes=prediction.pred_route_multimodal
        speeds=prediction.pred_target_speed_multimodal_scalar
        logits=prediction.pred_target_speed_multimodal_distribution
        conf=prediction.pred_route_conf
        selected=prediction.pred_route_selected_idx
        if any(value is None for value in (routes,speeds,logits,conf,selected)):
            raise RuntimeError('P8 prediction is missing paired route/speed candidates')
        if routes.shape[0]!=1 or conf.shape!=routes.shape[:2] or speeds.shape!=conf.shape or logits.shape[:2]!=conf.shape:
            raise RuntimeError('P8 closed loop expects one frame with K paired candidates')
        pick=int(selected.reshape(-1)[0])
        if pick!=int(conf.argmax(-1)[0]):
            raise RuntimeError('P8 selected mode differs from route confidence argmax')
        for name,actual,expected in (
            ('route',prediction.pred_route,routes[:,pick]),
            ('speed',prediction.pred_target_speed_scalar,speeds[:,pick]),
            ('speed logits',prediction.pred_target_speed_distribution,logits[:,pick]),
        ):
            if actual is None or actual.numel()!=expected.numel() or not torch.allclose(
                    actual.float().reshape(-1),expected.float().reshape(-1),atol=1e-5,rtol=1e-5):
                raise RuntimeError(f'P8 selected {name} does not belong to mode {pick}')
        self.last_trace.update(selected_idx=pick,selection='route_confidence',
            route_speed_pair_verified=True,candidate_speeds=speeds.detach().float().cpu().tolist(),
            route_conf_logits=conf.detach().float().cpu().tolist())

    def close(self):
        self.client.close()
