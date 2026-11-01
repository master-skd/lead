"""Live P7 image-to-semantic bridge, independent of CARLA imports."""
import time

import numpy as np

from lead.inference.qwendrive_intent_client import NAV_COMMANDS
from lead.inference.vlm_image_preprocess import carla_bgr_to_vlm_rgb
from lead.tfv6.semantic_intent_slots import parse_service_result, prepare_planner_intents, service_results_to_tensors


class P7ClosedLoopIntent:
    def __init__(self, client, *, capacity=6, jpeg_quality=30):
        self.client = client
        self.capacity = capacity
        self.jpeg_quality = jpeg_quality
        self.last_trace = None

    def prepare(self, image_bgr, command, speed):
        command = np.asarray(command)
        if command.shape != (6,) or not np.isin(command, [0, 1]).all() or command.sum() != 1:
            raise ValueError('P7 closed-loop navigation must be six-way one-hot')
        navigation = NAV_COMMANDS[int(command.argmax())]
        rgb = carla_bgr_to_vlm_rgb(np.asarray(image_bgr, dtype=np.uint8), self.jpeg_quality)
        started = time.monotonic()
        # Transport failures stop the route. Returned malformed/empty model answers
        # use the same explicit nonsemantic query fallback as training.
        results = self.client.generate_rgb(rgb, navigation, float(speed))
        tensors = service_results_to_tensors(results, [navigation], capacity=self.capacity)
        proposal = parse_service_result(results[0])
        packed = prepare_planner_intents(proposal, navigation, capacity=self.capacity)
        self.last_trace = {
            'navigation': navigation, 'speed_mps': float(speed),
            'seconds': time.monotonic() - started, 'jpeg_quality': self.jpeg_quality,
            'parsed': proposal, 'raw_text': results[0].get('raw_text'),
            'analysis': results[0].get('analysis'), 'error': results[0].get('error'),
            'real_valid_branches': sum(packed['valid']), 'rejected': packed['rejected'],
            'slots': packed['slots'], 'fallback': bool(tensors['semantic_intent_fallback'][0]),
        }
        return tensors

    def close(self):
        self.client.close()
