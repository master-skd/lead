"""Same-frame binary feature transport over Unix sockets; no feature files."""
import base64
import io
import socket
import struct
import numpy as np
from PIL import Image
from lead.inference.qwendrive_intent_client import _send_json, _receive_json, _receive_exact

MAX_FEATURE_BYTES = 512 * 1024**2


def send_features(sock, arrays, metadata):
    body = io.BytesIO()
    np.savez(body, **arrays)  # memory buffer, never a disk file; no pickle
    payload = body.getvalue()
    if len(payload) > MAX_FEATURE_BYTES:
        raise ValueError('feature response exceeds 512 MiB; reduce request batch')
    _send_json(sock, metadata)
    sock.sendall(struct.pack('<I', len(payload)))
    sock.sendall(payload)


class QwenDriveHiddenClient:
    def __init__(self, socket_path, timeout=600):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(str(socket_path))
        self.requests = 0
        self.last_metrics = None

    def _request(self, samples):
        if not 0 < len(samples) <= 256:
            raise ValueError('hidden-state batch must have 1..256 samples')
        for sample in samples:
            sample.setdefault('timestamp', None)
        request = {'request_id': self.requests, 'samples': samples}
        _send_json(self.sock, request)
        meta = _receive_json(self.sock)
        expected = [{key: item[key] for key in ('frame_id', 'command', 'speed', 'timestamp')} for item in samples]
        if meta.get('request_id') != self.requests or meta.get('frames') != expected:
            raise ValueError('Qwen hidden frame/navigation/speed alignment mismatch')
        length = struct.unpack('<I', _receive_exact(self.sock, 4))[0]
        if not 0 < length <= MAX_FEATURE_BYTES:
            raise ValueError('invalid hidden response size')
        with np.load(io.BytesIO(_receive_exact(self.sock, length)), allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in ('qwen_hidden', 'qwen_token_mask', 'qwen_view_ids')}
        hidden, mask, views = arrays.values()
        if (hidden.ndim != 3 or hidden.shape[0] != len(samples) or hidden.dtype != np.float16
                or mask.shape != hidden.shape[:2] or mask.dtype != bool or views.shape != mask.shape
                or not mask.any(1).all() or not np.isfinite(hidden).all()):
            raise ValueError('malformed hidden-state response')
        self.last_metrics = meta
        self.requests += 1
        return arrays

    def generate_batch(self, paths, commands, speeds):
        if not len(paths) == len(commands) == len(speeds):
            raise ValueError('hidden request lengths differ')
        return self._request([dict(image=str(path), frame_id=str(path), command=command, speed=float(speed))
                              for path, command, speed in zip(paths, commands, speeds, strict=True)])

    def extract_rgb(self, rgb, command, speed, frame_id, timestamp=None):
        if rgb.shape != (384, 1152, 3) or rgb.dtype != np.uint8:
            raise ValueError('expected 384x1152 uint8 RGB strip')
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format='PNG')
        return self._request([dict(image_png=base64.b64encode(stream.getvalue()).decode('ascii'),
                                   frame_id=str(frame_id), command=command, speed=float(speed), timestamp=timestamp)])

    def close(self):
        self.sock.close()
