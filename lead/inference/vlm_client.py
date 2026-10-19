"""Client for the Qwen-VL feature service (runs in the lead env).

The closed-loop agent (lead env) sends a front-camera RGB image over a Unix domain
socket to vlm_service.py (qwenvl env) and gets back the ``vlm_hidden`` array. No torch
or transformers import here -- only socket + numpy -- so it works in the lead env.

See vlm_service.py for the wire protocol.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
import struct

import numpy as np
import numpy.typing as npt


class VLMServiceClient:
    """Thin client that requests vlm_hidden for a front-camera image over a Unix socket."""

    def __init__(self, socket_path: str = "/tmp/vlm_service.sock", timeout: float = 60.0):
        self.socket_path = socket_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(socket_path)
        self.batch_requests = 0

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("vlm_service closed the connection")
            buf.extend(chunk)
        return bytes(buf)

    def extract(self, front_rgb: npt.NDArray) -> npt.NDArray:
        """Send a front-camera (H, W, 3) uint8 RGB image, return (h', w', D) float16.

        Args:
            front_rgb: front-camera image, contiguous uint8 RGB.

        Returns:
            vlm_hidden array as produced by the frozen Qwen-VL, dtype float16.
        """
        img = np.ascontiguousarray(front_rgb, dtype=np.uint8)
        h, w = img.shape[:2]
        self.sock.sendall(struct.pack("<II", h, w) + img.tobytes())

        fh, fw, fd = struct.unpack("<III", self._recv_exact(12))
        data = self._recv_exact(fh * fw * fd * 2)  # float16 = 2 bytes
        return np.frombuffer(data, dtype=np.float16).reshape(fh, fw, fd)

    def extract_batch(self, rgb: npt.NDArray) -> npt.NDArray:
        """Request one batched Qwen forward; input [B,H,W,3], output [B,12,36,D].

        The zero-height marker keeps the existing single-image closed-loop wire
        protocol unchanged. A training service handles this new request type.
        """
        images = np.ascontiguousarray(rgb, dtype=np.uint8)
        if images.ndim != 4 or images.shape[-1] != 3 or not len(images):
            raise ValueError("rgb must have shape [B,H,W,3] with B>0")
        count, h, w, _ = images.shape
        self.batch_requests += 1
        self.sock.sendall(struct.pack("<IIII", 0, count, h, w) + images.tobytes())
        returned, fh, fw, fd = struct.unpack("<IIII", self._recv_exact(16))
        if returned != count:
            raise ValueError(f"VLM service returned {returned} features for {count} images")
        data = self._recv_exact(returned * fh * fw * fd * 2)
        return np.frombuffer(data, dtype=np.float16).reshape(returned, fh, fw, fd)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class OnlineVLMPrefetcher:
    """Keep at most one next-batch VLM request in flight during planner training.

    The socket is owned by the single executor thread, so requests remain ordered.
    Only the next DataLoader batch is retained; no epoch-wide feature cache is made.
    """

    def __init__(self, client: VLMServiceClient):
        self.client = client
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-prefetch")
        self.current_data = None
        self.current_future = None
        self.current_request_index = None

    def _extract(self, data):
        return self.client.extract_batch(data["vlm_rgb"].numpy())

    def advance(self, batches):
        """Load and submit the following batch, or mark the iterator exhausted."""
        try:
            data = next(batches)
        except StopIteration:
            self.current_data = None
            self.current_future = None
            self.current_request_index = None
            return False
        self.current_data = data
        self.current_request_index = self.client.batch_requests
        self.current_future = self.executor.submit(self._extract, data)
        return True

    def result(self):
        if self.current_future is None:
            raise RuntimeError("No VLM batch has been prefetched")
        return self.current_future.result()

    def close(self):
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
