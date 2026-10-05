"""Client for the Qwen-VL feature service (runs in the lead env).

The closed-loop agent (lead env) sends a front-camera RGB image over a Unix domain
socket to vlm_service.py (qwenvl env) and gets back the ``vlm_hidden`` array. No torch
or transformers import here -- only socket + numpy -- so it works in the lead env.

See vlm_service.py for the wire protocol.
"""

from __future__ import annotations

import socket
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
