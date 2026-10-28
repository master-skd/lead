"""Length-prefixed JSON client for the frozen Qwen-Drive HF intent service."""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Sequence


NAV_COMMANDS = (
    "turn_left", "turn_right", "straight", "follow_lane",
    "change_lane_left", "change_lane_right",
)
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _receive_exact(sock: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        part = sock.recv(count - len(chunks))
        if not part:
            raise ConnectionError("Qwen-Drive service closed the connection")
        chunks.extend(part)
    return bytes(chunks)


def _send_json(sock: socket.socket, value: object) -> None:
    body = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError("Qwen-Drive request exceeds size limit")
    sock.sendall(struct.pack("<I", len(body)) + body)


def _receive_json(sock: socket.socket) -> object:
    length = struct.unpack("<I", _receive_exact(sock, 4))[0]
    if not 0 < length <= MAX_MESSAGE_BYTES:
        raise ValueError("invalid Qwen-Drive response length")
    return json.loads(_receive_exact(sock, length))


class QwenDriveIntentClient:
    def __init__(self, socket_path: str, *, timeout: float = 600.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(socket_path)
        self.requests = 0

    def generate_batch(
        self, paths: Sequence[str], commands: Sequence[str], speeds: Sequence[float],
    ) -> list[dict]:
        if not (len(paths) == len(commands) == len(speeds) and 0 < len(paths) <= 256):
            raise ValueError("Qwen-Drive batch lengths must agree and be within 1..256")
        if any(command not in NAV_COMMANDS for command in commands):
            raise ValueError("unknown navigation command")
        request = [
            {"image": str(path), "command": command, "speed": float(speed)}
            for path, command, speed in zip(paths, commands, speeds, strict=True)
        ]
        _send_json(self.sock, {"samples": request})
        response = _receive_json(self.sock)
        self.requests += 1
        if not isinstance(response, dict) or not isinstance(response.get("samples"), list):
            raise ValueError("malformed Qwen-Drive response")
        results = response["samples"]
        if len(results) != len(request) or any(not isinstance(item, dict) for item in results):
            raise ValueError("Qwen-Drive returned a different number of samples")
        return results

    def close(self) -> None:
        self.sock.close()
