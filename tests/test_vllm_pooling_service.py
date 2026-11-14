"""CPU checks for the speed-only pooling service's path protocol."""

import socket
import struct

import pytest

from lead.inference.vllm_pooling_service import receive_paths


def test_receive_paths_accepts_images_under_root(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"test")
    sender, receiver = socket.socketpair()
    try:
        value = str(image).encode()
        sender.sendall(struct.pack("<II", 0, 1) + struct.pack("<I", len(value)) + value)
        assert receive_paths(receiver, 1, tmp_path) == [image]
    finally:
        sender.close()
        receiver.close()


def test_receive_paths_rejects_outside_root(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"test")
    sender, receiver = socket.socketpair()
    try:
        value = str(outside).encode()
        sender.sendall(struct.pack("<II", 0, 1) + struct.pack("<I", len(value)) + value)
        with pytest.raises(ValueError, match="outside CARLA root"):
            receive_paths(receiver, 1, media_root)
    finally:
        sender.close()
        receiver.close()
