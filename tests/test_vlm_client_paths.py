import socket
import struct
import threading

import numpy as np

from lead.inference.vlm_client import OnlineVLMPrefetcher, VLMServiceClient


def _read_exact(conn, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunks.extend(conn.recv(count - len(chunks)))
    return bytes(chunks)


def test_path_batch_wire_protocol(tmp_path):
    socket_path = tmp_path / "vlm.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    observed = []

    def serve():
        with server.accept()[0] as conn:
            zero, count, height, width = struct.unpack("<IIII", _read_exact(conn, 16))
            assert (zero, count, height, width) == (0, 2, 0, 1)
            for _ in range(count):
                length, = struct.unpack("<I", _read_exact(conn, 4))
                observed.append(_read_exact(conn, length).decode("utf-8"))
            values = np.zeros((2, 12, 36, 2560), dtype=np.float16)
            conn.sendall(struct.pack("<IIII", 2, 12, 36, 2560) + values.tobytes())

    worker = threading.Thread(target=serve)
    worker.start()
    client = VLMServiceClient(str(socket_path))
    try:
        result = client.extract_paths(["/data/a.jpg", "/data/b.jpg"])
    finally:
        client.close()
        worker.join(timeout=5)
        server.close()
    assert observed == ["/data/a.jpg", "/data/b.jpg"]
    assert result.shape == (2, 12, 36, 2560)
    assert result.dtype == np.float16
    assert client.batch_requests == 1


def test_online_vlm_prefetch_keeps_one_ordered_request_in_flight():
    entered = threading.Event()
    release = threading.Event()

    class Client:
        batch_requests = 0

        def extract_paths(self, paths):
            self.batch_requests += 1
            if self.batch_requests == 2:
                entered.set()
                assert release.wait(timeout=5)
            return np.array([int(paths[0])], dtype=np.float16)

    client = Client()
    batches = iter([{"vlm_path": [str(i)]} for i in range(3)])
    with OnlineVLMPrefetcher(client, use_paths=True) as prefetch:
        assert prefetch.advance(batches)
        assert prefetch.current_request_index == 0
        np.testing.assert_array_equal(prefetch.result(), [0])
        assert prefetch.advance(batches)
        assert prefetch.current_request_index == 1
        assert entered.wait(timeout=5)  # Batch 2 runs while planner uses batch 1.
        release.set()
        np.testing.assert_array_equal(prefetch.result(), [1])
        assert prefetch.advance(batches)
        assert prefetch.current_request_index == 2
        np.testing.assert_array_equal(prefetch.result(), [2])
        assert not prefetch.advance(batches)
    assert client.batch_requests == 3


def test_online_vlm_prefetch_preserves_rgb_batch_and_propagates_errors():
    class Client:
        batch_requests = 0

        def extract_batch(self, rgb):
            self.batch_requests += 1
            assert rgb.shape == (2, 4, 5, 3)
            raise ValueError("service failure")

    class RGB:
        def numpy(self):
            return np.zeros((2, 4, 5, 3), dtype=np.uint8)

    with OnlineVLMPrefetcher(Client()) as prefetch:
        assert prefetch.advance(iter([{"vlm_rgb": RGB()}]))
        try:
            prefetch.result()
        except ValueError as exc:
            assert str(exc) == "service failure"
        else:
            raise AssertionError("VLM worker errors must reach the training rank")
