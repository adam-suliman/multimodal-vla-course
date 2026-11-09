import socket
import struct
import threading
import numpy as np
import pytest
from so101_lab.protocol import send, receive, MAX_MESSAGE
from so101_lab.policy import PolicyClient


def test_roundtrip_and_observation_allowlist():
    client_sock, server_sock = socket.socketpair()
    client = PolicyClient.__new__(PolicyClient)
    client.sock = client_sock
    recorded = {}

    def server():
        recorded.update(receive(server_sock))
        send(server_sock, action=np.zeros(6, dtype=np.float32))

    thread = threading.Thread(target=server)
    thread.start()
    try:
        obs = {
            "state": np.arange(6, dtype=np.float32),
            "task": "Move the cube.",
            "images": {
                "top": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist": np.ones((4, 4, 3), dtype=np.uint8),
            },
            "object_position": np.array([1, 2, 3]),
        }
        assert client.action(obs).shape == (6,)
        thread.join()
        assert set(recorded) == {"command", "state", "task", "top", "wrist"}
        assert recorded["task"].item() == "Move the cube."
    finally:
        client_sock.close()
        server_sock.close()


def test_bounded_message():
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!I", MAX_MESSAGE + 1))
        with pytest.raises(ValueError):
            receive(b)
    finally:
        a.close()
        b.close()
