"""Size-bounded, pickle-free messages over a private Unix domain socket."""

import io
import struct
import numpy as np

MAX_MESSAGE = 16 * 1024 * 1024


def send(sock, **values):
    stream = io.BytesIO()
    np.savez(stream, **values)
    data = stream.getvalue()
    if len(data) > MAX_MESSAGE:
        raise ValueError("IPC message exceeds 16 MiB.")
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive(sock):
    def exact(n):
        chunks = bytearray()
        while len(chunks) < n:
            block = sock.recv(n - len(chunks))
            if not block:
                raise EOFError("Policy process closed its connection.")
            chunks.extend(block)
        return bytes(chunks)

    (size,) = struct.unpack("!I", exact(4))
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid IPC message length.")
    with np.load(io.BytesIO(exact(size)), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
