"""Read an AERv1 file."""

import struct

import numpy as np


def read(path):
    """Read an AER file."""
    with open(path, "rb") as f:
        # Skip the header
        f.readline()

        # Read data bytes
        data = f.read()

        # Number of events (each event is 6 bytes)
        n = len(data) // 6

        # Read events into a data matrix
        X = np.empty((n, 4), dtype=np.int)
        for i in range(n):
            address, timestamp = struct.unpack_from(">HI", data, i * 6)

            x = (address & 0xFE) >> 1
            y = address >> 8
            parity = address & 0x1

            X[i, :] = [timestamp, x, y, parity]

        return X
