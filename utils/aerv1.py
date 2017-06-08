"""Read an AERv1 file."""

import struct

import numpy as np

EVENT_STRUCT = struct.Struct(">HI")


def read(f):
    """Read an AER file."""
    while True:
        event = f.read(6)
        if len(event) < 6:
            break

        address, timestamp = EVENT_STRUCT.unpack(event)

        x = (address & 0xFE) >> 1
        y = address >> 8
        polarity = address & 0x1

        yield ("DVS", timestamp, (x, y, polarity))
