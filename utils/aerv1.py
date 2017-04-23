"""Read an AERv1 file."""

import struct

import numpy as np

EVENT_STRUCT = struct.Struct(">HI")


def read(aedat_file):
    """Read an AER file."""
    with open(aedat_file, "rb") as f:
        version_line = f.readline()

        if version_line != b"#!AER-DAT1.0\r\n":
            raise Exception("Can only read version 1.0")

        # Discard comment lines
        while True:
            byte = f.peek(1)

            if len(byte) < 1:
                raise Exception("Could not read from file")

            if byte[0:1] == b"#":
                f.readline()
            else:
                break

        while True:
            event = f.read(6)
            if len(event) < 6:
                break

            address, timestamp = EVENT_STRUCT.unpack(event)

            x = (address & 0xFE) >> 1
            y = address >> 8
            parity = address & 0x1

            yield ("DVS", timestamp, (x, y, parity))
