"""Read an AER file."""

import re

import aerv1
import aerv2

VERSION_PATTERN = re.compile(b"#!AER-DAT([0-9.]+)\r\n")

def read(aedat_file, chip_class):
    """
    Arguments
    ---------
    aedat_file: Path to .aedat file
    chip_class: Either "DAVIS" or "DVS"
    """
    with open(aedat_file, "rb") as f:
        version_line = f.readline()

        match = VERSION_PATTERN.match(version_line)

        if not match:
            raise Exception("{} is not a valid AER file".format(aedat_file))

        # Discard comment lines
        while True:
            byte = f.peek(1)

            if len(byte) < 1:
                raise Exception("Could not read from file")

            if byte[0:1] == b"#":
                f.readline()
            else:
                break

        version = match.group(1)
        if version == b"1.0":
            events = aerv1.read(f)
        elif version == b"2.0":
            events = aerv2.read(f, chip_class)
        else:
            raise Exception("AER version {} is not supported".format(version))

        # Do it this way to keep file handle f open
        for e in events:
            yield e
