import struct

import numpy as np

EVENT_STRUCT = struct.Struct(">II")
TYPE_MASK = (0x1 << 31, 31)
Y_COORD_MASK = (0x1FF << 22, 22)
X_COORD_MASK = (0x1FF << 12, 12)
PARITY_MASK = (0x1 << 11, 11)
READ_EXT_MASK = (0x1 << 10, 10)
ADC_MASK = (0x3FF, 0)

TYPE_ADS = 0x1
TYPE_DVS = 0x0


def read(aedat_file):
    with open(aedat_file, "rb") as f:
        version_line = f.readline()

        if version_line != b"#!AER-DAT2.0\r\n":
            raise Exception("Can only read version 2.0")

        # Discard comment lines
        while True:
            byte = f.peek(1)

            if len(byte) < 1:
                raise Exception("Could not read from file")

            if byte[0:1] == b"#":
                f.readline()
            else:
                break

        PIXELS_PER_FRAME = 180 * 240
        frame_timestamp = None
        frame = np.empty((180, 240))
        npixels = 0
        reset_frame = np.empty((180, 240))
        reset_npixels = 0
        prev_was_read = False

        # Parse events
        while True:
            event = f.read(8)
            if len(event) < 8:
                break

            address, timestamp = EVENT_STRUCT.unpack(event)
            evt_type = (address & TYPE_MASK[0]) >> TYPE_MASK[1]
            x = (address & X_COORD_MASK[0]) >> X_COORD_MASK[1]
            y = (address & Y_COORD_MASK[0]) >> Y_COORD_MASK[1]
            parity = (address & PARITY_MASK[0]) >> PARITY_MASK[1]
            read_external = (address & READ_EXT_MASK[0]) >> READ_EXT_MASK[1]
            adc = (address & ADC_MASK[0]) >> ADC_MASK[1]

            if evt_type == TYPE_ADS:
                if parity:
                    # It is an IMU event. Ignore.
                    pass
                else:
                    signal_read = read_external == 0

                    if signal_read:
                        if frame_timestamp is None:
                            frame_timestamp = timestamp

                        frame[y, x] = adc
                        npixels += 1
                        prev_was_read = True
                    else:
                        # First reset signal. Yield frame if complete.
                        if prev_was_read:
                            reset_complete = reset_npixels == PIXELS_PER_FRAME
                            frame_complete = npixels == PIXELS_PER_FRAME
                            if frame_complete and reset_complete:
                                frame = frame - reset_frame
                                yield ("ADS", frame_timestamp, frame)

                            # Clear frame anyways. We might have received an
                            # incomplete frame if the recording starts in the
                            # middle of a frame.
                            frame_timestamp = None
                            frame = np.empty((180, 240))
                            npixels = 0
                            reset_frame = np.empty((180, 240))
                            reset_npixels = 0
                            prev_was_read = False

                        reset_frame[y, x] = adc
                        reset_npixels += 1
            else:
                # Filter non-parity events (external events)
                if not read_external:
                    yield ("DVS", timestamp, (x, y, parity))
