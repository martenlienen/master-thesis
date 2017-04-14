#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import aerv1
import aerv2
from scipy.misc import imsave


def main():
    parser = argparse.ArgumentParser(
        description="Split an .aedat file into a list of events and frames.")
    parser.add_argument("aedat", help=".aedat file to split")
    parser.add_argument("out_dir", help="Output directory")
    args = parser.parse_args()

    aedat_file = args.aedat
    out_dir = args.out_dir

    if not os.path.isfile(aedat_file):
        sys.exit("AEDat file {} does not exist".format(aedat_file))

    if os.path.exists(out_dir):
        sys.exit("Output directory {} already exists".format(out_dir))

    events_path = os.path.join(out_dir, "events.csv")
    frames_dir = os.path.join(out_dir, "frames")

    os.makedirs(frames_dir)

    with open(events_path, "w") as events_f:
        events_writer = csv.writer(events_f)
        events_writer.writerow(["timestamp", "x", "y", "parity"])

        try:
            events = aerv1.read(aedat_file)
        except:
            try:
                events = aerv2.read(aedat_file)
            except:
                sys.exit("Could not read AER file {}".format(aedat_file))

        for evt_type, timestamp, data in events:
            if evt_type == "ADS":
                # DVS puts bottom-left at (0, 0)
                data = np.flipud(data)

                frame_path = os.path.join(frames_dir,
                                          "{}.png".format(timestamp))
                imsave(frame_path, data)
            elif evt_type == "DVS":
                events_writer.writerow([timestamp, *data])


if __name__ == "__main__":
    main()
