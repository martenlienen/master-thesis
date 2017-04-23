#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import aer
from scipy.misc import imsave


def main():
    parser = argparse.ArgumentParser(
        description="Split an .aedat file into a list of events and frames.")
    parser.add_argument(
        "-c",
        "--chip",
        default="DVS",
        choices=["DVS", "DAVIS"],
        help="Type of chip that generated the events")
    parser.add_argument("aedat", help=".aedat file to split")
    parser.add_argument("out_dir", help="Output directory")
    args = parser.parse_args()

    chip_class = args.chip
    aedat_file = args.aedat
    out_dir = args.out_dir

    if not os.path.isfile(aedat_file):
        sys.exit("AEDat file {} does not exist".format(aedat_file))

    events_path = os.path.join(out_dir, "events.csv")
    frames_dir = os.path.join(out_dir, "frames")

    if os.path.exists(events_path) or os.path.exists(frames_dir):
        sys.exit(
            "Output directory {} already contains event data".format(out_dir))

    os.makedirs(frames_dir)

    with open(events_path, "w") as events_f:
        events_writer = csv.writer(events_f)
        events_writer.writerow(["timestamp", "x", "y", "parity"])

        try:
            events = aer.read(aedat_file, chip_class)
        except Exception as e:
            sys.exit(e)

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
