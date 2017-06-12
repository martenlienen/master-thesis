#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import aer
from scipy.misc import imsave
import pandas as pd


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
        events_writer.writerow(["timestamp", "x", "y", "polarity"])

        try:
            events = aer.read(aedat_file, chip_class)
        except Exception as e:
            sys.exit(e)

        frames = []
        for evt_type, timestamp, data in events:
            if evt_type == "ADS":
                # DVS puts bottom-left at (0, 0)
                data = np.flipud(data)

                frames.append((timestamp, data))
            elif evt_type == "DVS":
                events_writer.writerow([timestamp, *data])

    # Filter events that happended before the first event. This happended in a
    # DVS recording and the timestamps of some events were corrupted.
    events = pd.read_csv(events_path)
    first_ts = events.iloc[0]["timestamp"]
    events = events.loc[events["timestamp"] >= first_ts]

    # Ensure that the remaining events are sorted
    events = events.sort_values(by="timestamp")

    # Normalize timestamps so that each recording starts at 0. This also makes
    # timestamp fit into an int32 as long as the recording is shorter than ~40
    # minutes.
    min_time = events["timestamp"].min()
    events["timestamp"] -= min_time
    for i in range(len(frames)):
        frames[i][0] -= min_time

    # Save back to CSV
    events.to_csv(events_path, index=False)

    # Save frames
    for timestamp, data in frames:
        frame_path = os.path.join(frames_dir, "{}.png".format(timestamp))
        imsave(frame_path, data)


if __name__ == "__main__":
    main()
