#!/usr/bin/env python

import argparse
import math
import os

from acc_events_frames import binsearch
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Generate labels for fixed-length time windows")
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=10,
        help="Length of time window in milliseconds")
    parser.add_argument("events_path", help="Path to events CSV file")
    parser.add_argument("labels_path", help="Path to labels CSV file")
    args = parser.parse_args()

    length = args.length
    events_path = args.events_path
    labels_path = args.labels_path

    window_length = length * 10**3
    frame_labels_path = os.path.join(
        os.path.dirname(labels_path), "frame-labels-{}ms.csv".format(length))

    events = pd.read_csv(events_path)
    labels = pd.read_csv(labels_path)

    timestamps = events["timestamp"]
    min_t = timestamps.min()
    max_t = timestamps.max()

    num_windows = math.ceil((max_t - min_t) / window_length)
    window_labels = []

    labels_tend = labels["end"]
    for i in tqdm(range(num_windows)):
        tcenter = int(min_t + (i + 0.5) * (window_length))
        label_index = binsearch(labels_tend, tcenter)

        if tcenter >= labels.iloc[label_index]["start"]:
            window_labels.append((labels["name"].loc[label_index], ))
        else:
            window_labels.append(("none", ))

    labels_frame = pd.DataFrame(window_labels, columns=["label"])
    labels_frame.to_csv(frame_labels_path, index=False)


if __name__ == "__main__":
    main()
