#!/usr/bin/env python

import argparse
import csv
import math
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as ss
from tqdm import tqdm


def binsearch(arr, needle):
    a = 0
    b = len(arr) - 1

    while a < b:
        idx = (a + b) // 2

        if needle > arr[idx]:
            a = idx + 1
        elif needle < arr[idx]:
            b = idx - 1
        else:
            a = idx
            b = idx

    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--length",
        default=10,
        type=int,
        help="Length of frames in milliseconds")
    parser.add_argument(
        "-r", "--rows", default=180, type=int, help="Number of rows")
    parser.add_argument(
        "-c", "--columns", default=240, type=int, help="Number of columns")
    parser.add_argument("events_csv", help="Path to events.csv")
    args = parser.parse_args()

    length = args.length
    nrows = args.rows
    ncols = args.columns
    events_path = args.events_csv

    frames_path = os.path.join(
        os.path.dirname(events_path), "gray-frames-{}ms.mat".format(length))

    print("Read event data")
    events = pd.read_csv(events_path)
    events = events.sort_values(by="timestamp")
    timestamps = events["timestamp"]
    min_t = timestamps.iloc[0]
    max_t = timestamps.iloc[-1]

    window_length = length * 10**3
    num_frames = math.ceil((max_t - min_t) / window_length)

    print("Prepare data")
    events["parity"][events["parity"] == 0] = -1
    events["parity"] *= (1.0 / 200)

    print("Accumulate event frames")
    frames = [None for i in range(num_frames)]
    for i in tqdm(range(num_frames)):
        tstart = binsearch(timestamps, min_t + i * window_length)
        tend = binsearch(timestamps, min_t + (i + 1) * window_length)
        data = events.iloc[tstart:tend]

        # Sum up all pixels
        sums = data.groupby(by=("x", "y"))["parity"].sum()
        sums = sums.clip(lower=-0.5, upper=0.5)

        rows = [pair[1] for pair in sums.index.values]
        cols = [pair[0] for pair in sums.index.values]
        frame = ss.coo_matrix(
            (sums.values, (rows, cols)),
            shape=(nrows, ncols),
            dtype=np.float32)

        frames[i] = frame

    print("Save into {}".format(frames_path))
    sio.savemat(frames_path, {"frames": frames})


if __name__ == "__main__":
    main()
