#!/usr/bin/env python

import argparse
import os

import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import scipy.misc as sm
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", default=30, type=int, help="Frames per second")
    parser.add_argument("events", help="Path to events in CSV format")
    parser.add_argument("out", help="Directory to write frames to")
    args = parser.parse_args()

    fps = args.fps
    events_path = args.events
    out_path = args.out

    events = pd.read_csv(events_path)

    os.makedirs(out_path, exist_ok=True)

    ms_per_frame = 10**6 / fps
    ts = events["timestamp"]
    first_ts = ts.min()
    last_ts = ts.max()
    start_time = np.arange(first_ts, last_ts, ms_per_frame)
    end_time = start_time + ms_per_frame
    start = np.searchsorted(ts.values, start_time)
    end = np.searchsorted(ts.values, end_time)

    for i in tqdm(range(len(start))):
        s = start[i]
        e = end[i]

        frame = np.zeros((128, 128))

        for j in range(s, e):
            t, x, y, p = events.values[j]

            if p > 0:
                frame[x, y] += 1.0
            else:
                frame[x, y] -= 1.0

        sm.imsave(os.path.join(out_path, f"{i}.png"), sm.imresize(frame, 300, "nearest"))


if __name__ == "__main__":
    main()
