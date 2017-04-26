#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="File to save data to")
    parser.add_argument(
        "--drop-radius",
        default=None,
        type=float,
        help="Drop events that are further than a distance r from the mean of events"
    )
    parser.add_argument("events", help="CSV file with event data")
    args = parser.parse_args()

    out_path = args.out
    drop_radius = args.drop_radius
    events_path = args.events

    if out_path is None:
        ev_dir = os.path.dirname(os.path.realpath(events_path))
        out_path = os.path.join(ev_dir, "preprocessed.csv")

    # Read events
    events = pd.read_csv(events_path)

    # Preprocess events

    # Compute time since previous event that did not get the same timestamp
    delta_t = np.zeros(len(events))
    ts = events["timestamp"].as_matrix()
    prev_ts = ts[0]
    for i in range(1, len(events)):
        if ts[i] > ts[i - 1]:
            prev_ts = ts[i - 1]

        delta_t[i] = ts[i] - prev_ts
    events["delta-t"] = delta_t

    # Store distance to previous event
    events["delta-x-prev"] = events["x"] - events["x"].shift(1)
    events["delta-y-prev"] = events["y"] - events["y"].shift(1)

    # Store distance with regards to a lazy mean
    events["delta-x-mean"] = events["x"] - events["x"].ewm(halflife=100).mean()
    events["delta-y-mean"] = events["y"] - events["y"].ewm(halflife=100).mean()

    # Drop events that are further than some distance from the mean
    if drop_radius:
        dist = np.sqrt(events["delta-x-mean"]**2 + events["delta-y-mean"]**2)
        events = events[dist <= drop_radius]

    # Drop first event that had no predecessor
    events = events.drop([0])

    # Drop columns with absolute values. We keep the timestamp because that is
    # needed for labeling.
    events = events.drop(["x", "y"], axis=1)

    # Reorder columns just to have them in the original order
    events = events[[
        "timestamp", "delta-t", "delta-x-prev", "delta-x-mean", "delta-y-prev",
        "delta-y-mean", "parity"
    ]]

    # Store preprocessed events
    events.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
