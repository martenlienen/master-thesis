#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd


def ewm(data, dt, halflife):
    """Compute an exponentially-weighted mean over the columns of data.

    """
    # Perform the calculations in double because it involves numbers pretty
    # close to 1 (consider halflife=10**6)
    columns = data.columns
    data = data.values.astype(np.float64)
    alpha = np.exp(np.log(0.5) / halflife)

    for i in range(1, data.shape[0]):
        decay = np.power(alpha, dt[i])

        # Maybe the second formulation is more numerically stable since there
        # is no multiplication with a number very close to 0.
        # data[i] = (1 - decay) * data[i] + decay * data[i - 1]
        data[i] = data[i] + decay * (data[i - 1] - data[i])

    return pd.DataFrame(data=data.astype(np.float32), columns=columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="File to save data to")
    parser.add_argument("events", help="CSV file with event data")
    args = parser.parse_args()

    out_path = args.out
    events_path = args.events

    if out_path is None:
        ev_dir = os.path.dirname(os.path.realpath(events_path))
        out_path = os.path.join(ev_dir, "preprocessed.csv")

    # Read events
    events = pd.read_csv(events_path)

    # Preprocess events

    # Compute time since previous event
    ts = events["timestamp"].values
    dt = np.zeros(len(events), dtype=np.int32)
    dt[1:] = ts[1:] - ts[:-1]
    events["dt"] = dt

    # Compare events to an exponentially decaying mean position over the last
    # second to capture the major movement
    slow_mean = ewm(events[["x", "y"]], dt, halflife=10**6)
    events["dx-slow"] = events["x"] - slow_mean["x"]
    events["dy-slow"] = events["y"] - slow_mean["y"]

    # And also compare them to another mean over the last 50ms for quick
    # movements
    fast_mean = ewm(events[["x", "y"]], dt, halflife=50 * 10**3)
    events["dx-fast"] = events["x"] - fast_mean["x"]
    events["dy-fast"] = events["y"] - fast_mean["y"]

    # Drop first event that had no predecessor
    # events = events.drop([0])

    # Drop columns with absolute values. We keep the timestamp because that is
    # needed for labeling.
    events = events.drop(["x", "y"], axis=1)

    # Reorder columns just to have them in the original order
    events = events[["timestamp", "dt", "dx-slow", "dx-fast", "dy-slow", "dy-fast", "polarity"]]

    # Store preprocessed events. We do not need more than a few decimal places
    # (if even that) and would rather save space.
    events.to_csv(out_path, index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
