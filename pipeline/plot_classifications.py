#!/usr/bin/env python

import argparse

import h5py as h5
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--decoded", action="store_true", help="Plot decodings instead of probabilities")
    parser.add_argument("-s", "--start", type=float, help="Start timestamp")
    parser.add_argument("-e", "--end", type=float, help="end timestamp")
    parser.add_argument("-o", "--out", help="Optional output path")
    parser.add_argument("name", help="Name of a recording")
    parser.add_argument("dataset", help="HDF5 file with labeled classifications")
    args = parser.parse_args()

    is_decoded = args.decoded
    start_time = args.start
    end_time = args.end
    out_path = args.out
    recording_name = args.name
    dataset_path = args.dataset

    with h5.File(dataset_path, "r") as f:
        label_index = list(f["label_index"])
        timestamps = []
        logits = []
        labels = []

        def collect_logits(name, obj):
            if isinstance(obj, h5.Group) and name.endswith(recording_name):
                timestamps.append(np.array(obj["timestamps"]))
                logits.append(np.array(obj["data"]))
                labels.append(np.array(obj["labels"]))

                return True

        f.visititems(collect_logits)

    timestamps = timestamps[0]
    logits = logits[0]
    labels = labels[0]

    # Convert timestamps to seconds
    timestamps = timestamps / 10**6

    if start_time is not None:
        fltr = timestamps >= start_time
        timestamps = timestamps[fltr]
        logits = logits[fltr]
        labels = labels[fltr]

    if end_time is not None:
        fltr = timestamps <= end_time
        timestamps = timestamps[fltr]
        logits = logits[fltr]
        labels = labels[fltr]

    if is_decoded:
        p = logits
    else:
        # Softmax
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= np.sum(p, axis=1)[:, np.newaxis]

    norm = mpl.colors.Normalize(0, len(label_index) - 1)
    sm = mpl.cm.ScalarMappable(norm, "tab20c")

    fig, ax = pp.subplots(1, 1, figsize=(6, 1.5), dpi=200)

    # Plot true labels
    for i in range(len(label_index)):
        ax.fill_between(timestamps, -0.025, 1.025, where=(labels == i), color=sm.to_rgba(i), alpha=0.5, lw=0)

    # Plot class probabilities
    for i in range(len(label_index)):
        ax.plot(timestamps, p[:, i], lw=1, c=sm.to_rgba(i))

        if is_decoded:
            ax.fill_between(timestamps, 0, p[:, i], hatch="/", edgecolor=sm.to_rgba(i), facecolor="none")

    ax.set_xlabel("Time in seconds")
    ax.set_ylabel("Class probability")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim((-0.025, 1.025))

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    else:
        pp.show()


if __name__ == "__main__":
    main()
