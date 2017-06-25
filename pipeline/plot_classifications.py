#!/usr/bin/env python

import argparse

import h5py as h5
import numpy as np
import matplotlib.pyplot as pp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of a recording")
    parser.add_argument("dataset", help="HDF5 file with labeled classifications")
    args = parser.parse_args()

    recording_name = args.name
    dataset_path = args.dataset

    with h5.File(dataset_path, "r") as f:
        label_index = list(f["label_index"])
        logits = []
        labels = []

        def collect_logits(name, obj):
            if isinstance(obj, h5.Group) and name.endswith(recording_name):
                logits.append(np.array(obj["data"]))
                labels.append(np.array(obj["labels"]))

                return True

        f.visititems(collect_logits)

    logits = logits[0]
    labels = labels[0]

    p = np.exp(logits)
    p /= np.sum(p, axis=1)[:, np.newaxis]

    fig, axes = pp.subplots(2, 1, sharex=True)

    labels[labels == -1] = len(label_index)

    time = np.arange(len(labels))
    for i in range(len(label_index)):
        line = np.zeros(len(labels))
        line[labels == i] = 1.0

        axes[0].plot(time, line)

    for i in range(len(label_index)):
        axes[1].plot(time, p[:, i])

    pp.show()


if __name__ == "__main__":
    main()
