#!/usr/bin/env python

import argparse
import os

import h5py as h5
import matplotlib.pyplot as pp
import matplotlib.transforms as mpt
import numpy as np


def load_dataset(path):
    with h5.File(path, "r") as f:
        label_index = list(f["label_index"])
        labels = []
        logits = []

        def collect_logits(name, obj):
            if isinstance(obj, h5.Group) and "data" in obj and "labels" in obj:
                labels.append(np.array(obj["labels"]))
                logits.append(np.array(obj["data"]))

        f.visititems(collect_logits)

    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)

    labels[labels == -1] = len(label_index)
    label_index.append("<blank>")

    return label_index, labels, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="Path for confusion matrix")
    parser.add_argument("dataset", help="Path to HDF5 file with logits")
    args = parser.parse_args()

    out_path = args.out
    dataset_path = args.dataset

    label_index, labels, logits = load_dataset(dataset_path)

    cm = np.zeros((len(label_index), len(label_index)))
    predictions = np.argmax(logits, axis=-1)
    for i in range(len(label_index)):
        fltr = labels == i
        l = labels[fltr]
        p = predictions[fltr]

        for j in range(len(label_index)):
            cm[i, j] = np.count_nonzero(p == j) / len(l)

    fig, ax = pp.subplots(1, 1, figsize=(4, 4), dpi=300)

    ticks = np.arange(len(label_index))

    ax.set_xticks(ticks)
    ax.set_xticklabels(label_index, fontdict={"size": 5}, rotation=-45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(label_index, fontdict={"size": 5}, rotation=-45)

    ax.xaxis.set_tick_params(labeltop="on", labelbottom="off", top="on", bottom="off")

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")

    ax.imshow(cm)

    if out_path is None:
        pp.show()
    else:
        fig.savefig(out_path, bbox_inches=mpt.Bbox.from_bounds(-0.2, 0.4, 4.0, 3.8))


if __name__ == "__main__":
    main()
