#!/usr/bin/env python

import argparse

import h5py as h5
import matplotlib.pyplot as pp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="top", choices=["top", "softmax"], help="Plot either the top-1 accuracy of the probability distribution over classes")
    parser.add_argument("dataset")
    args = parser.parse_args()

    plot_type = args.type
    dataset_path = args.dataset

    with h5.File(dataset_path, "r") as f:
        label_index = list(f["label_index"])
        logits = []
        labels = []

        def collect_logits(name, obj):
            if isinstance(obj, h5.Group) and "data" in obj and "labels" in obj:
                logits.append(np.array(obj["data"]))
                labels.append(np.array(obj["labels"]))

        f.visititems(collect_logits)

    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)

    labels[labels == -1] = len(label_index)

    label_index.append("<blank>")

    cm = np.zeros((len(label_index), len(label_index)))

    if plot_type == "top":
        predictions = np.argmax(logits, axis=-1)

        for i in range(len(label_index)):
            fltr = labels == i
            l = labels[fltr]
            p = predictions[fltr]

            for j in range(len(label_index)):
                cm[i, j] = np.count_nonzero(p == j) / len(l)
    elif plot_type == "softmax":
        # Softmax
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=1)[:, np.newaxis]

        for i in range(len(label_index)):
            fltr = labels == i
            l = labels[fltr]
            p = probs[fltr]

            cm[i] = np.sum(p, axis=0) / len(l)

    ax = pp.gca()
    tm = np.arange(len(label_index))
    ax.xaxis.set_label_position("top")
    ax.set_xticks(tm)
    ax.set_xticklabels(label_index, rotation=90)
    ax.set_yticks(tm)
    ax.set_yticklabels(label_index)

    ax.imshow(cm)
    pp.show()


if __name__ == "__main__":
    main()
