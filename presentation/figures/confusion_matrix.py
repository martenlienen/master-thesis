#!/usr/bin/env python

import argparse
import json
import os

import matplotlib.pyplot as pp
import matplotlib.transforms as mpt
import numpy as np


def load_data(path):
    label_index = set()
    data = []

    with open(path, "r") as f:
        while True:
            try:
                pred = json.loads(f.readline())
                true = json.loads(f.readline())
                diff = json.loads(f.readline())

                data.append((pred, true, diff))

                label_index |= set(pred)
                label_index |= set(true)
            except json.JSONDecodeError as e:
                break

    label_index = list(label_index)
    label_index.sort()
    label_index.append("<blank>")

    return label_index, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="Path for confusion matrix")
    parser.add_argument("dataset", help="Path to json file")
    args = parser.parse_args()

    out_path = args.out
    dataset_path = args.dataset

    label_index, data = load_data(dataset_path)

    cm = np.zeros((len(label_index), len(label_index)))
    for pred, true, diff in data:
        diff = {int(op["path"][1:]): op for op in diff}
        for i, label in enumerate(pred):
            j = label_index.index(label)

            if i in diff:
                op = diff[i]

                if op["op"] == "replace":
                    cm[label_index.index(op["value"]), j] += 1
                elif op["op"] == "remove":
                    cm[-1, j] += 1
                else:
                    print("add better not happens")
            else:
                cm[j, j] += 1

    cm = cm / cm.sum(axis=1, keepdims=True)

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
