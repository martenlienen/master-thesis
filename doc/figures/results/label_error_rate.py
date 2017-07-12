#!/usr/bin/env python

import argparse

import editdistance
import h5py as h5
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

    for l in labels:
        l[l == -1] = len(label_index)

    label_index.append("<blank>")

    return label_index, labels, logits


def label_sequence(sequence, label_index):
    blank = label_index.index("<blank>")
    labels = []

    label = None
    for i in range(len(sequence)):
        if label is None or label != sequence[i]:
            label = sequence[i]
            labels.append(label)

    # Filter out blanks
    labels = [l for l in labels if l != blank]

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset")
    args = parser.parse_args()

    dataset_path = args.dataset

    label_index, labels, logits = load_dataset(dataset_path)

    pred = []
    for l in logits:
        pred.append(l.argmax(axis=-1))

    distances = []
    true_lenghts = []
    for tl, pl in zip(labels, pred):
        true_ls = label_sequence(tl, label_index)
        pred_ls = label_sequence(pl, label_index)

        ed = editdistance.eval(true_ls, pred_ls)

        distances.append(ed)
        true_lenghts.append(len(true_ls))

        # print(ed)
        # print(", ".join([label_index[l] for l in true_ls]))
        # print(", ".join([label_index[l] for l in pred_ls]))

    distances = np.array(distances)
    true_lenghts = np.array(true_lenghts)

    print("Mean Levenshtein distance: {}".format(distances.mean()))
    print("Label Error Rate: {}".format(distances.sum() / true_lenghts.sum()))

if __name__ == "__main__":
    main()
