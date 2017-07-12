#!/usr/bin/env python

import argparse

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

    labels = np.concatenate(labels, axis=0)
    logits = np.concatenate(logits, axis=0)

    labels[labels == -1] = len(label_index)

    label_index.append("<blank>")

    return label_index, labels, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset")
    args = parser.parse_args()

    dataset_path = args.dataset

    label_index, labels, logits = load_dataset(dataset_path)

    pred = np.argmax(logits, axis=-1)

    accuracy = np.mean(pred == labels)

    print("Accuracy: {:.3f}".format(accuracy))

    for l, name in enumerate(label_index):
        total = np.count_nonzero(labels == l)
        correct = np.count_nonzero(pred[labels == l] == l)

        print("{} accuracy: {:.3f}".format(name, correct / total))

if __name__ == "__main__":
    main()
