#!/usr/bin/env python

import argparse

import h5py as h5
import numpy as np
import scipy.io as sio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("out", help="Where to store the decoder")
    args = parser.parse_args()

    dataset_path = args.dataset
    out_path = args.out

    with h5.File(dataset_path, "r") as f:
        label_index = list(f["label_index"])

        labels = []
        data = []

        def collect(name, obj):
            if isinstance(obj, h5.Group) and "labels" in obj and "data" in obj:
                labels.append(np.array(obj["labels"]))
                data.append(np.array(obj["data"]))

        f.visititems(collect)

    ndata = len(data)

    labels = np.concatenate(labels, axis=0)
    data = np.concatenate(data, axis=0)

    # Convert logits to probabilites with softmax
    data = np.exp(data)
    data /= np.sum(data, axis=1)[:, np.newaxis]

    labels[labels == -1] = len(label_index)

    nclasses = len(label_index) + 1
    pi = np.ones(nclasses, np.float32) / nclasses
    A = np.zeros((nclasses, nclasses), np.float32)
    B = np.zeros((nclasses, nclasses), np.float32)

    for i in range(nclasses - 1):
        fltr = labels == i

        A[i, -1] = 1 / (np.count_nonzero(fltr) / ndata)
        A[i, i] = 1 - A[i, -1]

    A[-1, -1] = 0.999
    A[-1, :-1] = (1 - A[-1, -1]) / (nclasses - 1)
    # values, counts = np.unique(labels, return_counts=True)
    # for v, c in zip(values, counts):
    #     A[-1, v] = c / len(labels)

    for i in range(nclasses):
        fltr = labels == i
        B[i] = np.sum(data[fltr], axis=0) / np.count_nonzero(fltr)

    sio.savemat(out_path, {
        "pi": pi,
        "A": A,
        "B": B
    })


if __name__ == "__main__":
    main()
