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

        def collect(name, obj):
            if isinstance(obj, h5.Group) and "labels" in obj:
                labels.append(np.array(obj["labels"]))

        f.visititems(collect)

    nrecordings = len(labels)

    labels = np.concatenate(labels, axis=0)

    blank_filter = labels == -1
    nonblank_filter = ~blank_filter

    ninstances = int(np.count_nonzero(np.logical_xor(blank_filter[:-1], blank_filter[1:])) / 2)

    pi = np.full(2, 1/2, np.float32)
    A = np.zeros((2, 2), np.float32)

    A[0, 1] = ninstances / np.count_nonzero(nonblank_filter)
    A[0, 0] = 1 - A[0, 1]
    A[1, 0] = ninstances / np.count_nonzero(blank_filter)
    A[1, 1] = 1 - A[1, 0]

    sio.savemat(out_path, {"pi": pi, "A": A})


if __name__ == "__main__":
    main()
