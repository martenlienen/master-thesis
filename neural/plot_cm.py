#!/usr/bin/env python

import argparse

import matplotlib.pyplot as pp
import numpy as np
import scipy.io as sio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-5",
        help="Plot top-5 confusion matrix",
        dest="cm5",
        action="store_true")
    parser.add_argument("cm", help="Path to cm.mat")
    args = parser.parse_args()

    cm_path = args.cm

    matrices = sio.loadmat(cm_path)
    label_index = matrices["label_index"]
    cm = matrices["cm"]
    cm5 = matrices["cm5"]

    if args.cm5:
        cm = cm5

    print("{} / {}".format(np.sum(np.diagonal(cm)), np.sum(cm)))

    # Normalize rows
    cm = cm / np.sum(cm, axis=1)[:, np.newaxis]

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
