#!/usr/bin/env python

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Remove all checkpoints except the best performing and the last one")
    parser.add_argument("log_dir", help="Tensorflow log directory")
    args = parser.parse_args()

    log_dir = args.log_dir

    train_metrics = pd.read_csv(os.path.join(log_dir, "train.csv"))
    val_metrics = pd.read_csv(os.path.join(log_dir, "val.csv"))

    keep = set()
    keep.add(len(val_metrics) - 1)

    for c in train_metrics.columns:
        if c == "accuracy":
            keep.add(train_metrics[c].argmax())
        else:
            keep.add(train_metrics[c].argmin())

    for c in val_metrics.columns:
        if c == "accuracy":
            keep.add(val_metrics[c].argmax())
        else:
            keep.add(val_metrics[c].argmin())

    print("Keeping epochs {}".format(", ".join(map(str, sorted(keep)))))

    for i in range(len(val_metrics)):
        if not i in keep:
            files = glob.glob(os.path.join(log_dir, f"epoch-{i}.ckpt.*"))

            for f in files:
                os.remove(f)


if __name__ == "__main__":
    main()
