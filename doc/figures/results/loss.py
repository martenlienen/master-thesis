#!/usr/bin/env python

import argparse
import os

import matplotlib.pyplot as pp
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="Path to output file")
    parser.add_argument("log_dir", help="Tensorflow log directory")
    args = parser.parse_args()

    out_path = args.out
    log_dir = args.log_dir

    train_metrics = pd.read_csv(os.path.join(log_dir, "train.csv"))
    val_metrics = pd.read_csv(os.path.join(log_dir, "val.csv"))

    train_metrics.index += 1
    val_metrics.index += 1

    min_val_loss = val_metrics["loss"].argmin()

    fig, ax = pp.subplots(figsize=(3, 3), dpi=200)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    train_metrics["loss"].plot(ax=ax, label="Training", lw=1)
    val_metrics["loss"].plot(ax=ax, label="Validation", lw=1)

    ax.axvline(min_val_loss, alpha=0.5, lw=1, ls="--", label="Minimum", c="C1")

    ax.legend(loc="upper right", fontsize=6, framealpha=0.5)

    if out_path is None:
        pp.show()
    else:
        fig.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
