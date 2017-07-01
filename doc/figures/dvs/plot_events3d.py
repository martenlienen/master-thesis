#!/usr/bin/env python

import argparse

import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as pp
import matplotlib.transforms as mpt
import pandas as pd
import sklearn.neighbors as skn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events", help="Path to events in CSV format")
    args = parser.parse_args()

    events_path = args.events

    events = pd.read_csv(events_path)

    i = 19750
    events = events.iloc[i:i + 2000]

    events["timestamp"] -= events["timestamp"].min()

    coordinates = events[["x", "y"]].values
    kdtree = skn.KDTree(coordinates, metric="euclidean")
    nn_distance = kdtree.query(coordinates, k=5)[0][:, 4]

    events = events.loc[nn_distance <= 3]
    nn_distance = nn_distance[nn_distance <= 3]

    fig = pp.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.w_xaxis.line.set_visible(False)
    ax.w_yaxis.line.set_visible(False)
    ax.w_zaxis.line.set_visible(False)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$y$")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    ax.set_ylim(bottom=-1000, top=events["timestamp"].max())

    ax.view_init(10, 60)
    ax.scatter3D(events["x"], np.full(len(events), -1000), events["y"], s=10, depthshade=False)
    ax.scatter3D(events["x"], events["timestamp"], events["y"], s=8, c=events["timestamp"], cmap="Reds")

    fig.savefig("hand.pdf", bbox_inches=mpt.Bbox.from_bounds(1, 1.4, 6.0, 4.7))


if __name__ == "__main__":
    main()
