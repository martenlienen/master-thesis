#!/usr/bin/env python

import argparse

import h5py as h5
import matplotlib.pyplot as pp
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gesture", help="Gesture to plot")
    parser.add_argument("file", help="HDF5 file with labeled states")
    args = parser.parse_args()

    gesture_name = args.gesture
    file_path = args.file

    with h5.File(file_path) as f:
        label_index = list(f["label_index"])
        gesture_index = label_index.index(gesture_name)

        data = []
        rel_time = []
        num_events = []
        collection = f["gists"]
        for grp in collection.values():
            labels = np.array(grp["labels"])
            fltr = labels == gesture_index
            labels = labels[fltr]
            timestamps = np.array(grp["timestamps"][fltr])
            data.append(np.array(grp["data"][fltr, :]))

            rel_time.append(np.linspace(0.0, 1.0, num=np.count_nonzero(fltr)))
            num_events.append(np.log(1 + np.array(grp["num_events"][fltr])))
        data = np.concatenate(data)
        rel_time = np.concatenate(rel_time)
        num_events = np.concatenate(num_events)

    pca = IncrementalPCA(50)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=1000, verbose=2)
    pipe = Pipeline([("PCA", pca), ("t-SNE", tsne)])

    data2d = tsne.fit_transform(data)

    fig, ax = pp.subplots(1, 2)
    ax[0].set_title("By relative time")
    ax[0].scatter(data2d[:, 0], data2d[:, 1], c=rel_time)
    ax[1].set_title("By number of events")
    ax[1].scatter(data2d[:, 0], data2d[:, 1], c=num_events)

    pp.show()


if __name__ == "__main__":
    main()
