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
        colors = []
        collection = f["gists"]
        for grp in collection.values():
            labels = np.array(grp["labels"])
            fltr = labels == gesture_index
            labels = labels[fltr]
            timestamps = np.array(grp["timestamps"][fltr])
            data.append(np.array(grp["data"][fltr, :]))
            colors.append(np.linspace(0.0, 1.0, num=np.count_nonzero(fltr)))
        data = np.concatenate(data)
        colors = np.concatenate(colors)

    pca = IncrementalPCA(50)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=1000, verbose=2)
    pipe = Pipeline([("PCA", pca), ("t-SNE", tsne)])

    data2d = tsne.fit_transform(data)

    pp.scatter(data2d[:, 0], data2d[:, 1], c=colors)
    pp.show()


if __name__ == "__main__":
    main()
