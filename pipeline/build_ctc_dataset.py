#!/usr/bin/env python

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn.preprocessing as sp
from tqdm import tqdm


def label_events(events, label_spans, label_index):
    """Label events according to start and stop timestamps."""

    nevents = events.shape[0]
    labels = np.full(nevents, -1, dtype=np.int32)
    timestamps = np.array(events["timestamp"])

    for idx, start, end, name in label_spans.itertuples():
        fltr = (timestamps >= start) & (timestamps <= end)
        labels[fltr] = label_index.index(name)

    return labels


def read_data_and_labels(dir, events_filename, label_index):
    labels_path = os.path.join(dir, "labels.csv")
    events_path = os.path.join(dir, events_filename)

    label_spans = pd.read_csv(labels_path)
    events = pd.read_csv(events_path)

    labels = label_events(events, label_spans, label_index)

    # Convert event data to matrix
    features = [
        "delta-t", "delta-x-prev", "delta-x-mean", "delta-y-prev",
        "delta-y-mean", "polarity"
    ]
    data = events[features].as_matrix()

    return data, labels


def main():
    parser = argparse.ArgumentParser(
        description="Build a dataset from preprocessed events")
    parser.add_argument(
        "--based-on", help="Path to data file to reuse for normalization")
    parser.add_argument(
        "--load",
        default="preprocessed.csv",
        help="Event file to load from each directory")
    parser.add_argument("dirs", nargs="+", help="Directories with event data")
    parser.add_argument("out", help="File to save data to")
    args = parser.parse_args()

    based_on = args.based_on
    load_filename = args.load
    dirs = args.dirs
    out_path = args.out

    if based_on:
        based_on_values = sio.loadmat(
            based_on,
            variable_names=["mu", "sigma", "label_index"])

    # Create a label index
    if based_on:
        label_index = list(based_on_values["label_index"])

        # Storing string arrays in matlab files right-pads them with spaces
        label_index = [l.strip() for l in label_index]
    else:
        label_index = []
        for dir in dirs:
            labels_path = os.path.join(dir, "labels.csv")
            label_spans = pd.read_csv(labels_path)
            label_index += list(label_spans["name"])
        label_index = sorted(set(label_index))

    # Convert events into labeled slices. We use a thread pool here because
    # first, we can escape the GIL with numpy and pandas and second, you cannot
    # return python objects that large from a process pool. (MemoryError)
    pool = mp.pool.ThreadPool()
    buf = pool.starmap(read_data_and_labels, [(d, load_filename, label_index) for d in dirs])
    data = [d for d, l in buf]
    labels = [l for d, l in buf]

    # Normalize data
    if based_on:
        mu = based_on_values["mu"]
        sigma = based_on_values["sigma"]
    else:
        ndata = np.sum([d.shape[0] for d in data])
        mu = np.sum([np.mean(d, axis=0) * (d.shape[0] / ndata) for d in data], axis=0)
        sigma = np.sqrt(np.sum([np.var(d, axis=0) * (d.shape[0] / ndata) for d in data], axis=0))

    for i in range(len(data)):
        data[i] = (data[i] - mu) / sigma

    sio.savemat(out_path, {
        "labels": labels,
        "data": data,
        "label_index": label_index,
        "mu": mu,
        "sigma": sigma
    })


if __name__ == "__main__":
    main()
