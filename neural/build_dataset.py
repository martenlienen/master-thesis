#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


def label_events(events, label_spans, label_index):
    """Label events according to start and stop timestamps.

    The labels are returned in the form of a 1-hot encoded matrix that has one
    row per event.
    """

    nevents = events.shape[0]
    nlabels = len(label_index)
    labels = np.zeros((nevents, nlabels), dtype=np.float16)
    timestamps = np.array(events["timestamp"])
    nspans = label_spans.shape[0]
    curr_index = None
    curr_label = "none"
    curr_end = None
    next_start = label_spans.loc[0, "start"]
    next_index = 0
    for i in tqdm(range(nevents), desc="Label events"):
        t = timestamps[i]

        if next_start is not None and t >= next_start:
            curr_index = next_index
            curr_label = label_spans.loc[curr_index, "name"]
            curr_end = label_spans.loc[curr_index, "end"]

            if next_index < nspans - 1:
                next_index += 1
                next_start = label_spans.loc[next_index, "start"]
            else:
                next_index = None
                next_start = None
        elif curr_end is not None and t > curr_end:
            curr_label = "none"

        labels[i, label_index.index(curr_label)] = 1.0

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Build a dataset from preprocessed events")
    parser.add_argument(
        "--timesteps",
        default=2048,
        type=int,
        help="Number of events per sequence")
    parser.add_argument(
        "--based-on", help="Path to data file to reuse for normalization")
    parser.add_argument(
        "--load",
        default="preprocessed.csv",
        help="Event file to load from each directory")
    parser.add_argument(
        "--remove-none",
        default=False,
        dest="remove_none",
        action="store_true",
        help="Remove all sequences that were labeled with none")
    parser.add_argument(
        "--balance",
        default=False,
        dest="balance",
        action="store_true",
        help="Balance classes by under- and oversampling")
    parser.add_argument("dirs", nargs="+", help="Directories with event data")
    parser.add_argument("out", help="File to save data to")
    args = parser.parse_args()

    timesteps = args.timesteps
    based_on = args.based_on
    load_filename = args.load
    remove_none = args.remove_none
    balance_classes = args.balance
    dirs = args.dirs
    out_path = args.out

    if based_on:
        based_on_values = sio.loadmat(
            based_on,
            variable_names=["data", "mu", "sigma", "label_index", "has_none"])

        timesteps = based_on_values["data"].shape[1]
        remove_none = not based_on_values["has_none"]

        # Free memory
        del based_on_values["data"]

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
        label_index = sorted(set(label_index)) + ["none"]

    # Convert events into labeled slices
    data_buf = []
    label_buf = []
    for dir in dirs:
        labels_path = os.path.join(dir, "labels.csv")
        events_path = os.path.join(dir, load_filename)

        label_spans = pd.read_csv(labels_path)
        events = pd.read_csv(events_path)

        labels = label_events(events, label_spans, label_index)

        # Convert event data to matrix
        features = [
            "delta-t", "delta-x-prev", "delta-x-mean", "delta-y-prev",
            "delta-y-mean", "parity"
        ]
        data = events[features].as_matrix()

        # Drop events that cannot be evenly split into a time slice
        nrest = data.shape[0] % timesteps
        data = data[:-nrest, :]
        labels = labels[:-nrest, :]

        # Reshape into time slices
        data = np.reshape(data, (-1, timesteps, len(features)))
        labels = np.reshape(labels, (-1, timesteps, len(label_index)))

        # Select the middle label of each slice
        labels = labels[:, timesteps // 2, :]

        data_buf.append(data)
        label_buf.append(labels)

    data = np.concatenate(data_buf, axis=0)
    labels = np.concatenate(label_buf, axis=0)

    # Normalize data
    if based_on:
        mu = based_on_values["mu"]
        sigma = based_on_values["sigma"]
    else:
        mu = np.mean(data, axis=(0, 1))
        sigma = np.std(data, axis=(0, 1))
    data = (data - mu) / sigma

    if remove_none:
        # Remove all data points labeled with none
        none_index = label_index.index("none")
        filt = np.logical_not(labels[:, none_index] == 1.0)
        data = data[filt, :, :]
        labels = labels[filt, :]

        # Remove the none label from the index and its column
        labels = np.delete(labels, none_index, axis=1)
        del label_index[none_index]

    if balance_classes:
        counts = np.sum(labels, axis=0, dtype=np.int32)
        median = int(np.median(counts))
        too_few, = np.nonzero(counts < median)
        too_many, = np.nonzero(counts > median)

        # Oversample classes that are underrepresented
        indices = []
        for i in too_few:
            class_indices, = np.nonzero(labels[:, i] == 1.0)
            missing = median - len(class_indices)
            indices.append(
                np.random.choice(class_indices, size=missing, replace=True))
        to_replicate = np.concatenate(indices)
        data = np.concatenate((data, data[to_replicate]), axis=0)
        labels = np.concatenate((labels, labels[to_replicate]), axis=0)

        # Undersample classes that are overrepresented
        indices = []
        for i in too_many:
            class_indices, = np.nonzero(labels[:, i] == 1.0)
            abundant = len(class_indices) - median
            indices.append(
                np.random.choice(class_indices, size=abundant, replace=False))
        to_remove = np.concatenate(indices)
        data = np.delete(data, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)

    # Randomly permute sequences
    permutation = np.random.permutation(data.shape[0])
    data = data[permutation, :, :]
    labels = labels[permutation, :]

    sio.savemat(out_path, {
        "labels": labels,
        "data": data,
        "label_index": label_index,
        "mu": mu,
        "sigma": sigma,
        "has_none": not remove_none
    })


if __name__ == "__main__":
    main()
