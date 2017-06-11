#!/usr/bin/env python

import argparse
import concurrent.futures
import os

import h5py as h5
import numpy as np
import pandas as pd


def read_data(dir, events_filename):
    events_path = os.path.join(dir, events_filename)

    # Read everything as float32 except the timestamps
    columns = pd.read_csv(events_path, nrows=0).columns
    dtype = {c: np.float32 for c in columns}
    dtype["timestamp"] = np.int32

    return pd.read_csv(events_path, dtype=dtype)


def label_data(data, bins):
    digitized = data[sorted(bins.keys()) + ["polarity"]].copy()
    for c in bins.keys():
        digitized[c] = np.digitize(digitized[c], bins[c])

        # Make first and last bin open ended
        digitized.loc[digitized[c] == 0, c] = 1
        digitized.loc[digitized[c] == len(bins[c]), c] = len(bins[c]) - 1

        # Convert bin indices to array indices
        digitized[c] -= 1

    digitized.loc[digitized["polarity"] < 1, "polarity"] = 0

    # Convert to matrix for faster access
    digitized = digitized.values

    nbins = [len(bins[k]) - 1 for k in sorted(bins.keys())] + [2]
    base = np.roll(np.cumprod(nbins), 1)
    base[0] = 1
    labels = (digitized * base).sum(axis=-1, dtype=np.int16)

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Build a dataset from preprocessed events")
    parser.add_argument(
        "--load",
        default="preprocessed.csv",
        help="Event file to load from each directory")
    parser.add_argument("--x-bins", default=32, type=int, help="Number of bins in x-direction")
    parser.add_argument("--y-bins", default=32, type=int, help="Number of bins in y-direction")
    parser.add_argument("dirs", nargs="+", help="Directories with event data")
    parser.add_argument("out", help="File to save data to")
    args = parser.parse_args()

    load_filename = args.load
    dirs = args.dirs
    out_path = args.out

    # Read the data
    print("Read the data...")
    with concurrent.futures.ThreadPoolExecutor() as ex:
        data = list(ex.map(read_data, dirs, [load_filename] * len(dirs)))

    # Split off the timestamps
    timestamps = [d["timestamp"].values for d in data]
    data = [d.drop("timestamp", axis=1) for d in data]

    # Compute quantiles for digitization of labels
    print("Compute quantiles...")
    nbins = {"dt": 2, "dx-fast": 2, "dx-slow": 2, "dy-fast": 2, "dy-slow": 2}
    bins = {}
    for col, n in nbins.items():
        coldat = np.concatenate([d[col].values for d in data])
        bins[col] = pd.qcut(coldat, n, retbins=True)[1]

    # Create autoencoder labels. This is not parallelized because label_data
    # copies the data and running it in parallel can double the memory
    # requirements of this script
    print("Create autoencoder labels...")
    nclasses = np.product([len(b) - 1 for b in bins.values()]) * 2
    labels = list(map(label_data, data, [bins] * len(data)))

    # Normalize data columns
    print("Normalize data...")
    ndata = np.sum([d.shape[0] for d in data])
    with concurrent.futures.ThreadPoolExecutor() as ex:
        mu = np.sum(list(ex.map(lambda d: np.mean(d.values, axis=0, dtype=np.float64) * (d.shape[0] / ndata), data)), axis=0)
        sigma = np.sqrt(np.sum(list(ex.map(lambda d: np.var(d.values, axis=0, dtype=np.float64) * (d.shape[0] / ndata), data)), axis=0))

    # We computed mu and sigma in double precision, but now convert them back
    # so that the data will not be converted
    mu = mu.astype(np.float32)
    sigma = sigma.astype(np.float32)

    for i in range(len(data)):
        data[i] = (data[i] - mu) / sigma

    print("Save the dataset...")
    with h5.File(out_path, "w") as f:
        f.attrs["nclasses"] = nclasses

        for i in range(len(data)):
            grp = f.create_group("recording-{}".format(i))
            grp.attrs["directory"] = dirs[i]
            grp.create_dataset("timestamps", data=timestamps[i])
            grp.create_dataset("data", data=data[i])
            grp.create_dataset("labels", data=labels[i])


if __name__ == "__main__":
    main()
