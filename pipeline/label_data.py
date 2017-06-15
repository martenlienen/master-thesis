#!/usr/bin/env python

import argparse
import os

import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Label data in HDF5 files")
    parser.add_argument("file", help="HDF5 file with timestamped data")
    args = parser.parse_args()

    file_path = args.file

    with h5.File(file_path, "r+") as f:
        data_groups = []
        def collect_data_groups(name, obj):
            if "directory" in obj.attrs and "timestamps" in obj and "data" in obj:
                data_groups.append(obj)
        f.visititems(collect_data_groups)

        label_index = []
        for grp in data_groups:
            labels_path = os.path.join(grp.attrs["directory"], "labels.csv")
            labels = pd.read_csv(labels_path)
            label_index += list(labels["name"])
        label_index = list(sorted(set(label_index)))
        index_lookup = {name: i for i, name in enumerate(label_index)}

        str_type = h5.special_dtype(vlen=str)
        f.create_dataset("label_index", data=np.array(label_index, dtype=np.object), dtype=str_type)

        for grp in tqdm(data_groups, desc="Datasets"):
            labels_path = os.path.join(grp.attrs["directory"], "labels.csv")
            labels = pd.read_csv(labels_path)
            timestamps = np.array(grp["timestamps"])
            start_candidates = np.searchsorted(labels["start"], timestamps)
            end_candidates = np.searchsorted(labels["end"], timestamps)
            fltr = start_candidates - 1 == end_candidates
            names = labels.loc[end_candidates[fltr], "name"].values
            indices = np.full(len(timestamps), -1, np.int8)
            for i, j in enumerate(fltr.nonzero()[0]):
                indices[j] = index_lookup[names[i]]
            grp["labels"] = indices


if __name__ == "__main__":
    main()
