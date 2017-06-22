#!/usr/bin/env python

import argparse
import glob
import os
import re
import sys

import h5py as h5
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def read_data(path):
    directories = []
    timestamps = []
    data = []

    def collect_data(name, obj):
        if isinstance(obj, h5.Group) and "data" in obj and "timestamps" in obj:
            directories.append(obj.attrs["directory"])
            timestamps.append(np.array(obj["timestamps"]))
            data.append(np.array(obj["data"]))

    with h5.File(path, "r") as f:
        f.visititems(collect_data)

    return directories, timestamps, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", default=1000, type=int, help="Split sequences into chunks of length n")
    parser.add_argument("log_dir", help="Log directory")
    parser.add_argument("dataset", help="Preprocessed events to apply to")
    parser.add_argument("out", help="HDF5 file to write to")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    log_dir = args.log_dir
    dataset_path = args.dataset
    out_path = args.out

    meta_files = glob.glob(os.path.join(log_dir, "*.meta"))
    if len(meta_files) == 0:
        print("Could not find a MetaGraph in {}".format(log_dir))
        sys.exit(1)
    latest_meta_file = max(meta_files, key=lambda p: int(re.search("([0-9]+)\.meta$", p).group(1)))
    latest_checkpoint = latest_meta_file[:-5]

    saver = tf.train.import_meta_graph(latest_meta_file)
    g = tf.get_default_graph()
    sequences = g.get_tensor_by_name("sequences:0")
    sequence_lengths = g.get_tensor_by_name("sequence_lengths:0")
    initial_state = g.get_tensor_by_name("gru_classifier/initial_state:0")
    final_state = g.get_tensor_by_name("gru_classifier/final_state:0")
    logits = g.get_tensor_by_name("gru_classifier/logits:0")

    directories, timestamps, data = read_data(dataset_path)

    with tf.Session() as sess:
        saver.restore(sess, latest_checkpoint)

        data_lengths = np.array([d.shape[0] for d in data])
        max_length = max(data_lengths)
        feature_size = data[0].shape[-1]
        complete_data = np.zeros((len(data), max_length, feature_size), np.float32)
        for i, d in enumerate(data):
            complete_data[i, :len(d)] = d

        complete_logits = []
        remaining_lengths = data_lengths.copy()
        for i in tqdm(range(0, max_length, chunk_size)):
            chunk_data = complete_data[:, i:i + chunk_size]
            chunk_lengths = np.maximum(np.minimum(remaining_lengths, chunk_size), 0)

            feeds = {sequences: chunk_data,
                     sequence_lengths: chunk_lengths}

            chunk_logits = sess.run(logits, feeds)

            remaining_lengths -= chunk_size
            complete_logits.append(chunk_logits)

        complete_logits = np.concatenate(complete_logits, axis=1)

        with h5.File(out_path, "a") as f:
            collection = f.create_group("classifications")

            for i, d in enumerate(directories):
                name = os.path.basename(d)
                grp = collection.create_group(name)
                grp.attrs["directory"] = d
                grp.create_dataset("timestamps", data=timestamps[i])
                grp.create_dataset("data", data=complete_logits[i, :data_lengths[i]])


if __name__ == "__main__":
    main()
