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

    with h5.File(path, "r") as f:
        for group in f.values():
            directories.append(group.attrs["directory"])
            timestamps.append(np.array(group["timestamps"]))
            data.append(np.array(group["data"]))

    return directories, timestamps, data


def iterate_data(batch_size, length, timestamps, data):
    start_time = timestamps[0]
    end_time = start_time + length
    start = 0
    end = np.searchsorted(timestamps, end_time)

    while start < len(timestamps):
        batch_timestamps = np.zeros(batch_size, np.int32)
        ranges = []
        seq_lengths = np.zeros(batch_size, np.int32)
        for i in range(batch_size):
            batch_timestamps[i] = end_time - length // 2
            ranges.append((start, end))
            seq_lengths[i] = end - start

            end_time += length
            start = end
            end = np.searchsorted(timestamps, end_time)

            if start == len(timestamps):
                break

        max_length = max(seq_lengths)
        batch = np.zeros((batch_size, max_length, data.shape[-1]))
        for i in range(len(ranges)):
            batch[i, :seq_lengths[i], :] = data[ranges[i][0]:ranges[i][1]]

        yield batch_timestamps, seq_lengths, batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--chunk-size", default=100, type=int, help="Split sequences into chunks of length n")
    parser.add_argument("--length", default=1000, type=int, help="Length of frames in milliseconds")
    parser.add_argument("log_dir", help="Log directory")
    parser.add_argument("dataset", help="Dataset to apply to")
    parser.add_argument("out", help="h5 file to write to")
    args = parser.parse_args()

    batch_size = args.batch_size
    chunk_size = args.chunk_size
    length = args.length
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
    initial_state = g.get_tensor_by_name("encoder/initial_state:0")
    initial_output = g.get_tensor_by_name("encoder/initial_output:0")
    final_state = g.get_tensor_by_name("decoder/concat_2:0")
    final_output = g.get_tensor_by_name("decoder/concat_3:0")

    state_tensors = []
    try:
        i = 2
        while True:
            state_tensors.append(g.get_tensor_by_name("encoder/rnn/while/Exit_{}:0".format(i)))
            i += 1
    except KeyError:
        pass
    encoded_state = tf.concat([state_tensors[i]
                               for i in range(len(state_tensors))
                               if i % 2 == 0], axis=1)
    encoded_output = tf.concat([state_tensors[i]
                                for i in range(len(state_tensors))
                                if i % 2 == 1], axis=1)

    directories, timestamps, data = read_data(dataset_path)

    with tf.Session() as sess:
        saver.restore(sess, latest_checkpoint)

        for i in tqdm(range(len(directories)), desc="Directories"):
            encoded_timestamps = []
            encoded_states = []
            encoded_outputs = []

            total_batches = int(np.ceil((timestamps[i][-1] - timestamps[i][0]) / (length * batch_size)))
            for batch_timestamps, seq_lengths, batch in tqdm(iterate_data(batch_size, length, timestamps[i], data[i]), desc="Batches", total=total_batches):
                chunk_state = None
                chunk_input = None
                batch_encoded_states = None
                batch_encoded_outputs = None
                max_length = np.max(seq_lengths)
                offset = 0
                # We run this loop at least once so that the states and outputs
                # are computed correctly for batches that are completely empty
                while np.any(seq_lengths > 0) or offset == 0:
                    chunk_lengths = np.minimum(chunk_size, seq_lengths)
                    chunk_data = batch[:, offset:min(offset + chunk_size, max_length), :]

                    feeds = {sequences: chunk_data, sequence_lengths: chunk_lengths}
                    if chunk_state is not None:
                        feeds[initial_state] = chunk_state
                    if chunk_input is not None:
                        feeds[initial_output] = chunk_input

                    chunk_state, chunk_input, batch_encoded_states, batch_encoded_outputs = sess.run([final_state, final_output, encoded_state, encoded_output], feeds)

                    offset += chunk_size
                    seq_lengths = np.maximum(0, seq_lengths - chunk_size)

                encoded_timestamps.append(batch_timestamps)
                encoded_states.append(batch_encoded_states)
                encoded_outputs.append(batch_encoded_outputs)

            encoded_timestamps = np.concatenate(encoded_timestamps, axis=0)
            encoded_states = np.concatenate(encoded_states, axis=0)
            encoded_outputs = np.concatenate(encoded_outputs, axis=0)

            with h5.File(out_path, "a") as f:
                grp = f.create_group("recording-{}".format(i))
                grp.create_dataset("timestamps", data=encoded_timestamps)
                grp.create_dataset("states", data=encoded_states)
                grp.create_dataset("outputs", data=encoded_outputs)


if __name__ == "__main__":
    main()