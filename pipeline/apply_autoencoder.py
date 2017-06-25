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
        for group in f["events"].values():
            directories.append(group.attrs["directory"])
            timestamps.append(np.array(group["timestamps"]))
            data.append(np.array(group["data"]))

    return directories, timestamps, data


def num_iterations(length, chunk_size, timestamps, data):
    start_times = np.arange(timestamps[0], timestamps[-1], length)
    end_times = start_times + length

    start_indices = np.searchsorted(timestamps, start_times)
    end_indices = np.searchsorted(timestamps, end_times)

    seq_lengths = end_indices - start_indices
    num_chunks = np.ceil(seq_lengths / chunk_size)

    return int(np.sum(num_chunks))


def iterate_data(length, timestamps, data):
    start_times = np.arange(timestamps[0], timestamps[-1], length)
    end_times = start_times + length

    start_indices = np.searchsorted(timestamps, start_times)
    end_indices = np.searchsorted(timestamps, end_times)

    for start_time, start, end in zip(start_times, start_indices, end_indices):
        yield start_time + length // 2, end - start, data[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--chunk-size", default=100, type=int, help="Split sequences into chunks of length n")
    parser.add_argument("--length", default=1000, type=int, help="Length of frames in milliseconds")
    parser.add_argument("--clear", default=False, action="store_true", dest="clear", help="Clear state between gists")
    parser.add_argument("log_dir", help="Log directory")
    parser.add_argument("dataset", help="Preprocessed events to apply to")
    parser.add_argument("out", help="HDF5 file to write to")
    args = parser.parse_args()

    batch_size = args.batch_size
    chunk_size = args.chunk_size
    length = args.length
    clear_state = args.clear
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
    encoded_state = g.get_tensor_by_name("encoder/encoded_state:0")

    directories, timestamps, data = read_data(dataset_path)
    event_size = data[0].shape[-1]

    total_iterations = sum([num_iterations(length, chunk_size, t, d) for t, d in zip(timestamps, data)])
    iterators = [iterate_data(length, t, d) for t, d in zip(timestamps, data)]

    # Build a progress bar
    progress_bar = tqdm(total=total_iterations, desc="Chunks")

    encoded_timestamps = [[] for _ in range(len(data))]
    encoded_seq_lengths = [[] for _ in range(len(data))]
    encoded_states = [[] for _ in range(len(data))]

    with tf.Session() as sess:
        saver.restore(sess, latest_checkpoint)

        exhausted = np.array([False] * len(data))

        chunk_state = np.zeros((len(data), initial_state.shape[-1]), np.float32)
        current_seq_length = np.zeros(len(data), np.int32)
        current_data = [None] * len(data)
        offset = np.zeros(len(data), np.int32)
        while True:
            for i, it in enumerate(iterators):
                if exhausted[i]:
                    continue

                if offset[i] < current_seq_length[i]:
                    continue

                try:
                    # Store the encoding of the previous sequence
                    if len(encoded_timestamps[i]) > 0:
                        # But not on the first run through the loop
                        encoded_states[i].append(chunk_state[i])

                    timestamp, length, data_slice = next(it)

                    # Store the constant attributes of this sequence
                    encoded_timestamps[i].append(timestamp)
                    encoded_seq_lengths[i].append(length)

                    if clear_state:
                        chunk_state[i] = 0.0

                    current_seq_length[i] = length
                    current_data[i] = data_slice
                    offset[i] = 0
                except StopIteration:
                    exhausted[i] = True

            # If we have exhausted all iterators, we are done
            if np.all(exhausted):
                break

            # Compute the lengths of sequences that go into this chunk and
            # filter out exhausted/empty ones
            chunk_lengths = np.minimum(current_seq_length - offset, chunk_size)
            active_filter = (chunk_lengths > 0) & ~exhausted
            nactive = np.count_nonzero(active_filter)

            # Collect the data for this chunk
            max_length = np.max(chunk_lengths)
            chunk_data = np.zeros((nactive, max_length, event_size), np.float32)
            for i, j in enumerate(np.nonzero(active_filter)[0]):
                chunk_data[i, :chunk_lengths[j]] = current_data[j][offset[j]:offset[j] + chunk_size]

            # Run the encoder
            feeds = {sequences: chunk_data,
                     sequence_lengths: chunk_lengths[active_filter],
                     initial_state: chunk_state[active_filter]}
            encodings = sess.run(encoded_state, feeds)

            # Store the updated encoder states
            chunk_state[active_filter] = encodings

            # Advance each sequence by one chunk size
            offset += chunk_size

            # Update the progress bar
            progress_bar.update(nactive)

    with h5.File(out_path, "w") as f:
        collection = f.create_group("gists")

        for i in range(len(data)):
            name = os.path.basename(directories[i])
            grp = collection.create_group(name)
            grp.attrs["directory"] = directories[i]
            grp.create_dataset("timestamps", data=np.array(encoded_timestamps[i], np.int32))
            grp.create_dataset("num_events", data=np.array(encoded_seq_lengths[i], np.int32))
            grp.create_dataset("data", data=np.stack(encoded_states[i]))


if __name__ == "__main__":
    main()
