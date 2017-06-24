#!/usr/bin/env python3

import argparse
from collections import namedtuple
from datetime import datetime

import h5py as h5
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import runner


def read_data(path):
    data = []
    labels = []

    def collect_data(name, obj):
        if isinstance(obj, h5.Group) and "data" in obj and "labels" in obj:
            data.append(np.array(obj["data"]))
            labels.append(np.array(obj["labels"]))

    with h5.File(path, "r") as f:
        f.visititems(collect_data)

        label_index = list(f["label_index"])

    return label_index, data, labels


class DataGenerator:
    def __init__(self, batch_size, length, data, labels):
        self.batch_size = batch_size
        self.length = length
        self.data = data
        self.labels = labels

        self.slices = []
        for i, d in enumerate(data):
            for j in range(0, d.shape[0], length):
                if np.all(self.labels[i][j:j + length] == -1):
                    continue

                self.slices.append([i, j, min(j + length, d.shape[0])])
        self.slices = np.array(self.slices)
        np.random.shuffle(self.slices)

        self.feature_size = self.data[0].shape[-1]
        self.nclasses = max(np.max(l) for l in self.labels) + 1

    def num_batches(self):
        return int(np.ceil(len(self.slices) / self.batch_size))

    def iterate(self):
        for i in range(0, len(self.slices), self.batch_size):
            batch_slices = self.slices[i:i + self.batch_size]

            seq_lengths = np.array([end - start for _, start, end in batch_slices], np.int32)
            data = np.zeros((len(batch_slices), max(seq_lengths), self.feature_size), np.float32)
            labels = np.zeros((len(batch_slices), max(seq_lengths)), np.int32)

            for j, bounds in enumerate(batch_slices):
                data[j, :seq_lengths[j]] = self.data[bounds[0]][bounds[1]:bounds[2]]
                labels[j, :seq_lengths[j]] = self.labels[bounds[0]][bounds[1]:bounds[2]]

            # Convert labels into sparse tensor format
            nlabels = sum(seq_lengths)
            label_indices = np.empty((nlabels, 2), np.int64)
            label_values = np.empty(nlabels, np.int32)
            i = 0
            for j in range(len(labels)):
                label_indices[i:i + seq_lengths[j], 0] = j
                label_indices[i:i + seq_lengths[j], 1] = np.arange(seq_lengths[j], dtype=np.int64)
                label_values[i:i + seq_lengths[j]] = labels[j, :seq_lengths[j]]

                i += seq_lengths[j]

            # Filter <blank> labels
            fltr = label_values > -1
            label_indices = label_indices[fltr, :]
            label_values = label_values[fltr]

            yield seq_lengths, data, label_indices, label_values


CTCClassifier = namedtuple("CTCClassifier", ["seq_lengths", "inputs", "initial_state", "logits", "final_state", "batch_size", "chunk_size"])


def build_classifier(nlayers, ndense_layers, memory_size, feature_size, nclasses):
    inputs = tf.placeholder(tf.float32, shape=(None, None, feature_size), name="sequences")
    seq_lengths = tf.placeholder(tf.int32, shape=(None,), name="sequence_lengths")
    batch_size = tf.shape(seq_lengths)[0]
    chunk_size = tf.reduce_max(seq_lengths)

    with tf.variable_scope("ctc_classifier"):
        cells = [tf.contrib.rnn.GRUCell(memory_size) for _ in range(nlayers)]
        classifier = tf.contrib.rnn.MultiRNNCell(cells)
        initial_state = tf.placeholder_with_default(tf.zeros((batch_size, nlayers * memory_size), dtype=tf.float32),
                                                    shape=(None, nlayers * memory_size), name="initial_state")
        outputs, states = tf.nn.dynamic_rnn(classifier, inputs, sequence_length=seq_lengths,
                                            initial_state=tuple(tf.split(initial_state, nlayers, axis=1)),
                                            dtype=tf.float32)

        original_output_shape = tf.shape(outputs)
        outputs = tf.reshape(outputs, (-1, memory_size))
        for i in range(ndense_layers):
            with tf.variable_scope(f"dense-{i + 1}"):
                W = tf.get_variable("W", (memory_size, memory_size), tf.float32)
                b = tf.get_variable("b", (memory_size,), tf.float32)
                outputs = tf.nn.xw_plus_b(outputs, W, b)
                outputs = tf.nn.elu(outputs)

        with tf.variable_scope("logit-projection"):
            W = tf.get_variable("W", (memory_size, nclasses), tf.float32)
            b = tf.get_variable("b", (nclasses,), tf.float32)
            logits = tf.nn.xw_plus_b(outputs, W, b)

        logits = tf.reshape(logits, [original_output_shape[0], original_output_shape[1], nclasses], name="logits")
        final_state = tf.concat(states, axis=1, name="final_state")

    return CTCClassifier(seq_lengths, inputs, initial_state, logits, final_state, batch_size, chunk_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", help="TensorFlow logging directory")
    parser.add_argument("--learning-rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--length", default=1000, type=int, help="Length of sequences to train on")
    parser.add_argument("--memory", default=128, type=int, help="Number of memory cells")
    parser.add_argument("--layers", default=3, type=int, help="Number of recurrent layers")
    parser.add_argument("--dense-layers", default=0, type=int, help="Number of dense layers on top")
    parser.add_argument("--chunk-size", default=200, type=int, help="Length of chunks to process at once")
    parser.add_argument("--validation", help="HDF5 file with labeled validation data")
    parser.add_argument("dataset", help="HDF5 file with labeled features")
    args = parser.parse_args()

    log_dir = args.log_dir or f"ctc-classifier-log-{datetime.now():%Y%m%d-%H%M}"
    initial_learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    sequence_length = args.length
    memory = args.memory
    nlayers = args.layers
    ndense_layers = args.dense_layers
    chunk_size = args.chunk_size
    validation_path = args.validation
    dataset_path = args.dataset

    label_index, train_data, train_labels = read_data(dataset_path)

    if validation_path:
        _, val_data, val_labels = read_data(validation_path)

    clsfr = build_classifier(nlayers, ndense_layers, memory, train_data[0].shape[-1], len(label_index) + 1)

    labels = tf.sparse_placeholder(tf.int32, shape=(None, None), name="labels")
    time_major_logits = tf.transpose(clsfr.logits, perm=(1, 0, 2))
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(time_major_logits,
                                                               clsfr.seq_lengths,
                                                               beam_width=100,
                                                               top_paths=1,
                                                               merge_repeated=False)
    total_label_error = tf.reduce_sum(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels, normalize=False), name="total_label_error")
    label_error_rate = tf.divide(total_label_error, tf.maximum(tf.cast(tf.shape(labels.values)[0], tf.float32), 1.0), name="label_error_rate")
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels,
                                         inputs=time_major_logits,
                                         sequence_length=clsfr.seq_lengths,
                                         preprocess_collapse_repeated=False,
                                         ctc_merge_repeated=False,
                                         time_major=True))

    global_step = tf.get_variable("global_step", shape=(), dtype=tf.int64, trainable=False, initializer=tf.constant_initializer(0))
    learning_rate = tf.placeholder_with_default(np.float32(initial_learning_rate), shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
    train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("label_error_rate", label_error_rate)
    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    sv = tf.train.Supervisor(init_op=init, logdir=log_dir, summary_op=None, saver=saver, save_model_secs=300)
    with sv.managed_session() as sess:
        for epoch in range(epochs):
            print("# Epoch {}".format(epoch + 1))

            nbatches = 0
            epoch_loss = 0.0
            epoch_label_error = 0.0
            epoch_total_length = 0

            generator = DataGenerator(batch_size, sequence_length, train_data, train_labels)
            batches = tqdm(range(generator.num_batches()))
            iterator = generator.iterate()
            for batch in batches:
                seq_lengths, data, label_indices, label_values = next(iterator)

                batch_total_length = len(label_values)

                # The training data is split into fixed length chunks so that
                # sequences of arbitrary length can be handled
                chunk_state = None
                offset = 0

                # Metrics
                batch_loss = 0.0
                batch_label_error = 0

                # The loop always runs at least once, even if all sequences are
                # empty, so that the final states are properly set
                while np.any(seq_lengths > 0) or offset == 0:
                    chunk_lengths = np.minimum(chunk_size, seq_lengths)
                    chunk_data = data[:, offset:offset + max(chunk_lengths)]
                    fltr = (label_indices[:, 1] >= offset) & (label_indices[:, 1] < offset + max(chunk_lengths))
                    chunk_label_indices = label_indices[fltr, :]
                    chunk_label_values = label_values[fltr]

                    feeds = {clsfr.inputs: chunk_data,
                             clsfr.seq_lengths: chunk_lengths,
                             labels: (chunk_label_indices, chunk_label_values, (len(chunk_data), np.max(chunk_lengths))),
                             learning_rate: initial_learning_rate * 0.95**epoch}

                    if chunk_state is not None:
                        feeds[clsfr.initial_state] = chunk_state

                    chunk_state, chunk_loss, chunk_label_error, _ = sess.run([clsfr.final_state, loss, total_label_error, train_step], feeds)

                    offset += chunk_size
                    seq_lengths = np.maximum(0, seq_lengths - chunk_size)

                    batch_loss += chunk_loss
                    batch_label_error += chunk_label_error

                nbatches += 1
                epoch_loss += batch_loss
                epoch_label_error += batch_label_error
                epoch_total_length += batch_total_length
                batches.set_description(f"Loss {epoch_loss / nbatches:.3f} ({batch_loss:.1f}), LER {epoch_label_error / epoch_total_length:.3f} ({batch_label_error / batch_total_length:.2f})")

            if sv.should_stop():
                break

            if not validation_path:
                continue

            # Validation set
            nbatches = 0
            epoch_loss = 0.0
            epoch_label_error = 0.0
            epoch_total_length = 0

            generator = DataGenerator(batch_size, sequence_length, val_data, val_labels)
            batches = tqdm(range(generator.num_batches()))
            iterator = generator.iterate()
            for batch in batches:
                seq_lengths, data, label_indices, label_values = next(iterator)

                batch_total_length = len(label_values)

                # The training data is split into fixed length chunks so that
                # sequences of arbitrary length can be handled
                chunk_state = None
                offset = 0

                # Metrics
                batch_loss = 0.0
                batch_label_error = 0

                # The loop always runs at least once, even if all sequences are
                # empty, so that the final states are properly set
                while np.any(seq_lengths > 0) or offset == 0:
                    chunk_lengths = np.minimum(chunk_size, seq_lengths)
                    chunk_data = data[:, offset:offset + max(chunk_lengths)]
                    fltr = (label_indices[:, 1] >= offset) & (label_indices[:, 1] < offset + max(chunk_lengths))
                    chunk_label_indices = label_indices[fltr, :]
                    chunk_label_values = label_values[fltr]

                    feeds = {clsfr.inputs: chunk_data,
                             clsfr.seq_lengths: chunk_lengths,
                             labels: (chunk_label_indices, chunk_label_values, (len(chunk_data), np.max(chunk_lengths)))}

                    if chunk_state is not None:
                        feeds[clsfr.initial_state] = chunk_state

                    chunk_state, chunk_loss, chunk_label_error = sess.run([clsfr.final_state, loss, total_label_error], feeds)

                    offset += chunk_size
                    seq_lengths = np.maximum(0, seq_lengths - chunk_size)

                    batch_loss += chunk_loss
                    batch_label_error += chunk_label_error

                nbatches += 1
                epoch_loss += batch_loss
                epoch_label_error += batch_label_error
                epoch_total_length += batch_total_length
                batches.set_description(f"Loss {epoch_loss / nbatches:.3f} ({batch_loss:.1f}), LER {epoch_label_error / epoch_total_length:.3f} ({batch_label_error / batch_total_length:.2f})")

            if sv.should_stop():
                break


if __name__ == "__main__":
    main()
