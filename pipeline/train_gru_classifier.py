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


class RandomSliceGenerator:
    def __init__(self, batch_size, length, data, labels):
        self.batch_size = batch_size
        self.length = length
        self.data = data
        self.labels = labels

        self.slices = []
        for i, d in enumerate(data):
            for j in range(0, d.shape[0], length):
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

            labels[labels == -1] = self.nclasses

            yield seq_lengths, data, labels


GRUClassifier = namedtuple("GRUClassifier", ["seq_lengths", "inputs", "initial_state", "logits", "final_state", "batch_size", "chunk_size"])


def build_classifier(nlayers, memory_size, feature_size, nclasses):
    inputs = tf.placeholder(tf.float32, shape=(None, None, feature_size), name="sequences")
    seq_lengths = tf.placeholder(tf.int32, shape=(None,), name="sequence_lengths")
    batch_size = tf.shape(seq_lengths)[0]
    chunk_size = tf.reduce_max(seq_lengths)

    with tf.variable_scope("gru_classifier"):
        cells = [tf.contrib.rnn.GRUCell(memory_size) for _ in range(nlayers)]
        classifier = tf.contrib.rnn.MultiRNNCell(cells)
        initial_state = tf.placeholder_with_default(tf.zeros((batch_size, nlayers * memory_size), dtype=tf.float32),
                                                    shape=(None, nlayers * memory_size), name="initial_state")
        outputs, states = tf.nn.dynamic_rnn(classifier, inputs, sequence_length=seq_lengths,
                                            initial_state=tuple(tf.split(initial_state, nlayers, axis=1)),
                                            dtype=tf.float32)

        W = tf.get_variable("W", (memory_size, nclasses), tf.float32)
        b = tf.get_variable("b", (nclasses,), tf.float32)
        logits = tf.nn.xw_plus_b(tf.reshape(outputs, (-1, memory_size)), W, b)
        logits = tf.reshape(logits, [tf.shape(outputs)[0], tf.shape(outputs)[1], nclasses], name="logits")
        final_state = tf.concat(states, axis=1, name="final_state")

    return GRUClassifier(seq_lengths, inputs, initial_state, logits, final_state, batch_size, chunk_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", help="TensorFlow logging directory")
    parser.add_argument("--learning-rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--length", default=1000, type=int, help="Length of sequences to train on")
    parser.add_argument("--memory", default=128, type=int, help="Number of memory cells")
    parser.add_argument("--layers", default=3, type=int, help="Number of recurrent layers")
    parser.add_argument("--chunk-size", default=200, type=int, help="Length of chunks to process at once")
    parser.add_argument("dataset", help="HDF5 file with labeled features")
    args = parser.parse_args()

    log_dir = args.log_dir or f"gru-classifier-log-{datetime.now():%Y%m%d-%H%M}"
    initial_learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    sequence_length = args.length
    memory = args.memory
    nlayers = args.layers
    chunk_size = args.chunk_size
    dataset_path = args.dataset

    label_index, data, data_labels = read_data(dataset_path)

    clsfr = build_classifier(nlayers, memory, data[0].shape[-1], len(label_index) + 1)

    # Compute the cross-entropy between labels and predictions
    labels = tf.placeholder(tf.int32, shape=(None, None), name="labels")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=clsfr.logits)

    # Create a mask to block out log likelihoods beyond the end of a sequence
    sequence_position = tf.tile(tf.expand_dims(tf.range(clsfr.chunk_size, dtype=tf.int32), 0), [clsfr.batch_size, 1])
    mask = tf.tile(tf.expand_dims(clsfr.seq_lengths, 1), [1, clsfr.chunk_size]) > sequence_position

    masked_cross_entropy = tf.where(mask, cross_entropy, tf.zeros_like(cross_entropy))
    loss = tf.reduce_sum(masked_cross_entropy) / tf.cast(tf.reduce_sum(clsfr.seq_lengths), tf.float32)

    correct = tf.cast(tf.equal(tf.cast(tf.argmax(clsfr.logits, axis=-1), tf.int32), labels), tf.float32)
    masked_correct = tf.where(mask, correct, tf.zeros_like(correct))
    num_correct = tf.reduce_sum(masked_correct, name="num_correct")
    accuracy = num_correct / tf.cast(tf.reduce_sum(clsfr.seq_lengths), tf.float32)

    global_step = tf.get_variable("global_step", shape=(), dtype=tf.int64, trainable=False, initializer=tf.constant_initializer(0))
    learning_rate = tf.placeholder_with_default(np.float32(initial_learning_rate), shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
    train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    tf.summary.histogram("gradient_norm", gradient_norm)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("loss", accuracy)
    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)
    sv = tf.train.Supervisor(init_op=init, logdir=log_dir, summary_op=None, saver=saver, save_model_secs=300)
    with sv.managed_session() as sess:
        for epoch in range(epochs):
            print("# Epoch {}".format(epoch + 1))

            loss_metric = runner.TensorMetric("loss", loss)
            accuracy_metric = runner.AccuracyMetric("accuracy", num_correct)
            model = runner.Model(clsfr.inputs, clsfr.logits, clsfr.initial_state, clsfr.final_state, clsfr.seq_lengths, labels)

            train_generator = RandomSliceGenerator(batch_size, sequence_length, data[:-2], data_labels[:-2])
            val_generator = RandomSliceGenerator(batch_size, sequence_length, data[-2:], data_labels[-2:])

            train_runner = runner.Runner(metrics=[loss_metric, accuracy_metric])
            train_runner.run(sess, model, train_generator, chunk_size, extra_feeds={learning_rate: initial_learning_rate * 0.95**epoch}, extra_fetches={"train": train_step})

            val_runner = runner.Runner(metrics=[loss_metric, accuracy_metric])
            val_runner.run(sess, model, val_generator, chunk_size)

            if sv.should_stop():
                break


if __name__ == "__main__":
    main()
