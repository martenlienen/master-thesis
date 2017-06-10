#!/usr/bin/env python3

import argparse

import h5py as h5
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class DataGenerator:
    def __init__(self, path):
        with h5.File(path, "r") as f:
            self.nclasses = f.attrs["nclasses"]

            self.data = []
            self.labels = []
            self.timestamps = []
            for group in f.values():
                self.data.append(np.array(group["data"]))
                self.labels.append(np.array(group["labels"], dtype=np.int32))
                self.timestamps.append(np.array(group["timestamps"]))

        self.event_size = self.data[0].shape[-1]
        self.duration = [t[-1] - t[0] for t in self.timestamps]
        self.total_time = np.sum(self.duration)
        self.weights = np.array(self.duration, dtype=np.float32) / self.total_time

    def generate(self, batch_size, length):
        data_indices = np.random.choice(np.arange(len(self.data)), size=batch_size,
                                        replace=True, p=self.weights)
        ranges = []
        seq_lengths = np.empty(batch_size, np.int32)
        for i in range(batch_size):
            k = data_indices[i]
            start_index, end_index = None, None
            while start_index == end_index:
                start = np.random.randint(self.duration[k] - length + 1)
                end = start + length
                start_index, end_index = np.searchsorted(self.timestamps[k], [start, end + 1])
            seq_lengths[i] = end_index - start_index
            ranges.append((start_index, end_index))

        max_length = max(seq_lengths)
        data = np.zeros((batch_size, max_length, self.event_size), np.float32)
        labels = np.zeros((batch_size, max_length), np.int32)
        for i in range(batch_size):
            k = data_indices[i]
            data[i, :seq_lengths[i], :] = self.data[k][ranges[i][0]:ranges[i][1]]
            labels[i, :seq_lengths[i]] = self.labels[k][ranges[i][0]:ranges[i][1]]

        return seq_lengths, data, labels


class FrameEncoder:
    def __init__(self, batch_size, event_size, nclasses, memory_size):
        self.inputs = tf.placeholder(tf.float32, shape=(batch_size, None, event_size), name="sequences")
        self.seq_lengths = tf.placeholder(tf.int32, shape=(batch_size,), name="sequence_lengths")

        with tf.variable_scope("encoder"):
            cells = [
                tf.contrib.rnn.LSTMCell(memory_size, use_peepholes=True),
                tf.contrib.rnn.LSTMCell(memory_size, use_peepholes=True),
                tf.contrib.rnn.LSTMCell(memory_size, num_proj=nclasses, use_peepholes=True),
            ]
            encoder = tf.contrib.rnn.MultiRNNCell(cells)
            self.initial_state = tf.placeholder_with_default(np.zeros((batch_size, len(cells) * memory_size), np.float32), shape=(batch_size, len(cells) * memory_size), name="initial_state")
            self.initial_output = tf.placeholder_with_default(np.zeros((batch_size, (len(cells) - 1) * memory_size + nclasses), np.float32), shape=(batch_size, (len(cells) - 1) * memory_size + nclasses), name="initial_output")
            complete_state = tuple([tf.contrib.rnn.LSTMStateTuple(c, h) for c, h in zip(tf.split(self.initial_state, len(cells), axis=1), tf.split(self.initial_output, [memory_size] * (len(cells) - 1) + [nclasses], axis=1))])
            _, self.encoded_state = tf.nn.dynamic_rnn(encoder, self.inputs,
                                                      sequence_length=self.seq_lengths,
                                                      initial_state=complete_state, dtype=tf.float32)

        with tf.variable_scope("decoder"):
            decode_inputs = tf.concat([tf.constant(np.zeros((self.inputs.shape[0], 1, event_size), np.float32)),
                                       self.inputs[:, :-1, :]], axis=1)
            cells = [
                tf.contrib.rnn.LSTMCell(memory_size, use_peepholes=True),
                tf.contrib.rnn.LSTMCell(memory_size, use_peepholes=True),
                tf.contrib.rnn.LSTMCell(memory_size, num_proj=nclasses, use_peepholes=True),
            ]
            decoder = tf.contrib.rnn.MultiRNNCell(cells)
            self.outputs, state = tf.nn.dynamic_rnn(decoder, decode_inputs,
                                                    sequence_length=self.seq_lengths,
                                                    initial_state=self.encoded_state, dtype=tf.float32)

            self.final_state = tf.concat([c for c, h in state], axis=1)
            self.final_output = tf.concat([h for c, h in state], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", help="TensorFlow logging directory")
    parser.add_argument("--learning-rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--length", default=1000, type=int, help="Length of time window to encode in microseconds")
    parser.add_argument("--memory", default=128, type=int, help="Number of LSTM memory cells")
    parser.add_argument("--chunk-size", default=100, type=int, help="Length of chunks to process at once")
    parser.add_argument("dataset", help="Path to preprocessed dataset")
    args = parser.parse_args()

    log_dir = args.log_dir or "frame-encoder-log-{}".format(0)
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    window_length = args.length
    memory = args.memory
    chunk_size = args.chunk_size
    dataset_path = args.dataset

    generator = DataGenerator(dataset_path)

    num_batches = int(np.floor(generator.total_time / (batch_size * window_length)))

    encoder = FrameEncoder(batch_size, generator.event_size, generator.nclasses, memory)

    # The masking makes sure that loss values after the end of a sequence are ignored
    sparse_labels = tf.placeholder(tf.int32, shape=(batch_size, None), name="sparse_labels")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_labels, logits=encoder.outputs)
    sequence_position = tf.constant(np.tile(np.arange(chunk_size, dtype=np.int32), [batch_size, 1]))
    zeros = tf.constant(np.zeros((batch_size, chunk_size), np.float32))
    mask = tf.tile(tf.expand_dims(encoder.seq_lengths, 1), [1, chunk_size]) > sequence_position
    masked_cross_entropy = tf.where(mask, cross_entropy, zeros)
    loss = tf.reduce_sum(masked_cross_entropy) / tf.reduce_sum(tf.cast(encoder.seq_lengths, tf.float32))

    global_step = tf.get_variable("global_step", shape=(), dtype=tf.int64, trainable=False, initializer=tf.constant_initializer(0))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
    train_step = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    tf.summary.histogram("gradient_norm", gradient_norm)
    tf.summary.scalar("loss", loss)
    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir)

    init = tf.global_variables_initializer()
    sv = tf.train.Supervisor(init_op=init, logdir=log_dir, summary_op=None)
    with sv.managed_session() as sess:
        for epoch in range(epochs):
            print("# Epoch {}".format(epoch + 1))

            loss_values = []
            batches = tqdm(range(num_batches))
            for batch in batches:
                seq_lengths, data, labels = generator.generate(batch_size, window_length)

                # Split the training data into chunks of fixed length
                chunk_state = None
                chunk_input = None
                max_length = np.max(seq_lengths)
                offset = 0
                batch_loss = 0.0
                nchunks = 0
                while np.any(seq_lengths > 0):
                    chunk_lengths = np.minimum(chunk_size, seq_lengths)
                    chunk_data = data[:, offset:min(offset + chunk_size, max_length), :]
                    chunk_labels = labels[:, offset:min(offset + chunk_size, max_length)]

                    feeds = {encoder.inputs: chunk_data, encoder.seq_lengths: chunk_lengths, sparse_labels: chunk_labels}
                    if chunk_state is not None:
                        feeds[encoder.initial_state] = chunk_state
                    if chunk_input is not None:
                        feeds[encoder.initial_output] = chunk_input
                    if batch % 200 == 0:
                        chunk_loss, chunk_state, chunk_input, _, summary, step = sess.run([loss, encoder.final_state, encoder.final_output, train_step, summaries, global_step], feeds)
                        summary_writer.add_summary(summary, step)
                    else:
                        chunk_loss, chunk_state, chunk_input, _ = sess.run([loss, encoder.final_state, encoder.final_output, train_step], feeds)
                    batch_loss += chunk_loss

                    nchunks += 1
                    offset += chunk_size
                    seq_lengths = np.maximum(0, seq_lengths - chunk_size)

                if batch_loss == np.nan:
                    print("Got nan loss: seq_lengths={}".format(seq_lengths))

                    if len(loss_values) > 0:
                        batch_loss = np.mean(loss_values)
                    else:
                        batch_loss = 0.0

                batch_loss /= nchunks
                loss_values.append(batch_loss)
                batches.set_description("Loss {:.3f} ({:.5f})".format(np.mean(loss_values), batch_loss))

                if sv.should_stop():
                    break


if __name__ == "__main__":
    main()
