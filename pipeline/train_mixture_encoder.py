#!/usr/bin/env python3

import argparse

import h5py as h5
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class DataGenerator:
    def __init__(self, path):
        with h5.File(path, "r") as f:
            self.timestamps = []
            self.data = []
            for group in f["events"].values():
                self.timestamps.append(np.array(group["timestamps"]))
                self.data.append(np.array(group["data"]))

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
        for i in range(batch_size):
            k = data_indices[i]
            data[i, :seq_lengths[i], :] = self.data[k][ranges[i][0]:ranges[i][1]]

        return seq_lengths, data


class FrameEncoder:
    def __init__(self, event_size, memory_size, nlayers, ncomponents):
        self.ncomponents = ncomponents
        self.inputs = tf.placeholder(tf.float32, shape=(None, None, event_size), name="sequences")
        self.seq_lengths = tf.placeholder(tf.int32, shape=(None,), name="sequence_lengths")
        self.batch_size = tf.shape(self.seq_lengths)[0]
        self.chunk_size = tf.reduce_max(self.seq_lengths)

        # Each component has 5 values for mu, 5 for sigma, 1 mixture weight and
        # the whole thing has a probability for the polarity
        nparameters = self.ncomponents * 2 * 5 + self.ncomponents + 1
        with tf.variable_scope("encoder"):
            cells = [tf.contrib.rnn.GRUCell(memory_size) for _ in range(nlayers)]
            encoder = tf.contrib.rnn.MultiRNNCell(cells)
            self.initial_state = tf.placeholder_with_default(tf.zeros((self.batch_size, nlayers * memory_size), dtype=tf.float32),
                                                             shape=(None, nlayers * memory_size), name="initial_state")
            _, self.encoded_state = tf.nn.dynamic_rnn(encoder, self.inputs,
                                                      sequence_length=self.seq_lengths,
                                                      initial_state=tuple(tf.split(self.initial_state, nlayers, axis=1)),
                                                      dtype=tf.float32)

            # Create nodes for easy loading of the graph
            tf.concat(self.encoded_state, axis=1, name="encoded_state")

        with tf.variable_scope("decoder"):
            decode_inputs = tf.concat([tf.zeros((self.batch_size, 1, event_size), tf.float32),
                                       self.inputs[:, :-1, :]], axis=1)
            cells = [tf.contrib.rnn.GRUCell(memory_size) for _ in range(nlayers)]
            decoder = tf.contrib.rnn.MultiRNNCell(cells)
            final_outputs, final_state = tf.nn.dynamic_rnn(decoder, decode_inputs,
                                                           sequence_length=self.seq_lengths,
                                                           initial_state=self.encoded_state,
                                                           dtype=tf.float32)

            self.final_state = tf.concat(final_state, axis=1, name="final_state")
            self.outputs = tf.contrib.keras.layers.Dense(nparameters, activation=None, name="outputs")(final_outputs)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", help="TensorFlow logging directory")
    parser.add_argument("--learning-rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--length", default=1000, type=int, help="Length of time window to encode in microseconds")
    parser.add_argument("--memory", default=128, type=int, help="Number of LSTM memory cells")
    parser.add_argument("--components", default=10, type=int, help="Number of mixture components")
    parser.add_argument("--layers", default=3, type=int, help="Number of recurrent layers")
    parser.add_argument("--chunk-size", default=100, type=int, help="Length of chunks to process at once")
    parser.add_argument("dataset", help="Path to preprocessed dataset")
    args = parser.parse_args()

    log_dir = args.log_dir or "mixture-encoder-log-{}".format(0)
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    window_length = args.length
    memory = args.memory
    ncomponents = args.components
    nlayers = args.layers
    chunk_size = args.chunk_size
    dataset_path = args.dataset

    generator = DataGenerator(dataset_path)

    num_batches = int(np.floor(generator.total_time / (batch_size * window_length)))

    encoder = FrameEncoder(generator.event_size, memory, nlayers, ncomponents)

    # mu = mean, sigma = variance, pi = mixture weights, tau = polarity distribution
    N = encoder.ncomponents
    mu = tf.split(encoder.outputs[:, :, :5 * N], N, axis=2)
    sigma = tf.split(encoder.outputs[:, :, 5 * N:2 * 5 * N], N, axis=2)
    pi = encoder.outputs[:, :, 2 * 5 * N:2 * 5 * N + N]
    tau = encoder.outputs[:, :, -1]
    cat = tf.contrib.distributions.Categorical(logits=pi)
    mvns = [tf.contrib.distributions.MultivariateNormalDiagWithSoftplusScale(mu[i], sigma[i])
            for i in range(encoder.ncomponents)]
    mixture = tf.contrib.distributions.Mixture(cat, mvns)
    bernoulli = tf.contrib.distributions.BernoulliWithSigmoidProbs(tau)
    continuous_attrs = encoder.inputs[:, :, :5]
    polarities = encoder.inputs[:, :, 5]
    polarities = tf.where(polarities > 0, tf.ones_like(polarities), tf.zeros_like(polarities))
    log_likelihood = mixture.log_prob(continuous_attrs) + bernoulli.log_prob(polarities)

    # Create a mask to block out log likelihoods beyond the end of a sequence
    sequence_position = tf.tile(tf.expand_dims(tf.range(encoder.chunk_size, dtype=tf.int32), 0), [encoder.batch_size, 1])
    mask = tf.tile(tf.expand_dims(encoder.seq_lengths, 1), [1, encoder.chunk_size]) > sequence_position

    masked_ll = tf.where(tf.slice(mask, [0, 0], tf.shape(log_likelihood)), log_likelihood, tf.zeros_like(log_likelihood))
    loss = -tf.reduce_sum(masked_ll) / tf.reduce_sum(tf.cast(encoder.seq_lengths, tf.float32))

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
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    sv = tf.train.Supervisor(init_op=init, logdir=log_dir, summary_op=None, saver=saver)
    with sv.managed_session() as sess:
        for epoch in range(epochs):
            print("# Epoch {}".format(epoch + 1))

            loss_values = []
            batches = tqdm(range(num_batches))
            for batch in batches:
                seq_lengths, data = generator.generate(batch_size, window_length)

                # Split the training data into chunks of fixed length
                chunk_state = None
                max_length = np.max(seq_lengths)
                offset = 0
                batch_loss = 0.0
                nchunks = 0
                while np.any(seq_lengths > 0):
                    chunk_lengths = np.minimum(chunk_size, seq_lengths)
                    chunk_data = data[:, offset:min(offset + chunk_size, max_length), :]

                    feeds = {encoder.inputs: chunk_data, encoder.seq_lengths: chunk_lengths}
                    if chunk_state is not None:
                        feeds[encoder.initial_state] = chunk_state
                    if batch % 200 == 0:
                        chunk_loss, chunk_state, _, summary, step = sess.run([loss, encoder.final_state, train_step, summaries, global_step], feeds)
                        summary_writer.add_summary(summary, step)
                    else:
                        chunk_loss, chunk_state, _ = sess.run([loss, encoder.final_state, train_step], feeds)
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
