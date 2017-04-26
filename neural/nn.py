#!/usr/bin/env python

import argparse
import datetime
import os

# Fix the bug
import keras.backend


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    if dtype is None:
        dtype = floatx()
    if not shape:
        if ndim:
            shape = tuple([None for _ in range(ndim)])
    if sparse:
        # Convert input shape into an int64 array. This is a workaround for a
        # tensorflow bug where tf converts tuples into int32 tensors and later
        # complains that it expected an int64 tensor.
        if not any(i is None for i in shape):
            shape = np.array(shape, dtype=np.int64)

        x = tf.sparse_placeholder(dtype, shape=shape, name=name)
    else:
        x = tf.placeholder(dtype, shape=shape, name=name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


keras.backend.tensorflow_backend.__dict__["placeholder"] = placeholder

import keras as ks
from keras.layers import TimeDistributed, Lambda, Conv2D, MaxPool2D, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf

length = 30
directory = "/home/cqql/Seafile/thesis/davis/marten"
nrows = 180
ncols = 240
timesteps = 10
nchannels = 3
batch_size = 25
nepochs = 10
learning_rate = 0.02


def load_data(directory, length, timesteps, nrows, ncols):
    labels_path = os.path.join(directory,
                               "frame-labels-{}ms.csv".format(length))
    frames_path = os.path.join(directory, "abs-frames-{}ms.mat".format(length))
    flow_path = os.path.join(directory, "flow-{}ms.mat".format(length))

    labels = pd.read_csv(labels_path)
    frames = np.squeeze(sio.loadmat(frames_path)["frames"])
    flow = sio.loadmat(flow_path)
    flowx = np.squeeze(flow["flowx"])
    flowy = np.squeeze(flow["flowy"])
    labels_index = sorted(labels["label"].unique())
    label_indices = [labels_index.index(l) for l in labels["label"]]

    # Generate input data. We do it this way to conserve memory.
    NCHANNELS = 3
    data = np.empty((len(frames), nrows, ncols, NCHANNELS), dtype=np.float32)
    for i, f in enumerate(frames):
        data[i, :, :, 0] = f.todense()
    for i, f in enumerate(flowx):
        data[i, :, :, 1] = f.todense()
    for i, f in enumerate(flowy):
        data[i, :, :, 2] = f.todense()

    # Reshape into time sequences
    nframes = len(frames)
    nrest_frames = nframes % timesteps
    data = data[:-nrest_frames, :, :, :]
    data = np.reshape(data, (-1, timesteps, nrows, ncols, NCHANNELS))
    label_indices = label_indices[:-nrest_frames]
    label_indices = np.reshape(label_indices, (-1, timesteps))

    # Convert labels to 1-hot encoding
    labels_1hot = np.zeros((data.shape[0], len(labels_index)))
    labels_1hot[range(labels_1hot.shape[0]), label_indices[:, 0]] = 1

    return data, labels_1hot, labels_index


def build_model(timesteps, nrows, ncols, nchannels, nlabels, learning_rate):
    m = ks.models.Sequential()
    m.add(
        ks.layers.InputLayer(input_shape=(timesteps, nrows, ncols, nchannels)))
    m.add(
        TimeDistributed(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(3, 4),
                activation="relu")))
    m.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3))))
    m.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
    m.add(TimeDistributed(ks.layers.Activation("relu")))
    m.add(
        TimeDistributed(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu")))
    m.add(
        TimeDistributed(
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))))
    m.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
    m.add(TimeDistributed(ks.layers.Activation("relu")))
    m.add(
        TimeDistributed(
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))))
    m.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
    m.add(TimeDistributed(ks.layers.Activation("relu")))
    m.add(
        TimeDistributed(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu")))
    m.add(TimeDistributed(ks.layers.Reshape((-1, ))))
    m.add(ks.layers.LSTM(2048, activation="tanh"))
    m.add(Dense(1024, activation="tanh"))
    m.add(Dense(nlabels, activation="softmax"))

    m.compile(
        keras.optimizers.SGD(lr=learning_rate, momentum=0.99, nesterov=False),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=10,
        help="Length of time window in milliseconds")
    parser.add_argument(
        "--rows", type=int, default=180, help="Number of rows in frames")
    parser.add_argument(
        "--cols", type=int, default=240, help="Number of columns in frames")
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=25, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("directory", help="Directory with event data")
    args = parser.parse_args()

    length = args.length
    nrows = args.rows
    ncols = args.cols
    directory = args.directory
    nchannels = 3
    timesteps = 10
    batch_size = args.batch_size
    nepochs = args.epochs
    learning_rate = args.lr

    data, labels_1hot, labels_index = load_data(directory, length, timesteps,
                                                nrows, ncols)
    nlabels = len(labels_index)
    m = build_model(timesteps, nrows, ncols, nchannels, nlabels, learning_rate)

    basedir = "./run-{}".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(basedir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        os.path.join(basedir, "checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5"),
        save_best_only=True)
    csv_logger = CSVLogger(os.path.join(basedir, "log.csv"))
    plateau = ReduceLROnPlateau(factor=0.333, patience=4, verbose=1)

    m.fit(
        x=data,
        y=labels_1hot,
        batch_size=batch_size,
        epochs=nepochs,
        callbacks=[checkpointer, csv_logger, plateau],
        validation_split=0.05,
        shuffle=True)


if __name__ == "__main__":
    main()
