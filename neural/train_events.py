#!/usr/bin/env python

import argparse
import datetime
import os
import shutil

import keras as ks
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import scipy.io as sio
import tensorflow as tf


def build_lstm_model(timesteps, nlabels, nfeatures, learning_rate):
    m = ks.models.Sequential()
    m.add(ks.layers.InputLayer(input_shape=(timesteps, nfeatures)))
    # m.add(ks.layers.LSTM(128, activation="relu", return_sequences=True))
    # m.add(ks.layers.LSTM(32, activation="tanh", return_sequences=True))
    m.add(ks.layers.LSTM(512, activation="tanh"))
    m.add(Dense(256, activation="tanh"))
    m.add(Dense(256, activation="tanh"))
    m.add(Dense(nlabels, activation="softmax"))

    m.compile(
        ks.optimizers.Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    return m


def build_cnn_model(timesteps, nlabels, nfeatures, learning_rate):
    m = ks.models.Sequential()
    m.add(ks.layers.InputLayer(input_shape=(timesteps, nfeatures)))
    m.add(ks.layers.Reshape((timesteps, nfeatures, 1)))
    m.add(ks.layers.Conv2D(16, (1, nfeatures), activation="relu"))
    m.add(ks.layers.Flatten())
    m.add(Dense(128, activation="tanh"))
    m.add(ks.layers.Dropout(0.5))
    m.add(Dense(128, activation="tanh"))
    m.add(ks.layers.Dropout(0.5))
    m.add(Dense(nlabels, activation="softmax"))

    m.compile(
        ks.optimizers.RMSprop(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--model", help="Path to model to load")
    parser.add_argument(
        "--type", default="lstm", choices=["lstm", "cnn"], help="Type of NN")
    parser.add_argument("--val-set", help="Path to validation set")
    parser.add_argument("data", help="Path to preprocessed event data")
    args = parser.parse_args()

    batch_size = args.batch_size
    nepochs = args.epochs
    learning_rate = args.lr
    nn_type = args.type
    model_path = args.model
    val_path = args.val_set
    data_path = args.data

    # Load data
    matrices = sio.loadmat(data_path)
    data = matrices["data"]
    labels = matrices["labels"]
    label_index = matrices["label_index"]
    nlabels = len(label_index)
    nfeatures = data.shape[-1]
    timesteps = data.shape[1]

    # Load model
    if model_path:
        m = ks.models.load_model(model_path)
    else:
        if nn_type == "lstm":
            m = build_lstm_model(timesteps, nlabels, nfeatures, learning_rate)
        else:
            m = build_cnn_model(timesteps, nlabels, nfeatures, learning_rate)

    basedir = "./run-{}".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(basedir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        os.path.join(
            basedir,
            "checkpoint.{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"),
        monitor="val_categorical_accuracy")
    csv_logger = CSVLogger(os.path.join(basedir, "log.csv"))
    plateau = ReduceLROnPlateau(factor=0.333, patience=4, verbose=1)

    # Store a copy of the data in the run directory for plotting purposes
    shutil.copy2(data_path, basedir)

    validation_data = None
    if val_path:
        values = sio.loadmat(val_path, variable_names=["labels", "data"])
        validation_data = (values["data"], values["labels"])

    m.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=nepochs,
        callbacks=[checkpointer, csv_logger, plateau],
        validation_split=0.1,
        validation_data=validation_data,
        shuffle=True)


if __name__ == "__main__":
    main()
