#!/usr/bin/env python

import argparse
import os

import keras as ks
import nn
import numpy as np
import scipy.io as sio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to saved model")
    parser.add_argument("data", help="Path to preprocessed event data")
    parser.add_argument("out", help="Where to store the confusion matrix")
    args = parser.parse_args()

    model_path = args.model
    data_path = args.data
    out_path = args.out

    # Load model
    m = ks.models.load_model(model_path)

    # Load data
    matrices = sio.loadmat(data_path)
    data = matrices["data"]
    labels = matrices["labels"]
    label_index = matrices["label_index"]
    nlabels = len(label_index)
    timesteps = data.shape[1]

    y_pred = m.predict(data, verbose=1)
    y_true = np.argmax(labels, axis=-1)

    k = 5
    top1 = np.argmax(y_pred, axis=-1)
    topk = np.argpartition(y_pred, k)[:, -k:]

    nlabels = len(label_index)
    cm = np.zeros((nlabels, nlabels), dtype=np.float32)
    cm5 = np.zeros((nlabels, nlabels), dtype=np.float32)
    ncorrect = 0
    ntopkcorrect = 0
    for i in range(topk.shape[0]):
        label = y_true[i]
        pred = top1[i]
        correct = label == pred
        topkcorrect = label in topk[i, :]

        cm[label, pred] += 1

        if topkcorrect:
            cm5[label, label] += 1
        else:
            for j in topk[i, :]:
                cm5[label, j] += 1.0 / float(k)

        if correct:
            ncorrect += 1

        if topkcorrect:
            ntopkcorrect += 1

    print("Total {}; Correct {}; Top-k Correct {}"
          .format(topk.shape[0], ncorrect, ntopkcorrect)) # yapf: disable

    sio.savemat(out_path, {"label_index": label_index, "cm": cm, "cm5": cm5})


if __name__ == "__main__":
    main()
