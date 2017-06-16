#!/usr/bin/env python

import argparse
import os

import h5py as h5
from keras_models.inception_v3 import InceptionV3
import numpy as np
from scipy.misc import imresize, imread
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract features with a deep network like Resnet or InceptionV3")
    parser.add_argument("--batch-size", default=32, help="Batch size")
    parser.add_argument("dirs", nargs="*", help="Directories with reconstructed frames")
    parser.add_argument("out", help="HDF5 file to write the features into")
    args = parser.parse_args()

    batch_size = args.batch_size
    directories = args.dirs
    out_path = args.out

    input_shape = (299, 299, 3)
    model = InceptionV3(include_top=False, weights="imagenet", pooling="max")

    with h5.File(out_path, "w") as f:
        collection = f.create_group("inceptionv3")

        for directory in tqdm(directories, desc="Directories"):
            frame_dir = os.path.join(directory, "reconstructed-frames")
            frame_files = os.listdir(frame_dir)
            frame_files = sorted(frame_files, key=lambda ff: int(os.path.splitext(ff)[0]))

            n = len(frame_files)
            timestamps = np.empty(n, np.int32)
            data = np.empty((n, model.output_shape[1]), np.float32)
            for i in tqdm(range(0, n, batch_size), desc="Batches", leave=False):
                batch_indices = list(range(i, min(i + batch_size, n)))
                batch_inputs = np.empty((len(batch_indices), *input_shape), np.float32)

                for k, j in enumerate(batch_indices):
                    filename = frame_files[j]
                    timestamps[j] = int(os.path.splitext(filename)[0])

                    path = os.path.join(frame_dir, filename)
                    img = imread(path)
                    img = imresize(img, input_shape[0:2])
                    img = np.stack([img] * input_shape[-1], axis=-1)
                    batch_inputs[k] = img

                features = model.predict(batch_inputs)

                data[batch_indices[0]:batch_indices[-1] + 1] = features

            grp_name = os.path.basename(directory)
            grp = collection.create_group(grp_name)
            grp.attrs["directory"] = directory
            grp["timestamps"] = timestamps
            grp["data"] = data


if __name__ == "__main__":
    main()
