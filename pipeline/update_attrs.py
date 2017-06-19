#!/usr/bin/env python

import argparse
import os

import h5py as h5


def main():
    parser = argparse.ArgumentParser(description="Update directory attributes in HDF5 files that were generated on another host")
    parser.add_argument("file", help="HDF5 file to fix")
    parser.add_argument("basedir", help="Base directory to search for data directories")
    args = parser.parse_args()

    file_path = args.file
    base_dir = args.basedir

    with h5.File(file_path, "r+") as f:
        data_groups = []
        def collect_data_groups(name, obj):
            if "directory" in obj.attrs and "timestamps" in obj and "data" in obj:
                data_groups.append(obj)
        f.visititems(collect_data_groups)

        for grp in data_groups:
            name = os.path.basename(grp.name)
            new_dir = os.path.join(base_dir, name)

            if os.path.isdir(new_dir):
                grp.attrs["directory"] = new_dir
            else:
                print("Could not find data directory for group {}".format(grp.name))
                return 1


if __name__ == "__main__":
    main()
