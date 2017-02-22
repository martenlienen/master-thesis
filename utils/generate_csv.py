#!/usr/bin/env python3

import argparse
import csv
import glob
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int, default=1, help="Iterations")
    parser.add_argument("directory")
    args = parser.parse_args()

    iterations = args.i
    directory = args.directory

    with open(os.path.join(directory, "instructions.csv"), "w") as csvfile:
        writer = csv.writer(csvfile)

        files = glob.glob(os.path.join(directory, "*.mkv"))
        for f in files:
            filename = os.path.basename(f)
            id, _ = os.path.splitext(filename)
            title = " ".join([word.capitalize() for word in id.split("-")])

            if iterations == 1:
                writer.writerow([title, id, filename])
            else:
                for i in range(1, iterations + 1):
                    writer.writerow(["{} ({})".format(title, i),
                                     "{}-{}".format(id, i), filename])


if __name__ == "__main__":
    main()
