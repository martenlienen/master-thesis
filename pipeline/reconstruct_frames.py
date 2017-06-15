#!/usr/bin/env python

import argparse
import os
import subprocess
import sys

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frame-interval", default=1000, type=int, help="Frame interval in milliseconds")
    parser.add_argument("-W", "--width", type=int, help="Image width")
    parser.add_argument("-H", "--height", type=int, help="Image height")
    parser.add_argument("--out_dir", default="reconstructed-frames", help="Directory name to reconstruct into")
    parser.add_argument("dirs", nargs="*", help="Event directories to reconstruct")
    args = parser.parse_args()

    frame_interval = args.frame_interval
    width = args.width
    height = args.height
    out_dir = args.out_dir
    directories = args.dirs

    extra_args = []
    if width is not None:
        extra_args += ["-W", str(width)]
    if height is not None:
        extra_args += ["-H", str(height)]

    reconstruct_bin = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dvs-reconstruction", "build", "reconstruct"))

    for d in tqdm(directories):
        events_path = os.path.join(d, "events.csv")
        frames_path = os.path.join(d, out_dir)

        os.makedirs(frames_path, exist_ok=True)

        subprocess.run([reconstruct_bin, "--frame-interval", str(frame_interval), *extra_args, events_path, frames_path], stdout=sys.stdout)

if __name__ == '__main__':
    main()
