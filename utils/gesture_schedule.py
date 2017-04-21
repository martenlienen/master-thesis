#!/usr/bin/env python

import argparse
import random

DIRECTIONS = ["up", "down", "left", "right"]
GESTURES = []
GESTURES += ["hand-{}".format(d) for d in DIRECTIONS]
GESTURES += ["two-fingers-{}".format(d) for d in DIRECTIONS]
GESTURES += ["tap-index", "tap-two-fingers"]
GESTURES += ["beckoning"]
GESTURES += ["open-hand", "close-hand", "close-hand-twice"]
GESTURES += ["extend-one", "extend-two", "extend-three"]
GESTURES += ["push-hand-{}".format(d) for d in DIRECTIONS]
GESTURES += ["rotate-{}".format(d) for d in ["clockwise", "counterclockwise"]]
GESTURES += ["thumbs-up"]
GESTURES += ["ok"]
GESTURES += ["swipe-{}".format(d) for d in DIRECTIONS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations",
        default=1,
        type=int,
        help="Number of recordings per gesture")
    parser.add_argument(
        "-r",
        "--randomized",
        default=False,
        dest="randomized",
        action="store_true",
        help="Randomize order")
    args = parser.parse_args()

    iterations = args.iterations
    randomized = args.randomized

    gestures = GESTURES * iterations

    if randomized:
        random.shuffle(gestures)
    else:
        gestures = sorted(gestures)

    print("\n".join(gestures))


if __name__ == "__main__":
    main()
