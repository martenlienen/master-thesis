#!/usr/bin/env python3

import argparse
import os
import subprocess

directions = ["up", "down", "left", "right"]
gestures = [("Hand {}".format(d), "hand-{}.mkv".format(d)) for d in directions]
gestures += [("Two fingers {}".format(d), "two-fingers-{}.mkv".format(d))
             for d in directions]
gestures += [("Tap with index finger", "tap-index.mkv"),
             ("Tap with two fingers", "tap-two-fingers.mkv")]
gestures += [("Beckoning", "beckoning.mkv")]
gestures += [("Open hand", "open-hand.mkv"), ("Close hand", "close-hand.mkv"),
             ("Close hand twice", "close-hand-twice.mkv")]
gestures += [("Extend index finger", "extend-one.mkv"),
             ("Extend index and middle finger", "extend-two.mkv"),
             ("Extend two fingers plus thumb", "extend-three.mkv")]
gestures += [("Push hand {}".format(d), "push-hand-{}.mkv".format(d))
             for d in directions]
gestures += [("Rotate {}".format(d), "rotate-{}.mkv".format(d))
             for d in ["clockwise", "counterclockwise"]]
gestures += [("Thumbs up", "thumbs-up.mkv")]
gestures += [("Ok", "ok.mkv")]
gestures += [("Swipe {}".format(d), "swipe-{}.mkv".format(d))
             for d in directions]


def record(path):
    subprocess.run(["ffmpeg", "-f", "v4l2", "-video_size", "1280x720", "-i",
                    "/dev/video1", "-an", path])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    directory = args.directory

    os.makedirs(directory)

    for name, filename in gestures:
        path = os.path.join(directory, filename)

        print(name)
        input("Press enter to record...")
        record(path)

if __name__ == "__main__":
    main()
