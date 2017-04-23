#!/usr/bin/env python3

import argparse
import os
import subprocess

directions = ["up", "down", "left", "right"]
gestures = []
gestures += [("Two fingers {}".format(d), "two-fingers-{}.mkv".format(d))
             for d in directions]
gestures += [("Tap with index finger", "tap-index.mkv"),
             ("Tap with two fingers", "tap-two-fingers.mkv")]
gestures += [("Extend index finger", "extend-one.mkv"),
             ("Extend index and middle finger", "extend-two.mkv"),
             ("Extend two fingers plus thumb", "extend-three.mkv")]
gestures += [("Push hand {}".format(d), "push-hand-{}.mkv".format(d))
             for d in directions]
gestures += [("Swipe {}".format(d), "swipe-{}.mkv".format(d))
             for d in directions]
gestures += [("Zoom In", "zoom-in.mkv"), ("Zoom Out", "zoom-out.mkv")]
gestures += [("Grab", "grab.mkv")]
gestures += [("Finger Snapping", "finger-snap.mkv")]
gestures += [("Beckoning", "beckoning.mkv")]
gestures += [("Rotate Outward", "rotate-outward.mkv")]
gestures += [("Thumbs up", "thumbs-up.mkv")]
gestures += [("Ok", "ok.mkv")]
# gestures += [("Accelerate", "accelerate.mkv"),
#              ("Decelerate", "decelerate.mkv"),
#              ("Switch lane to the left", "switch-left.mkv"),
#              ("Switch lane to the right", "switch-right.mkv"),
#              ("Take over left", "take-over-left.mkv"),
#              ("Take over right", "take-over-right.mkv"),
#              ("Turn left", "turn-left.mkv"),
#              ("Turn right", "turn-right.mkv"),
#              ("Park on the front left", "park-front-left.mkv"),
#              ("Park on the front right", "park-front-right.mkv"),
#              ("Park on the back left", "park-back-left.mkv"),
#              ("Park on the back right", "park-back-right.mkv"),
#              ("Merge", "merge.mkv"),
#              ("Start engine", "start-engine.mkv"),
#              ("Stop engine", "stop-engine.mkv")] # yapf: disable
# gestures += [("Zone {}".format(i), "zone-{}.mkv".format(i))
#              for i in range(1, 9)]
# gestures += [("Yawning", "yawning.mkv"), ("Blinking", "blinking.mkv")]


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
