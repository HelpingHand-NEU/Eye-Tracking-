#!/usr/bin/env python3
"""
Run JEOresearch 3D Eye Tracker using the official repo with a CSI camera.

This script uses someone else's open source repo: JEOresearch/EyeTracker (3DTracker).
We do not claim authorship of the eye-tracking algorithm; it is vendored from
https://github.com/JEOresearch/EyeTracker and used here with a CSI camera feed.

- Uses the vendored JEOresearch/EyeTracker 3DTracker (Orlosky3DEyeTracker) directly.
- CSI camera is opened via GStreamer; each frame is passed to the repo's process_frame().
- Output: gaze_vector.txt (origin + direction), same as the original tracker.

Usage:
  python3 run_csi_jeo_3dtracker.py [--sensor-id 0]

Requires: vendor/EyeTracker (clone with git clone or submodule).
  From project root: git submodule add https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker
  Or: git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker
"""

import argparse
import os
import sys
import time

import cv2

# Add vendored JEO 3DTracker so we import their API directly
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
JEO_3D_PATH = os.path.join(PROJECT_ROOT, "vendor", "EyeTracker", "3DTracker")
if not os.path.isdir(JEO_3D_PATH):
    print("JEO 3DTracker not found. Clone the repo into vendor/EyeTracker:")
    print("  git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker")
    print("  or: git submodule add https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker")
    sys.exit(1)
sys.path.insert(0, JEO_3D_PATH)

# Use the repo's API directly
from Orlosky3DEyeTracker import process_frame

# CSI pipeline (same as project's test_csi_glass_frame_pupil / eye_tracker_jetson)
CSI_W, CSI_H, CSI_FPS = 1280, 720, 60


def build_csi_pipeline(sensor_id, width=CSI_W, height=CSI_H, fps=CSI_FPS):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx, width=(int){width}, height=(int){height} ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def open_csi(sensor_id):
    pipeline = build_csi_pipeline(sensor_id)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            return cap, f"CSI sensor-id={sensor_id} {CSI_W}x{CSI_H}@{CSI_FPS}"
    if cap.isOpened():
        cap.release()
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Run JEOresearch 3D Eye Tracker with CSI camera (official repo API)"
    )
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor id (0 = single camera; 1 for second)")
    args = parser.parse_args()

    cap, desc = open_csi(args.sensor_id)
    if cap is None:
        print("Could not open CSI camera. Close other apps using the camera and try again.")
        print("  cam0: --sensor-id 0   cam1: --sensor-id 1")
        return 1

    print(f"CSI camera: {desc}")
    print("Using JEOresearch 3DTracker (Orlosky3DEyeTracker) from vendor/EyeTracker/3DTracker")
    print("Output: gaze_vector.txt (origin + direction). Press Q to quit, Space to pause.")

    paused = False
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            continue

        frame = cv2.flip(frame, 1)
        if not paused:
            # Call the repo's API directly: processes frame, updates gaze_vector.txt, shows window
            process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
