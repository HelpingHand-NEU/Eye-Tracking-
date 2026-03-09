#!/usr/bin/env python3
"""
Test USB camera on Jetson for AprilTag detection only.
No eye tracking. Use this to verify the USB camera works for tag detection.
Run from project root: python test_usb_apriltag.py [--device /dev/video0]
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Allow importing from eye_tracker package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_tracker.object_hover.apriltag_detector import AprilTagDetector


def is_jetson():
    return os.path.exists("/etc/nv_tegra_release")


def gstreamer_available():
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    return "GStreamer" in info and "YES" in info.split("GStreamer")[1].splitlines()[0]


def build_v4l2_pipeline(device_path, width=640, height=480, fps=30):
    return (
        f"v4l2src device={device_path} ! "
        f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def open_usb_camera(device_path=None, camera_id=0, width=640, height=480, fps=30):
    """
    Open USB camera. On Jetson, prefer V4L2 GStreamer pipeline.
    Returns (cap, description) or (None, None) on failure.
    """
    if device_path is None:
        device_path = f"/dev/video{camera_id}"

    if is_jetson() and gstreamer_available() and os.path.exists(device_path):
        pipeline = build_v4l2_pipeline(device_path, width, height, fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, f"V4L2 {device_path}"
        cap.release()

    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap, f"OpenCV ID {camera_id}"
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Test USB camera for AprilTag detection on Jetson")
    parser.add_argument("--device", type=str, default=None,
                        help="V4L2 device (e.g. /dev/video0). If not set, try video0 then video1.")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    args = parser.parse_args()

    detector = AprilTagDetector(families="tag36h11")

    # Open camera: explicit device, or try video0 then video1
    cap, desc = None, None
    if args.device:
        cap, desc = open_usb_camera(device_path=args.device, width=args.width, height=args.height, fps=args.fps)
    else:
        for dev in ["/dev/video0", "/dev/video1"]:
            cap, desc = open_usb_camera(device_path=dev, width=args.width, height=args.height, fps=args.fps)
            if cap is not None:
                break
        if cap is None:
            cap, desc = open_usb_camera(camera_id=0, width=args.width, height=args.height, fps=args.fps)

    if cap is None or not cap.isOpened():
        print("Error: Could not open USB camera. Try --device /dev/video0 or /dev/video1")
        return 1

    print(f"USB camera opened: {desc}")
    print("Show an AprilTag (tag36h11) to the camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        dets = detector.detect(frame)
        for d in dets:
            x, y, w, h = d["bbox"]
            tag_id = d.get("tag_id", d["label"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"id={tag_id}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"  Tag id={tag_id} bbox=({x},{y},{w},{h}) score={d['score']:.1f}")

        cv2.putText(frame, "USB camera - AprilTag test (q=quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("USB camera - AprilTag", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
