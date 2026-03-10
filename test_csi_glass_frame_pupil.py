#!/usr/bin/env python3
"""
Test CSI camera (cam1, sensor-id=1) on Jetson with glass-frame pupil tracking only.
Uses the same algorithm as eye_tracker_jetson: eye-only frame -> _detect_pupil_center.
No calibration, no second camera. Run this to verify CSI + pupil detection before calibration.
Usage: python3 test_csi_glass_frame_pupil.py [--sensor-id 1]
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Use the Jetson eye tracker's glass-frame pupil detection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_tracker_jetson import EyeTracker


def open_csi_camera(tracker, sensor_id, width=1280, height=720, fps=60):
    """Open CSI camera by sensor-id using the same pipeline as EyeTracker."""
    if not tracker._is_jetson():
        print("Not a Jetson platform; CSI pipeline may not work.")
    if not tracker._gstreamer_available():
        print("OpenCV built without GStreamer. Cannot use CSI.")
        return None, None
    # Temporarily set resolution for CSI (supported mode)
    tracker.camera_width, tracker.camera_height, tracker.camera_fps = width, height, fps
    pipeline = tracker._build_csi_gstreamer_pipeline(sensor_id)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return None, None
    return cap, f"CSI sensor-id={sensor_id}"


def main():
    parser = argparse.ArgumentParser(
        description="Test CSI camera (cam1) with glass-frame pupil tracking on Jetson"
    )
    parser.add_argument(
        "--sensor-id",
        type=int,
        default=1,
        help="CSI camera sensor ID (cam1 = 1, cam0 = 0). Default: 1",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="CSI frame width (use supported mode)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="CSI frame height"
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="CSI frame rate"
    )
    args = parser.parse_args()

    # Reuse EyeTracker only for its glass-frame pupil detection (no cameras opened via init)
    tracker = EyeTracker(
        front_camera_id=0,
        back_camera_id=0,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
        front_sensor_id=None,
        back_sensor_id=args.sensor_id,
    )

    cap, desc = open_csi_camera(tracker, args.sensor_id, args.width, args.height, args.fps)
    if cap is None:
        print(f"Failed to open CSI camera sensor-id={args.sensor_id}. Check cam1 connection and GStreamer.")
        return 1

    print(f"CSI camera opened: {desc}")
    print("Glass-frame pupil tracking: show your eye to the CSI camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Use the same glass-frame algorithm as in eye_tracker_jetson (eye-only frame)
        pupil_norm = tracker._detect_pupil_center(frame)
        h, w = frame.shape[:2]

        if pupil_norm is not None:
            px = int(pupil_norm[0] * w)
            py = int(pupil_norm[1] * h)
            cv2.circle(frame, (px, py), 12, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"pupil ({pupil_norm[0]:.2f}, {pupil_norm[1]:.2f})",
                (px + 14, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        else:
            cv2.putText(
                frame,
                "No pupil detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            frame,
            "CSI cam1 - Glass-frame pupil (q=quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("CSI Glass-Frame Pupil Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
