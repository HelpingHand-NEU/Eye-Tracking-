#!/usr/bin/env python3
"""
Test USB camera on Jetson for AprilTag detection only.
This is the tag/board camera (outward-facing). Pupil camera is CSI (IMX219) — use test_csi_glass_frame_pupil.py.
No eye tracking. Use this to verify the USB camera works for tag detection.
Run from project root: python test_usb_apriltag.py [--device /dev/video0]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Allow importing from eye_tracker package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from eye_tracker.object_hover.apriltag_detector import AprilTagDetector
except ImportError as e:
    print(f"Error: Could not import AprilTagDetector. Install: pip install pupil-apriltags\n{e}")
    sys.exit(1)


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


def open_usb_opencv(camera_id, width=640, height=480, fps=30):
    """Open USB camera with plain OpenCV (works on most systems)."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Reduce buffer so we get fresher frames and avoid stale data
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap, f"OpenCV ID {camera_id}"


def open_usb_gstreamer(device_path, width=640, height=480, fps=30):
    """Open USB camera with GStreamer V4L2 (Jetson)."""
    if not os.path.exists(device_path):
        return None, None
    pipeline = build_v4l2_pipeline(device_path, width, height, fps)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap, f"V4L2 {device_path}"
    if cap is not None:
        cap.release()
    return None, None


def open_usb_camera(device_path=None, camera_id=0, width=640, height=480, fps=30, prefer_opencv=False):
    """
    Open USB camera. Tries OpenCV first (more reliable for USB), then GStreamer on Jetson.
    Returns (cap, description) or (None, None) on failure.
    """
    if device_path is not None:
        if is_jetson() and gstreamer_available() and not prefer_opencv:
            cap, desc = open_usb_gstreamer(device_path, width, height, fps)
            if cap is not None:
                return cap, desc
        # Map /dev/videoN to camera_id N for OpenCV
        try:
            cid = int(device_path.replace("/dev/video", ""))
        except ValueError:
            cid = 0
        return open_usb_opencv(cid, width, height, fps)

    # No device specified: try OpenCV indices first (avoids GStreamer pipeline errors)
    for cid in [0, 1, 2]:
        cap, desc = open_usb_opencv(cid, width, height, fps)
        if cap is not None:
            return cap, desc
    # Then try GStreamer on Jetson for /dev/video0 and /dev/video1
    if is_jetson() and gstreamer_available():
        for dev in ["/dev/video0", "/dev/video1"]:
            cap, desc = open_usb_gstreamer(dev, width, height, fps)
            if cap is not None:
                return cap, desc
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Test USB camera for AprilTag detection on Jetson")
    parser.add_argument("--device", type=str, default=None,
                        help="V4L2 device (e.g. /dev/video0). If not set, try OpenCV 0,1,2 then GStreamer.")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--headless-frames", type=int, default=0,
                        help="If no DISPLAY: run this many frames and print detections (default 0 = exit immediately)")
    parser.add_argument("--prefer-opencv", action="store_true", default=True,
                        help="Prefer OpenCV over GStreamer for USB (default: True)")
    parser.add_argument("--no-prefer-opencv", action="store_false", dest="prefer_opencv",
                        help="Try GStreamer first on Jetson")
    args = parser.parse_args()

    try:
        detector = AprilTagDetector(families="tag36h11")
    except Exception as e:
        print(f"Error initializing AprilTag detector: {e}")
        return 1

    cap, desc = open_usb_camera(
        device_path=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        prefer_opencv=args.prefer_opencv,
    )
    if cap is None or not cap.isOpened():
        print("Error: Could not open USB camera. Try --device /dev/video0 or /dev/video1")
        return 1

    # Warmup: some USB cams need a moment and a few discarded frames
    time.sleep(0.5)
    for _ in range(5):
        cap.read()
    time.sleep(0.2)

    has_display = bool(os.environ.get("DISPLAY"))
    headless_frames = args.headless_frames if not has_display else 0

    if not has_display and headless_frames <= 0:
        print("No DISPLAY set (e.g. SSH/headless). Camera opened OK; run on a machine with a display to see the window.")
        print("Or run with --headless-frames 50 to test detection without a window.")
        cap.release()
        return 0

    print(f"USB camera opened: {desc}")
    if has_display:
        print("Show an AprilTag (tag36h11) to the camera. Press 'q' to quit.")
    else:
        print(f"Headless: processing {headless_frames} frames and printing detections.")

    frame_count = 0
    read_failures = 0
    max_read_failures = 30  # ~1 second of retries at 30 fps
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures >= max_read_failures:
                    print("Failed to read frame repeatedly. Is the camera in use or unplugged?")
                    break
                time.sleep(0.033)
                continue
            read_failures = 0
            frame_count += 1

            try:
                dets = detector.detect(frame)
            except Exception as e:
                print(f"  Detection error: {e}")
                dets = []

            for d in dets:
                x, y, w, h = d["bbox"]
                tag_id = d.get("tag_id", d.get("label", "?"))
                score = d.get("score", 0.0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text_y = max(20, y - 8)
                cv2.putText(frame, f"id={tag_id}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if has_display or frame_count % 10 == 1:
                    print(f"  Tag id={tag_id} bbox=({x},{y},{w},{h}) score={score:.1f}")

            cv2.putText(frame, "USB camera - AprilTag test (q=quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if has_display:
                cv2.imshow("USB camera - AprilTag", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if headless_frames > 0 and frame_count >= headless_frames:
                    break
                time.sleep(0.033)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if has_display:
            cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
