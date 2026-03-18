#!/usr/bin/env python3
"""
Test CSI camera (e.g. IMX219) on Jetson for glass-frame eye tracking.

Uses the latest 3D tracker implementation: JEOGlassFrameTracker, which runs the
JEOresearch/EyeTracker 3DTracker from vendor/ and reads gaze_vector.txt (see CREDITS.md).
Same code path as calibration and main.py glass-frame mode.

Pupil camera = CSI; AprilTag camera = USB (test_usb_apriltag.py).

Usage:
  python3 test_csi_glass_frame_pupil.py [--sensor-id 0]
  (Use 0 for a single CSI camera; use 1 only if you have a second CSI camera on cam1.)
"""

import argparse
import os
import sys
import time

# Reduce Qt font warnings from OpenCV GUI (optional)
if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir("/usr/share/fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

# On Jetson, prefer system OpenCV (built with GStreamer) so CSI nvarguscamerasrc works.
# Pip OpenCV is often built without GStreamer and cannot open CSI pipelines.
if os.path.exists("/etc/nv_tegra_release") and os.path.exists("/usr/lib/python3/dist-packages"):
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
import numpy as np

# Suppress vendor OpenCV windows so only our 3 windows show (no 4th duplicate).
# calibrate_glass_frame_rois.py no longer imports this module, so its window is unaffected.
_OUR_WINDOWS = {"1. CSI camera (full frame)", "2. Left eye CROPPED", "3. Right eye CROPPED"}
_imshow_orig = cv2.imshow
def _imshow_filter(name, img):
    if name not in _OUR_WINDOWS:
        return
    _imshow_orig(name, img)
cv2.imshow = _imshow_filter

# Use the same glass-frame tracker as calibration and main (vendor 3DTracker)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_tracker.glass_frame_jeo_tracker import (
    JEOGlassFrameTracker,
    jeo_tracker_available,
    CSI_W,
    CSI_H,
    CSI_FPS,
    GLASS_FRAME_ROIS_FILENAME,
    _roi_to_pixel_rect,
)

# Height of the info strip under each cropped frame (coordinates and gaze)
INFO_STRIP_H = 72


def _crop_with_info_strip(crop_bgr, color_bgr, lines):
    """Stack crop image and an info strip with the given text lines. Returns one BGR image."""
    h, w = crop_bgr.shape[:2]
    strip = np.zeros((INFO_STRIP_H, w, 3), dtype=np.uint8)
    strip[:] = (40, 40, 40)
    y_offset = 18
    for line in lines:
        if line:
            cv2.putText(strip, line, (6, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
        y_offset += 20
    return np.vstack([crop_bgr, strip])


def load_rois_for_display():
    """Load glass_frame_rois.json for display. Returns (rois_dict or None, path_found or None)."""
    import json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        script_dir,
        os.getcwd(),
        os.path.dirname(script_dir),  # e.g. Desktop if script is in Eye-Tracking-
        os.path.expanduser("~"),
        os.path.join(script_dir, ".."),
    ]
    seen = set()
    for base in search_dirs:
        if not base:
            continue
        base = os.path.abspath(base)
        if base in seen:
            continue
        seen.add(base)
        path = os.path.join(base, GLASS_FRAME_ROIS_FILENAME)
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and ("left_eye_roi" in data or "right_eye_roi" in data):
                    return data, path
            except Exception:
                pass
    return None, None


def open_csi(sensor_id):
    """Open CSI camera (for use by calibrate_glass_frame_rois.py). Returns (cap, desc) or (None, None)."""
    from eye_tracker.glass_frame_jeo_tracker import _build_csi_pipeline
    pipeline = _build_csi_pipeline(sensor_id)
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
        description="Test CSI glass-frame camera with latest 3DTracker (JEO vendor)"
    )
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI port: 0 (default) or 1. Use 0 if you have one camera.")
    parser.add_argument("--usb", type=int, default=None, help="Use USB camera index instead of CSI (e.g. --usb 0)")
    parser.add_argument("--roi-file", type=str, default=None, help=f"Path to {GLASS_FRAME_ROIS_FILENAME} if not found automatically")
    args = parser.parse_args()

    # Resolve ROI file path first so the tracker runs 3D on both cropped frames (not full frame)
    rois_display, rois_path = None, None
    if args.roi_file and os.path.isfile(args.roi_file):
        try:
            import json
            with open(args.roi_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and ("left_eye_roi" in data or "right_eye_roi" in data):
                rois_display, rois_path = data, args.roi_file
        except Exception:
            pass
    if not rois_display:
        rois_display, rois_path = load_rois_for_display()

    if not jeo_tracker_available():
        print("JEO 3DTracker not found. Clone the repo:")
        print("  git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker")
        return 1

    eye_sensor_id = None if args.usb is not None else args.sensor_id
    eye_camera_id = args.usb if args.usb is not None else 0
    tracker = JEOGlassFrameTracker(
        eye_camera_id=eye_camera_id,
        eye_sensor_id=eye_sensor_id,
        rois_path=rois_path,
    )

    if not tracker.is_opened():
        print("Could not open eye camera. Close other apps using the camera and try again.")
        if eye_sensor_id is not None:
            print("  CSI failed. Use --sensor-id 0 for a single CSI camera (sensor-id 1 = invalid if only one camera).")
        print("  On Jetson, if camera stays busy, run: sudo systemctl restart nvargus-daemon")
        print("  Then: --sensor-id 0   or   --usb 0 for V4L2")
        return 1

    desc = f"CSI sensor-id={eye_sensor_id}" if tracker.using_csi() else f"USB camera {eye_camera_id}"
    print(f"Eye camera: {desc}")
    if tracker.has_rois():
        print("3D tracker running on LEFT and RIGHT cropped frames (pupil tracking per crop).")
    if rois_display and rois_path:
        print(f"ROI file found: {rois_path}")
        if rois_display.get("left_eye_roi"):
            print("  Left ROI: OK")
        if rois_display.get("right_eye_roi"):
            print("  Right ROI: OK")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        expected = os.path.join(script_dir, GLASS_FRAME_ROIS_FILENAME)
        print(f"ROI file not found. Expected at: {expected}")
        print(f"  Copy your saved file there, or run: python3 test_csi_glass_frame_pupil.py --roi-file /path/to/{GLASS_FRAME_ROIS_FILENAME}")
    print("Space=start/pause eye tracking. Q=quit.")

    win_full = "1. CSI camera (full frame)"
    win_left = "2. Left eye CROPPED"
    win_right = "3. Right eye CROPPED"
    cv2.namedWindow(win_full, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_left, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)

    tracking_active = False  # Eye tracking runs only after Space (start); Space again = pause
    frame_count = 0
    while True:
        tracker.process_frame_binocular()
        full = tracker.get_last_full_frame()
        has_valid_frame = full is not None and full.size > 0
        if not has_valid_frame:
            full = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(full, "No frame from camera", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(full, "Check camera / CSI", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        h, w = full.shape[:2]
        frame_count += 1

        # Coordinates only when tracking is active and not blink; otherwise show --
        if tracking_active and has_valid_frame and not tracker.get_last_is_blink():
            pupil_full = tracker.get_last_left_right_pupil_full()
            pupils_crop = tracker.get_last_left_right_pupil_in_crop()
            full_left = pupil_full[0] if pupil_full else None
            full_right = pupil_full[1] if pupil_full else None
            pupil_left_crop = pupils_crop[0] if pupils_crop else None
            pupil_right_crop = pupils_crop[1] if pupils_crop else None
        else:
            full_left = full_right = pupil_left_crop = pupil_right_crop = None

            # Window 1: full frame from CSI (with ROI boxes, pupil dots, and all three coordinate sets)
            disp_full = full.copy()
            if rois_display:
                for key, color, label in (
                    ("left_eye_roi", (0, 255, 0), "L"),
                    ("right_eye_roi", (255, 0, 0), "R"),
                ):
                    roi_norm = rois_display.get(key)
                    if roi_norm and len(roi_norm) == 4:
                        rect = _roi_to_pixel_rect(roi_norm, w, h)
                        if rect:
                            x, y, rw, rh = rect
                            cv2.rectangle(disp_full, (x, y), (x + rw, y + rh), color, 3)
                            cv2.putText(disp_full, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Dots on main frame at real-time pupil coordinates (same style as cropped windows)
                if full_left is not None:
                    lx, ly = int(full_left[0]), int(full_left[1])
                    cv2.circle(disp_full, (lx, ly), 6, (0, 255, 0), 2)
                if full_right is not None:
                    rx, ry = int(full_right[0]), int(full_right[1])
                    cv2.circle(disp_full, (rx, ry), 6, (0, 0, 255), 2)
                # Three coordinate sets: main frame (L,R), left crop, right crop — updated every frame
                y_text = disp_full.shape[0] - 72
                cv2.rectangle(disp_full, (0, y_text - 4), (disp_full.shape[1], disp_full.shape[0]), (32, 32, 32), -1)
                cv2.putText(disp_full, f"Live #%d" % frame_count, (disp_full.shape[1] - 100, y_text + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if tracking_active and has_valid_frame and tracker.get_last_is_blink():
                    cv2.putText(disp_full, "Blink (x:-- y:--)", (disp_full.shape[1] - 280, y_text + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                cv2.putText(disp_full, "1. Main frame (calculated):", (10, y_text + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                main_l = f"L: ({full_left[0]:.1f}, {full_left[1]:.1f})" if full_left else "L: --"
                main_r = f"R: ({full_right[0]:.1f}, {full_right[1]:.1f})" if full_right else "R: --"
                cv2.putText(disp_full, main_l, (10, y_text + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(disp_full, main_r, (180, y_text + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(disp_full, "2. Left crop (rect):", (10, y_text + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                left_c = f"({pupil_left_crop[0]:.1f}, {pupil_left_crop[1]:.1f})" if pupil_left_crop else "--"
                cv2.putText(disp_full, left_c, (10, y_text + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(disp_full, "3. Right crop (rect):", (380, y_text + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                right_c = f"({pupil_right_crop[0]:.1f}, {pupil_right_crop[1]:.1f})" if pupil_right_crop else "--"
                cv2.putText(disp_full, right_c, (380, y_text + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(disp_full, "1. Full frame", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not tracking_active:
                cv2.putText(disp_full, "Press SPACE to start eye tracking", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(disp_full, "Tracking | SPACE=pause Q=quit", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win_full, disp_full)

            # Window 2: CROP of left ROI only (from full frame)
            # Window 3: CROP of right ROI only (from full frame)
            if rois_display:
                left_roi = rois_display.get("left_eye_roi")
                right_roi = rois_display.get("right_eye_roi")
                left_rect = _roi_to_pixel_rect(left_roi, w, h) if left_roi and len(left_roi) == 4 else None
                right_rect = _roi_to_pixel_rect(right_roi, w, h) if right_roi and len(right_roi) == 4 else None

                if left_rect:
                    xl, yl, wl, hl = left_rect
                    left_crop = full[yl : yl + hl, xl : xl + wl].copy()
                    gaze = tracker.get_last_left_right_gaze()
                    left_gaze, _ = gaze if gaze is not None else (None, None)
                    gxL, gyL = left_gaze if left_gaze is not None else (None, None)
                    if pupil_left_crop is not None:
                        px, py = int(pupil_left_crop[0]), int(pupil_left_crop[1])
                        cv2.circle(left_crop, (px, py), 6, (0, 255, 0), 2)
                    cv2.putText(left_crop, "2. LEFT CROP", (4, left_crop.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # Two of the three sets: main frame (calculated) and this cropped rectangle
                    lines = [
                        f"Main frame: ({full_left[0]:.1f}, {full_left[1]:.1f})" if full_left else "Main frame: --",
                        f"Left crop: ({pupil_left_crop[0]:.1f}, {pupil_left_crop[1]:.1f})" if pupil_left_crop else "Left crop: --",
                        f"Gaze: ({gxL:.2f}, {gyL:.2f})" if gxL is not None and gyL is not None else "Gaze: --",
                    ]
                    disp_left = _crop_with_info_strip(left_crop, (0, 255, 0), lines)
                    cv2.imshow(win_left, disp_left)
                else:
                    no_crop = np.zeros((200, 320, 3), dtype=np.uint8)
                    cv2.putText(no_crop, "No left ROI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow(win_left, no_crop)

                if right_rect:
                    xr, yr, wr, hr = right_rect
                    right_crop = full[yr : yr + hr, xr : xr + wr].copy()
                    gaze = tracker.get_last_left_right_gaze()
                    _, right_gaze = gaze if gaze is not None else (None, None)
                    gxR, gyR = right_gaze if right_gaze is not None else (None, None)
                    if pupil_right_crop is not None:
                        px, py = int(pupil_right_crop[0]), int(pupil_right_crop[1])
                        cv2.circle(right_crop, (px, py), 6, (0, 0, 255), 2)
                    cv2.putText(right_crop, "3. RIGHT CROP", (4, right_crop.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # Two of the three sets: main frame (calculated) and this cropped rectangle
                    lines = [
                        f"Main frame: ({full_right[0]:.1f}, {full_right[1]:.1f})" if full_right else "Main frame: --",
                        f"Right crop: ({pupil_right_crop[0]:.1f}, {pupil_right_crop[1]:.1f})" if pupil_right_crop else "Right crop: --",
                        f"Gaze: ({gxR:.2f}, {gyR:.2f})" if gxR is not None and gyR is not None else "Gaze: --",
                    ]
                    disp_right = _crop_with_info_strip(right_crop, (0, 0, 255), lines)
                    cv2.imshow(win_right, disp_right)
                else:
                    no_crop = np.zeros((200, 320, 3), dtype=np.uint8)
                    cv2.putText(no_crop, "No right ROI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow(win_right, no_crop)
            else:
                no_roi = np.zeros((200, 320, 3), dtype=np.uint8)
                cv2.putText(no_roi, "Run calibrate_glass_frame_rois.py", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(no_roi, "and save ROIs", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(win_left, no_roi)
                cv2.imshow(win_right, no_roi)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            tracking_active = not tracking_active

    tracker.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
