#!/usr/bin/env python3
"""
Define your own eye regions ("landmarks") for the glass-frame camera.
Use when the frame blocks the area between the eyes and you want to restrict
pupil detection to fixed left/right eye regions.

Usage:
  1. Run: python3 calibrate_glass_frame_rois.py [--sensor-id 0]
  2. Put on the glasses and look at the camera so both eyes are visible.
  3. Press '1', then click-and-drag a rectangle around your LEFT eye. Release to set.
  4. Press '2', then click-and-drag a rectangle around your RIGHT eye. Release to set.
  5. Press 's' to save to glass_frame_rois.json (in current directory).
  6. Press 'r' to clear and redraw. 'q' to quit.

Once saved to glass_frame_rois.json (project root or cwd), the glass-frame 3D tracker loads
these ROIs and runs pupil detection on each cropped region; pupil coordinates are converted
to full-frame (glass-frame) coordinates.
"""

import argparse
import json
import os
import sys
import time

# On Jetson, prefer system OpenCV (GStreamer) so CSI works
if os.path.exists("/etc/nv_tegra_release") and os.path.exists("/usr/lib/python3/dist-packages"):
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root = folder containing this script (so ROI file lives next to test_csi_glass_frame_pupil.py)
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, SCRIPT_DIR)
GLASS_FRAME_ROIS_FILENAME = "glass_frame_rois.json"

# CSI opening: use glass_frame_jeo_tracker only (do not import test script — it can patch cv2.imshow)
try:
    from eye_tracker.glass_frame_jeo_tracker import _build_csi_pipeline, CSI_W, CSI_H
except ImportError:
    _build_csi_pipeline = None
    CSI_W, CSI_H = 1280, 720

# ROI state: (x, y, w, h) in pixels, or None
left_roi = None
right_roi = None
draw_mode = None  # 'left', 'right', or None
start_pt = None
end_pt = None
frame_size = (720, 1280)  # (h, w), updated from first frame


def _normalize_roi(roi, h, w):
    if roi is None or len(roi) != 4:
        return None
    x, y, rw, rh = roi
    return [
        round(x / w, 4),
        round(y / h, 4),
        round(rw / w, 4),
        round(rh / h, 4),
    ]


def _mouse_callback(event, x, y, flags, param):
    global start_pt, end_pt, left_roi, right_roi, draw_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pt = (x, y)
        end_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and start_pt is not None:
        end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and start_pt is not None:
        x0, y0 = start_pt
        x1, y1 = end_pt
        rx = min(x0, x1)
        ry = min(y0, y1)
        rw = abs(x1 - x0)
        rh = abs(y1 - y0)
        if rw < 10 or rh < 10:
            start_pt = end_pt = None
            return
        if draw_mode == "left":
            left_roi = (rx, ry, rw, rh)
            draw_mode = None
        elif draw_mode == "right":
            right_roi = (rx, ry, rw, rh)
            draw_mode = None
        start_pt = end_pt = None


def main():
    global frame_size, left_roi, right_roi, draw_mode
    parser = argparse.ArgumentParser(description="Define left/right eye regions for glass-frame (custom landmarks).")
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor (0 = default for single camera; 1 for second). Ignored if using USB.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: glass_frame_rois.json in cwd)")
    args = parser.parse_args()

    cap = None
    if _build_csi_pipeline is not None:
        csi_cap = None
        try:
            pipeline = _build_csi_pipeline(args.sensor_id)
            csi_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if csi_cap.isOpened():
                for _ in range(20):
                    ret, _ = csi_cap.read()
                    if ret:
                        cap = csi_cap
                        csi_cap = None
                        break
                    time.sleep(0.1)
        except Exception:
            pass
        if csi_cap is not None:
            csi_cap.release()
            csi_cap = None
            time.sleep(1.0)
    if cap is None:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open any camera.")
            print("  If using Jetson CSI: try closing other apps, then run: sudo systemctl restart nvargus-daemon")
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read first frame.")
        cap.release()
        return 1
    frame_size = (frame.shape[0], frame.shape[1])
    h, w = frame_size

    # Default: save to project root so eye tracker (main.py, test_csi_glass_frame_pupil.py) finds it
    out_path = args.output or os.path.join(PROJECT_ROOT, GLASS_FRAME_ROIS_FILENAME)
    win = "Glass-frame ROI calibration (1=left 2=right s=save r=reset q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _mouse_callback)

    print("Draw LEFT eye: press '1' then click-and-drag. Draw RIGHT eye: press '2' then click-and-drag.")
    print("Save: 's'. Reset: 'r'. Quit: 'q'.")

    try:
        import gi
        gi.require_version("GLib", "2.0")
        from gi.repository import GLib
        main_loop_ref = [None]
        def tick():
            global draw_mode, left_roi, right_roi
            ret, frame = cap.read()
            if not ret:
                return True
            disp = frame.copy()
            # Draw saved ROIs with thick lines and clear labels
            if left_roi is not None:
                x, y, rw, rh = left_roi
                cv2.rectangle(disp, (x, y), (x + rw, y + rh), (0, 255, 0), 4)
                cv2.putText(disp, "LEFT eye ROI", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if right_roi is not None:
                x, y, rw, rh = right_roi
                cv2.rectangle(disp, (x, y), (x + rw, y + rh), (255, 0, 0), 4)
                cv2.putText(disp, "RIGHT eye ROI", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if draw_mode and start_pt is not None and end_pt is not None:
                cv2.rectangle(disp, start_pt, end_pt, (0, 255, 255), 4)
            mode_str = f" [drawing: {draw_mode}]" if draw_mode else ""
            cv2.putText(disp, f"1=left 2=right s=save r=reset q=quit{mode_str}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Status: what you have drawn
            status = []
            if left_roi is not None:
                status.append("Left: drawn")
            if right_roi is not None:
                status.append("Right: drawn")
            if status:
                cv2.putText(disp, " | ".join(status), (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                if main_loop_ref[0]:
                    main_loop_ref[0].quit()
                return False
            if k == ord("1"):
                draw_mode = "left"
            if k == ord("2"):
                draw_mode = "right"
            if k == ord("r"):
                left_roi = right_roi = None
                draw_mode = None
            if k == ord("s"):
                data = {}
                if left_roi is not None:
                    data["left_eye_roi"] = _normalize_roi(left_roi, h, w)
                if right_roi is not None:
                    data["right_eye_roi"] = _normalize_roi(right_roi, h, w)
                if data:
                    with open(out_path, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved to {out_path}")
                else:
                    print("Draw at least one ROI (1 or 2) before saving.")
            return True
        GLib.timeout_add(33, tick)
        loop = GLib.MainLoop()
        main_loop_ref[0] = loop
        loop.run()
    except ImportError:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            h, w = disp.shape[:2]
            if left_roi is not None:
                x, y, rw, rh = left_roi
                cv2.rectangle(disp, (x, y), (x + rw, y + rh), (0, 255, 0), 4)
                cv2.putText(disp, "LEFT eye ROI", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if right_roi is not None:
                x, y, rw, rh = right_roi
                cv2.rectangle(disp, (x, y), (x + rw, y + rh), (255, 0, 0), 4)
                cv2.putText(disp, "RIGHT eye ROI", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            if draw_mode and start_pt is not None and end_pt is not None:
                cv2.rectangle(disp, start_pt, end_pt, (0, 255, 255), 4)
            mode_str = f" [drawing: {draw_mode}]" if draw_mode else ""
            cv2.putText(disp, f"1=left 2=right s=save r=reset q=quit{mode_str}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status = []
            if left_roi is not None:
                status.append("Left: drawn")
            if right_roi is not None:
                status.append("Right: drawn")
            if status:
                cv2.putText(disp, " | ".join(status), (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("1"):
                draw_mode = "left"
            if k == ord("2"):
                draw_mode = "right"
            if k == ord("r"):
                left_roi = right_roi = None
                draw_mode = None
            if k == ord("s"):
                data = {}
                if left_roi is not None:
                    data["left_eye_roi"] = _normalize_roi(left_roi, h, w)
                if right_roi is not None:
                    data["right_eye_roi"] = _normalize_roi(right_roi, h, w)
                if data:
                    with open(out_path, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved to {out_path}")
                else:
                    print("Draw at least one ROI (1 or 2) before saving.")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
