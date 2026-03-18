#!/usr/bin/env python3
"""
Dual CSI test: glass-frame eye tracking + back-camera AprilTag detection.

  - CSI sensor-id 1 (cam1 / 15-pin): front-facing eye camera → JEO 3D tracker,
    same three windows as test_csi_glass_frame_pupil.py (full, left crop, right crop).
  - CSI sensor-id 0 (cam0 / 24-pin): outward-facing camera → AprilTag detection only
    (no gaze→cursor mapping).

Requires: glass_frame_rois.json, vendor JEO EyeTracker, pupil-apriltags, Jetson GStreamer OpenCV.

Usage (project root):
  python3 test_dual_csi_glass_apriltag.py
  python3 test_dual_csi_glass_apriltag.py --full-tag-quality
  python3 test_dual_csi_glass_apriltag.py --sync-tags          # block on AprilTag each cycle (old behavior)
  python3 test_dual_csi_glass_apriltag.py --fast-tag

By default AprilTag runs on a **background thread** while the eye tracker runs — same accuracy
(scale / quad_decimate / decimation), but video stays fluid. `--sync-tags` disables that.

Calibration CSV already stores tag position from the back camera (tag_cam_x/y, tag_center_*, corners,
label_x_norm/y_norm, board_distance_m) — see docs/CALIBRATION_DATA_AND_CAMERAS.md §1.1.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
import time

if "QT_QPA_FONTDIR" not in os.environ and os.path.isdir("/usr/share/fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

if os.path.exists("/etc/nv_tegra_release") and os.path.exists("/usr/lib/python3/dist-packages"):
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress vendor/JEO duplicate imshow; only our four windows are shown
_OUR_WINDOWS = {
    "1. Eye CSI (full frame)",
    "2. Left eye CROPPED",
    "3. Right eye CROPPED",
    "4. Back CSI (AprilTag)",
}
_imshow_orig = cv2.imshow


def _imshow_filter(name, img):
    if name not in _OUR_WINDOWS:
        return
    _imshow_orig(name, img)


cv2.imshow = _imshow_filter

from eye_tracker.glass_frame_jeo_tracker import (
    JEOGlassFrameTracker,
    jeo_tracker_available,
    CSI_W,
    CSI_H,
    CSI_FPS,
    GLASS_FRAME_ROIS_FILENAME,
    _build_csi_pipeline,
    _roi_to_pixel_rect,
)

try:
    from eye_tracker.object_hover.apriltag_detector import AprilTagDetector
except ImportError as e:
    AprilTagDetector = None
    _APRILTAG_ERR = e

INFO_STRIP_H = 72


def _crop_with_info_strip(crop_bgr, color_bgr, lines):
    h, w = crop_bgr.shape[:2]
    strip = np.zeros((INFO_STRIP_H, w, 3), dtype=np.uint8)
    strip[:] = (40, 40, 40)
    y_offset = 18
    for line in lines:
        if line:
            cv2.putText(strip, line, (6, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
        y_offset += 20
    return np.vstack([crop_bgr, strip])


def load_rois_for_display(roi_file=None):
    import json

    if roi_file and os.path.isfile(roi_file):
        try:
            with open(roi_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and ("left_eye_roi" in data or "right_eye_roi" in data):
                return data, roi_file
        except Exception:
            pass
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for base in (script_dir, os.getcwd(), os.path.expanduser("~")):
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


def open_tag_csi(sensor_id: int, width: int, height: int, fps: int):
    pipeline = _build_csi_pipeline(sensor_id, width=width, height=height, fps=fps)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ret, _ = cap.read()
        if ret:
            return cap, f"CSI sensor-id={sensor_id} {width}x{height}@{fps}"
    if cap.isOpened():
        cap.release()
    return None, None


def grab_latest_frame(cap, drain_grabs: int):
    """Drop buffered frames so the displayed frame is fresher (reduces 'stuck' lag)."""
    drain_grabs = max(0, int(drain_grabs))
    for _ in range(drain_grabs):
        if not cap.grab():
            break
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        ret, frame = cap.read()
    return ret, frame


def scale_detections(dets, inv_scale: float):
    """Map bboxes/centers from downscaled detect() back to full frame."""
    if inv_scale == 1.0 or not dets:
        return dets
    s = float(inv_scale)
    out = []
    for d in dets:
        x, y, w, h = d["bbox"]
        cx, cy = d["center"]
        corners = d.get("corners")
        nd = {
            **d,
            "bbox": (int(round(x * s)), int(round(y * s)), int(round(w * s)), int(round(h * s))),
            "center": (cx * s, cy * s),
        }
        if corners:
            nd["corners"] = [(c[0] * s, c[1] * s) for c in corners]
        out.append(nd)
    return out


class _AsyncTagWorker:
    """
    Runs AprilTag in a worker thread so the GUI loop is not blocked.
    Same detector and same input prep (resize/decimate schedule) as sync path — accuracy unchanged.
    Overlaps detection with JEO by submitting the tag frame before process_frame_binocular().
    """

    def __init__(self, detector, dscale: float, inv_scale: float):
        self.detector = detector
        self.dscale = dscale
        self.inv_scale = inv_scale
        self._lock = threading.Lock()
        self._pending = None  # BGR image to detect (copy)
        self._last_dets: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="apriltag-worker", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def submit_if_due(self, tag_frame_bgr, frame_count: int, every: int) -> None:
        if every > 1 and (frame_count % every) != 0:
            return
        if self.dscale < 1.0:
            small = cv2.resize(
                tag_frame_bgr,
                None,
                fx=self.dscale,
                fy=self.dscale,
                interpolation=cv2.INTER_AREA,
            )
            payload = small.copy()
        else:
            payload = tag_frame_bgr.copy()
        with self._lock:
            self._pending = payload

    def clear_dets(self):
        with self._lock:
            self._last_dets = []
            self._pending = None

    def get_dets(self):
        with self._lock:
            return list(self._last_dets)

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                if self._stop.wait(0.0015):
                    break
                continue
            try:
                raw = self.detector.detect(job)
                dets = scale_detections(raw, self.inv_scale) if self.dscale < 1.0 else raw
            except Exception:
                dets = []
            with self._lock:
                self._last_dets = dets


def draw_tag_overlays(frame, dets):
    for d in dets:
        x, y, bw, bh = d["bbox"]
        tid = d.get("tag_id", "?")
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"id={tid} ({d['center'][0]:.0f},{d['center'][1]:.0f})",
            (x, max(22, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )


def draw_eye_windows(
    tracker,
    full,
    rois_display,
    w,
    h,
    frame_count,
    tracking_active,
    has_valid_frame,
    win_full,
    win_left,
    win_right,
):
    if tracking_active and has_valid_frame and not tracker.get_last_is_blink():
        pupil_full = tracker.get_last_left_right_pupil_full()
        pupils_crop = tracker.get_last_left_right_pupil_in_crop()
        full_left = pupil_full[0] if pupil_full else None
        full_right = pupil_full[1] if pupil_full else None
        pupil_left_crop = pupils_crop[0] if pupils_crop else None
        pupil_right_crop = pupils_crop[1] if pupils_crop else None
    else:
        full_left = full_right = pupil_left_crop = pupil_right_crop = None

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
        if full_left is not None:
            cv2.circle(disp_full, (int(full_left[0]), int(full_left[1])), 6, (0, 255, 0), 2)
        if full_right is not None:
            cv2.circle(disp_full, (int(full_right[0]), int(full_right[1])), 6, (0, 0, 255), 2)
        y_text = disp_full.shape[0] - 72
        cv2.rectangle(disp_full, (0, y_text - 4), (disp_full.shape[1], disp_full.shape[0]), (32, 32, 32), -1)
        cv2.putText(
            disp_full,
            f"Live #{frame_count}",
            (disp_full.shape[1] - 100, y_text + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        if tracking_active and has_valid_frame and tracker.get_last_is_blink():
            cv2.putText(
                disp_full,
                "Blink",
                (disp_full.shape[1] - 280, y_text + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )
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
    cv2.putText(disp_full, "1. Eye CSI — full frame (cam1)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if not tracking_active:
        cv2.putText(disp_full, "SPACE = start/pause eye tracking | Q = quit", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    else:
        cv2.putText(disp_full, "Tracking | SPACE=pause", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow(win_full, disp_full)

    if not rois_display:
        no_roi = np.zeros((200, 320, 3), dtype=np.uint8)
        cv2.putText(no_roi, "Save glass_frame_rois.json", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win_left, no_roi)
        cv2.imshow(win_right, no_roi)
        return

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
        lines = [
            f"Main: ({full_left[0]:.1f}, {full_left[1]:.1f})" if full_left else "Main: --",
            f"Crop: ({pupil_left_crop[0]:.1f}, {pupil_left_crop[1]:.1f})" if pupil_left_crop else "Crop: --",
            f"Gaze: ({gxL:.2f}, {gyL:.2f})" if gxL is not None and gyL is not None else "Gaze: --",
        ]
        cv2.imshow(win_left, _crop_with_info_strip(left_crop, (0, 255, 0), lines))
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
        lines = [
            f"Main: ({full_right[0]:.1f}, {full_right[1]:.1f})" if full_right else "Main: --",
            f"Crop: ({pupil_right_crop[0]:.1f}, {pupil_right_crop[1]:.1f})" if pupil_right_crop else "Crop: --",
            f"Gaze: ({gxR:.2f}, {gyR:.2f})" if gxR is not None and gyR is not None else "Gaze: --",
        ]
        cv2.imshow(win_right, _crop_with_info_strip(right_crop, (0, 0, 255), lines))
    else:
        no_crop = np.zeros((200, 320, 3), dtype=np.uint8)
        cv2.putText(no_crop, "No right ROI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win_right, no_crop)


def main():
    parser = argparse.ArgumentParser(
        description="Dual CSI: eye (cam1) + AprilTag back (cam0). No gaze→cursor."
    )
    parser.add_argument(
        "--eye-sensor-id",
        type=int,
        default=1,
        help="CSI sensor for glass-frame eye camera (default 1 = cam1 / front)",
    )
    parser.add_argument(
        "--tag-sensor-id",
        type=int,
        default=0,
        help="CSI sensor for back/tag camera (default 0 = cam0)",
    )
    parser.add_argument("--tag-width", type=int, default=1280, help="Back CSI width")
    parser.add_argument("--tag-height", type=int, default=720, help="Back CSI height")
    parser.add_argument("--roi-file", type=str, default=None, help="Path to glass_frame_rois.json")
    parser.add_argument(
        "--fast-tag",
        action="store_true",
        help="Extra-smooth: detect every 3rd frame (alias for heavy Jetson load)",
    )
    parser.add_argument(
        "--full-tag-quality",
        action="store_true",
        help="AprilTag every frame at full resolution (slower, may stutter)",
    )
    parser.add_argument(
        "--tag-detect-every",
        type=int,
        default=2,
        help="Run AprilTag every N frames (default 2; higher = smoother UI)",
    )
    parser.add_argument(
        "--tag-detect-scale",
        type=float,
        default=0.5,
        help="Resize frame before detect (default 0.5; 1.0 = full res, slower)",
    )
    parser.add_argument(
        "--tag-quad-decimate",
        type=float,
        default=2.0,
        help="pupil-apriltags quad_decimate (default 2; 1 = finer, slower)",
    )
    parser.add_argument(
        "--tag-drain",
        type=int,
        default=2,
        help="Drop buffered back-camera frames before display (default 2 = fresher video)",
    )
    parser.add_argument(
        "--sync-tags",
        action="store_true",
        help="Run AprilTag on the main thread (blocks; default is background thread)",
    )
    args = parser.parse_args()

    if args.full_tag_quality:
        args.tag_detect_every = 1
        args.tag_detect_scale = 1.0
        args.tag_quad_decimate = 1.0
        args.tag_drain = 0
    elif args.fast_tag:
        args.tag_detect_every = max(3, args.tag_detect_every)

    if not os.environ.get("DISPLAY"):
        print("Need a display (DISPLAY set) for OpenCV windows.")
        return 1

    if not jeo_tracker_available():
        print("JEO 3DTracker not found. Clone:")
        print("  git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker")
        return 1

    if AprilTagDetector is None:
        print(f"AprilTag detector unavailable: {_APRILTAG_ERR}")
        print("  pip install pupil-apriltags")
        return 1

    rois_display, rois_path = load_rois_for_display(args.roi_file)
    if rois_display and rois_path:
        print(f"ROI file: {rois_path}")
    else:
        print(f"Warning: {GLASS_FRAME_ROIS_FILENAME} not found — eye crops will be placeholders.")

    tag_cap, tag_desc = open_tag_csi(args.tag_sensor_id, args.tag_width, args.tag_height, CSI_FPS)
    if tag_cap is None:
        print(f"Could not open back CSI (sensor-id={args.tag_sensor_id}). Try another --tag-sensor-id.")
        print("  If busy: sudo systemctl restart nvargus-daemon")
        return 1
    print(f"Back (AprilTag) camera: {tag_desc}")

    tracker = JEOGlassFrameTracker(
        eye_camera_id=0,
        eye_sensor_id=args.eye_sensor_id,
        rois_path=rois_path,
    )
    if not tracker.is_opened():
        tag_cap.release()
        print(f"Could not open eye CSI (sensor-id={args.eye_sensor_id}).")
        print("  Front eye camera should be on cam1 → default --eye-sensor-id 1")
        return 1
    print(f"Eye camera: CSI sensor-id={args.eye_sensor_id} (JEO 3D tracker)")

    try:
        detector = AprilTagDetector(
            families="tag36h11",
            quad_decimate=float(args.tag_quad_decimate),
        )
    except Exception as e:
        tracker.release()
        tag_cap.release()
        print(f"AprilTag init failed: {e}")
        return 1

    win_full = "1. Eye CSI (full frame)"
    win_left = "2. Left eye CROPPED"
    win_right = "3. Right eye CROPPED"
    win_tag = "4. Back CSI (AprilTag)"
    cv2.namedWindow(win_full, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_left, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_tag, cv2.WINDOW_NORMAL)

    tracking_active = False
    frame_count = 0
    last_tag_dets: list = []
    de = max(1, int(args.tag_detect_every))
    dscale = float(args.tag_detect_scale)
    if dscale <= 0 or dscale > 1.0:
        dscale = 1.0
    inv_scale = 1.0 / dscale
    use_async = not args.sync_tags
    tag_worker = None
    if use_async:
        tag_worker = _AsyncTagWorker(detector, dscale, inv_scale)
        tag_worker.start()

    print("SPACE = start/pause eye tracking. Q = quit. Back window: tag36h11.")
    print(
        f"Tag: every {de} frame(s), scale={dscale}, quad_decimate={args.tag_quad_decimate}, "
        f"drain={args.tag_drain}, async={'on' if use_async else 'off (sync)'}"
    )

    try:
        while True:
            frame_count += 1
            ret, tag_frame = grab_latest_frame(tag_cap, args.tag_drain)
            tag_ok = ret and tag_frame is not None
            if not tag_ok:
                tag_frame = np.zeros((args.tag_height, args.tag_width, 3), dtype=np.uint8)
                cv2.putText(
                    tag_frame,
                    "No back camera frame",
                    (40, tag_frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )
                last_tag_dets = []
                if tag_worker:
                    tag_worker.clear_dets()
            elif tag_worker:
                tag_worker.submit_if_due(tag_frame, frame_count, de)

            tracker.process_frame_binocular()
            full = tracker.get_last_full_frame()
            has_valid_frame = full is not None and full.size > 0
            if not has_valid_frame:
                full = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(full, "No eye camera frame", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            h, w = full.shape[:2]

            draw_eye_windows(
                tracker,
                full,
                rois_display,
                w,
                h,
                frame_count,
                tracking_active,
                has_valid_frame,
                win_full,
                win_left,
                win_right,
            )

            if tag_worker:
                last_tag_dets = tag_worker.get_dets()
            elif tag_ok:
                if frame_count % de == 0:
                    try:
                        if dscale < 1.0:
                            small = cv2.resize(
                                tag_frame,
                                None,
                                fx=dscale,
                                fy=dscale,
                                interpolation=cv2.INTER_AREA,
                            )
                            raw = detector.detect(small)
                            last_tag_dets = scale_detections(raw, inv_scale)
                        else:
                            last_tag_dets = detector.detect(tag_frame)
                    except Exception:
                        last_tag_dets = []
            draw_tag_overlays(tag_frame, last_tag_dets)
            sub = f" every{de}f" if de > 1 else ""
            async_lbl = " async" if tag_worker else ""
            cv2.putText(
                tag_frame,
                f"4. Back CSI — AprilTag{sub}{async_lbl} (sensor {args.tag_sensor_id})",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(win_tag, tag_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                tracking_active = not tracking_active
    finally:
        if tag_worker:
            tag_worker.stop()
        tracker.release()
        tag_cap.release()
        cv2.destroyAllWindows()

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
