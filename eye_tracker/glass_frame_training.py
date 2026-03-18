import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .calibration import (
    Calibration,
    _get_screen_size,
    PolyRidgeRegressor,
    _fit_binocular_ridge,
    _remove_outliers_mad,
    _train_val_split,
    MIN_BINOC_DOTS,
    ML_POLY_DEGREE,
)
from .object_hover.apriltag_detector import AprilTagDetector

# Glass-frame ML data comes from the JEOresearch/EyeTracker 3DTracker (vendored). See CREDITS.md.
# ML data is recorded ONLY when the user runs this calibration (GlassFrameTraining.run()).
# The test script (test_csi_glass_frame_pupil) and the hover app never write training CSV/npz.
from .glass_frame_jeo_tracker import (
    JEOGlassFrameTracker,
    jeo_tracker_available,
    load_rois_and_signature,
    _build_csi_pipeline,
    _roi_to_pixel_rect,
)

# No recording until user has started and quality is sufficient
MIN_RECORDING_CONF = 0.6  # Only record when tracker confidence >= this (avoid noisy data)
WARMUP_SEC = 2.0          # Seconds after calibration window opens before first sample (avoid settling noise)
# Safety: never record when no pupil detected in (cropped) frame — _collect_feature returns None in that case

# Front camera display (same as test_csi_glass_frame_pupil)
WIN_FRONT_FULL = "1. Front camera (full frame)"
WIN_FRONT_LEFT = "2. Left eye CROPPED"
WIN_FRONT_RIGHT = "3. Right eye CROPPED"
WIN_BACK = "4. Back camera (AprilTag)"
INFO_STRIP_H = 72


def _crop_with_info_strip(crop_bgr, color_bgr, lines):
    """Stack crop and info strip with text lines (same as test script)."""
    h, w = crop_bgr.shape[:2]
    strip = np.zeros((INFO_STRIP_H, w, 3), dtype=np.uint8)
    strip[:] = (40, 40, 40)
    y_offset = 18
    for line in lines:
        if line:
            cv2.putText(strip, line, (6, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
        y_offset += 20
    return np.vstack([crop_bgr, strip])


class GlassFrameTrainingConfig:
    def __init__(
        self,
        calibration_mode="automated_calibration",  # automated_calibration | manual_calibration
        screen_size=None,
        board_size=None,
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        training_mode="glass_frame",
        eye_camera_id=0,
        eye_sensor_id=None,  # CSI sensor id on Jetson (e.g. 0 or 1); if set, eye camera uses CSI
        tag_camera_id=1,
        tag_camera_csi_sensor_id=None,  # if set, back/tag camera uses CSI (e.g. 0 = cam0 24-pin on Jetson)
        apriltag_length=0.04,  # meters
        apriltag_width=None,   # meters (if None, same as length)
        apriltag_focal_px=700.0,
        tag_ids=(1, 2, 3, 4, 5),
        apriltag_families="tag36h11",
        tag_image_dir=None,
        tag_display_px=260,
        dwell_sec=1.5,
        manual_duration_sec=45.0,
        num_automated_positions=5,  # number of screen targets in automated mode (5 = center + 4 corners; use 9 or 13 for more)
        glass_quality_profile="balanced",  # unused; kept for API compatibility
        warmup_sec=WARMUP_SEC,
        min_recording_conf=MIN_RECORDING_CONF,
        rois_path=None,  # path to glass_frame_rois.json; None = search default locations
        validation_fraction=0.15,  # hold out for validation RMSE (0 = no split)
        outlier_mad_multiplier=2.5,  # MAD-based outlier removal (0 = disable)
        min_train_samples=10,
        tag_camera_buffer_drain=2,  # grab N buffered frames before decode — fresher tag video (CSI)
        overlap_tag_eye_detection=True,  # run AprilTag on tag frame while JEO runs (same frame, smoother loop)
        tag_quad_decimate=1.0,  # pupil-apriltags speedup; 1.0 = full accuracy, 2.0 = faster
    ):
        self.calibration_mode = calibration_mode
        self.screen_size = screen_size
        self.board_size = board_size
        self.training_data_name = training_data_name
        self.training_data_dir = training_data_dir
        self.training_mode = training_mode
        self.eye_camera_id = int(eye_camera_id)
        self.eye_sensor_id = int(eye_sensor_id) if eye_sensor_id is not None else None
        self.tag_camera_id = int(tag_camera_id)
        self.tag_camera_csi_sensor_id = int(tag_camera_csi_sensor_id) if tag_camera_csi_sensor_id is not None else None
        self.apriltag_length = float(apriltag_length)
        self.apriltag_width = float(apriltag_width) if apriltag_width is not None else float(apriltag_length)
        self.apriltag_focal_px = float(apriltag_focal_px)
        self.tag_ids = list(tag_ids)
        self.apriltag_families = apriltag_families
        if tag_image_dir is None:
            tag_image_dir = os.path.expanduser("~/Downloads/apriltags_tag36h11")
        self.tag_image_dir = tag_image_dir
        self.tag_display_px = int(tag_display_px)
        self.dwell_sec = float(dwell_sec)
        self.manual_duration_sec = float(manual_duration_sec)
        self.num_automated_positions = int(num_automated_positions)
        self.glass_quality_profile = str(glass_quality_profile).strip().lower()
        self.warmup_sec = float(warmup_sec)
        self.min_recording_conf = float(min_recording_conf)
        self.rois_path = rois_path
        self.validation_fraction = float(validation_fraction)
        self.outlier_mad_multiplier = float(outlier_mad_multiplier)
        self.min_train_samples = int(min_train_samples)
        self.tag_camera_buffer_drain = int(tag_camera_buffer_drain)
        self.overlap_tag_eye_detection = bool(overlap_tag_eye_detection)
        self.tag_quad_decimate = float(tag_quad_decimate)


class GlassFrameTraining:
    def __init__(self, config):
        if config.training_mode != "glass_frame":
            raise ValueError("GlassFrameTraining requires training_mode='glass_frame'.")
        if config.calibration_mode not in ("automated_calibration", "manual_calibration"):
            raise ValueError("calibration_mode must be 'automated_calibration' or 'manual_calibration'.")
        self.config = config
        if config.screen_size is None:
            self.screen_w, self.screen_h = _get_screen_size()
        else:
            self.screen_w, self.screen_h = int(config.screen_size[0]), int(config.screen_size[1])
        # ROI-based paths: when ROIs are loaded, CSV/npz use _roi_<signature> so redrawing ROI creates new files
        rois_dict, rois_path, roi_signature = load_rois_and_signature(getattr(config, "rois_path", None))
        self._rois_dict = rois_dict
        self._rois_path = rois_path
        self.calib = Calibration(
            training_data_name=self.config.training_data_name,
            training_data_dir=self.config.training_data_dir,
            screen_size=self.config.board_size or self.config.screen_size,
            fullscreen=True,
            window_name="glass_frame_calibration",
            training_mode="glass_frame",
            roi_signature=roi_signature,
        )
        self._write_roi_meta_if_needed()
        self.tag_detector = AprilTagDetector(
            families=self.config.apriltag_families,
            quad_decimate=max(1.0, float(self.config.tag_quad_decimate)),
        )

    @staticmethod
    def _grab_latest_tag_frame(cap, drain: int):
        """Drop stale CSI/USB buffers so tag frame matches recent video (same idea as dual-CSI test)."""
        n = max(0, int(drain))
        for _ in range(n):
            if not cap.grab():
                break
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            ret, frame = cap.read()
        return ret, frame

    def _detect_tag_frame(self, frame):
        """AprilTag on one BGR frame; safe to call from worker thread (one call at a time)."""
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return []
        try:
            return self.tag_detector.detect(frame)
        except Exception:
            return []

    def _write_roi_meta_if_needed(self):
        """Write ROI metadata next to the training CSV so you can see which ROI set a file belongs to (for fallback/reuse)."""
        if not self._rois_dict or not self.calib.current_roi_signature:
            return
        path = self.calib.current_training_data_path
        if not path or not path.endswith(".csv"):
            return
        import json
        meta_path = path.replace(".csv", "_roi_meta.json")
        try:
            with open(meta_path, "w") as f:
                json.dump({
                    "left_eye_roi": self._rois_dict.get("left_eye_roi"),
                    "right_eye_roi": self._rois_dict.get("right_eye_roi"),
                    "roi_signature": self.calib.current_roi_signature,
                }, f, indent=2)
        except Exception:
            pass

    def _estimate_distance_m(self, bbox_w_px, bbox_h_px=None):
        # Pin-hole estimate: Z = f * tag_size / observed_size
        if bbox_w_px is None or bbox_w_px <= 1:
            return None
        if self.config.apriltag_focal_px <= 0:
            return None
        z_w = None
        z_h = None
        if self.config.apriltag_width > 0:
            z_w = float((self.config.apriltag_focal_px * self.config.apriltag_width) / float(bbox_w_px))
        if bbox_h_px is not None and bbox_h_px > 1 and self.config.apriltag_length > 0:
            z_h = float((self.config.apriltag_focal_px * self.config.apriltag_length) / float(bbox_h_px))
        if z_w is not None and z_h is not None:
            return float(0.5 * (z_w + z_h))
        return z_w if z_w is not None else z_h

    def _load_tag_images(self):
        out = {}
        for tag_id in self.config.tag_ids:
            name = f"tag36_11_{tag_id:05d}.png"
            path = os.path.join(self.config.tag_image_dir, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out[tag_id] = img
        return out

    def _target_positions(self, n):
        """Screen positions for calibration targets. 5 = center + 4 corners; 9 = 3x3 grid; 13 = 3x3 + 4 edge midpoints."""
        center = (self.screen_w // 2, self.screen_h // 2)
        pts_5 = [
            center,
            (self.screen_w // 4, self.screen_h // 4),
            (3 * self.screen_w // 4, self.screen_h // 4),
            (self.screen_w // 4, 3 * self.screen_h // 4),
            (3 * self.screen_w // 4, 3 * self.screen_h // 4),
        ]
        if n <= 5:
            return pts_5[:n]
        grid_9 = [
            (self.screen_w // 4, self.screen_h // 4),
            (self.screen_w // 2, self.screen_h // 4),
            (3 * self.screen_w // 4, self.screen_h // 4),
            (self.screen_w // 4, self.screen_h // 2),
            (self.screen_w // 2, self.screen_h // 2),
            (3 * self.screen_w // 4, self.screen_h // 2),
            (self.screen_w // 4, 3 * self.screen_h // 4),
            (self.screen_w // 2, 3 * self.screen_h // 4),
            (3 * self.screen_w // 4, 3 * self.screen_h // 4),
        ]
        if n <= 9:
            return grid_9[:n]
        edge_4 = [
            (0, self.screen_h // 2),
            (self.screen_w // 2, 0),
            (self.screen_w - 1, self.screen_h // 2),
            (self.screen_w // 2, self.screen_h - 1),
        ]
        pts_13 = grid_9 + edge_4
        if n <= 13:
            return pts_13[:n]
        out = list(pts_13)
        while len(out) < n:
            out.append(center)
        return out[:n]

    def _render_tag_canvas(self, tag_img, center_xy):
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        s = self.config.tag_display_px
        tag = cv2.resize(tag_img, (s, s), interpolation=cv2.INTER_NEAREST)
        tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        x0 = int(np.clip(center_xy[0] - s // 2, 0, self.screen_w - s))
        y0 = int(np.clip(center_xy[1] - s // 2, 0, self.screen_h - s))
        canvas[y0:y0 + s, x0:x0 + s] = tag
        return canvas

    def _pick_detection(self, detections, desired_id=None, allow_any_tag=False):
        # In manual mode (allow_any_tag=True), accept any detected tag so you can use as many tags on the board as you want.
        if allow_any_tag:
            filtered = list(detections) if detections else []
        else:
            filtered = [d for d in detections if d.get("tag_id") in self.config.tag_ids]
        if desired_id is not None:
            filtered = [d for d in filtered if d.get("tag_id") == desired_id]
        if not filtered:
            return None
        filtered.sort(key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
        return filtered[0]

    def _make_training_row(
        self,
        fv,
        conf,
        stage,
        screen_xy,
        screen_size,
        tag_det=None,
        tag_frame_size=None,
    ):
        sw, sh = screen_size
        sx, sy = screen_xy
        row = {
            "timestamp": time.time(),
            "dot_index": "",
            "stage": stage,
            "screen_x": float(sx),
            "screen_y": float(sy),
            "screen_x_norm": float(sx) / max(1.0, float(sw)),
            "screen_y_norm": float(sy) / max(1.0, float(sh)),
            "screen_w": float(sw),
            "screen_h": float(sh),
            "gx": "",
            "gy": "",
            "yaw": 0.0,
            "pitch": 0.0,
            "gxL": float(fv[0]),
            "gyL": float(fv[1]),
            "gxR": float(fv[2]),
            "gyR": float(fv[3]),
            "iod": float(fv[6]) if len(fv) > 6 else 0.0,
            "roll": float(fv[7]) if len(fv) > 7 else 0.0,
            "lid_hL": float(fv[8]) if len(fv) > 8 else 0.0,
            "lid_hR": float(fv[9]) if len(fv) > 9 else 0.0,
            "face_w": float(fv[10]) if len(fv) > 10 else 0.0,
            "face_h": float(fv[11]) if len(fv) > 11 else 0.0,
            "nose_x": float(fv[12]) if len(fv) > 12 else 0.0,
            "nose_y": float(fv[13]) if len(fv) > 13 else 0.0,
            "wL": float(fv[14]) if len(fv) > 14 else 0.5,
            "wR": float(fv[15]) if len(fv) > 15 else 0.5,
            "quality": float(np.clip(conf, 0.0, 1.0)),
            "eye_w": "",
            "lid_h": "",
            "conf": float(conf),
            "bbox_x": "",
            "bbox_y": "",
            "bbox_w": "",
            "bbox_h": "",
            "label_x": "",
            "label_y": "",
            "label_x_norm": "",
            "label_y_norm": "",
            "tag_id": "",
            "board_distance_m": "",
            "apriltag_length_m": float(self.config.apriltag_length),
            "apriltag_width_m": float(self.config.apriltag_width),
            "tag_cam_x": "",
            "tag_cam_y": "",
            "tag_bbox_w_px": "",
            "tag_bbox_h_px": "",
            "pupil_L_x_norm": "",
            "pupil_L_y_norm": "",
            "pupil_R_x_norm": "",
            "pupil_R_y_norm": "",
            "pupil_L_x_full": "",
            "pupil_L_y_full": "",
            "pupil_R_x_full": "",
            "pupil_R_y_full": "",
        }
        # Pupil coordinates and gaze are recorded for data training (from tracker after process_frame_binocular)
        if getattr(self, "_last_pupil_left_norm", None) is not None:
            nx, ny = self._last_pupil_left_norm
            row["pupil_L_x_norm"] = float(nx)
            row["pupil_L_y_norm"] = float(ny)
        if getattr(self, "_last_pupil_right_norm", None) is not None:
            nx, ny = self._last_pupil_right_norm
            row["pupil_R_x_norm"] = float(nx)
            row["pupil_R_y_norm"] = float(ny)
        if getattr(self, "_last_pupil_left_full", None) is not None:
            fx, fy = self._last_pupil_left_full
            row["pupil_L_x_full"] = float(fx)
            row["pupil_L_y_full"] = float(fy)
        if getattr(self, "_last_pupil_right_full", None) is not None:
            fx, fy = self._last_pupil_right_full
            row["pupil_R_x_full"] = float(fx)
            row["pupil_R_y_full"] = float(fy)
        if tag_det is not None:
            x, y, w, h = tag_det["bbox"]
            # Prefer corner-mean center for higher accuracy; fallback to bbox center
            center = tag_det.get("center")
            if center is not None and len(center) >= 2:
                cx, cy = float(center[0]), float(center[1])
            else:
                cx = x + w * 0.5
                cy = y + h * 0.5
            row["bbox_x"] = float(x)
            row["bbox_y"] = float(y)
            row["bbox_w"] = float(w)
            row["bbox_h"] = float(h)
            row["label_x"] = float(cx)
            row["label_y"] = float(cy)
            if tag_frame_size is not None:
                tw, th = tag_frame_size
                row["label_x_norm"] = float(cx) / max(1.0, float(tw))
                row["label_y_norm"] = float(cy) / max(1.0, float(th))
            row["tag_id"] = int(tag_det.get("tag_id")) if tag_det.get("tag_id") is not None else ""
            row["tag_cam_x"] = float(cx)
            row["tag_cam_y"] = float(cy)
            row["tag_center_x"] = float(cx)
            row["tag_center_y"] = float(cy)
            row["tag_bbox_w_px"] = float(w)
            row["tag_bbox_h_px"] = float(h)
            corners = tag_det.get("corners")
            if corners is not None and len(corners) >= 4:
                for i in range(4):
                    row[f"tag_c{i}_x"] = float(corners[i][0])
                    row[f"tag_c{i}_y"] = float(corners[i][1])
            z_m = self._estimate_distance_m(w, h)
            if z_m is not None:
                row["board_distance_m"] = float(z_m)
        return row

    def _collect_feature(self, tracker):
        """Get one sample from the tracker. Returns (fv, conf) or None.
        Returns None when no pupil is detected in the (cropped) frame — no data is recorded then."""
        gx, gy, yaw, pitch, fv, conf, is_blink = tracker.process_frame_binocular()
        if is_blink or gx is None or gy is None or fv is None:
            return None
        conf = float(conf)
        if conf <= 0:
            return None  # No pupil detected; do not record
        return fv, conf

    def _draw_front_camera_windows(self, tracker):
        """Draw the same three front-camera windows as test_csi_glass_frame_pupil (full, left crop, right crop + coords)."""
        full = tracker.get_last_full_frame()
        rois_display = self._rois_dict
        if full is None or full.size == 0:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No frame from front camera", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(WIN_FRONT_FULL, placeholder)
            cv2.imshow(WIN_FRONT_LEFT, placeholder)
            cv2.imshow(WIN_FRONT_RIGHT, placeholder)
            return
        h, w = full.shape[:2]
        disp_full = full.copy()
        pupil_full = tracker.get_last_left_right_pupil_full()
        pupils_crop = tracker.get_last_left_right_pupil_in_crop()
        full_left = pupil_full[0] if pupil_full else None
        full_right = pupil_full[1] if pupil_full else None
        pupil_left_crop = pupils_crop[0] if pupils_crop else None
        pupil_right_crop = pupils_crop[1] if pupils_crop else None
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
                lx, ly = int(full_left[0]), int(full_left[1])
                cv2.circle(disp_full, (lx, ly), 6, (0, 255, 0), 2)
            if full_right is not None:
                rx, ry = int(full_right[0]), int(full_right[1])
                cv2.circle(disp_full, (rx, ry), 6, (0, 0, 255), 2)
            y_text = disp_full.shape[0] - 72
            cv2.rectangle(disp_full, (0, y_text - 4), (disp_full.shape[1], disp_full.shape[0]), (32, 32, 32), -1)
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
        cv2.putText(disp_full, "1. Front (full frame)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WIN_FRONT_FULL, disp_full)
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
                lines = [
                    f"Main frame: ({full_left[0]:.1f}, {full_left[1]:.1f})" if full_left else "Main frame: --",
                    f"Left crop: ({pupil_left_crop[0]:.1f}, {pupil_left_crop[1]:.1f})" if pupil_left_crop else "Left crop: --",
                    f"Gaze: ({gxL:.2f}, {gyL:.2f})" if gxL is not None and gyL is not None else "Gaze: --",
                ]
                disp_left = _crop_with_info_strip(left_crop, (0, 255, 0), lines)
                cv2.imshow(WIN_FRONT_LEFT, disp_left)
            else:
                no_crop = np.zeros((200, 320, 3), dtype=np.uint8)
                cv2.putText(no_crop, "No left ROI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(WIN_FRONT_LEFT, no_crop)
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
                    f"Main frame: ({full_right[0]:.1f}, {full_right[1]:.1f})" if full_right else "Main frame: --",
                    f"Right crop: ({pupil_right_crop[0]:.1f}, {pupil_right_crop[1]:.1f})" if pupil_right_crop else "Right crop: --",
                    f"Gaze: ({gxR:.2f}, {gyR:.2f})" if gxR is not None and gyR is not None else "Gaze: --",
                ]
                disp_right = _crop_with_info_strip(right_crop, (0, 0, 255), lines)
                cv2.imshow(WIN_FRONT_RIGHT, disp_right)
            else:
                no_crop = np.zeros((200, 320, 3), dtype=np.uint8)
                cv2.putText(no_crop, "No right ROI", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(WIN_FRONT_RIGHT, no_crop)
        else:
            no_roi = np.zeros((200, 320, 3), dtype=np.uint8)
            cv2.putText(no_roi, "No ROIs loaded", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WIN_FRONT_LEFT, no_roi)
            cv2.imshow(WIN_FRONT_RIGHT, no_roi)

    def _init_tag_camera(self):
        csi_sensor = getattr(self.config, "tag_camera_csi_sensor_id", None)
        if csi_sensor is not None:
            pipeline = _build_csi_pipeline(csi_sensor, width=1280, height=720, fps=30)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.config.tag_camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def _train_model_from_rows(self):
        import math
        feature_vecs, screen_targets = self.calib._load_training_data()
        n_total = len(feature_vecs)
        if n_total < self.config.min_train_samples:
            print(f"[glass_frame] Not enough samples to train model: {n_total} (min {self.config.min_train_samples})")
            return False
        n_dropped = 0
        if self.config.outlier_mad_multiplier > 0:
            feature_vecs, screen_targets, n_dropped = _remove_outliers_mad(
                feature_vecs, screen_targets, k_mad=self.config.outlier_mad_multiplier
            )
            if n_dropped > 0:
                print(f"[glass_frame] Outlier removal (MAD k={self.config.outlier_mad_multiplier}): dropped {n_dropped} samples")
            if len(feature_vecs) < self.config.min_train_samples:
                print(f"[glass_frame] Too few samples after outlier removal: {len(feature_vecs)}")
                return False
        train_fv, train_tgt, val_fv, val_tgt = feature_vecs, screen_targets, [], []
        if self.config.validation_fraction > 0 and len(feature_vecs) >= 20:
            train_fv, train_tgt, val_fv, val_tgt = _train_val_split(
                feature_vecs, screen_targets,
                val_fraction=self.config.validation_fraction,
                seed=42,
            )
        if len(train_fv) >= MIN_BINOC_DOTS:
            mu, std, W = _fit_binocular_ridge(train_fv, train_tgt, lam=1e-1)
            self.calib.binoc_mu = mu
            self.calib.binoc_std = std
            self.calib.binoc_W = W
        self.calib.ml = PolyRidgeRegressor(lam=5e-3, degree=ML_POLY_DEGREE)
        ok = self.calib.ml.fit(train_fv, train_tgt)
        if not ok:
            print("[glass_frame] ML fit failed.")
            return False
        def rmse_norm(fv_list, tgt_list):
            if not fv_list or not tgt_list:
                return None
            pred = []
            for fv in fv_list:
                out = self.calib.ml.predict(fv)
                if out is None:
                    continue
                pred.append(out)
            if len(pred) != len(tgt_list):
                return None
            pred = np.asarray(pred, dtype=np.float64)
            tgt = np.asarray(tgt_list, dtype=np.float64)
            return float(np.sqrt(np.mean(np.sum((pred - tgt) ** 2, axis=1))))
        train_rmse = rmse_norm(train_fv, train_tgt)
        val_rmse = rmse_norm(val_fv, val_tgt) if val_fv else None
        if train_rmse is not None:
            print(f"[glass_frame] Train RMSE (norm): {train_rmse:.4f}")
        if val_rmse is not None:
            print(f"[glass_frame] Validation RMSE (norm): {val_rmse:.4f}")
        sw = getattr(self.calib, "screen_w", 1920) or 1920
        sh = getattr(self.calib, "screen_h", 1080) or 1080
        viewing_mm = 500.0
        pixel_err = (train_rmse or 0) * math.sqrt(sw * sw + sh * sh) / math.sqrt(2)
        approx_deg = math.degrees(math.atan(pixel_err / viewing_mm)) if viewing_mm > 0 else 0.0
        if val_rmse is not None:
            pixel_err_val = val_rmse * math.sqrt(sw * sw + sh * sh) / math.sqrt(2)
            approx_deg_val = math.degrees(math.atan(pixel_err_val / viewing_mm)) if viewing_mm > 0 else 0.0
        else:
            pixel_err_val = approx_deg_val = None
        path = self.calib._ml_model_path()
        self.calib.ml.save(path, extra={
            "binoc_mu": self.calib.binoc_mu if self.calib.binoc_mu is not None else np.array([]),
            "binoc_std": self.calib.binoc_std if self.calib.binoc_std is not None else np.array([]),
            "binoc_W": self.calib.binoc_W if self.calib.binoc_W is not None else np.array([]),
        })
        print(f"[glass_frame] Model saved: {path}")
        report_lines = [
            "Glass-frame calibration report",
            "==============================",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
            f"Total samples loaded: {n_total}",
            f"Outliers removed (MAD): {n_dropped}",
            f"Train samples: {len(train_fv)}",
            f"Validation samples: {len(val_fv)}",
            f"Train RMSE (norm): {train_rmse:.4f}" if train_rmse is not None else "Train RMSE: N/A",
            f"Validation RMSE (norm): {val_rmse:.4f}" if val_rmse is not None else "Validation RMSE: N/A",
            f"Approx. train pixel error (px): {pixel_err:.1f}",
            f"Approx. train angular error (deg, {viewing_mm}mm): {approx_deg:.2f}",
        ]
        if val_rmse is not None:
            report_lines.append(f"Approx. val pixel error (px): {pixel_err_val:.1f}")
            report_lines.append(f"Approx. val angular error (deg): {approx_deg_val:.2f}")
        report_lines.append(f"Model path: {path}")
        report_text = "\n".join(report_lines)
        print(report_text)
        report_path = path.replace(".npz", "_report.txt")
        try:
            with open(report_path, "w") as f:
                f.write(report_text)
            print(f"[glass_frame] Report saved: {report_path}")
        except Exception as e:
            print(f"[glass_frame] Could not write report file: {e}")
        return True

    def _validate_rows(self, rows):
        valid = []
        dropped = 0
        for r in rows:
            try:
                # Essential training targets/features must be finite.
                sxn = float(r["screen_x_norm"])
                syn = float(r["screen_y_norm"])
                gxL = float(r["gxL"])
                gyL = float(r["gyL"])
                gxR = float(r["gxR"])
                gyR = float(r["gyR"])
                _ = (sxn, syn, gxL, gyL, gxR, gyR)
                if not np.isfinite([sxn, syn, gxL, gyL, gxR, gyR]).all():
                    dropped += 1
                    continue
                # Keep targets within a broad bound (manual mode may come from tag camera).
                if abs(sxn) > 2.0 or abs(syn) > 2.0:
                    dropped += 1
                    continue
                valid.append(r)
            except Exception:
                dropped += 1
        if dropped > 0:
            print(f"[glass_frame] Dropped invalid rows: {dropped}")
        return valid

    def _run_automated(self, tracker):
        tag_images = self._load_tag_images()
        missing = [tid for tid in self.config.tag_ids if tid not in tag_images]
        if missing:
            raise FileNotFoundError(
                f"Missing tag images for IDs {missing} in {self.config.tag_image_dir}. "
                "Download tags first (IDs 1-5)."
            )
        tag_cap = self._init_tag_camera()
        if not tag_cap.isOpened():
            raise RuntimeError(f"Could not open tag camera id={self.config.tag_camera_id}.")
        name = "automated_calibration"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        num_positions = self.config.num_automated_positions
        positions = self._target_positions(num_positions)
        rows = []
        warmup_sec = self.config.warmup_sec
        min_conf = self.config.min_recording_conf
        tag_ids = self.config.tag_ids
        drain = self.config.tag_camera_buffer_drain
        overlap = self.config.overlap_tag_eye_detection
        try:
            with ThreadPoolExecutor(max_workers=1) as tag_pool:
                for idx in range(num_positions):
                    target_xy = positions[idx]
                    # Cycle through configured tag images (e.g. 5 tags for 9 or 13 positions)
                    tag_id = tag_ids[idx % len(tag_ids)]
                    t0 = time.perf_counter()
                    recording_started = t0 + warmup_sec
                    while (time.perf_counter() - t0) < self.config.dwell_sec:
                        now = time.perf_counter()
                        ret, tag_frame = self._grab_latest_tag_frame(tag_cap, drain)
                        if not ret:
                            continue
                        if overlap:
                            fut = tag_pool.submit(self._detect_tag_frame, tag_frame.copy())
                            fv_conf = self._collect_feature(tracker)
                            try:
                                all_dets = fut.result(timeout=15.0)
                            except Exception:
                                all_dets = []
                        else:
                            fv_conf = self._collect_feature(tracker)
                            all_dets = self._detect_tag_frame(tag_frame)
                        det = self._pick_detection(all_dets, desired_id=tag_id)
                    display = self._render_tag_canvas(tag_images[tag_id], target_xy)
                    if now < recording_started:
                        cv2.putText(
                            display,
                            f"Get ready - recording in {max(0, int(recording_started - now))}s (target {idx + 1}/{num_positions})",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            display,
                            f"Automated calibration: target {idx + 1}/{num_positions} - RECORDING",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )
                    cv2.imshow(name, display)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return rows
                    if now < recording_started or fv_conf is None or det is None:
                        continue
                    fv, conf = fv_conf
                    if conf < min_conf:
                        continue
                    # Record pupil coordinates and gaze for data training (from tracker)
                    norm = tracker.get_last_left_right_pupil_normalized_crop() if hasattr(tracker, "get_last_left_right_pupil_normalized_crop") else (None, None)
                    full = tracker.get_last_left_right_pupil_full() if hasattr(tracker, "get_last_left_right_pupil_full") else (None, None)
                    self._last_pupil_left_norm = norm[0] if norm else None
                    self._last_pupil_right_norm = norm[1] if norm else None
                    self._last_pupil_left_full = full[0] if full else None
                    self._last_pupil_right_full = full[1] if full else None
                    row = self._make_training_row(
                        fv=fv,
                        conf=conf,
                        stage="auto_tag",
                        screen_xy=target_xy,
                        screen_size=(self.screen_w, self.screen_h),
                        tag_det=det,
                        tag_frame_size=(tag_frame.shape[1], tag_frame.shape[0]),
                    )
                    rows.append(row)
        finally:
            tag_cap.release()
            cv2.destroyWindow(name)
        return rows

    def _run_manual(self, tracker):
        """Manual calibration: front camera = 3 windows (full + L/R crop + coords), back camera = 1 window.
        AprilTag IDs in order 0, 1, 2, 3, ... SPACE: first = start recording for tag 0; second = pause;
        third = move to tag 1 and record; etc. Only records when the back camera sees the current tag ID."""
        tag_cap = self._init_tag_camera()
        if not tag_cap.isOpened():
            raise RuntimeError(
                f"Could not open back (tag) camera. Check tag_camera_id / tag_camera_csi_sensor_id."
            )
        cv2.namedWindow(WIN_FRONT_FULL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_FRONT_LEFT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_FRONT_RIGHT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_BACK, cv2.WINDOW_NORMAL)
        rows = []
        current_tag_id = 0
        recording = False
        just_paused = False
        min_conf = self.config.min_recording_conf
        drain = self.config.tag_camera_buffer_drain
        overlap = self.config.overlap_tag_eye_detection
        try:
            with ThreadPoolExecutor(max_workers=1) as tag_pool:
                while True:
                    if overlap:
                        ret, tag_frame = self._grab_latest_tag_frame(tag_cap, drain)
                        if not ret:
                            tag_frame = None
                        if tag_frame is not None:
                            fut = tag_pool.submit(self._detect_tag_frame, tag_frame.copy())
                            fv_conf = self._collect_feature(tracker)
                            try:
                                all_dets = fut.result(timeout=15.0)
                            except Exception:
                                all_dets = []
                        else:
                            fv_conf = self._collect_feature(tracker)
                            all_dets = []
                    else:
                        fv_conf = self._collect_feature(tracker)
                        ret, tag_frame = self._grab_latest_tag_frame(tag_cap, drain)
                        if not ret:
                            tag_frame = None
                        all_dets = self._detect_tag_frame(tag_frame) if tag_frame is not None else []
                    det_current = next((d for d in all_dets if d.get("tag_id") == current_tag_id), None)
                    det_for_display = det_current if det_current is not None else (all_dets[0] if all_dets else None)
                    self._draw_front_camera_windows(tracker)
                    preview = tag_frame.copy() if tag_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    tw, th = preview.shape[1], preview.shape[0]
                    cv2.putText(
                        preview,
                        f"Current tag ID: {current_tag_id}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )
                    if recording:
                        cv2.putText(preview, "Recording - look at tag %d" % current_tag_id, (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(preview, "Paused", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(preview, "SPACE = start / pause / next tag    Q = quit", (20, th - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    if det_for_display is not None:
                        x, y, w, h = det_for_display["bbox"]
                        color = (0, 255, 0) if det_for_display.get("tag_id") == current_tag_id else (128, 128, 128)
                        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(preview, "tag_%d" % det_for_display.get("tag_id", -1), (x, max(20, y - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.imshow(WIN_BACK, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        if recording:
                            recording = False
                            just_paused = True
                        else:
                            if just_paused:
                                current_tag_id += 1
                                just_paused = False
                            recording = True
                    if not recording or fv_conf is None or det_current is None:
                        continue
                    fv, conf = fv_conf
                    if conf < min_conf:
                        continue
                    center = det_current.get("center")
                    if center is not None and len(center) >= 2:
                        cx, cy = float(center[0]), float(center[1])
                    else:
                        x, y, w, h = det_current["bbox"]
                        cx = x + w * 0.5
                        cy = y + h * 0.5
                    norm = tracker.get_last_left_right_pupil_normalized_crop() if hasattr(tracker, "get_last_left_right_pupil_normalized_crop") else (None, None)
                    full_p = tracker.get_last_left_right_pupil_full() if hasattr(tracker, "get_last_left_right_pupil_full") else (None, None)
                    self._last_pupil_left_norm = norm[0] if norm else None
                    self._last_pupil_right_norm = norm[1] if norm else None
                    self._last_pupil_left_full = full_p[0] if full_p else None
                    self._last_pupil_right_full = full_p[1] if full_p else None
                    row = self._make_training_row(
                        fv=fv,
                        conf=conf,
                        stage="manual_tag",
                        screen_xy=(cx, cy),
                        screen_size=(tw, th),
                        tag_det=det_current,
                        tag_frame_size=(tw, th),
                    )
                    rows.append(row)
        finally:
            tag_cap.release()
            cv2.destroyWindow(WIN_FRONT_FULL)
            cv2.destroyWindow(WIN_FRONT_LEFT)
            cv2.destroyWindow(WIN_FRONT_RIGHT)
            cv2.destroyWindow(WIN_BACK)
        return rows

    def run(self):
        if not jeo_tracker_available():
            raise RuntimeError(
                "JEO 3DTracker not available. Clone the repo: "
                "git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker"
            )
        tracker = JEOGlassFrameTracker(
            eye_camera_id=self.config.eye_camera_id,
            eye_sensor_id=self.config.eye_sensor_id,
            rois_path=self._rois_path,
        )
        if not tracker.is_opened():
            raise RuntimeError(
                f"Could not open eye camera (camera_id={self.config.eye_camera_id}, "
                f"sensor_id={self.config.eye_sensor_id}). For CSI on Jetson set eye_sensor_id=0 or 1."
            )
        print(
            f"[glass_frame] Dual-camera tuning: overlap_tag_eye={self.config.overlap_tag_eye_detection}, "
            f"tag_buffer_drain={self.config.tag_camera_buffer_drain}, tag_quad_decimate={self.config.tag_quad_decimate}"
        )
        try:
            if self.config.calibration_mode == "automated_calibration":
                rows = self._run_automated(tracker)
            else:
                rows = self._run_manual(tracker)
            rows = self._validate_rows(rows)
            self.calib._append_training_rows(rows)
            print(f"[glass_frame] Collected valid rows: {len(rows)}")
            print(f"[glass_frame] CSV path: {self.calib.current_training_data_path}")
            trained = self._train_model_from_rows()
            if trained:
                print("[glass_frame] Calibration/training completed.")
        finally:
            tracker.release()


def main():
    cfg = GlassFrameTrainingConfig(
        calibration_mode="automated_calibration",
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        eye_camera_id=0,
        eye_sensor_id=0,  # CSI cam on Jetson (0 = single camera; 1 = second CSI); set to None for USB
        tag_camera_id=1,
        apriltag_length=0.04,
        tag_ids=(1, 2, 3, 4, 5),
        num_automated_positions=9,  # 5, 9, or 13 for more samples
        glass_quality_profile="balanced",
    )
    GlassFrameTraining(cfg).run()


if __name__ == "__main__":
    main()
