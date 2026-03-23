import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque

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
    CSI_FPS,
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
        # If set: fixed tag list (testing / no discovery). If None (default): count & order come from
        # the back camera during session start (unique IDs seen while you show the board).
        tag_calibration_order=None,
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
        tag_detect_scale=1.0,  # <1.0 runs AprilTag on a downscaled frame (faster, slightly less precise)
        record_every_n_frames=2,  # downsample writes to reduce IO/CPU while preserving coverage
        stability_window_frames=5,  # only record after this many recent samples are available
        stability_std_threshold=0.025,  # max std on gxL/gyL/gxR/gyR for stable recording
        aggregate_group_size=4,  # median-aggregate this many valid frames into one row
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
        self.tag_calibration_order = (
            tuple(int(x) for x in tag_calibration_order) if tag_calibration_order is not None else None
        )
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
        self.tag_detect_scale = float(tag_detect_scale)
        self.record_every_n_frames = int(record_every_n_frames)
        self.stability_window_frames = int(stability_window_frames)
        self.stability_std_threshold = float(stability_std_threshold)
        self.aggregate_group_size = int(aggregate_group_size)


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
        # On-screen feedback after S (manual calibration)
        self._checkpoint_banner_until = 0.0
        self._checkpoint_banner_line1 = ""
        self._checkpoint_banner_line2 = ""
        self._checkpoint_banner_ok = True

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
            scale = float(getattr(self.config, "tag_detect_scale", 1.0))
            if 0 < scale < 1.0:
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                dets = self.tag_detector.detect(small)
                inv = 1.0 / scale
                out = []
                for d in dets:
                    x, y, w, h = d["bbox"]
                    cx, cy = d.get("center", (x + 0.5 * w, y + 0.5 * h))
                    nd = {
                        **d,
                        "bbox": (int(round(x * inv)), int(round(y * inv)), int(round(w * inv)), int(round(h * inv))),
                        "center": (float(cx) * inv, float(cy) * inv),
                    }
                    corners = d.get("corners")
                    if corners:
                        nd["corners"] = [(float(c[0]) * inv, float(c[1]) * inv) for c in corners]
                    out.append(nd)
                return out
            return self.tag_detector.detect(frame)
        except Exception:
            return []

    def _is_stable(self, recent_fv4):
        if len(recent_fv4) < max(2, self.config.stability_window_frames):
            return False
        arr = np.asarray(recent_fv4, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4:
            return False
        std = np.std(arr, axis=0)
        return bool(np.max(std) <= float(self.config.stability_std_threshold))

    @staticmethod
    def _median_pair(values):
        vals = [v for v in values if v is not None and len(v) >= 2]
        if not vals:
            return None
        a = np.asarray(vals, dtype=np.float64)
        return float(np.median(a[:, 0])), float(np.median(a[:, 1]))

    def _aggregate_candidates_to_row(self, candidates, stage, screen_xy, screen_size, tag_frame_size):
        """Aggregate several nearby samples into one robust row (median of fv/conf/pupil/tag geometry)."""
        if not candidates:
            return None
        fv = np.median(np.asarray([c["fv"] for c in candidates], dtype=np.float64), axis=0)
        conf = float(np.median(np.asarray([c["conf"] for c in candidates], dtype=np.float64)))

        # Aggregate pupil points to damp micro-jitter in the saved rows.
        self._last_pupil_left_norm = self._median_pair([c.get("pupil_left_norm") for c in candidates])
        self._last_pupil_right_norm = self._median_pair([c.get("pupil_right_norm") for c in candidates])
        self._last_pupil_left_full = self._median_pair([c.get("pupil_left_full") for c in candidates])
        self._last_pupil_right_full = self._median_pair([c.get("pupil_right_full") for c in candidates])

        dets = [c.get("det") for c in candidates if c.get("det") is not None]
        det_agg = None
        if dets:
            b = np.asarray([d["bbox"] for d in dets], dtype=np.float64)
            x, y, w, h = np.median(b, axis=0).tolist()
            centers = []
            for d in dets:
                center = d.get("center")
                if center is not None and len(center) >= 2:
                    centers.append((float(center[0]), float(center[1])))
                else:
                    bx, by, bw, bh = d["bbox"]
                    centers.append((bx + 0.5 * bw, by + 0.5 * bh))
            cx, cy = self._median_pair(centers) if centers else (x + 0.5 * w, y + 0.5 * h)
            ids = [d.get("tag_id") for d in dets if d.get("tag_id") is not None]
            tag_id = Counter(ids).most_common(1)[0][0] if ids else None
            det_agg = {"bbox": (int(x), int(y), int(w), int(h)), "center": (float(cx), float(cy)), "tag_id": tag_id}
            corners_all = [d.get("corners") for d in dets if d.get("corners") is not None and len(d.get("corners")) >= 4]
            if corners_all:
                med_corners = []
                for i in range(4):
                    pts = [(float(c[i][0]), float(c[i][1])) for c in corners_all]
                    med_corners.append(self._median_pair(pts))
                det_agg["corners"] = med_corners

        return self._make_training_row(
            fv=fv,
            conf=conf,
            stage=stage,
            screen_xy=screen_xy,
            screen_size=screen_size,
            tag_det=det_agg,
            tag_frame_size=tag_frame_size,
        )

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

    def _ordered_tag_id_list(self):
        """Current calibration tag order (from camera discovery or config override)."""
        ids = getattr(self, "_ordered_tag_ids", None)
        return list(ids) if ids else list(self.config.tag_ids)

    def _load_tag_images(self):
        out = {}
        for tag_id in self._ordered_tag_id_list():
            name = f"tag36_11_{tag_id:05d}.png"
            path = os.path.join(self.config.tag_image_dir, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out[tag_id] = img
        return out

    def _resolve_ordered_tag_ids_from_config_only(self):
        """Fixed list from config only (no camera). Used when tag_calibration_order is set."""
        explicit = getattr(self.config, "tag_calibration_order", None)
        if explicit is None:
            return None
        if len(explicit) == 0:
            raise ValueError("calibration.tag_ids must be non-empty when used as override.")
        return list(explicit)

    def _discover_tag_ids_from_camera(self, tag_cap, tracker):
        """Build sorted tag ID list from unique AprilTags the back camera has seen (user shows board).

        Caller must create all four calibration windows first. Updates front-camera views every frame
        so manual calibration always shows the same 4 windows as the main loop.
        """
        seen = set()
        drain = self.config.tag_camera_buffer_drain
        while True:
            # Must run JEO pipeline before drawing — get_last_full_frame() is only set inside
            # process_frame_binocular() (same as the main calibration loop via _collect_feature).
            if tracker is not None:
                tracker.process_frame_binocular()
                self._draw_front_camera_windows(tracker)
            ret, frame = self._grab_latest_tag_frame(tag_cap, drain)
            if not ret or frame is None:
                preview = np.zeros((480, 640, 3), dtype=np.uint8)
                dets = []
            else:
                preview = frame.copy()
                dets = self._detect_tag_frame(frame)
                for d in dets:
                    tid = d.get("tag_id")
                    if tid is not None:
                        seen.add(int(tid))
                for d in dets:
                    x, y, w, h = d["bbox"]
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    t = d.get("tag_id", "?")
                    cv2.putText(preview, str(t), (x, max(20, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            th, tw = preview.shape[:2]
            ids_sorted = sorted(seen)
            id_str = str(ids_sorted) if ids_sorted else "(none yet)"
            if len(id_str) > 90:
                id_str = id_str[:87] + "..."
            lines = [
                "TAG DISCOVERY (back camera)",
                "Show the calibration board; pan so each tag is visible at least once.",
                f"Unique IDs seen: {len(seen)}  {id_str}",
                "SPACE = done   R = clear list   Q = quit",
            ]
            y = 24
            for line in lines:
                cv2.putText(preview, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
                y += 24
            cv2.imshow(WIN_BACK, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise RuntimeError("Calibration cancelled during tag discovery (Q).")
            if key == ord("r"):
                seen.clear()
            if key == ord(" "):
                if len(seen) < 1:
                    print("[glass_frame] No tags detected yet — show the board to the back camera, or press R and retry.")
                    continue
                order = sorted(seen)
                print(f"[glass_frame] Discovery done: {len(order)} tag(s) {order}")
                return order

    def _resolve_tag_order_after_camera_open(self, tag_cap, tracker):
        """Return ordered tag ID list: config override, else live discovery."""
        fixed = self._resolve_ordered_tag_ids_from_config_only()
        if fixed is not None:
            self._ordered_tag_ids = tuple(fixed)
            self.config.tag_ids = list(fixed)
            return fixed
        order = self._discover_tag_ids_from_camera(tag_cap, tracker)
        self._ordered_tag_ids = tuple(order)
        self.config.tag_ids = list(order)
        return order

    def _manual_flush_new_rows(self, rows_all, already_written: int) -> tuple[int, int]:
        """Validate and append rows_all[already_written:] to CSV.

        Returns:
            (new_already_written, num_appended)
        Only advances ``new_already_written`` past pending rows when at least one row was
        written — otherwise the same pending slice is retried on the next S (fixes
        \"must press S many times\" when validation first dropped everything).
        """
        if already_written >= len(rows_all):
            return already_written, 0
        pending = rows_all[already_written:]
        chunk = self._validate_rows(pending)
        if not chunk:
            print(
                "[glass_frame] Checkpoint (S): no valid rows in pending buffer yet — "
                "nothing written; try again after more recording or check gaze/tag data."
            )
            return already_written, 0
        self.calib._append_training_rows(chunk)
        n_pending = len(pending)
        n_ok = len(chunk)
        if n_ok < n_pending:
            print(f"[glass_frame] Checkpoint: wrote {n_ok} of {n_pending} pending row(s); rest failed validation.")
        print(
            f"[glass_frame] Saved +{n_ok} row(s). Session rows: {len(rows_all)}. "
            f"CSV: {self.calib.current_training_data_path}"
        )
        return already_written + n_pending, n_ok

    def _set_checkpoint_banner(
        self, *, ok: bool, line1: str, line2: str = "", duration_sec: float = 4.5
    ) -> None:
        """Large on-screen feedback on the back-camera preview after S (or auto-flush)."""
        self._checkpoint_banner_until = time.monotonic() + float(duration_sec)
        self._checkpoint_banner_line1 = line1
        self._checkpoint_banner_line2 = line2
        self._checkpoint_banner_ok = ok

    def _draw_manual_back_camera_ui(
        self,
        preview,
        tag_images,
        all_dets,
        *,
        recording,
        session_complete,
        just_paused,
        tag_idx,
        tag_order,
        checkpoint_saved=False,
        unsaved_pending=False,
    ):
        """Back camera: RED = next tag to look at (paused). LIME = tag to record (running) or just recorded (paused)."""
        th, tw = preview.shape[:2]
        current_id = tag_order[tag_idx] if tag_idx < len(tag_order) else None

        def _q_hint():
            if session_complete:
                return None
            if checkpoint_saved and not unsaved_pending:
                return "Q = quit (data saved to CSV)"
            if checkpoint_saved and unsaved_pending:
                return "Q = quit (checkpoint saved; press S again to save newest rows)"
            return "Q = quit without saving — press S first to save"

        q_line = _q_hint()

        # red_next_id: upcoming tag (only when paused — what to prepare for)
        # lime_id: tag user should fixate while recording, OR tag just finished when paused
        red_next_id = None
        lime_id = None

        if session_complete:
            lines = [
                "CALIBRATION COMPLETE",
                "All tags recorded; data flushed to CSV.",
                "Q = exit and run training   S = re-save checkpoint if needed",
            ]
            if tag_order:
                lime_id = tag_order[-1]
        elif recording:
            lines = [
                f"RECORDING — look at AprilTag ID {current_id} (lime)",
                f"Progress: tag {tag_idx + 1} of {len(tag_order)}",
                "SPACE = pause",
            ]
            if q_line:
                lines.append(q_line)
            lime_id = current_id
        elif just_paused:
            lime_id = tag_order[tag_idx] if tag_idx < len(tag_order) else None
            if tag_idx + 1 < len(tag_order):
                red_next_id = tag_order[tag_idx + 1]
                nid = red_next_id
                lines = [
                    "PAUSED",
                    f"LIME = tag you just recorded  |  RED = next: AprilTag ID {nid}",
                    f"Press SPACE to record tag {tag_idx + 2}/{len(tag_order)}",
                    "S = save progress to CSV",
                ]
            else:
                lines = [
                    "PAUSED — last tag recorded",
                    "LIME = tag you just finished  |  Press SPACE to end session",
                ]
            if q_line:
                lines.append(q_line)
        else:
            red_next_id = tag_order[tag_idx] if tag_idx < len(tag_order) else None
            nid = red_next_id
            lines = [
                "PAUSED — get ready",
                f"RED = look at AprilTag ID {nid} next, then SPACE to start recording",
                "S = save progress to CSV",
            ]
            if q_line:
                lines.append(q_line)

        y = 32
        for line in lines:
            cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            y += 28

        # Big temporary banner after pressing S (success or failure)
        now = time.monotonic()
        if now < getattr(self, "_checkpoint_banner_until", 0):
            b1 = getattr(self, "_checkpoint_banner_line1", "") or ""
            b2 = getattr(self, "_checkpoint_banner_line2", "") or ""
            ok = getattr(self, "_checkpoint_banner_ok", True)
            bar_top = min(y + 4, th - 130)
            bar_h = 56 if b2 else 44
            bg = (50, 170, 50) if ok else (50, 70, 230)
            cv2.rectangle(preview, (0, bar_top), (tw, bar_top + bar_h), bg, -1)
            cv2.rectangle(preview, (0, bar_top), (tw, bar_top + bar_h), (255, 255, 255), 2)
            cv2.putText(
                preview, b1, (12, bar_top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2,
            )
            if b2:
                cv2.putText(
                    preview, b2, (12, bar_top + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 255, 240), 1,
                )

        # Always-visible save state (stack above footer; avoid clash with tag-not-visible at th-68)
        ps_y = th - 48
        if session_complete:
            if unsaved_pending:
                ps, pc = "SAVE STATUS: Rows not on disk — press S, then Q.", (0, 165, 255)
            elif checkpoint_saved:
                ps, pc = "SAVE STATUS: All rows saved to CSV — safe to press Q.", (80, 255, 100)
            else:
                ps, pc = "SAVE STATUS: No CSV write yet — press S or Q.", (100, 200, 255)
            cv2.putText(preview, ps, (12, ps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, pc, 2)
        else:
            if checkpoint_saved and not unsaved_pending:
                ps, pc = "SAVE STATUS: All data in memory is saved to CSV.", (80, 255, 100)
            elif checkpoint_saved and unsaved_pending:
                ps, pc = "SAVE STATUS: New rows not saved — press S again before Q.", (0, 255, 255)
            else:
                ps, pc = "SAVE STATUS: Nothing on disk yet — press S while paused.", (100, 200, 255)
            cv2.putText(preview, ps, (12, ps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, pc, 2)

        footer = "SPACE = start / pause / advance tag     S = save checkpoint"
        if session_complete:
            footer += "     Q = quit"
        elif checkpoint_saved and not unsaved_pending:
            footer += "     Q = quit (saved)"
        elif checkpoint_saved:
            footer += "     Q = quit (S again if more rows)"
        else:
            footer += "     Q = quit (use S to save first)"
        cv2.putText(
            preview,
            footer,
            (12, th - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        thumb = 130
        margin = 8
        gap = 6
        color_next_red = (0, 0, 255)
        color_lime = (50, 255, 80)
        x0 = tw - thumb - margin
        y_thumb = margin

        def _paste_thumb(tag_id, y_top, border_color, caption):
            if tag_id is None or tag_id not in tag_images:
                return y_top
            tg = cv2.resize(tag_images[tag_id], (thumb, thumb), interpolation=cv2.INTER_NEAREST)
            tgc = cv2.cvtColor(tg, cv2.COLOR_GRAY2BGR)
            preview[y_top : y_top + thumb, x0 : x0 + thumb] = tgc
            cv2.rectangle(preview, (x0, y_top), (x0 + thumb, y_top + thumb), border_color, 3)
            cv2.putText(
                preview,
                caption,
                (x0, min(th - 6, y_top + thumb + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                border_color,
                2,
            )
            return y_top + thumb + gap

        # Thumbnails: while recording = lime only; paused = red (next) above lime (just recorded)
        if recording:
            if lime_id is not None:
                _paste_thumb(lime_id, margin, color_lime, f"LIME look {lime_id}")
        else:
            if red_next_id is not None:
                y_thumb = _paste_thumb(red_next_id, y_thumb, color_next_red, f"RED next {red_next_id}")
            if lime_id is not None and (just_paused or session_complete):
                cap = f"LIME last {lime_id}" if session_complete else f"LIME done {lime_id}"
                y_thumb = _paste_thumb(lime_id, y_thumb, color_lime, cap)

        def _draw_one_tag_box(tid, color, label):
            if tid is None or not all_dets:
                return False
            for d in all_dets:
                if d.get("tag_id") != tid:
                    continue
                x, y, w, h = d["bbox"]
                cv2.rectangle(preview, (x, y), (x + w, y + h), color, 3)
                cv2.putText(
                    preview,
                    label,
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
                return True
            return False

        if recording:
            _draw_one_tag_box(lime_id, color_lime, f"LOOK (lime) id {lime_id}")
        elif session_complete:
            _draw_one_tag_box(lime_id, color_lime, f"LAST (lime) id {lime_id}")
        else:
            if lime_id is not None:
                _draw_one_tag_box(lime_id, color_lime, f"JUST DONE (lime) id {lime_id}")
            if red_next_id is not None:
                _draw_one_tag_box(red_next_id, color_next_red, f"NEXT (red) id {red_next_id}")

        if not recording and red_next_id is not None and all_dets:
            visible_next = any(d.get("tag_id") == red_next_id for d in all_dets)
            if not visible_next:
                cv2.putText(
                    preview,
                    f"RED next tag {red_next_id} not visible — show it to this camera",
                    (12, th - 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 140, 255),
                    2,
                )

        if all_dets:
            skip = {tid for tid in (lime_id, red_next_id) if tid is not None}
            if recording and current_id is not None:
                skip.add(current_id)
            for d in all_dets:
                tid = d.get("tag_id")
                x, y, w, h = d["bbox"]
                muted = (80, 80, 80)
                if tid in skip:
                    continue
                cv2.rectangle(preview, (x, y), (x + w, y + h), muted, 1)
                cv2.putText(preview, f"id {tid}", (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, muted, 1)

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
            pipeline = _build_csi_pipeline(csi_sensor, width=1280, height=720, fps=CSI_FPS)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.config.tag_camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, CSI_FPS)
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
        tag_cap = self._init_tag_camera()
        if not tag_cap.isOpened():
            raise RuntimeError(f"Could not open tag camera id={self.config.tag_camera_id}.")
        # Automated calibration uses one fullscreen target window later; discovery is back-cam only.
        tag_order = self._resolve_tag_order_after_camera_open(tag_cap, None)
        if not tag_order:
            raise ValueError("No tags in calibration list after discovery.")
        tag_images = self._load_tag_images()
        missing = [tid for tid in tag_order if tid not in tag_images]
        if missing:
            raise FileNotFoundError(
                f"Automated mode needs a PNG per tag in {self.config.tag_image_dir}. "
                f"Missing tag36_11_*.png for IDs {missing}."
            )
        name = "automated_calibration"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # One calibration round per AprilTag present (same count as tag PNGs in tag_image_dir).
        num_positions = len(tag_order)
        positions = self._target_positions(num_positions)
        rows = []
        warmup_sec = self.config.warmup_sec
        min_conf = self.config.min_recording_conf
        drain = self.config.tag_camera_buffer_drain
        overlap = self.config.overlap_tag_eye_detection
        record_every = max(1, int(self.config.record_every_n_frames))
        group_size = max(1, int(self.config.aggregate_group_size))
        stable_n = max(2, int(self.config.stability_window_frames))
        try:
            with ThreadPoolExecutor(max_workers=1) as tag_pool:
                for idx in range(num_positions):
                    target_xy = positions[idx]
                    tag_id = tag_order[idx]
                    t0 = time.perf_counter()
                    recording_started = t0 + warmup_sec
                    frame_idx = 0
                    recent_fv4 = deque(maxlen=stable_n)
                    group = []
                    while (time.perf_counter() - t0) < self.config.dwell_sec:
                        now = time.perf_counter()
                        frame_idx += 1
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
                        if frame_idx % record_every != 0:
                            continue
                        recent_fv4.append(np.asarray(fv[:4], dtype=np.float64))
                        if not self._is_stable(recent_fv4):
                            continue
                        norm = tracker.get_last_left_right_pupil_normalized_crop() if hasattr(tracker, "get_last_left_right_pupil_normalized_crop") else (None, None)
                        full = tracker.get_last_left_right_pupil_full() if hasattr(tracker, "get_last_left_right_pupil_full") else (None, None)
                        group.append({
                            "fv": np.asarray(fv, dtype=np.float64),
                            "conf": float(conf),
                            "det": det,
                            "pupil_left_norm": norm[0] if norm else None,
                            "pupil_right_norm": norm[1] if norm else None,
                            "pupil_left_full": full[0] if full else None,
                            "pupil_right_full": full[1] if full else None,
                        })
                        if len(group) >= group_size:
                            row = self._aggregate_candidates_to_row(
                                group,
                                stage="auto_tag",
                                screen_xy=target_xy,
                                screen_size=(self.screen_w, self.screen_h),
                                tag_frame_size=(tag_frame.shape[1], tag_frame.shape[0]),
                            )
                            if row is not None:
                                rows.append(row)
                            group.clear()
                    if group:
                        row = self._aggregate_candidates_to_row(
                            group,
                            stage="auto_tag",
                            screen_xy=target_xy,
                            screen_size=(self.screen_w, self.screen_h),
                            tag_frame_size=(tag_frame.shape[1], tag_frame.shape[0]),
                        )
                        if row is not None:
                            rows.append(row)
        finally:
            tag_cap.release()
            cv2.destroyWindow(name)
        return rows

    def _run_manual(self, tracker):
        """Manual calibration: front camera = 3 windows; back camera shows next-tag progress.

        Unless ``tag_calibration_order`` is set in config, the **number of rounds** is the count of **unique
        AprilTag IDs the back camera has seen** during the discovery step (you show the board, then SPACE).

        Keys (SPACE / S only count when back-camera window is focused):
        - SPACE: start recording current tag → pause → advance to next tag and record (same triple-press as before).
        - S (while paused or after all tags done): checkpoint — append new rows to CSV so a later Q does not lose them.
        - Q: quit. Rows not yet checkpointed with S are discarded (unless you finished all tags, which auto-saves).

        When paused, the back view shows the **next** AprilTag to record (thumbnail + highlight if visible).
        """
        tag_cap = self._init_tag_camera()
        if not tag_cap.isOpened():
            raise RuntimeError(
                "Could not open back (tag) camera. Check tag_camera_id / tag_camera_csi_sensor_id."
            )
        # All four OpenCV windows must exist for the whole session (including tag discovery), same as before.
        cv2.namedWindow(WIN_FRONT_FULL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_FRONT_LEFT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_FRONT_RIGHT, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_BACK, cv2.WINDOW_NORMAL)

        tag_order = self._resolve_tag_order_after_camera_open(tag_cap, tracker)
        if not tag_order:
            tag_cap.release()
            try:
                cv2.destroyWindow(WIN_FRONT_FULL)
                cv2.destroyWindow(WIN_FRONT_LEFT)
                cv2.destroyWindow(WIN_FRONT_RIGHT)
                cv2.destroyWindow(WIN_BACK)
            except Exception:
                pass
            raise ValueError("No tags in calibration list after discovery.")

        tag_images = self._load_tag_images()
        missing = [t for t in tag_order if t not in tag_images]
        if missing:
            print(
                f"[glass_frame] No tag36_11_*.png in {self.config.tag_image_dir} for IDs {missing} — "
                "on-screen thumbnails skipped for those IDs."
            )

        print(
            f"[glass_frame] Manual calibration: {len(tag_order)} tag(s) in order {tag_order}. "
            "Paused view shows the next tag to look at. S while paused saves CSV checkpoint; Q without S drops unsaved rows."
        )

        rows = []
        already_disk = 0
        committed_session = False  # True if any CSV write this run, or session finished all tags
        checkpoint_saved = False  # True after at least one successful S checkpoint (updates Q hint)
        tag_idx = 0
        recording = False
        just_paused = False
        session_complete = False
        cx, cy = 0.0, 0.0
        current_tag_id = tag_order[0]

        min_conf = self.config.min_recording_conf
        drain = self.config.tag_camera_buffer_drain
        overlap = self.config.overlap_tag_eye_detection
        record_every = max(1, int(self.config.record_every_n_frames))
        group_size = max(1, int(self.config.aggregate_group_size))
        stable_n = max(2, int(self.config.stability_window_frames))
        frame_idx = 0
        recent_fv4 = deque(maxlen=stable_n)
        group = []

        def _flush_group_to_rows():
            nonlocal group
            if not group:
                return
            row = self._aggregate_candidates_to_row(
                group,
                stage="manual_tag",
                screen_xy=(cx, cy) if det_current is not None else (0.0, 0.0),
                screen_size=(tw, th),
                tag_frame_size=(tw, th),
            )
            if row is not None:
                rows.append(row)
            group.clear()

        try:
            with ThreadPoolExecutor(max_workers=1) as tag_pool:
                while True:
                    frame_idx += 1
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

                    if not session_complete and recording:
                        current_tag_id = tag_order[tag_idx]
                    det_current = (
                        next((d for d in all_dets if d.get("tag_id") == current_tag_id), None)
                        if (recording and not session_complete)
                        else None
                    )

                    self._draw_front_camera_windows(tracker)
                    preview = tag_frame.copy() if tag_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    tw, th = preview.shape[1], preview.shape[0]

                    self._draw_manual_back_camera_ui(
                        preview,
                        tag_images,
                        all_dets,
                        recording=recording and not session_complete,
                        session_complete=session_complete,
                        just_paused=just_paused,
                        tag_idx=tag_idx,
                        tag_order=tag_order,
                        checkpoint_saved=checkpoint_saved,
                        unsaved_pending=(already_disk < len(rows)),
                    )
                    if recording and not session_complete and det_current is None:
                        cv2.putText(
                            preview,
                            f"Tag {current_tag_id} not visible — show it to the back camera",
                            (12, th - 68),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 140, 255),
                            2,
                        )

                    cv2.imshow(WIN_BACK, preview)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        _flush_group_to_rows()
                        break

                    if key == ord("s") and (not recording or session_complete):
                        already_disk, n_appended = self._manual_flush_new_rows(rows, already_disk)
                        csv_bn = os.path.basename(self.calib.current_training_data_path or "") or "training.csv"
                        if n_appended > 0:
                            committed_session = True
                            checkpoint_saved = True
                            self._set_checkpoint_banner(
                                ok=True,
                                line1=f"SAVED {n_appended} row(s) to disk",
                                line2=csv_bn,
                            )
                        else:
                            self._set_checkpoint_banner(
                                ok=False,
                                line1="NOTHING SAVED (this press)",
                                line2="No new valid rows — buffer empty or already checkpointed.",
                            )
                        continue

                    if key == ord(" ") and session_complete:
                        continue

                    if key == ord(" "):
                        if recording:
                            recording = False
                            just_paused = True
                            _flush_group_to_rows()
                        else:
                            if just_paused:
                                tag_idx += 1
                                if tag_idx >= len(tag_order):
                                    session_complete = True
                                    recording = False
                                    just_paused = False
                                    _flush_group_to_rows()
                                    already_disk, n_appended = self._manual_flush_new_rows(rows, already_disk)
                                    committed_session = True
                                    csv_bn = os.path.basename(
                                        self.calib.current_training_data_path or ""
                                    ) or "training.csv"
                                    if n_appended > 0:
                                        checkpoint_saved = True
                                        self._set_checkpoint_banner(
                                            ok=True,
                                            line1=f"SESSION COMPLETE — saved {n_appended} row(s)",
                                            line2=csv_bn,
                                        )
                                    else:
                                        self._set_checkpoint_banner(
                                            ok=False,
                                            line1="SESSION COMPLETE — CSV not updated",
                                            line2="No valid rows to write; check recording quality.",
                                        )
                                    if n_appended > 0:
                                        print(
                                            "[glass_frame] Recorded all tags; "
                                            f"wrote {n_appended} row(s) to CSV. Press Q to exit."
                                        )
                                    else:
                                        print(
                                            "[glass_frame] Recorded all tags; no valid rows written to CSV. "
                                            "Press S while paused to retry, or Q to exit."
                                        )
                                else:
                                    recording = True
                                    just_paused = False
                            else:
                                recording = True

                    if session_complete or not recording or fv_conf is None or det_current is None:
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
                    if frame_idx % record_every != 0:
                        continue
                    recent_fv4.append(np.asarray(fv[:4], dtype=np.float64))
                    if not self._is_stable(recent_fv4):
                        continue
                    norm = (
                        tracker.get_last_left_right_pupil_normalized_crop()
                        if hasattr(tracker, "get_last_left_right_pupil_normalized_crop")
                        else (None, None)
                    )
                    full_p = (
                        tracker.get_last_left_right_pupil_full()
                        if hasattr(tracker, "get_last_left_right_pupil_full")
                        else (None, None)
                    )
                    group.append({
                        "fv": np.asarray(fv, dtype=np.float64),
                        "conf": float(conf),
                        "det": det_current,
                        "pupil_left_norm": norm[0] if norm else None,
                        "pupil_right_norm": norm[1] if norm else None,
                        "pupil_left_full": full_p[0] if full_p else None,
                        "pupil_right_full": full_p[1] if full_p else None,
                    })
                    if len(group) >= group_size:
                        row = self._aggregate_candidates_to_row(
                            group,
                            stage="manual_tag",
                            screen_xy=(cx, cy),
                            screen_size=(tw, th),
                            tag_frame_size=(tw, th),
                        )
                        if row is not None:
                            rows.append(row)
                        group.clear()
        finally:
            tag_cap.release()
            cv2.destroyWindow(WIN_FRONT_FULL)
            cv2.destroyWindow(WIN_FRONT_LEFT)
            cv2.destroyWindow(WIN_FRONT_RIGHT)
            cv2.destroyWindow(WIN_BACK)

        return committed_session

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
            # Jetson dual-CSI mapping can vary by board/cable order. Try the alternate sensor once.
            alt_sensor = None
            if self.config.eye_sensor_id in (0, 1):
                alt_sensor = 1 - int(self.config.eye_sensor_id)
            if alt_sensor is not None:
                print(
                    f"[glass_frame] Eye camera failed on sensor {self.config.eye_sensor_id}; "
                    f"retrying with sensor {alt_sensor}..."
                )
                try:
                    tracker.release()
                except Exception:
                    pass
                tracker = JEOGlassFrameTracker(
                    eye_camera_id=self.config.eye_camera_id,
                    eye_sensor_id=alt_sensor,
                    rois_path=self._rois_path,
                )
                if tracker.is_opened():
                    self.config.eye_sensor_id = alt_sensor
                    print(f"[glass_frame] Eye camera opened on fallback sensor {alt_sensor}.")
            if not tracker.is_opened():
                raise RuntimeError(
                    f"Could not open eye camera (camera_id={self.config.eye_camera_id}, "
                    f"sensor_id={self.config.eye_sensor_id}). For CSI on Jetson set eye_sensor_id=0 or 1."
                )
        print(
            f"[glass_frame] Dual-camera tuning: overlap_tag_eye={self.config.overlap_tag_eye_detection}, "
            f"tag_buffer_drain={self.config.tag_camera_buffer_drain}, tag_quad_decimate={self.config.tag_quad_decimate}"
        )
        fixed = self._resolve_ordered_tag_ids_from_config_only()
        if fixed is not None:
            self._ordered_tag_ids = tuple(fixed)
            self.config.tag_ids = list(fixed)
            print(
                f"[glass_frame] Tag order from config override: {len(self._ordered_tag_ids)} tag(s) "
                f"{list(self._ordered_tag_ids)}"
            )
        else:
            self._ordered_tag_ids = None
            print(
                "[glass_frame] Tag list: from back camera at session start (show board; SPACE when done). "
                "Override: set calibration.tag_ids in JSON if you cannot run discovery."
            )
        # Hide vendor debug windows (ellipse/rays) and keep only calibration UI windows.
        allowed_windows = {WIN_FRONT_FULL, WIN_FRONT_LEFT, WIN_FRONT_RIGHT, WIN_BACK, "automated_calibration"}
        _imshow_orig = cv2.imshow
        def _imshow_filter(name, img):
            if name in allowed_windows:
                _imshow_orig(name, img)
        cv2.imshow = _imshow_filter
        try:
            if self.config.calibration_mode == "manual_calibration":
                committed = self._run_manual(tracker)
                if not committed:
                    print(
                        "[glass_frame] Quit without saving this session's new rows. "
                        "While paused, press S to checkpoint to CSV, or finish all tags (auto-saves at end)."
                    )
                    return
                print(f"[glass_frame] CSV path: {self.calib.current_training_data_path}")
                trained = self._train_model_from_rows()
            else:
                rows = self._run_automated(tracker)
                rows = self._validate_rows(rows)
                self.calib._append_training_rows(rows)
                print(f"[glass_frame] Collected valid rows: {len(rows)}")
                print(f"[glass_frame] CSV path: {self.calib.current_training_data_path}")
                trained = self._train_model_from_rows()
            if trained:
                print("[glass_frame] Calibration/training completed.")
        finally:
            cv2.imshow = _imshow_orig
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
