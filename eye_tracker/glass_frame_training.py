import os
import time
import cv2
import numpy as np

from .calibration import (
    Calibration,
    _get_screen_size,
    PolyRidgeRegressor,
    _fit_binocular_ridge,
    MIN_BINOC_DOTS,
    ML_POLY_DEGREE,
)
from .eye_tracker import EyeTracker
from .object_hover.apriltag_detector import AprilTagDetector


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
        tag_camera_id=1,
        apriltag_length=0.04,  # meters
        apriltag_width=None,   # meters (if None, same as length)
        apriltag_focal_px=700.0,
        tag_ids=(1, 2, 3, 4, 5),
        apriltag_families="tag36h11",
        tag_image_dir=None,
        tag_display_px=260,
        dwell_sec=1.5,
        manual_duration_sec=45.0,
        glass_quality_profile="balanced",  # fast | balanced | max_accuracy
    ):
        self.calibration_mode = calibration_mode
        self.screen_size = screen_size
        self.board_size = board_size
        self.training_data_name = training_data_name
        self.training_data_dir = training_data_dir
        self.training_mode = training_mode
        self.eye_camera_id = int(eye_camera_id)
        self.tag_camera_id = int(tag_camera_id)
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
        self.glass_quality_profile = str(glass_quality_profile).strip().lower()


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
        self.calib = Calibration(
            training_data_name=self.config.training_data_name,
            training_data_dir=self.config.training_data_dir,
            screen_size=self.config.board_size or self.config.screen_size,
            fullscreen=True,
            window_name="glass_frame_calibration",
            training_mode="glass_frame",
        )
        self.tag_detector = AprilTagDetector(families=self.config.apriltag_families)

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
        pts = [
            (self.screen_w // 2, self.screen_h // 2),
            (self.screen_w // 4, self.screen_h // 4),
            (3 * self.screen_w // 4, self.screen_h // 4),
            (self.screen_w // 4, 3 * self.screen_h // 4),
            (3 * self.screen_w // 4, 3 * self.screen_h // 4),
        ]
        if n <= len(pts):
            return pts[:n]
        # Repeat center for extra tags.
        while len(pts) < n:
            pts.append((self.screen_w // 2, self.screen_h // 2))
        return pts

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

    def _pick_detection(self, detections, desired_id=None):
        # Filter to configured IDs.
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
        }
        if tag_det is not None:
            x, y, w, h = tag_det["bbox"]
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
            row["tag_bbox_w_px"] = float(w)
            row["tag_bbox_h_px"] = float(h)
            z_m = self._estimate_distance_m(w, h)
            if z_m is not None:
                row["board_distance_m"] = float(z_m)
        return row

    def _collect_feature(self, tracker):
        gx, gy, yaw, pitch, fv, conf, is_blink = tracker.process_frame_binocular()
        if is_blink or gx is None or gy is None or fv is None:
            return None
        return fv, float(conf)

    def _init_tag_camera(self):
        cap = cv2.VideoCapture(self.config.tag_camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _train_model_from_rows(self):
        feature_vecs, screen_targets = self.calib._load_training_data()
        if len(feature_vecs) < 10:
            print(f"[glass_frame] Not enough samples to train model: {len(feature_vecs)}")
            return False
        if len(feature_vecs) >= MIN_BINOC_DOTS:
            mu, std, W = _fit_binocular_ridge(feature_vecs, screen_targets, lam=1e-1)
            self.calib.binoc_mu = mu
            self.calib.binoc_std = std
            self.calib.binoc_W = W
        self.calib.ml = PolyRidgeRegressor(lam=5e-3, degree=ML_POLY_DEGREE)
        ok = self.calib.ml.fit(feature_vecs, screen_targets)
        if not ok:
            print("[glass_frame] ML fit failed.")
            return False
        # Quick in-sample quality metric for sanity.
        pred = []
        for fv in feature_vecs:
            out = self.calib.ml.predict(fv)
            if out is None:
                continue
            pred.append(out)
        if pred:
            pred = np.asarray(pred, dtype=np.float64)
            tgt = np.asarray(screen_targets[: len(pred)], dtype=np.float64)
            rmse = float(np.sqrt(np.mean(np.sum((pred - tgt) ** 2, axis=1))))
            print(f"[glass_frame] ML train RMSE (norm): {rmse:.4f}")
        path = self.calib._ml_model_path()
        self.calib.ml.save(path, extra={
            "binoc_mu": self.calib.binoc_mu if self.calib.binoc_mu is not None else np.array([]),
            "binoc_std": self.calib.binoc_std if self.calib.binoc_std is not None else np.array([]),
            "binoc_W": self.calib.binoc_W if self.calib.binoc_W is not None else np.array([]),
        })
        print(f"[glass_frame] Model saved: {path}")
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
        positions = self._target_positions(len(self.config.tag_ids))
        rows = []
        try:
            for idx, tag_id in enumerate(self.config.tag_ids):
                target_xy = positions[idx]
                t0 = time.perf_counter()
                while (time.perf_counter() - t0) < self.config.dwell_sec:
                    fv_conf = self._collect_feature(tracker)
                    ret, tag_frame = tag_cap.read()
                    if not ret:
                        continue
                    det = self._pick_detection(self.tag_detector.detect(tag_frame), desired_id=tag_id)
                    # Show displayed target tag
                    display = self._render_tag_canvas(tag_images[tag_id], target_xy)
                    cv2.putText(
                        display,
                        f"Automated calibration: tag {tag_id} ({idx + 1}/{len(self.config.tag_ids)})",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(name, display)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return rows
                    if fv_conf is None or det is None:
                        continue
                    fv, conf = fv_conf
                    # In automated mode, screen target is known (displayed tag position),
                    # and outward-camera tag center is also logged for distance/jitter modeling.
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
        tag_cap = self._init_tag_camera()
        if not tag_cap.isOpened():
            raise RuntimeError(f"Could not open tag camera id={self.config.tag_camera_id}.")
        name = "manual_calibration"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        rows = []
        t0 = time.perf_counter()
        try:
            while (time.perf_counter() - t0) < self.config.manual_duration_sec:
                fv_conf = self._collect_feature(tracker)
                ret, tag_frame = tag_cap.read()
                if not ret:
                    continue
                det = self._pick_detection(self.tag_detector.detect(tag_frame), desired_id=None)
                preview = tag_frame.copy()
                cv2.putText(
                    preview,
                    "Manual calibration: show your tag board (Q to stop)",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                if det is not None:
                    x, y, w, h = det["bbox"]
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(preview, f"tag_id={det.get('tag_id')}", (x, max(20, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(name, preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                if fv_conf is None or det is None:
                    continue
                fv, conf = fv_conf
                # In manual mode, no known screen-ground-truth target from software.
                # Use outward-camera tag center as the supervised target.
                x, y, w, h = det["bbox"]
                cx = x + w * 0.5
                cy = y + h * 0.5
                row = self._make_training_row(
                    fv=fv,
                    conf=conf,
                    stage="manual_tag",
                    screen_xy=(cx, cy),
                    screen_size=(tag_frame.shape[1], tag_frame.shape[0]),
                    tag_det=det,
                    tag_frame_size=(tag_frame.shape[1], tag_frame.shape[0]),
                )
                rows.append(row)
        finally:
            tag_cap.release()
            cv2.destroyWindow(name)
        return rows

    def run(self):
        tracker = EyeTracker(
            front_camera_id=self.config.eye_camera_id,
            reset=True,
            training_mode="glass_frame",
            glass_quality_profile=self.config.glass_quality_profile,
        )
        if self.config.calibration_mode == "automated_calibration":
            rows = self._run_automated(tracker)
        else:
            rows = self._run_manual(tracker)
        rows = self._validate_rows(rows)
        self.calib._append_training_rows(rows)
        print(f"[glass_frame] Collected valid rows: {len(rows)}")
        print(f"[glass_frame] CSV path: {self.calib.training_data_path}")
        trained = self._train_model_from_rows()
        if trained:
            print("[glass_frame] Calibration/training completed.")


def main():
    cfg = GlassFrameTrainingConfig(
        calibration_mode="automated_calibration",
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        eye_camera_id=0,
        tag_camera_id=1,
        apriltag_length=0.04,
        tag_ids=(1, 2, 3, 4, 5),
        glass_quality_profile="balanced",
    )
    GlassFrameTraining(cfg).run()


if __name__ == "__main__":
    main()
