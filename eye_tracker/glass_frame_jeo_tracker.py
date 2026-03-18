"""
Glass-frame eye tracker that uses the JEOresearch/EyeTracker 3DTracker (vendored).

All ML data for the camera mounted on the glass frame comes from this repo's algorithm.
We do not use the old in-tree glass-frame implementation; this module is the single
source for glass-frame tracking when using the 3D tracker.

See CREDITS.md and vendor/EyeTracker — we do not claim authorship of the tracking algorithm.
"""

from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

# Add vendored JEO 3DTracker so we use their API directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_JEO_3D_PATH = os.path.join(_PROJECT_ROOT, "vendor", "EyeTracker", "3DTracker")
if os.path.isdir(_JEO_3D_PATH):
    if _JEO_3D_PATH not in sys.path:
        sys.path.insert(0, _JEO_3D_PATH)

try:
    from Orlosky3DEyeTracker import process_frame as _jeo_process_frame
except ImportError:
    _jeo_process_frame = None

# 16-D feature vector order (must match Calibration.FEATURE_COLUMNS)
FEATURE_DIM = 16
GazeVectorPathDefault = "gaze_vector.txt"
CSI_W, CSI_H, CSI_FPS = 1280, 720, 30
# ROI crop is resized to this before passing to JEO (vendor expects ~640x480)
JEO_CROP_W, JEO_CROP_H = 640, 480
GLASS_FRAME_ROIS_FILENAME = "glass_frame_rois.json"


def _build_csi_pipeline(sensor_id: int, width: int = CSI_W, height: int = CSI_H, fps: int = CSI_FPS) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx, width=(int){width}, height=(int){height} ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def _read_gaze_vector(path: str):
    """Read last line of gaze_vector.txt.
    Format: origin(3), direction(3), [pupil_x, pupil_y], [pupil_major_px, pupil_minor_px].
    Returns (origin_3d, direction_3d, pupil_xy, pupil_radius_px).
    pupil_xy = (x,y) in frame; pupil_radius_px = (major, minor) ellipse axes in px for distance estimation, or None."""
    if not path or not os.path.isfile(path):
        return None, None, None, None
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if not lines:
            return None, None, None, None
        last = lines[-1].strip()
        parts = [float(x) for x in last.split(",")]
        if len(parts) >= 6:
            origin = np.array(parts[:3], dtype=np.float64)
            direction = np.array(parts[3:6], dtype=np.float64)
            pupil_xy = (float(parts[6]), float(parts[7])) if len(parts) >= 8 else None
            pupil_radius_px = (float(parts[8]), float(parts[9])) if len(parts) >= 10 else None  # (major, minor) in px
            return origin, direction, pupil_xy, pupil_radius_px
    except Exception:
        pass
    return None, None, None, None


def _load_rois(rois_path: str | None = None):
    """Load left/right eye ROIs from glass_frame_rois.json.
    If rois_path is given and a valid file, load from it. Otherwise search project root, cwd, script dir, Desktop, home.
    Returns dict with 'left_eye_roi' and/or 'right_eye_roi' (normalized [x,y,w,h]) or None if not found."""
    import json
    if rois_path and os.path.isfile(rois_path):
        try:
            with open(rois_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and ("left_eye_roi" in data or "right_eye_roi" in data):
                return data
        except Exception:
            pass
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else None
    search_bases = [
        _PROJECT_ROOT,
        os.getcwd(),
        script_dir,
        os.path.dirname(script_dir) if script_dir else None,
        os.path.expanduser("~"),
    ]
    seen = set()
    for base in search_bases:
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
                    return data
            except Exception:
                pass
    return None


def get_roi_signature(rois_dict: dict | None) -> str | None:
    """Stable 8-char hex signature from ROI coords. None if no valid ROIs. Used for ROI-specific CSV/npz paths."""
    if not rois_dict:
        return None
    import hashlib
    parts = []
    for key in ("left_eye_roi", "right_eye_roi"):
        v = rois_dict.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            parts.append(f"{key}={v}")
    if not parts:
        return None
    h = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return h[:8]


def load_rois_and_signature(rois_path: str | None = None) -> tuple[dict | None, str | None, str | None]:
    """Load ROIs and compute signature. Returns (rois_dict, path_where_found, roi_signature)."""
    if rois_path and os.path.isfile(rois_path):
        try:
            with open(rois_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and ("left_eye_roi" in data or "right_eye_roi" in data):
                sig = get_roi_signature(data)
                return data, rois_path, sig
        except Exception:
            pass
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv else None
    search_bases = [
        _PROJECT_ROOT,
        os.getcwd(),
        script_dir,
        os.path.dirname(script_dir) if script_dir else None,
        os.path.expanduser("~"),
    ]
    seen = set()
    for base in search_bases:
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
                    sig = get_roi_signature(data)
                    return data, path, sig
            except Exception:
                pass
    return None, None, None


def _pupil_crop_to_main_frame(
    pupil_xy_in_resized_crop: tuple[float, float],
    crop_rect_main_frame: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Convert pupil position from crop space to main-frame (IMX219) pixel coordinates.

    Professional approach: we know the crop rectangle's position in the main frame
    (x, y, rw, rh) = left, top, width, height. Pupil position in main frame is:
      main_x = crop_left + pupil_x_in_crop
      main_y = crop_top  + pupil_y_in_crop
    The tracker returns pupil in resized crop space (JEO_CROP_W x JEO_CROP_H), so we
    first scale to actual crop pixels, then add the crop's top-left offset.
    """
    px, py = pupil_xy_in_resized_crop
    x, y, rw, rh = crop_rect_main_frame
    pupil_crop_x = px * (rw / JEO_CROP_W)
    pupil_crop_y = py * (rh / JEO_CROP_H)
    return (x + pupil_crop_x, y + pupil_crop_y)


def _roi_to_pixel_rect(roi_norm: list, frame_w: int, frame_h: int):
    """Convert normalized [x,y,w,h] to pixel (x,y,w,h) clipped to frame."""
    if not roi_norm or len(roi_norm) != 4:
        return None
    x = int(roi_norm[0] * frame_w)
    y = int(roi_norm[1] * frame_h)
    w = int(roi_norm[2] * frame_w)
    h = int(roi_norm[3] * frame_h)
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return (x, y, w, h)


def _gaze_3d_to_feature_vector(origin_3d, direction_3d, confidence: float = 0.9) -> np.ndarray:
    """Build 16-D feature vector for calibration from JEO 3D output. Single-eye: gxL=gxR, gyL=gyR from direction."""
    if origin_3d is None or direction_3d is None:
        gx = gy = 0.0
    else:
        gx = float(direction_3d[0])
        gy = float(direction_3d[1])
    return np.array([
        gx, gy, gx, gy,
        0.0, 0.0, 0.01, 0.0,
        0.02, 0.02, 1.0, 1.0,
        0.5, 0.5, 0.5, 0.5,
    ], dtype=np.float64)


def _gaze_binocular_to_feature_vector(left_direction, right_direction) -> np.ndarray:
    """Build 16-D feature vector from left and right gaze directions (3D)."""
    gxL = float(left_direction[0]) if left_direction is not None else 0.0
    gyL = float(left_direction[1]) if left_direction is not None else 0.0
    gxR = float(right_direction[0]) if right_direction is not None else 0.0
    gyR = float(right_direction[1]) if right_direction is not None else 0.0
    return np.array([
        gxL, gyL, gxR, gyR,
        0.0, 0.0, 0.01, 0.0,
        0.02, 0.02, 1.0, 1.0,
        0.5, 0.5, 0.5, 0.5,
    ], dtype=np.float64)


class JEOGlassFrameTracker:
    """
    Glass-frame tracker using the vendored JEOresearch/EyeTracker 3DTracker only.
    Exposes process_frame_binocular() so calibration and hover app can use it.
    """

    def __init__(
        self,
        eye_camera_id: int = 0,
        eye_sensor_id: int | None = None,
        gaze_vector_path: str | None = GazeVectorPathDefault,
        rois_path: str | None = None,
    ):
        if _jeo_process_frame is None:
            raise RuntimeError(
                "JEO 3DTracker not found. Clone the repo: "
                "git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker"
            )
        self._jeo_process_frame = _jeo_process_frame
        self.eye_camera_id = int(eye_camera_id)
        self.eye_sensor_id = eye_sensor_id
        self.gaze_vector_path = gaze_vector_path or GazeVectorPathDefault
        self.cap = None
        self._using_csi = False
        self._rois = _load_rois(rois_path)  # left_eye_roi, right_eye_roi (normalized) or None
        self._last_left_pupil_full = None
        self._last_right_pupil_full = None
        self._last_left_pupil_radius_px = None   # (major, minor) ellipse axes in px for distance estimation
        self._last_right_pupil_radius_px = None
        self._last_left_crop = None
        self._last_right_crop = None
        self._last_full_frame = None   # last full CSI frame for display (before any crop)
        self._last_left_gaze_2d = None
        self._last_right_gaze_2d = None
        self._last_left_pupil_in_crop = None
        self._last_right_pupil_in_crop = None
        self._last_is_blink = False  # True when no pupil detected (treat as blink for display/recording)
        self._open_camera()

    def _open_camera(self):
        if self.eye_sensor_id is not None:
            pipeline = _build_csi_pipeline(self.eye_sensor_id)
            csi_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if csi_cap.isOpened():
                for _ in range(25):
                    ret, _ = csi_cap.read()
                    if ret:
                        self.cap = csi_cap
                        self._using_csi = True
                        return
                    time.sleep(0.1)
                csi_cap.release()
            time.sleep(1.0)
            self.cap = None
            self._using_csi = False
            if self.eye_camera_id == 0:
                time.sleep(0.5)
                fallback = cv2.VideoCapture(0, getattr(cv2, "CAP_V4L2", 0))
                if not fallback.isOpened():
                    fallback = cv2.VideoCapture(0)
                if fallback.isOpened():
                    for _ in range(15):
                        if fallback.read()[0]:
                            self.cap = fallback
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CSI_W)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CSI_H)
                            self.cap.set(cv2.CAP_PROP_FPS, CSI_FPS)
                            return
                        time.sleep(0.1)
                    fallback.release()
            return
        self.cap = cv2.VideoCapture(self.eye_camera_id)
        self._using_csi = False
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CSI_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CSI_H)
            self.cap.set(cv2.CAP_PROP_FPS, CSI_FPS)

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def using_csi(self) -> bool:
        return self._using_csi

    def has_rois(self) -> bool:
        """True if glass_frame_rois.json was loaded with at least one ROI."""
        return bool(self._rois and (self._rois.get("left_eye_roi") or self._rois.get("right_eye_roi")))

    def get_roi_info(self, frame_w: int | None = None, frame_h: int | None = None) -> dict | None:
        """Expose ROI data for fallback/reuse. Returns dict with:
        - left_eye_roi, right_eye_roi: normalized [x, y, w, h] (0..1)
        - left_rect_px, right_rect_px: (x, y, w, h) in pixels if frame_w and frame_h are provided
        - roi_signature: 8-char hex id for this ROI set (for matching training files)
        Returns None if no ROIs loaded."""
        if not self._rois:
            return None
        out = {
            "left_eye_roi": list(self._rois["left_eye_roi"]) if self._rois.get("left_eye_roi") else None,
            "right_eye_roi": list(self._rois["right_eye_roi"]) if self._rois.get("right_eye_roi") else None,
            "roi_signature": get_roi_signature(self._rois),
        }
        if frame_w is not None and frame_h is not None:
            out["left_rect_px"] = _roi_to_pixel_rect(out["left_eye_roi"], frame_w, frame_h) if out["left_eye_roi"] else None
            out["right_rect_px"] = _roi_to_pixel_rect(out["right_eye_roi"], frame_w, frame_h) if out["right_eye_roi"] else None
        return out

    def get_pupil_radius_px(self):
        """Return last pupil ellipse axes in pixels (major, minor) for left and right.
        Use for approximate camera–eye distance: apparent size vs known real pupil size.
        Returns ((left_major, left_minor) or None, (right_major, right_minor) or None)."""
        return (self._last_left_pupil_radius_px, self._last_right_pupil_radius_px)

    def get_last_crops(self):
        """Return the last left and right ROI crops (numpy BGR images) for display.
        Only set when ROIs are loaded and process_frame_binocular() was called in ROI mode.
        Returns (left_crop or None, right_crop or None)."""
        return (self._last_left_crop, self._last_right_crop)

    def get_last_left_right_gaze(self):
        """Return 2D gaze (gx, gy) for left and right eye from last frame (ROI mode only).
        Returns ((gxL, gyL) or None, (gxR, gyR) or None)."""
        return (self._last_left_gaze_2d, self._last_right_gaze_2d)

    def get_last_left_right_pupil_in_crop(self):
        """Return pupil position in crop pixel coords for drawing on crop windows (ROI mode only).
        Returns ((xL, yL) or None, (xR, yR) or None)."""
        return (self._last_left_pupil_in_crop, self._last_right_pupil_in_crop)

    def get_last_left_right_pupil_full(self):
        """Return pupil position in full-frame pixel coords (ROI mode only). For display and training.
        Returns ((xL, yL) or None, (xR, yR) or None)."""
        return (self._last_left_pupil_full, self._last_right_pupil_full)

    def get_last_left_right_pupil_normalized_crop(self):
        """Return pupil position normalized in crop (0–1). For training CSV. ROI mode only.
        Returns ((x_normL, y_normL) or None, (x_normR, y_normR) or None)."""
        out_left = None
        if self._last_left_pupil_in_crop is not None and self._last_left_crop_w and self._last_left_crop_h:
            px, py = self._last_left_pupil_in_crop
            out_left = (px / max(1, self._last_left_crop_w), py / max(1, self._last_left_crop_h))
        out_right = None
        if self._last_right_pupil_in_crop is not None and self._last_right_crop_w and self._last_right_crop_h:
            px, py = self._last_right_pupil_in_crop
            out_right = (px / max(1, self._last_right_crop_w), py / max(1, self._last_right_crop_h))
        return (out_left, out_right)

    def get_last_full_frame(self):
        """Return the last full frame from the camera (before any ROI crop). For display in test."""
        return self._last_full_frame

    def get_last_is_blink(self) -> bool:
        """True when no pupil was detected last frame (treat as blink: show x:--, y:-- and do not record)."""
        return getattr(self, "_last_is_blink", False)

    def process_frame_binocular(self):
        """
        Get one frame from the eye camera, run JEO 3DTracker, return (gx, gy, yaw, pitch, fv, conf, is_blink).
        If ROIs are loaded, crops to each eye ROI, runs tracker on each crop, and converts pupil coordinates
        to full-frame (glass-frame) coordinates.
        """
        if self.cap is None or not self.cap.isOpened():
            self._last_is_blink = True
            return None, None, None, None, None, 0.0, True
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._last_is_blink = True
            return None, None, None, None, None, 0.0, True
        frame = cv2.flip(frame, 1)
        self._last_full_frame = frame.copy()
        h, w = frame.shape[:2]

        if self._rois and (self._rois.get("left_eye_roi") or self._rois.get("right_eye_roi")):
            # ROI mode: crop by ROI, run 3D tracker on each cropped frame, then combine
            left_origin, left_dir, left_pupil_full = None, None, None
            right_origin, right_dir, right_pupil_full = None, None, None
            left_pupil_radius_px = right_pupil_radius_px = None
            self._last_left_crop = None
            self._last_right_crop = None
            self._last_left_gaze_2d = None
            self._last_right_gaze_2d = None
            self._last_left_pupil_in_crop = None
            self._last_right_pupil_in_crop = None
            self._last_left_crop_w = self._last_left_crop_h = None
            self._last_right_crop_w = self._last_right_crop_h = None

            for label, key in (("left", "left_eye_roi"), ("right", "right_eye_roi")):
                roi_norm = self._rois.get(key) if self._rois else None
                if not roi_norm or len(roi_norm) != 4:
                    continue
                rect = _roi_to_pixel_rect(roi_norm, w, h)
                if not rect:
                    continue
                x, y, rw, rh = rect
                crop = frame[y : y + rh, x : x + rw].copy()
                if crop.size == 0:
                    continue
                if label == "left":
                    self._last_left_crop = crop
                    self._last_left_crop_w, self._last_left_crop_h = rw, rh
                else:
                    self._last_right_crop = crop
                    self._last_right_crop_w, self._last_right_crop_h = rw, rh
                crop_resized = cv2.resize(crop, (JEO_CROP_W, JEO_CROP_H), interpolation=cv2.INTER_LINEAR)
                self._jeo_process_frame(crop_resized)
                origin, direction, pupil_xy, pupil_radius_px = _read_gaze_vector(self.gaze_vector_path)
                if origin is None or direction is None:
                    continue
                if label == "left":
                    left_origin, left_dir = origin, direction
                    self._last_left_gaze_2d = (float(direction[0]), float(direction[1]))
                    if pupil_xy is not None:
                        rect_main = (x, y, rw, rh)
                        left_pupil_full = _pupil_crop_to_main_frame(pupil_xy, rect_main)
                        self._last_left_pupil_in_crop = (
                            pupil_xy[0] * (rw / JEO_CROP_W),
                            pupil_xy[1] * (rh / JEO_CROP_H),
                        )
                    else:
                        self._last_left_pupil_in_crop = None
                    if pupil_radius_px is not None:
                        left_pupil_radius_px = pupil_radius_px
                else:
                    right_origin, right_dir = origin, direction
                    self._last_right_gaze_2d = (float(direction[0]), float(direction[1]))
                    if pupil_xy is not None:
                        rect_main = (x, y, rw, rh)
                        right_pupil_full = _pupil_crop_to_main_frame(pupil_xy, rect_main)
                        self._last_right_pupil_in_crop = (
                            pupil_xy[0] * (rw / JEO_CROP_W),
                            pupil_xy[1] * (rh / JEO_CROP_H),
                        )
                    else:
                        self._last_right_pupil_in_crop = None
                    if pupil_radius_px is not None:
                        right_pupil_radius_px = pupil_radius_px

            # Prefer both eyes; else use whichever we got
            if left_dir is not None and right_dir is not None:
                gx = 0.5 * (float(left_dir[0]) + float(right_dir[0]))
                gy = 0.5 * (float(left_dir[1]) + float(right_dir[1]))
                fv = _gaze_binocular_to_feature_vector(left_dir, right_dir)
                conf = 0.9
            elif left_dir is not None:
                gx, gy = float(left_dir[0]), float(left_dir[1])
                fv = _gaze_binocular_to_feature_vector(left_dir, left_dir)
                conf = 0.9
            elif right_dir is not None:
                gx, gy = float(right_dir[0]), float(right_dir[1])
                fv = _gaze_binocular_to_feature_vector(right_dir, right_dir)
                conf = 0.9
            else:
                self._last_left_gaze_2d = self._last_right_gaze_2d = None
                self._last_left_pupil_in_crop = self._last_right_pupil_in_crop = None
                self._last_left_pupil_full = self._last_right_pupil_full = None
                self._last_is_blink = True
                return None, None, None, None, None, 0.0, True
            self._last_left_pupil_full = left_pupil_full
            self._last_right_pupil_full = right_pupil_full
            self._last_left_pupil_radius_px = left_pupil_radius_px
            self._last_right_pupil_radius_px = right_pupil_radius_px
            self._last_is_blink = False
            return gx, gy, 0.0, 0.0, fv, conf, False
        else:
            # Full-frame mode (no ROIs or ROIs not loaded)
            self._jeo_process_frame(frame)
            origin, direction, _, pupil_radius_px = _read_gaze_vector(self.gaze_vector_path)
            self._last_left_pupil_radius_px = pupil_radius_px
            self._last_right_pupil_radius_px = None
            if origin is None or direction is None:
                self._last_is_blink = True
                return None, None, None, None, None, 0.0, True
            self._last_is_blink = False
            gx = float(direction[0])
            gy = float(direction[1])
            fv = _gaze_3d_to_feature_vector(origin, direction, 0.9)
            conf = 0.9
            return gx, gy, 0.0, 0.0, fv, conf, False

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def jeo_tracker_available() -> bool:
    return _jeo_process_frame is not None
