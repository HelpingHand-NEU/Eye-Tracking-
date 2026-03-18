import os
import cv2
import numpy as np
import time
from collections import deque
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:
    mp = None
    mp_python = None
    mp_vision = None
try:
    from pupil_detectors import Detector2D as PuReDetector2D
except Exception:
    PuReDetector2D = None
from eye_tracker.calibration import HEAD_YAW_CLIP, HEAD_PITCH_CLIP, HEAD_ROLL_CLIP

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_IRIS_IDX = [474, 475, 476, 477]
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_IRIS_IDX = [469, 470, 471, 472]

LEFT_EYE = {
    "outer": LEFT_EYE_OUTER,
    "inner": LEFT_EYE_INNER,
    "top": LEFT_EYE_TOP,
    "bottom": LEFT_EYE_BOTTOM,
    "iris": LEFT_IRIS_IDX,
}
RIGHT_EYE = {
    "outer": RIGHT_EYE_OUTER,
    "inner": RIGHT_EYE_INNER,
    "top": RIGHT_EYE_TOP,
    "bottom": RIGHT_EYE_BOTTOM,
    "iris": RIGHT_IRIS_IDX,
}

# Head pose estimation (solvePnP) landmarks
# MediaPipe: 263 = LEFT eye outer, 33 = RIGHT eye outer; order must match FACE_3D_MODEL
HEAD_LANDMARK_IDS = [1, 152, 263, 33, 61, 291]  # nose, chin, left eye, right eye, left/right mouth
FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -63.6, -12.5),    # Chin
        (-43.3, 32.7, -26.0),   # Left eye corner
        (43.3, 32.7, -26.0),    # Right eye corner
        (-28.9, -28.9, -24.1),  # Left mouth
        (28.9, -28.9, -24.1),   # Right mouth
    ],
    dtype=np.float64,
)
USE_HEAD_POSE_PNP = True
USE_HEAD_POSE_NORM = False  # head norm on 2D gaze features is not physically correct; hurts accuracy
NOSE_TIP = 1

# Eye Aspect Ratio: below this the eye is considered closed (blink)
EAR_CLOSED_THRESH = 0.18
EAR_CALIBRATION_SECONDS = 2.0

DISPLAY_HEIGHT_BASE = 350
DISPLAY_HEIGHT_MAX = 600

# Glass-frame pupil tracking tuning
GLASS_ROI_Y0 = 0.12
GLASS_ROI_Y1 = 0.88
GLASS_MIN_PUPIL_AREA = 20.0
GLASS_MAX_PUPIL_AREA_FRAC = 0.25
GLASS_LOCAL_SEARCH_RADIUS = 70
GLASS_RAY_COUNT = 24
GLASS_RAY_STEP = 2
GLASS_EDGE_GRAD_THRESH = 7.0
GLASS_EDGE_MIN_BRIGHT = 45.0
# Stability: reject gaze jumps larger than this (normalized); use previous value instead
GLASS_GAZE_JUMP_THRESH = 0.08
GLASS_MEDIAN_LEN = 5  # temporal median over this many frames to reduce jitter
GLASS_QUALITY_PRESETS = {
    "fast": {
        "local_search_radius": 52,
        "ray_count": 16,
        "ray_step": 3,
        "edge_grad_thresh": 8.5,
        "edge_min_bright": 52.0,
        "clahe_clip": 1.8,
        "blur_ksize": 5,
        "kalman_process_noise": 2.2e-2,
        "kalman_base_measurement_noise": 8.5,
    },
    "balanced": {
        "local_search_radius": GLASS_LOCAL_SEARCH_RADIUS,
        "ray_count": GLASS_RAY_COUNT,
        "ray_step": GLASS_RAY_STEP,
        "edge_grad_thresh": GLASS_EDGE_GRAD_THRESH,
        "edge_min_bright": GLASS_EDGE_MIN_BRIGHT,
        "clahe_clip": 2.0,
        "blur_ksize": 7,
        "kalman_process_noise": 5.0e-3,
        "kalman_base_measurement_noise": 9.0,
    },
    "max_accuracy": {
        "local_search_radius": 86,
        "ray_count": 36,
        "ray_step": 1,
        "edge_grad_thresh": 5.2,
        "edge_min_bright": 36.0,
        "clahe_clip": 2.4,
        "blur_ksize": 9,
        "kalman_process_noise": 3.0e-3,
        "kalman_base_measurement_noise": 12.0,
    },
}


class _AdaptiveKalman2D:
    """Constant-velocity 2D Kalman with confidence-adaptive measurement noise."""

    def __init__(self, process_noise=1e-2, base_measurement_noise=6.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        self.base_r = float(base_measurement_noise)
        self.initialized = False

    def _set_r(self, conf):
        conf = float(np.clip(conf, 1e-3, 1.0))
        r = self.base_r / (conf * conf)
        r = float(np.clip(r, 1.0, 120.0))
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r

    def update(self, x, y, conf=1.0):
        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.kf.statePre = self.kf.statePost.copy()
            self.initialized = True
        self._set_r(conf)
        self.kf.predict()
        m = np.array([[float(x)], [float(y)]], dtype=np.float32)
        est = self.kf.correct(m)
        return float(est[0, 0]), float(est[1, 0])

    def predict_only(self):
        if not self.initialized:
            return None
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])


class UsbCameraDetector:
    """
    Detect the USB camera by location (index), check if it can be used.
    Use .camera_id after .detect() to get the camera number for EyeTracker.
    """

    def __init__(self, max_try=4):
        self.max_try = max_try
        self.camera_id = None
        self.available = False

    def detect(self):
        """
        Find USB camera: try indices 1, 2, 3 first (typical USB), then 0.
        Set self.camera_id and self.available if a usable camera is found.
        """
        order = list(range(1, self.max_try)) + [0]
        for i in order:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.camera_id = i
                    self.available = True
                    return
        self.camera_id = 0
        self.available = False


class _LandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _ResultCompat:
    def __init__(self, result):
        self.multi_face_landmarks = []
        if result and getattr(result, "face_landmarks", None):
            for landmarks in result.face_landmarks:
                self.multi_face_landmarks.append(_LandmarkList(landmarks))


class _FaceMeshCompat:
    def __init__(self, model_path, num_faces=1):
        if mp_vision is None or mp_python is None:
            raise ImportError("MediaPipe is required for webcam mode but is not installed.")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame, timestamp_ms):
        if rgb_frame is None:
            return _ResultCompat(None)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return _ResultCompat(self._landmarker.detect_for_video(image, int(timestamp_ms)))


class EyeTracker:
    def __init__(
        self,
        front_camera_id=0,
        back_camera_id=1,
        reset=False,
        training_mode="webcam",
        glass_quality_profile="balanced",
    ):
        self.training_mode = training_mode
        if self.training_mode not in ("webcam", "glass_frame"):
            raise ValueError("EyeTracker training_mode must be 'webcam' or 'glass_frame'.")
        if self.training_mode == "glass_frame":
            raise ValueError(
                "For glass_frame (camera on glasses) use JEOGlassFrameTracker from eye_tracker.glass_frame_jeo_tracker. "
                "ML data is taken from the JEOresearch/EyeTracker 3DTracker (vendor)."
            )
        self.front_camera_id = front_camera_id
        self.back_camera_id = back_camera_id
        self.cap = cv2.VideoCapture(self.front_camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Back camera is optional; use it only if available
        self.back_cap = cv2.VideoCapture(self.back_camera_id)
        if self.back_cap.isOpened():
            self.back_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.back_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.back_cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.back_cap.isOpened():
            self.back_cap = None
        self._left_eye_was_closed = False
        self.left_eye_crop_x = None
        self.left_eye_crop_y = None
        self.left_eye_crop_width = None
        self.left_eye_crop_height = None
        self.left_eye_pupil_x = None
        self.left_eye_pupil_y = None
        self.left_eye_pupil_x_norm = None
        self.left_eye_pupil_y_norm = None
        self._last_gx = None
        self._last_gy = None
        self._frame_index = 0
        self.ear_closed_thresh = EAR_CLOSED_THRESH
        self.last_yaw = None
        self.last_pitch = None
        self.last_iod = None
        self.last_roll = None
        self._hp_ema = None
        self._last_head_pose = None
        self._last_head_rmat = None
        self._buf_L = deque(maxlen=12)
        self._buf_R = deque(maxlen=12)
        self._pred = None
        self.reset = reset
        if self.training_mode == "webcam":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "face_landmarker.task")
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(script_dir), "face_landmarker.task")
            self.face_mesh = _FaceMeshCompat(model_path=model_path, num_faces=1)
        else:
            # Glass-frame mode: pupil-only tracking (no face landmarks / MediaPipe dependency).
            self.face_mesh = None
            self.glass_quality_profile = "balanced"
            self._apply_glass_quality_profile(glass_quality_profile)
            self._glass_kf_left = _AdaptiveKalman2D(
                process_noise=self._glass_kf_process_noise,
                base_measurement_noise=self._glass_kf_base_measurement_noise,
            )
            self._glass_kf_right = _AdaptiveKalman2D(
                process_noise=self._glass_kf_process_noise,
                base_measurement_noise=self._glass_kf_base_measurement_noise,
            )
            self._glass_gaze_buf = deque(maxlen=GLASS_MEDIAN_LEN)
            self._last_glass_gaze = None
            # PuRe (Pupil Labs) detector for more accurate pupil detection when available
            self._pupil_detector_2d = PuReDetector2D() if PuReDetector2D is not None else None

    def _apply_glass_quality_profile(self, profile_name):
        key = str(profile_name).strip().lower()
        if key not in GLASS_QUALITY_PRESETS:
            allowed = ", ".join(sorted(GLASS_QUALITY_PRESETS.keys()))
            raise ValueError(f"glass_quality_profile must be one of: {allowed}. Got: {profile_name}")
        p = GLASS_QUALITY_PRESETS[key]
        self.glass_quality_profile = key
        self._glass_local_search_radius = int(p["local_search_radius"])
        self._glass_ray_count = int(p["ray_count"])
        self._glass_ray_step = int(p["ray_step"])
        self._glass_edge_grad_thresh = float(p["edge_grad_thresh"])
        self._glass_edge_min_bright = float(p["edge_min_bright"])
        self._glass_clahe_clip = float(p["clahe_clip"])
        k = int(p["blur_ksize"])
        self._glass_blur_ksize = k if (k % 2 == 1) else (k + 1)
        self._glass_kf_process_noise = float(p["kalman_process_noise"])
        self._glass_kf_base_measurement_noise = float(p["kalman_base_measurement_noise"])

    def _fit_to_frame_ratio(self, img, frame_w, frame_h, max_height=400):
        out_h = min(max_height, frame_h)
        out_w = int(out_h * frame_w / frame_h)
        if out_w < 1 or out_h < 1:
            out_w, out_h = max(1, out_w), max(1, out_h)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if img.size == 0:
            return canvas
        in_h, in_w = img.shape[0], img.shape[1]
        scale = min(out_w / in_w, out_h / in_h)
        new_w, new_h = max(1, int(in_w * scale)), max(1, int(in_h * scale))
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        y0 = (out_h - new_h) // 2
        x0 = (out_w - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = scaled
        return canvas

    def _dynamic_crop_size(self, lw, lh):
        eye_span = lw + lh
        scale = eye_span / 80.0
        target_w = max(30, min(200, int(100 * scale)))
        target_h = max(15, min(80, int(40 * scale)))
        display_max_h = min(DISPLAY_HEIGHT_MAX, max(100, int(DISPLAY_HEIGHT_BASE * scale)))
        return target_w, target_h, display_max_h

    def _eye_ear_from_landmarks(self, lms, eye):
        # Use a simple eye aspect ratio based on normalized coords.
        left = lms[eye["outer"]]
        right = lms[eye["inner"]]
        top = lms[eye["top"]]
        bottom = lms[eye["bottom"]]
        h = np.hypot(right.x - left.x, right.y - left.y)
        v = np.hypot(top.x - bottom.x, top.y - bottom.y)
        if h == 0:
            return 0.0
        return v / h

    def _crop_and_mask_eye(self, frame, lms, eye, pad=6):
        h, w = frame.shape[:2]
        pts = np.array(
            [
                [lms[eye["outer"]].x * w, lms[eye["outer"]].y * h],
                [lms[eye["top"]].x * w, lms[eye["top"]].y * h],
                [lms[eye["inner"]].x * w, lms[eye["inner"]].y * h],
                [lms[eye["bottom"]].x * w, lms[eye["bottom"]].y * h],
            ],
            dtype=np.int32,
        )
        x, y, ww, hh = cv2.boundingRect(pts)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)
        roi = frame[y0:y1, x0:x1].copy()
        pts_roi = pts - np.array([x0, y0])
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_roi], 255)
        return roi, mask, (x0, y0)

    def _clamp01(self, x):
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _suppress_glints(self, gray_roi):
        """Suppress bright specular highlights (glints) via inpainting."""
        if gray_roi is None or gray_roi.size == 0:
            return gray_roi
        # Dynamic threshold: very bright tail only.
        p = float(np.percentile(gray_roi, 99.3))
        thr = int(max(220, min(250, p)))
        mask = (gray_roi >= thr).astype(np.uint8) * 255
        if cv2.countNonZero(mask) == 0:
            return gray_roi
        return cv2.inpaint(gray_roi, mask, 3, cv2.INPAINT_TELEA)

    def _ellipse_support_score(self, ellipse, points):
        """Return [0,1] support score: how many points lie near ellipse."""
        if ellipse is None or points is None or len(points) == 0:
            return 0.0
        (cx, cy), (ma, mi), angle = ellipse
        axes = (max(1, int(ma * 0.5)), max(1, int(mi * 0.5)))
        poly = cv2.ellipse2Poly((int(cx), int(cy)), axes, int(angle), 0, 360, 6)
        if poly is None or len(poly) < 6:
            return 0.0
        contour = poly.reshape((-1, 1, 2)).astype(np.int32)
        inliers = 0
        for p in points:
            d = abs(float(cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)))
            if d <= 3.0:
                inliers += 1
        return float(inliers / max(1, len(points)))

    def _refine_pupil_center_with_rays(self, gray_blur, seed_xy):
        """
        Starburst-like boundary refinement:
        cast rays from candidate center and collect strongest dark->bright edges,
        then fit ellipse and use it to refine center.
        Returns (cx, cy, quality) in ROI coordinates, or None.
        """
        if gray_blur is None or gray_blur.size == 0 or seed_xy is None:
            return None
        h, w = gray_blur.shape[:2]
        cx0, cy0 = float(seed_xy[0]), float(seed_xy[1])
        if cx0 < 1 or cy0 < 1 or cx0 >= (w - 1) or cy0 >= (h - 1):
            return None
        max_r = int(max(12, min(w, h) * 0.45))
        pts = []
        angles = np.linspace(0.0, 2.0 * np.pi, self._glass_ray_count, endpoint=False)
        for ang in angles:
            dx = float(np.cos(ang))
            dy = float(np.sin(ang))
            px = int(round(cx0))
            py = int(round(cy0))
            prev = float(gray_blur[py, px])
            best_pt = None
            best_grad = 0.0
            s = self._glass_ray_step
            while s <= max_r:
                x = int(round(cx0 + dx * s))
                y = int(round(cy0 + dy * s))
                if x < 1 or y < 1 or x >= (w - 1) or y >= (h - 1):
                    break
                val = float(gray_blur[y, x])
                grad = val - prev
                if grad > self._glass_edge_grad_thresh and val > self._glass_edge_min_bright and grad > best_grad:
                    best_grad = grad
                    best_pt = (x, y)
                prev = val
                s += self._glass_ray_step
            if best_pt is not None:
                pts.append(best_pt)
        if len(pts) < 8:
            return None
        cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(cnt)
        (ecx, ecy), (ma, mi), _ = ellipse
        a = max(float(ma), float(mi))
        b = max(1e-6, min(float(ma), float(mi)))
        aspect = b / a
        if a < 6.0 or b < 3.0 or aspect < 0.18:
            return None
        support = self._ellipse_support_score(ellipse, pts)
        quality = float(np.clip(0.5 * support + 0.5 * np.clip(aspect / 0.6, 0.0, 1.0), 0.0, 1.0))
        return float(ecx), float(ecy), quality

    def _detect_pupil_pure(self, gray_roi):
        """
        Pupil detection using PuRe (Pupil Labs) when pupil-detectors is installed.
        Returns (x, y, conf) in ROI coordinates, or None if unavailable or no pupil.
        """
        if self._pupil_detector_2d is None or gray_roi is None or gray_roi.size == 0:
            return None
        h, w = gray_roi.shape[:2]
        if h < 40 or w < 40:
            return None
        try:
            result = self._pupil_detector_2d.detect(gray_roi)
            ellipse = result.get("ellipse") if isinstance(result, dict) else getattr(result, "ellipse", None)
            if ellipse is None:
                return None
            center = ellipse.get("center") if isinstance(ellipse, dict) else getattr(ellipse, "center", None)
            if center is None or len(center) < 2:
                return None
            cx, cy = float(center[0]), float(center[1])
            axes = ellipse.get("axes") if isinstance(ellipse, dict) else getattr(ellipse, "axes", (10.0, 10.0))
            if axes is not None and len(axes) >= 2:
                a, b = max(float(axes[0]), float(axes[1])), min(float(axes[0]), float(axes[1]))
                area_frac = (np.pi * a * b) / float(h * w)
                aspect = b / max(a, 1e-6)
                if area_frac > 0.6 or area_frac < 0.001 or aspect < 0.2:
                    return None
                conf = float(np.clip(0.5 + 0.4 * min(aspect / 0.7, 1.0) + 0.1 * min(area_frac / 0.15, 1.0), 0.5, 1.0))
            else:
                conf = 0.85
            if cx < 2 or cy < 2 or cx > (w - 3) or cy > (h - 3):
                return None
            return (cx, cy, conf)
        except Exception:
            return None

    def _detect_pupil_in_roi(self, gray_roi, prior_xy=None, allow_retry=True):
        """
        Robust pupil detector for glass-frame mode.
        Tries PuRe (Pupil Labs) first when available, then contour + ray refinement.
        Returns (x, y, conf) in ROI coordinates.
        """
        if gray_roi is None or gray_roi.size == 0:
            return None
        roi_full = gray_roi
        h, w = roi_full.shape[:2]
        if h < 8 or w < 8:
            return None

        ox = 0
        oy = 0
        used_local_search = False

        # Try PuRe first for higher accuracy when available
        if self._pupil_detector_2d is not None:
            roi_for_pure = roi_full
            if prior_xy is not None:
                px, py = int(prior_xy[0]), int(prior_xy[1])
                r = int(max(40, min(self._glass_local_search_radius, min(w, h) // 2)))
                x0 = max(0, px - r)
                x1 = min(w, px + r)
                y0 = max(0, py - r)
                y1 = min(h, py + r)
                if (x1 - x0) >= 40 and (y1 - y0) >= 40:
                    used_local_search = True
                    ox, oy = x0, y0
                    roi_for_pure = roi_full[y0:y1, x0:x1]
            pure_result = self._detect_pupil_pure(roi_for_pure)
            if pure_result is not None:
                cx, cy, conf = pure_result
                return (cx + float(ox), cy + float(oy), conf)

        # PuREST-style local search around prediction from previous frame (for fallback path).
        ox = 0
        oy = 0
        used_local_search = False
        if prior_xy is not None:
            px, py = int(prior_xy[0]), int(prior_xy[1])
            r = int(max(24, min(self._glass_local_search_radius, min(w, h) // 2)))
            x0 = max(0, px - r)
            x1 = min(w, px + r)
            y0 = max(0, py - r)
            y1 = min(h, py + r)
            if (x1 - x0) >= 16 and (y1 - y0) >= 16:
                used_local_search = True
                ox, oy = x0, y0
                roi = roi_full[y0:y1, x0:x1]
            else:
                roi = roi_full
        else:
            roi = roi_full

        clahe = cv2.createCLAHE(clipLimit=self._glass_clahe_clip, tileGridSize=(8, 8))
        no_glint = self._suppress_glints(roi)
        norm = clahe.apply(no_glint)
        blur = cv2.GaussianBlur(norm, (self._glass_blur_ksize, self._glass_blur_ksize), 0)

        # Dark blobs (pupil) become foreground.
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        rh, rw = roi.shape[:2]
        max_area = GLASS_MAX_PUPIL_AREA_FRAC * float(rh * rw)

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < GLASS_MIN_PUPIL_AREA or area > max_area:
                continue
            per = float(cv2.arcLength(c, True))
            if per <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (per * per))
            if circularity <= 0.15:
                continue
            m = cv2.moments(c)
            if abs(m["m00"]) < 1e-6:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            ecc_pen = 1.0
            if len(c) >= 5:
                (_, _), (ma, mi), _ = cv2.fitEllipse(c)
                a = max(float(ma), float(mi))
                b = max(1e-6, min(float(ma), float(mi)))
                ecc = float(np.sqrt(max(0.0, 1.0 - (b * b) / (a * a))))
                # Prefer less eccentric pupil candidates.
                ecc_pen = float(np.clip(1.0 - 0.5 * ecc, 0.5, 1.0))
            # Penalize boundary blobs (eyelid/lashes at borders).
            border_pen = 1.0
            if cx < 4 or cx > (rw - 4) or cy < 4 or cy > (rh - 4):
                border_pen = 0.5
            score = area * max(0.0, circularity) * border_pen * ecc_pen
            if score > best_score:
                best_score = score
                best = (cx, cy, area, circularity)

        if best is not None:
            cx, cy, area, circularity = best
            conf = float(np.clip((circularity / 0.9) * min(1.0, area / 250.0), 0.05, 1.0))
            refined = self._refine_pupil_center_with_rays(blur, (cx, cy))
            if refined is not None:
                rcx, rcy, rq = refined
                alpha = float(np.clip(0.25 + 0.55 * rq, 0.25, 0.8))
                cx = (1.0 - alpha) * cx + alpha * rcx
                cy = (1.0 - alpha) * cy + alpha * rcy
                conf = float(np.clip(conf * (0.75 + 0.45 * rq), 0.03, 1.0))
            # If local-search result is weak or near borders, retry full ROI once.
            if used_local_search and allow_retry:
                near_border = (cx < 6.0) or (cy < 6.0) or (cx > (rw - 6.0)) or (cy > (rh - 6.0))
                if conf < 0.55 or near_border:
                    retry = self._detect_pupil_in_roi(roi_full, prior_xy=None, allow_retry=False)
                    if retry is not None and retry[2] > conf:
                        return retry
            return cx + float(ox), cy + float(oy), conf

        # If local-search failed, fall back to full ROI once to recover from fast motion.
        if used_local_search and allow_retry:
            return self._detect_pupil_in_roi(roi_full, prior_xy=None, allow_retry=False)

        # Fallback to darkest-point detector if contour route fails.
        min_val, _, min_loc, _ = cv2.minMaxLoc(blur)
        y0 = max(0, min_loc[1] - 6)
        y1 = min(blur.shape[0], min_loc[1] + 7)
        x0 = max(0, min_loc[0] - 6)
        x1 = min(blur.shape[1], min_loc[0] + 7)
        patch = blur[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        local_mean = float(np.mean(patch))
        conf = float(np.clip((local_mean - float(min_val)) / 40.0, 0.0, 0.6))
        return float(min_loc[0] + ox), float(min_loc[1] + oy), conf

    def _stabilize_glass_gaze(self, gx, gy):
        """Apply temporal median and jump rejection to reduce jitter and spikes."""
        self._glass_gaze_buf.append((float(gx), float(gy)))
        arr = np.array(list(self._glass_gaze_buf), dtype=np.float64)
        gx_m = float(np.median(arr[:, 0]))
        gy_m = float(np.median(arr[:, 1]))
        if self._last_glass_gaze is not None:
            lgx, lgy = self._last_glass_gaze
            if abs(gx_m - lgx) > GLASS_GAZE_JUMP_THRESH or abs(gy_m - lgy) > GLASS_GAZE_JUMP_THRESH:
                gx_m, gy_m = lgx, lgy
        self._last_glass_gaze = (gx_m, gy_m)
        return gx_m, gy_m

    def _split_glass_rois(self, gray):
        h, w = gray.shape[:2]
        y0 = int(max(0, min(h - 1, h * GLASS_ROI_Y0)))
        y1 = int(max(y0 + 1, min(h, h * GLASS_ROI_Y1)))
        band = gray[y0:y1, :]
        half = band.shape[1] // 2
        left_roi = band[:, :half]
        right_roi = band[:, half:]
        return left_roi, right_roi, y0, half, band.shape[1], band.shape[0]

    def _kalman_smooth_pupils(self, left_det, right_det):
        # Adaptive R: higher confidence => stronger correction, lower lag.
        if left_det is None:
            l_pred = self._glass_kf_left.predict_only()
            lx, ly, lconf = (l_pred[0], l_pred[1], 0.2) if l_pred is not None else (None, None, 0.0)
        else:
            lx, ly, lconf = left_det
            lx, ly = self._glass_kf_left.update(lx, ly, conf=lconf)
        if right_det is None:
            r_pred = self._glass_kf_right.predict_only()
            rx, ry, rconf = (r_pred[0], r_pred[1], 0.2) if r_pred is not None else (None, None, 0.0)
        else:
            rx, ry, rconf = right_det
            rx, ry = self._glass_kf_right.update(rx, ry, conf=rconf)
        return (lx, ly, lconf), (rx, ry, rconf)

    def _process_frame_glass_frame_basic(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, True
        self._frame_index += 1
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        left_roi, right_roi, y0_band, half, _, band_h = self._split_glass_rois(gray)
        left_prior = self._glass_kf_left.predict_only()
        right_prior = self._glass_kf_right.predict_only()
        left_det = self._detect_pupil_in_roi(left_roi, prior_xy=left_prior)
        right_det = self._detect_pupil_in_roi(right_roi, prior_xy=right_prior)
        left_kf, right_kf = self._kalman_smooth_pupils(left_det, right_det)
        lx, ly, lconf = left_kf
        rx, ry, rconf = right_kf
        if lx is None and rx is None:
            return None, None, True

        # Use whichever side is available; if only one is detected, mirror to both.
        if lx is None:
            lx, ly, lconf = rx, ry, rconf
        elif rx is None:
            rx, ry, rconf = lx, ly, lconf

        # Convert ROI pupil coordinates to normalized eye-local features in [-0.5, 0.5].
        gxL = (float(lx) / max(1.0, float(half))) - 0.5
        gyL = (float(ly) / max(1.0, float(band_h))) - 0.5
        gxR = (float(rx) / max(1.0, float(half))) - 0.5
        gyR = (float(ry) / max(1.0, float(band_h))) - 0.5

        gx = float(0.5 * (gxL + gxR))
        gy = float(0.5 * (gyL + gyR))
        gx, gy = self._stabilize_glass_gaze(gx, gy)
        self.left_eye_pupil_x = int(lx)
        self.left_eye_pupil_y = int(ly + y0_band)
        self.left_eye_pupil_x_norm = gx
        self.left_eye_pupil_y_norm = gy
        self.last_left_gx = gxL
        self.last_left_gy = gyL
        self.last_right_gx = gxR
        self.last_right_gy = gyR
        self.last_w_left = 0.5
        self.last_w_right = 0.5
        self.last_raw_sum = float(lconf + rconf)
        return gx, gy, False

    def _process_frame_binocular_glass_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None, None, 0.0, True
        self._frame_index += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        left_roi, right_roi, y0_band, half, _, band_h = self._split_glass_rois(gray)
        left_prior = self._glass_kf_left.predict_only()
        right_prior = self._glass_kf_right.predict_only()
        left_det = self._detect_pupil_in_roi(left_roi, prior_xy=left_prior)
        right_det = self._detect_pupil_in_roi(right_roi, prior_xy=right_prior)
        left_kf, right_kf = self._kalman_smooth_pupils(left_det, right_det)
        lx, ly, lconf = left_kf
        rx, ry, rconf = right_kf
        if lx is None and rx is None:
            return None, None, None, None, None, 0.0, True

        if lx is None:
            lx, ly, lconf = rx, ry, rconf
        elif rx is None:
            rx, ry, rconf = lx, ly, lconf

        gxL = (float(lx) / max(1.0, float(half))) - 0.5
        gyL = (float(ly) / max(1.0, float(band_h))) - 0.5
        gxR = (float(rx) / max(1.0, float(half))) - 0.5
        gyR = (float(ry) / max(1.0, float(band_h))) - 0.5
        gx = float(0.5 * (gxL + gxR))
        gy = float(0.5 * (gyL + gyR))
        gx, gy = self._stabilize_glass_gaze(gx, gy)

        # Glass-frame mode has no face pose; keep pose-related fields neutral.
        yaw = 0.0
        pitch = 0.0
        iod = float(np.hypot((lx + half) - rx, ly - ry) / max(1.0, float(w)))
        roll = float(np.degrees(np.arctan2((ry - ly), max(1.0, (rx - lx)))))
        lid_hL = 0.02
        lid_hR = 0.02
        face_w = 1.0
        face_h = 1.0
        nose_x = 0.5
        nose_y = 0.5
        iod_safe = max(iod, 1e-4)
        nx = (nose_x - 0.5) / iod_safe
        ny = (nose_y - 0.5) / iod_safe
        yaw_c = float(np.clip(yaw, -HEAD_YAW_CLIP, HEAD_YAW_CLIP))
        pitch_c = float(np.clip(pitch, -HEAD_PITCH_CLIP, HEAD_PITCH_CLIP))
        roll_c = float(np.clip(roll, -HEAD_ROLL_CLIP, HEAD_ROLL_CLIP))

        self.last_left_gx = gxL
        self.last_left_gy = gyL
        self.last_right_gx = gxR
        self.last_right_gy = gyR
        self.last_left_eye_w = 0.04
        self.last_right_eye_w = 0.04
        self.last_left_lid_h = lid_hL
        self.last_right_lid_h = lid_hR
        self.last_face_w = face_w
        self.last_face_h = face_h
        self.last_nose_x = nose_x
        self.last_nose_y = nose_y
        self.last_w_left = 0.5
        self.last_w_right = 0.5
        self.last_raw_sum = float(lconf + rconf)
        self.last_iod = iod
        self.last_roll = roll
        self.last_yaw = yaw
        self.last_pitch = pitch

        fv = np.array(
            [
                gxL, gyL, gxR, gyR,
                yaw_c, pitch_c, iod_safe, roll_c,
                lid_hL, lid_hR, face_w, face_h,
                nx, ny, 0.5, 0.5,
            ],
            dtype=np.float64,
        )
        conf = float(np.clip(0.5 * (lconf + rconf), 0.0, 1.0))
        return gx, gy, yaw, pitch, fv, conf, False

    def _iris_center(self, lms, iris_idx):
        cx = float(np.mean([lms[i].x for i in iris_idx]))
        cy = float(np.mean([lms[i].y for i in iris_idx]))
        return np.array([cx, cy], dtype=np.float32)

    def _pose_proxies(self, lms):
        nose = np.array([lms[NOSE_TIP].x, lms[NOSE_TIP].y], dtype=np.float32)
        left_eye_center = (
            np.array([lms[LEFT_EYE_OUTER].x, lms[LEFT_EYE_OUTER].y], dtype=np.float32)
            + np.array([lms[LEFT_EYE_INNER].x, lms[LEFT_EYE_INNER].y], dtype=np.float32)
        ) * 0.5
        right_eye_center = (
            np.array([lms[RIGHT_EYE_OUTER].x, lms[RIGHT_EYE_OUTER].y], dtype=np.float32)
            + np.array([lms[RIGHT_EYE_INNER].x, lms[RIGHT_EYE_INNER].y], dtype=np.float32)
        ) * 0.5
        iod = float(np.linalg.norm(left_eye_center - right_eye_center))
        iod = max(iod, 1e-4)
        eye_center = (left_eye_center + right_eye_center) * 0.5
        yaw = float((nose[0] - eye_center[0]) / iod)
        pitch = float((nose[1] - eye_center[1]) / iod)
        return yaw, pitch

    def _roll_proxy(self, lms):
        def eye_angle(outer, inner):
            a = np.array([lms[outer].x, lms[outer].y], dtype=np.float32)
            b = np.array([lms[inner].x, lms[inner].y], dtype=np.float32)
            v = b - a
            return float(np.arctan2(v[1], v[0]))
        ang_l = eye_angle(LEFT_EYE_OUTER, LEFT_EYE_INNER)
        ang_r = eye_angle(RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
        return float(np.degrees(0.5 * (ang_l + ang_r)))  # degrees to match PnP roll

    def _iod_proxy(self, lms):
        le = (
            np.array([lms[LEFT_EYE_OUTER].x, lms[LEFT_EYE_OUTER].y], dtype=np.float32)
            + np.array([lms[LEFT_EYE_INNER].x, lms[LEFT_EYE_INNER].y], dtype=np.float32)
        ) * 0.5
        re = (
            np.array([lms[RIGHT_EYE_OUTER].x, lms[RIGHT_EYE_OUTER].y], dtype=np.float32)
            + np.array([lms[RIGHT_EYE_INNER].x, lms[RIGHT_EYE_INNER].y], dtype=np.float32)
        ) * 0.5
        return float(np.linalg.norm(le - re))

    def _estimate_head_pose(self, lms, frame_w, frame_h):
        image_points = np.array(
            [(lms[i].x * frame_w, lms[i].y * frame_h) for i in HEAD_LANDMARK_IDS],
            dtype=np.float64,
        )
        focal_length = float(frame_w)
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        pitch = float(np.arctan2(-rmat[2, 0], sy))
        yaw = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        roll = float(np.arctan2(rmat[1, 0], rmat[0, 0]))
        yaw = float(np.degrees(yaw))
        pitch = float(np.degrees(pitch))
        roll = float(np.degrees(roll))
        return yaw, pitch, roll, rmat

    def _normalize_gaze_by_head(self, gx, gy):
        if not USE_HEAD_POSE_NORM:
            return gx, gy
        rmat = self._last_head_rmat
        if rmat is None:
            return gx, gy
        vec = np.array([gx, gy, 1.0], dtype=np.float64)
        v = rmat.T @ vec
        if abs(v[2]) < 1e-4:
            return gx, gy
        return float(v[0] / v[2]), float(v[1] / v[2])

    def _gaze_features(self, lms, eye):
        left = np.array([lms[eye["outer"]].x, lms[eye["outer"]].y], dtype=np.float32)
        right = np.array([lms[eye["inner"]].x, lms[eye["inner"]].y], dtype=np.float32)
        top = np.array([lms[eye["top"]].x, lms[eye["top"]].y], dtype=np.float32)
        bottom = np.array([lms[eye["bottom"]].x, lms[eye["bottom"]].y], dtype=np.float32)
        eye_vec = right - left
        eye_w = float(np.linalg.norm(eye_vec))
        if eye_w < 1e-4:
            return None
        lid_h = float(np.linalg.norm(bottom - top))
        iris = self._iris_center(lms, eye["iris"])
        u = eye_vec / eye_w
        # Force consistent axes across eyes.
        if u[0] < 0:
            u = -u
        v = np.array([-u[1], u[0]], dtype=np.float32)
        if v[1] > 0:
            v = -v
        center = (left + right) * 0.5
        d = iris - center
        gx = float(np.dot(d, u) / eye_w)
        gy = float(np.dot(d, v) / eye_w)
        gx, gy = self._normalize_gaze_by_head(gx, gy)
        if not np.isfinite(gx) or not np.isfinite(gy):
            return None
        gx = float(np.clip(gx, -0.6, 0.6))
        gy = float(np.clip(gy, -0.6, 0.6))
        if gx < -1.5 or gx > 1.5 or gy < -1.5 or gy > 1.5:
            return None
        if self._last_head_pose is not None:
            yaw, pitch, _ = self._last_head_pose
        else:
            yaw, pitch = 0.0, 0.0  # don't substitute pose proxies (different scale) when PnP fails
        return gx, gy, yaw, pitch, eye_w, lid_h

    def _left_eye_features(self, lms):
        return self._gaze_features(lms, LEFT_EYE)

    def _right_eye_features(self, lms):
        return self._gaze_features(lms, RIGHT_EYE)

    def _eye_reliability(self, gx, gy, eye_w, lid_h, blink, gx_limit=0.35, gy_limit=0.25):
        if blink or gx is None or gy is None:
            return 0.0
        ex = max(0.0, abs(gx) - gx_limit) / max(1e-6, (0.6 - gx_limit))
        ey = max(0.0, abs(gy) - gy_limit) / max(1e-6, (0.6 - gy_limit))
        extreme_penalty = float(np.exp(-4.0 * (ex * ex + ey * ey)))
        geom = (eye_w / 0.08) * (lid_h / 0.02)
        geom = float(np.clip(geom, 0.2, 2.5))
        return float(np.clip(extreme_penalty * geom, 0.0, 2.5))

    def _direction_bias(self, yaw, gx, eye_side, k_yaw=0.6, k_gaze=0.4):
        if yaw is None:
            yaw = 0.0
        yaw_term = 1.0 + k_yaw * (-eye_side * yaw)
        gaze_term = 1.0 + k_gaze * (eye_side * (-gx))
        return float(np.clip(yaw_term * gaze_term, 0.2, 2.0))

    def _robust_var2(self, buf):
        if len(buf) < 6:
            return 1e-4
        arr = np.array(buf, dtype=np.float64)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        sig = 1.4826 * mad
        return float(sig[0] ** 2 + sig[1] ** 2 + 1e-6)

    def _innovation_penalty(self, gx, gy, pred, sigma=0.04):
        if pred is None:
            return 1.0
        dx = gx - pred[0]
        dy = gy - pred[1]
        r2 = dx * dx + dy * dy
        return float(np.exp(-0.5 * r2 / (sigma * sigma)))

    def _fuse_eyes(self, left_feats, right_feats, left_blink, right_blink):
        if left_feats is None and right_feats is None:
            self.last_w_left = 0.0
            self.last_w_right = 0.0
            return None
        if left_feats is None:
            gx, gy, yaw, pitch, eye_w, lid_h = right_feats
            self.last_w_left = 0.0
            self.last_w_right = 1.0
            return gx, gy, yaw, pitch, eye_w, lid_h
        if right_feats is None:
            gx, gy, yaw, pitch, eye_w, lid_h = left_feats
            self.last_w_left = 1.0
            self.last_w_right = 0.0
            return gx, gy, yaw, pitch, eye_w, lid_h

        gxL, gyL, yaw, pitch, eye_wL, lid_hL = left_feats
        gxR, gyR, _, _, eye_wR, lid_hR = right_feats

        rawL = self._eye_reliability(gxL, gyL, eye_wL, lid_hL, left_blink)
        rawR = self._eye_reliability(gxR, gyR, eye_wR, lid_hR, right_blink)
        biasL = self._direction_bias(yaw, gxL, eye_side=-1)
        biasR = self._direction_bias(yaw, gxR, eye_side=1)
        rawL *= biasL
        rawR *= biasR
        self._buf_L.append((gxL, gyL))
        self._buf_R.append((gxR, gyR))
        vL = self._robust_var2(self._buf_L)
        vR = self._robust_var2(self._buf_R)
        nL = 1.0 / vL
        nR = 1.0 / vR
        pL = self._innovation_penalty(gxL, gyL, self._pred)
        pR = self._innovation_penalty(gxR, gyR, self._pred)
        rawL = rawL * pL * nL
        rawR = rawR * pR * nR

        wsum = rawL + rawR
        if wsum < 1e-6:
            self.last_w_left = 0.0
            self.last_w_right = 0.0
            return None
        wL = rawL / wsum
        wR = rawR / wsum
        gx = wL * gxL + wR * gxR
        gy = wL * gyL + wR * gyR
        eye_w = wL * eye_wL + wR * eye_wR
        lid_h = wL * lid_hL + wR * lid_hR
        self.last_w_left = wL
        self.last_w_right = wR
        self.last_raw_sum = wsum
        self._pred = (gx, gy)
        return gx, gy, yaw, pitch, eye_w, lid_h

    def calibrate_ear_threshold(self, seconds=EAR_CALIBRATION_SECONDS):
        vals = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb, time.perf_counter() * 1000.0)
            if not results.multi_face_landmarks:
                continue
            lms = results.multi_face_landmarks[0].landmark
            ear_left = self._eye_ear_from_landmarks(lms, LEFT_EYE)
            ear_right = self._eye_ear_from_landmarks(lms, RIGHT_EYE)
            ear = float((ear_left + ear_right) * 0.5)
            vals.append(ear)
        if len(vals) >= 10:
            open_mean = float(np.median(vals))
            self.ear_closed_thresh = max(0.10, min(0.22, open_mean * 0.50))
        else:
            self.ear_closed_thresh = EAR_CLOSED_THRESH

    def process_frame(self):
        if self.training_mode == "glass_frame":
            return self._process_frame_glass_frame_basic()
        ret, frame = self.cap.read()
        if not ret:
            return None, None, True
        self._frame_index += 1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb, time.perf_counter() * 1000.0)
        if not results.multi_face_landmarks:
            self.left_eye_pupil_x = self.left_eye_pupil_y = None
            self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
            return None, None, True
        lms = results.multi_face_landmarks[0].landmark
        frame_h, frame_w = frame.shape[:2]
        try:
            self.last_face_w = float(abs(lms[33].x - lms[263].x))
            self.last_face_h = float(abs(lms[1].y - lms[152].y))
            self.last_nose_x = float(lms[1].x)
            self.last_nose_y = float(lms[1].y)
        except Exception:
            self.last_face_w = None
            self.last_face_h = None
            self.last_nose_x = None
            self.last_nose_y = None
        if USE_HEAD_POSE_PNP:
            pose = self._estimate_head_pose(lms, frame_w, frame_h)
            if pose is not None:
                yaw_p, pitch_p, roll_p, rmat = pose
                self._last_head_pose = (yaw_p, pitch_p, roll_p)
                self._last_head_rmat = rmat
            else:
                self._last_head_pose = None
                self._last_head_rmat = None
        else:
            self._last_head_pose = None
            self._last_head_rmat = None
        ear_left = self._eye_ear_from_landmarks(lms, LEFT_EYE)
        ear_right = self._eye_ear_from_landmarks(lms, RIGHT_EYE)
        left_blink = ear_left < self.ear_closed_thresh
        right_blink = ear_right < self.ear_closed_thresh
        if left_blink and right_blink:
            self._left_eye_was_closed = True
            self.left_eye_pupil_x = self.left_eye_pupil_y = None
            self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
            return None, None, True
        self._left_eye_was_closed = False
        left_feats = self._left_eye_features(lms) if not left_blink else None
        right_feats = self._right_eye_features(lms) if not right_blink else None
        self.last_left_gx = left_feats[0] if left_feats else None
        self.last_left_gy = left_feats[1] if left_feats else None
        self.last_left_eye_w = left_feats[4] if left_feats else None
        self.last_left_lid_h = left_feats[5] if left_feats else None
        self.last_right_gx = right_feats[0] if right_feats else None
        self.last_right_gy = right_feats[1] if right_feats else None
        self.last_right_eye_w = right_feats[4] if right_feats else None
        self.last_right_lid_h = right_feats[5] if right_feats else None
        fused = self._fuse_eyes(left_feats, right_feats, left_blink, right_blink)
        if fused is None:
            return None, None, False
        gx, gy, yaw, pitch, eye_w, lid_h = fused
        iris = self._iris_center(lms, LEFT_EYE["iris"])
        self.left_eye_pupil_x = int(iris[0] * frame.shape[1])
        self.left_eye_pupil_y = int(iris[1] * frame.shape[0])
        if eye_w < 0.012 or lid_h < 0.004:
            return None, None, False
        self._last_gx, self._last_gy = gx, gy
        self.last_yaw = yaw
        self.last_pitch = pitch
        self.left_eye_pupil_x_norm = gx
        self.left_eye_pupil_y_norm = gy
        return gx, gy, False

    # Must match Calibration.FEATURE_COLUMNS length and order (single source of truth for ML).
    BINOCULAR_FV_DIM = 16

    def process_frame_binocular(self):
        """
        Returns:
          gx, gy, yaw, pitch, fv(16,), conf, is_blink
        fv = [gxL, gyL, gxR, gyR, yaw_c, pitch_c, iod, roll_c,
              lid_hL, lid_hR, face_w, face_h, nx, ny, wL, wR]
        - yaw_c, pitch_c, roll_c: clamped to HEAD_*_CLIP (degrees); match calibration for ML.
        - nx, ny: (nose_x - 0.5) / iod_safe, (nose_y - 0.5) / iod_safe.
        - wL, wR: fusion weights from _fuse_eyes.
        Same format as training; do not change without retraining.
        conf in [0..1].
        """
        if self.training_mode == "glass_frame":
            return self._process_frame_binocular_glass_frame()
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None, None, 0.0, True
        self._frame_index += 1
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb, time.perf_counter() * 1000.0)
        if not results.multi_face_landmarks:
            return None, None, None, None, None, 0.0, True

        lms = results.multi_face_landmarks[0].landmark
        try:
            self.last_face_w = float(abs(lms[33].x - lms[263].x))
            self.last_face_h = float(abs(lms[1].y - lms[152].y))
            self.last_nose_x = float(lms[1].x)
            self.last_nose_y = float(lms[1].y)
        except Exception:
            self.last_face_w = None
            self.last_face_h = None
            self.last_nose_x = None
            self.last_nose_y = None
        if USE_HEAD_POSE_PNP:
            pose = self._estimate_head_pose(lms, frame_w, frame_h)
            if pose is not None:
                yaw_p, pitch_p, roll_p, rmat = pose
                self._last_head_pose = (yaw_p, pitch_p, roll_p)
                self._last_head_rmat = rmat
            else:
                self._last_head_pose = None
                self._last_head_rmat = None
        else:
            self._last_head_pose = None
            self._last_head_rmat = None
        ear_left = self._eye_ear_from_landmarks(lms, LEFT_EYE)
        ear_right = self._eye_ear_from_landmarks(lms, RIGHT_EYE)
        left_blink = ear_left < self.ear_closed_thresh
        right_blink = ear_right < self.ear_closed_thresh
        is_blink = left_blink and right_blink
        if is_blink:
            self.last_w_left = 0.0
            self.last_w_right = 0.0
            return None, None, None, None, None, 0.0, True

        left_feats = self._left_eye_features(lms) if not left_blink else None
        right_feats = self._right_eye_features(lms) if not right_blink else None

        self.last_left_gx = left_feats[0] if left_feats else None
        self.last_left_gy = left_feats[1] if left_feats else None
        self.last_left_eye_w = left_feats[4] if left_feats else None
        self.last_left_lid_h = left_feats[5] if left_feats else None
        self.last_right_gx = right_feats[0] if right_feats else None
        self.last_right_gy = right_feats[1] if right_feats else None
        self.last_right_eye_w = right_feats[4] if right_feats else None
        self.last_right_lid_h = right_feats[5] if right_feats else None

        fused = self._fuse_eyes(left_feats, right_feats, left_blink, right_blink)
        if fused is None:
            # Fallback: use available eye(s) to avoid dropping all samples.
            if left_feats is None and right_feats is None:
                return None, None, None, None, None, 0.0, False
            if left_feats is None:
                gx, gy, yaw, pitch, _, _ = right_feats
                self.last_w_left = 0.0
                self.last_w_right = 1.0
                self.last_raw_sum = 1.0
            elif right_feats is None:
                gx, gy, yaw, pitch, _, _ = left_feats
                self.last_w_left = 1.0
                self.last_w_right = 0.0
                self.last_raw_sum = 1.0
            else:
                gx = float((left_feats[0] + right_feats[0]) * 0.5)
                gy = float((left_feats[1] + right_feats[1]) * 0.5)
                yaw, pitch = left_feats[2], left_feats[3]
                self.last_w_left = 0.5
                self.last_w_right = 0.5
                self.last_raw_sum = 0.5
        else:
            gx, gy, yaw, pitch, _, _ = fused
        iod = self._iod_proxy(lms)
        if USE_HEAD_POSE_PNP and self._last_head_pose is not None:
            yaw, pitch, roll = self._last_head_pose
        else:
            roll = self._roll_proxy(lms)
        hp = np.array([yaw, pitch, iod, roll], dtype=np.float64)
        if self._hp_ema is None:
            self._hp_ema = hp
        else:
            self._hp_ema = 0.4 * hp + 0.6 * self._hp_ema  # more responsive than 0.25/0.75
        yaw, pitch, iod, roll = [float(x) for x in self._hp_ema]
        self.last_iod = iod
        self.last_roll = roll
        self.last_yaw = float(yaw)
        self.last_pitch = float(pitch)

        gxL = self.last_left_gx if self.last_left_gx is not None else gx
        gyL = self.last_left_gy if self.last_left_gy is not None else gy
        gxR = self.last_right_gx if self.last_right_gx is not None else gx
        gyR = self.last_right_gy if self.last_right_gy is not None else gy

        lid_hL = self.last_left_lid_h if self.last_left_lid_h is not None else 0.0
        lid_hR = self.last_right_lid_h if self.last_right_lid_h is not None else 0.0
        face_w = self.last_face_w if self.last_face_w is not None else 0.0
        face_h = self.last_face_h if self.last_face_h is not None else 0.0
        nose_x = self.last_nose_x if self.last_nose_x is not None else 0.5
        nose_y = self.last_nose_y if self.last_nose_y is not None else 0.5
        iod_safe = max(float(iod), 1e-4)
        nx = (float(nose_x) - 0.5) / iod_safe
        ny = (float(nose_y) - 0.5) / iod_safe
        yaw_c = float(np.clip(yaw, -HEAD_YAW_CLIP, HEAD_YAW_CLIP))
        pitch_c = float(np.clip(pitch, -HEAD_PITCH_CLIP, HEAD_PITCH_CLIP))
        roll_c = float(np.clip(roll, -HEAD_ROLL_CLIP, HEAD_ROLL_CLIP))
        wL = float(getattr(self, "last_w_left", 0.5))
        wR = float(getattr(self, "last_w_right", 0.5))
        fv = np.array(
            [
                gxL, gyL, gxR, gyR,
                yaw_c, pitch_c, float(iod), roll_c,
                float(lid_hL), float(lid_hR), float(face_w), float(face_h),
                nx, ny, wL, wR,
            ],
            dtype=np.float64,
        )
        assert fv.shape == (self.BINOCULAR_FV_DIM,), "fv must match training dimension"
        raw_sum = float(getattr(self, "last_raw_sum", 0.0))
        conf = float(np.clip(np.tanh(raw_sum), 0.0, 1.0))
        return float(gx), float(gy), float(yaw), float(pitch), fv, conf, is_blink

    def crop_left_eye_region(self):
        if not self.cap.isOpened():
            print(f"Error: Could not open camera (ID: {self.front_camera_id})")
            raise ValueError(f"Could not open camera (ID: {self.front_camera_id})")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb, time.perf_counter() * 1000.0)
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                roi, _, (x0, y0) = self._crop_and_mask_eye(frame, lms, LEFT_EYE)
                target_w, target_h, display_max_h = self._dynamic_crop_size(
                    roi.shape[1], roi.shape[0]
                )
                roi_resized = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_AREA)
                iris = self._iris_center(lms, LEFT_EYE["iris"])
                ix = int(iris[0] * frame_w)
                iy = int(iris[1] * frame_h)
                cv2.circle(frame, (ix, iy), 3, (0, 255, 0), 1)
                if x0 <= ix < x0 + roi.shape[1] and y0 <= iy < y0 + roi.shape[0]:
                    rx = int((ix - x0) * (target_w / roi.shape[1]))
                    ry = int((iy - y0) * (target_h / roi.shape[0]))
                    cv2.circle(roi_resized, (rx, ry), 3, (0, 255, 0), 1)
                feats = self._left_eye_features(lms)
                if feats is not None:
                    gx, gy, yaw, pitch, _, _ = feats
                    self.left_eye_pupil_x_norm = gx
                    self.left_eye_pupil_y_norm = gy
                    cv2.putText(
                        roi_resized,
                        f"gx={gx:.3f} gy={gy:.3f}",
                        (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                display_crop = self._fit_to_frame_ratio(
                    roi_resized, frame_w, frame_h, max_height=display_max_h
                )
                cv2.imshow("Left Eye Region", display_crop)
            else:
                display_crop = self._fit_to_frame_ratio(
                    np.zeros((1, 1, 3), dtype=np.uint8), frame_w, frame_h
                )
                h, w = display_crop.shape[0], display_crop.shape[1]
                cv2.putText(
                    display_crop,
                    "No face detected",
                    (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("Left Eye Region", display_crop)
                # No threshold view when using iris landmarks.
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        if self.back_cap is not None:
            self.back_cap.release()
        cv2.destroyAllWindows()
