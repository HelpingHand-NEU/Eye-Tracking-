import os
import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

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
HEAD_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]  # nose, chin, left/right eye, left/right mouth
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
USE_HEAD_POSE_NORM = True
NOSE_TIP = 1

# Eye Aspect Ratio: below this the eye is considered closed (blink)
EAR_CLOSED_THRESH = 0.18
EAR_CALIBRATION_SECONDS = 2.0

DISPLAY_HEIGHT_BASE = 350
DISPLAY_HEIGHT_MAX = 600


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
    def __init__(self, front_camera_id=0, back_camera_id=1, reset=False):
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "face_landmarker.task")
        self.face_mesh = _FaceMeshCompat(model_path=model_path, num_faces=1)

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
        return 0.5 * (ang_l + ang_r)

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
        if lid_h < 1e-4:
            return None
        iris = self._iris_center(lms, eye["iris"])
        u = eye_vec / eye_w
        # Force consistent axes across eyes.
        if u[0] < 0:
            u = -u
        v = np.array([-u[1], u[0]], dtype=np.float32)
        if v[1] > 0:
            v = -v
        center = (left + right) * 0.5
        gx = float(np.dot(iris - center, u) / eye_w)
        gy = float(np.dot(iris - center, v) / max(lid_h, 1e-4))
        gx, gy = self._normalize_gaze_by_head(gx, gy)
        if not np.isfinite(gx) or not np.isfinite(gy):
            return None
        if gx < -1.5 or gx > 1.5 or gy < -1.5 or gy > 1.5:
            return None
        if self._last_head_pose is not None:
            yaw, pitch, _ = self._last_head_pose
        else:
            yaw, pitch = self._pose_proxies(lms)
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
        # Temporal innovation penalty (disabled for now until axis consistency verified)
        pL = 1.0
        pR = 1.0
        self._buf_L.append((gxL, gyL))
        self._buf_R.append((gxR, gyR))
        vL = self._robust_var2(self._buf_L)
        vR = self._robust_var2(self._buf_R)
        nL = 1.0 / vL
        nR = 1.0 / vR
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
        if eye_w < 0.015 or lid_h < 0.005:
                return None, None, False
        self._last_gx, self._last_gy = gx, gy
        self.last_yaw = yaw
        self.last_pitch = pitch
        self.left_eye_pupil_x_norm = gx
        self.left_eye_pupil_y_norm = gy
        return gx, gy, False

    def process_frame_binocular(self):
        """
        Returns:
          gx, gy, yaw, pitch, fv(6,), conf, is_blink
        fv = [gxL, gyL, gxR, gyR, yaw, pitch]
        conf in [0..1]
        """
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
            elif right_feats is None:
                gx, gy, yaw, pitch, _, _ = left_feats
                self.last_w_left = 1.0
                self.last_w_right = 0.0
            else:
                gx = float((left_feats[0] + right_feats[0]) * 0.5)
                gy = float((left_feats[1] + right_feats[1]) * 0.5)
                yaw, pitch = left_feats[2], left_feats[3]
                self.last_w_left = 0.5
                self.last_w_right = 0.5
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
            self._hp_ema = 0.25 * hp + 0.75 * self._hp_ema
        yaw, pitch, iod, roll = [float(x) for x in self._hp_ema]
        self.last_iod = iod
        self.last_roll = roll
        self.last_yaw = float(yaw)
        self.last_pitch = float(pitch)

        gxL = self.last_left_gx if self.last_left_gx is not None else gx
        gyL = self.last_left_gy if self.last_left_gy is not None else gy
        gxR = self.last_right_gx if self.last_right_gx is not None else gx
        gyR = self.last_right_gy if self.last_right_gy is not None else gy

        fv = np.array([gxL, gyL, gxR, gyR, float(yaw), float(pitch), float(iod), float(roll)], dtype=np.float64)
        wL = float(getattr(self, "last_w_left", 0.0))
        wR = float(getattr(self, "last_w_right", 0.0))
        conf = float(np.clip(wL + wR, 0.0, 1.0))
        return float(gx), float(gy), float(yaw), float(pitch), fv, conf, False

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
    
