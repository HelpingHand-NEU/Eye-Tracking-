"""
Calibration and eye-controlled cursor for left-eye gaze.
20 dots on screen; calibrate by looking at each for 1s (no blink); then cursor follows gaze.
"""
import os
import sys
import csv
import cv2
import numpy as np
import time
import math
import json
from collections import deque

# Minimum valid samples per dot (pupil_x/crop_w, pupil_y/crop_h) so fit isn't degenerate
MIN_SAMPLES_PER_DOT = 12
MIN_SAMPLES_FALLBACK = 8
MIN_BINOC_DOTS = 8
MAX_DOT_ATTEMPTS = None  # unused; calibration now loops until a dot passes
USE_POLYNOMIAL = True  # affine alone is insufficient; real gaze->screen is at least quadratic
FORCE_DISABLE_POLY = False
USE_BINOCULAR_RIDGE = True
USE_HEAD_COMP = True
USE_POSE_MODEL = False
ALLOW_HEAD_MOTION_TRAINING = True
DISABLE_DEADZONE_DURING_TUNING = True  # set False once mapping is good
CALIB_IGNORE_BLINK = False
TRAIN_MIN_LID_H = 0.006
TRAIN_MIN_EYE_W = 0.015
TRAIN_MIN_CONF = 0.1
BLINK_LID_H_MIN = 0.010
BLINK_EYE_W_MIN = 0.02
BLINK_LID_RATIO = 0.20
BLINK_HOLD_SEC = 0.18
BLINK_RESUME_FRAMES = 3
USE_ML_ONLY = True
# Head-pose clip range (degrees) for ML features; wider = better with moving head (retrain after changing).
HEAD_YAW_CLIP = 30.0
HEAD_PITCH_CLIP = 25.0
HEAD_ROLL_CLIP = 30.0
# Polynomial degree for gaze ML; 3 gives more capacity for head-motion / nonlinear mapping.
ML_POLY_DEGREE = 3


def _get_screen_size():
    """Real screen size; avoid hardcoded 1920x1080 on other displays."""
    if sys.platform == "darwin":
        try:
            from Quartz import CGMainDisplayID, CGDisplayPixelsWide, CGDisplayPixelsHigh
            did = CGMainDisplayID()
            return int(CGDisplayPixelsWide(did)), int(CGDisplayPixelsHigh(did))
        except Exception:
            pass
    try:
        import pyautogui
        w, h = pyautogui.size()
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            from ctypes import windll
            w = windll.user32.GetSystemMetrics(0)
            h = windll.user32.GetSystemMetrics(1)
            return w, h
        except Exception:
            pass
    return 1920, 1080


def _training_suffix(training_mode):
    if training_mode == "webcam":
        return "webcam"
    if training_mode == "glass_frame":
        return "glassframe"
    return None


def _normalize_training_name(name, training_mode):
    if not name:
        return None
    if name.endswith(".csv"):
        name = name[:-4]
    suffix = _training_suffix(training_mode)
    if suffix and not name.endswith(f"_{suffix}"):
        name = f"{name}_{suffix}"
    return name


class TargetProvider:
    """
    Provides calibration targets and optional neighbor relationships.
    External camera/tag-based calibration can implement this interface.
    """

    def get_targets(self, screen_w, screen_h):
        raise NotImplementedError

    def get_neighbor_pairs(self, num_targets):
        return []


NUM_DOTS = 24


class ScreenDotTargetProvider(TargetProvider):
    def __init__(self, rows=4, cols=5, margin_ratio=0.05, corner_pad=20, add_corners=True, add_center=True):
        self.rows = rows
        self.cols = cols
        self.margin_ratio = margin_ratio
        self.corner_pad = corner_pad
        self.add_corners = add_corners
        self.add_center = add_center

    def get_targets(self, screen_w, screen_h):
        margin_x = screen_w * self.margin_ratio
        margin_y = screen_h * self.margin_ratio
        usable_w = screen_w - 2 * margin_x
        usable_h = screen_h - 2 * margin_y
        positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                x = margin_x + (col + 0.5) * (usable_w / self.cols)
                y = margin_y + (row + 0.5) * (usable_h / self.rows)
                positions.append((int(x), int(y)))
        # If a true center is desired, add it (do not replace a grid dot).
        if self.add_center:
            positions.append((int(screen_w // 2), int(screen_h // 2)))
        if self.add_corners:
            cp = int(self.corner_pad)
            positions.extend(
                [
                    (cp, cp),
                    (screen_w - cp, cp),
                    (cp, screen_h - cp),
                    (screen_w - cp, screen_h - cp),
                ]
            )
        return positions

    def get_neighbor_pairs(self, num_targets):
        pairs = []
        grid_count = self.rows * self.cols
        for i in range(min(num_targets, grid_count)):
            row, col = i // self.cols, i % self.cols
            if col < self.cols - 1:
                pairs.append((i, i + 1))
            if row < self.rows - 1:
                pairs.append((i, i + self.cols))
        return pairs
FIXATION_SEC = 3.5  # longer fixation to gather stable samples
SETTLE_SEC = 0.3  # ignore samples at start of fixation
NEXT_DOT_TRANSITION_SEC = 1.0  # show arrow for this long between dots; no data recorded
MIN_VALID_DOTS = 16  # require at least this many valid dots for fit
MIN_ROWS_COLS = 3    # require at least 3 distinct rows and 3 distinct columns (coverage)
# Dead zone: cap so cursor stays responsive (large dead zone = dot barely moves)
DEAD_ZONE_MULTIPLIER = 2.0
DEAD_ZONE_MAX = 0.04   # max movement threshold in normalized gaze; larger = stickier cursor
DEAD_ZONE_MIN = 0.008  # min to avoid jitter
DEAD_PX_MAX = 6.0  # cap pixel dead zone while tuning mapping stability
CURSOR_SMOOTH = 0.10   # 0=instant, 1=no move; lower = faster follow (0.25 = responsive)
CURSOR_RADIUS = 20
# When calibration.py lives in eye_tracker/, use repo root for data paths so eyetracking_ml/ and npz are at repo root
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
LOG_DIR = os.path.join(PACKAGE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
CALIBRATE_TXT = os.path.join(LOG_DIR, "calibrate.txt")  # export path for calibration data
DEBUG_OUTPUT_FILE = os.path.join(LOG_DIR, "eye_control_debug.txt")  # runtime gx, gy, mapped for first N frames
DIAGNOSIS_OUTPUT_FILE = os.path.join(LOG_DIR, "eye_control_diagnosis.txt")  # valid count, map_matrix, first ~20 debug lines
DEBUG_LINES_FOR_DIAGNOSIS = 20
MAX_JUMP = 0.12  # normalized; reject frame if gaze jump > this (outlier rejection)
MEDIAN_FILTER_LEN = 5  # number of frames for median filter on gaze
GAZE_EMA_NEW_WEIGHT = 0.35  # EMA new-sample weight (0=sticky, 1=raw)
MAX_STD_X_FOR_DOT = 0.035
MAX_STD_Y_FOR_DOT = 0.050
MAX_STD_IOD = 0.02
MAX_STD_ROLL = 6.0
MAX_STD_YAW = 8.0
MAX_STD_PITCH = 8.0
DOT_RADIUS_CALIB = 15
# Flip is applied in EyeTracker.process_frame() so calibration and runtime use same gaze coords.
FLIP_GAZE_X = True  # kept for reference; actual flip done in eye_tracker_win


class _LowPass:
    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)
        self.y = None

    def reset(self):
        self.y = None

    def filter(self, x, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if self.y is None:
            self.y = x
            return x
        self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


def _smoothing_factor(dt, cutoff_hz):
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


def _exponential_cutoff(dx, min_cutoff, beta):
    return min_cutoff + beta * abs(dx)


class OneEuroFilter1D:
    def __init__(self, min_cutoff=2.0, beta=0.10, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_lp = _LowPass()
        self.dx_lp = _LowPass()
        self.prev_x = None
        self.prev_t = None

    def reset(self):
        self.x_lp.reset()
        self.dx_lp.reset()
        self.prev_x = None
        self.prev_t = None

    def __call__(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        if self.prev_t is None:
            self.prev_t = t
            self.prev_x = x
            return self.x_lp.filter(x, alpha=1.0)

        dt = t - self.prev_t
        self.prev_t = t
        dx = 0.0 if self.prev_x is None else (x - self.prev_x) / max(dt, 1e-6)
        self.prev_x = x

        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx_hat = self.dx_lp.filter(dx, alpha=a_d)

        cutoff = _exponential_cutoff(dx_hat, self.min_cutoff, self.beta)
        a = _smoothing_factor(dt, cutoff)
        return self.x_lp.filter(x, alpha=a)


class OneEuroFilter2D:
    def __init__(self, min_cutoff=2.0, beta=0.10, d_cutoff=1.0):
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x, y, t=None):
        return self.fx(x, t), self.fy(y, t)


class GazeML:
    def __init__(self, lam=1e-1):
        self.lam = float(lam)
        self.mu = None
        self.std = None
        self.W = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.size == 0 or Y.size == 0:
            return
        self.mu = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), 1e-4)
        Xz = (X - self.mu) / self.std
        Xb = np.hstack([Xz, np.ones((Xz.shape[0], 1))])
        reg = self.lam * np.eye(Xb.shape[1])
        self.W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)

    def predict(self, x):
        if self.W is None or self.mu is None or self.std is None:
            return None
        x = np.asarray(x, dtype=np.float64)
        xz = (x - self.mu) / self.std
        xb = np.hstack([xz, 1.0])
        y = xb @ self.W
        return float(y[0]), float(y[1])

    def save(self, path=None):
        if self.W is None or self.mu is None or self.std is None:
            return
        if path is None:
            path = os.path.join(PROJECT_ROOT, "gaze_ml.npz")
        np.savez(path, mu=self.mu, std=self.std, W=self.W)

    def load(self, path=None):
        if path is None:
            path = os.path.join(PROJECT_ROOT, "gaze_ml.npz")
        d = np.load(path)
        self.mu = d["mu"]
        self.std = d["std"]
        self.W = d["W"]


class PolyRidgeRegressor:
    """
    Polynomial feature expansion + ridge regression for gaze->screen_norm.
    More accurate than plain linear ridge, still fast.
    """

    def __init__(self, lam=1e-2, degree=2):
        self.lam = float(lam)
        self.degree = int(degree)
        self.mu = None
        self.std = None
        self.W = None

    def _poly_expand(self, x):
        x = np.asarray(x, dtype=np.float64)
        feats = [x, np.ones(1)]
        if self.degree >= 2:
            D = x.shape[0]
            quad = []
            for i in range(D):
                quad.append(x[i] * x[i])
            for i in range(D):
                for j in range(i + 1, D):
                    quad.append(x[i] * x[j])
            feats.append(np.array(quad, dtype=np.float64))
        return np.concatenate(feats)

    def _design(self, X):
        return np.stack([self._poly_expand(row) for row in X], axis=0)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if len(X) < 10:
            return False
        self.input_dim = int(X.shape[1])
        self.mu = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), 1e-4)
        Xz = (X - self.mu) / self.std
        Phi = self._design(Xz)
        reg = self.lam * np.eye(Phi.shape[1])
        self.W = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y)
        return True

    def predict(self, x):
        if self.W is None or self.mu is None or self.std is None:
            return None
        x = np.asarray(x, dtype=np.float64)
        if getattr(self, "input_dim", None) is not None and len(x) != self.input_dim:
            x = x[:self.input_dim]
        xz = (x - self.mu) / self.std
        phi = self._poly_expand(xz)
        y = phi @ self.W
        return float(y[0]), float(y[1])

    def save(self, path, extra=None):
        if self.W is None:
            return
        d = {
            "mu": self.mu, "std": self.std, "W": self.W,
            "degree": np.array(self.degree),
            "input_dim": np.array(getattr(self, "input_dim", len(self.mu))),
            "feature_version": np.array(2),
        }
        if extra:
            d.update(extra)
        np.savez(path, **d)

    def load(self, path):
        data = np.load(path)
        self.mu = data["mu"]
        self.std = data["std"]
        self.W = data["W"]
        self.degree = int(data.get("degree", 2))
        self.input_dim = int(data.get("input_dim", len(self.mu)))
        if self.input_dim != 16:
            import warnings
            warnings.warn(
                f"ML model input_dim={self.input_dim}; runtime expects 16-D. Retrain for correct gaze.",
                UserWarning,
                stacklevel=2,
            )


class Kalman2D:
    """
    Constant-velocity Kalman filter for cursor smoothing.
    State: [x, y, vx, vy]
    """

    def __init__(self, q=200.0, r=1200.0):
        self.q = float(q)
        self.r = float(r)
        self.x = None
        self.P = None
        self.t_prev = None

    def reset(self):
        self.x = None
        self.P = None
        self.t_prev = None

    def update(self, meas_x, meas_y, t=None, meas_noise=None):
        if t is None:
            t = time.perf_counter()
        z = np.array([[meas_x], [meas_y]], dtype=np.float64)
        if self.x is None:
            self.x = np.array([[meas_x], [meas_y], [0.0], [0.0]], dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64) * 1e3
            self.t_prev = t
            return meas_x, meas_y
        dt = max(1e-3, min(0.05, t - self.t_prev))
        self.t_prev = t
        F = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float64,
        )
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = self.q * np.array(
            [[dt4 / 4, 0, dt3 / 2, 0],
             [0, dt4 / 4, 0, dt3 / 2],
             [dt3 / 2, 0, dt2, 0],
             [0, dt3 / 2, 0, dt2]],
            dtype=np.float64,
        )
        r = self.r if meas_noise is None else float(meas_noise)
        R = np.eye(2, dtype=np.float64) * r
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ H) @ P_pred
        return float(self.x[0, 0]), float(self.x[1, 0])


class _FixationGate:
    def __init__(self, vel_thresh_px_s=800.0, window=6):
        self.vel_thresh = float(vel_thresh_px_s)
        self.buf = deque(maxlen=window)

    def reset(self):
        self.buf.clear()

    def update(self, x, y, t):
        self.buf.append((t, x, y))
        if len(self.buf) < 2:
            return True
        t0, x0, y0 = self.buf[0]
        t1, x1, y1 = self.buf[-1]
        dt = max(1e-6, t1 - t0)
        v = math.hypot(x1 - x0, y1 - y0) / dt
        return v < self.vel_thresh


def _huber_weights(residuals, delta):
    w = np.ones_like(residuals, dtype=np.float64)
    mask = residuals > delta
    w[mask] = delta / (residuals[mask] + 1e-9)
    return w


def _fit_affine_irls_huber(gxgy, xy, base_w=None, iters=12, delta=25.0, lam=1e-6):
    gxgy = np.asarray(gxgy, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    n = gxgy.shape[0]
    X = np.hstack([gxgy, np.ones((n, 1), dtype=np.float64)])
    if base_w is None:
        base_w = np.ones((n,), dtype=np.float64)
    else:
        base_w = np.asarray(base_w, dtype=np.float64)

    def wls_fit(y, w):
        W = w[:, None]
        XtW = X.T * W.T
        A = XtW @ X + lam * np.eye(3)
        b = XtW @ y
        return np.linalg.solve(A, b)

    ax = wls_fit(xy[:, 0], base_w)
    ay = wls_fit(xy[:, 1], base_w)
    for _ in range(iters):
        pred_x = X @ ax
        pred_y = X @ ay
        rx = np.abs(pred_x - xy[:, 0])
        ry = np.abs(pred_y - xy[:, 1])
        wx = _huber_weights(rx, delta)
        wy = _huber_weights(ry, delta)
        ax = wls_fit(xy[:, 0], base_w * wx)
        ay = wls_fit(xy[:, 1], base_w * wy)
    return np.array([[ax[0], ax[1], ax[2]], [ay[0], ay[1], ay[2]]], dtype=np.float64)


def _fit_ridge_linear(F, Y, lam=1e-2):
    FtF = F.T @ F
    reg = lam * np.eye(F.shape[1])
    W = np.linalg.solve(FtF + reg, F.T @ Y)
    return W


def _fit_binocular_ridge(feature_vecs, screen_xy, lam=1e-1):
    F = np.asarray(feature_vecs, dtype=np.float64)
    Y = np.asarray(screen_xy, dtype=np.float64)
    mu = F.mean(axis=0)
    std = np.maximum(F.std(axis=0), 1e-4)
    Fz = (F - mu) / std
    X = np.hstack([Fz, np.ones((Fz.shape[0], 1), dtype=np.float64)])
    XtX = X.T @ X
    reg = lam * np.eye(X.shape[1])
    W = np.linalg.solve(XtX + reg, X.T @ Y)
    return mu, std, W


def _remove_outliers_mad(feature_vecs, screen_targets, k_mad=2.5):
    """Remove samples whose target (screen_x_norm, screen_y_norm) is beyond k_mad * MAD from median.
    Reduces noisy data from fixation drift or mis-labels. Returns (fv_keep, tgt_keep, n_dropped)."""
    if not feature_vecs or not screen_targets or len(feature_vecs) != len(screen_targets):
        return feature_vecs, screen_targets, 0
    F = np.asarray(feature_vecs, dtype=np.float64)
    Y = np.asarray(screen_targets, dtype=np.float64)
    med = np.median(Y, axis=0)
    mad = np.median(np.abs(Y - med), axis=0)
    mad = np.maximum(mad, 1e-6)
    keep = np.all(np.abs(Y - med) <= k_mad * mad, axis=1)
    n_drop = int(np.sum(~keep))
    if n_drop == 0:
        return feature_vecs, screen_targets, 0
    F = F[keep]
    Y = Y[keep]
    return [f.tolist() for f in F], [tuple(y) for y in Y], n_drop


def _train_val_split(feature_vecs, screen_targets, val_fraction=0.15, seed=42):
    """Split into train/val. val_fraction in (0,1). Returns (train_fv, train_tgt, val_fv, val_tgt)."""
    n = len(feature_vecs)
    if n < 20 or val_fraction <= 0 or val_fraction >= 1:
        return feature_vecs, screen_targets, [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    train_fv = [feature_vecs[i] for i in train_idx]
    train_tgt = [screen_targets[i] for i in train_idx]
    val_fv = [feature_vecs[i] for i in val_idx]
    val_tgt = [screen_targets[i] for i in val_idx]
    return train_fv, train_tgt, val_fv, val_tgt


def _training_row_quality(gx, gy, lid_hL, lid_hR, eye_w, conf=None, iod_std=0.0, roll_std=0.0):
    """Quality score for a training row (higher is better). Used to filter bad samples."""
    q = 1.0
    lid_min = min(float(lid_hL or 0), float(lid_hR or 0))
    if lid_min and lid_min < 0.009:
        q *= 0.3
    if eye_w is not None and float(eye_w) < 0.02:
        q *= 0.5
    if conf is not None and float(conf) < 0.25:
        q *= 0.4
    if abs(float(gx or 0)) > 0.38 or abs(float(gy or 0)) > 0.30:
        q *= 0.6
    if iod_std > 0.02:
        q *= 0.6
    if roll_std > 6.0:
        q *= 0.6
    return q


class Calibration:
    """
    20 dots at fixed positions on screen. Fullscreen. Calibrate by 1s fixation per dot
    (no blink); record pupil normalized by crop size; compute dead zone from fixation std.
    """

    # Order and semantics must match EyeTracker.process_frame_binocular() fv exactly (16-D).
    # nose_x/nose_y in training CSV = nx, ny = (nose - 0.5) / iod_safe; yaw/pitch = clamped degrees.
    FEATURE_COLUMNS = [
        "gxL",
        "gyL",
        "gxR",
        "gyR",
        "yaw",
        "pitch",
        "iod",
        "roll",
        "lid_hL",
        "lid_hR",
        "face_w",
        "face_h",
        "nose_x",
        "nose_y",
        "wL",
        "wR",
    ]
    FEATURE_DIM = len(FEATURE_COLUMNS)  # 16; single source of truth for ML/training
    NORM_TARGET_COLUMNS = ["screen_x_norm", "screen_y_norm"]
    TRAINING_COLUMNS = [
        "timestamp",
        "dot_index",
        "stage",
        "screen_x",
        "screen_y",
        "screen_x_norm",
        "screen_y_norm",
        "screen_w",
        "screen_h",
        "gx",
        "gy",
        "yaw",
        "pitch",
        "gxL",
        "gyL",
        "gxR",
        "gyR",
        "iod",
        "roll",
        "lid_hL",
        "lid_hR",
        "face_w",
        "face_h",
        "nose_x",
        "nose_y",
        "wL",
        "wR",
        "quality",
        "eye_w",
        "lid_h",
        "conf",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "label_x",
        "label_y",
        "label_x_norm",
        "label_y_norm",
    ]
    # Only used for glass_frame training; not written in webcam CSVs
    GLASS_FRAME_EXTRA_COLUMNS = [
        "tag_id",
        "board_distance_m",
        "apriltag_length_m",
        "apriltag_width_m",
        "tag_cam_x",
        "tag_cam_y",
        "tag_bbox_w_px",
        "tag_bbox_h_px",
        "tag_center_x",
        "tag_center_y",
        "tag_c0_x", "tag_c0_y", "tag_c1_x", "tag_c1_y", "tag_c2_x", "tag_c2_y", "tag_c3_x", "tag_c3_y",
        "pupil_L_x_norm",
        "pupil_L_y_norm",
        "pupil_R_x_norm",
        "pupil_R_y_norm",
        "pupil_L_x_full",
        "pupil_L_y_full",
        "pupil_R_x_full",
        "pupil_R_y_full",
    ]

    @property
    def training_columns(self):
        """Columns to write in training CSV; glass_frame adds tag_id and board_distance_m."""
        if self.training_mode == "glass_frame":
            return list(self.TRAINING_COLUMNS) + list(self.GLASS_FRAME_EXTRA_COLUMNS)
        return list(self.TRAINING_COLUMNS)

    def __init__(
        self,
        target_provider=None,
        training_data_dir="eyetracking_ml",
        training_data_name=None,
        screen_size=None,
        fullscreen=True,
        window_name="calibration",
        training_mode="webcam",
        roi_signature=None,
    ):
        if training_mode not in ("webcam", "glass_frame"):
            raise ValueError("training_mode must be 'webcam' or 'glass_frame'.")
        self._roi_signature = roi_signature  # 8-char hex; when set, CSV/npz paths include _roi_<sig>
        self._training_path_override = None  # when set, use this CSV (and derived npz) for load/save
        if screen_size is None:
            self.screen_w, self.screen_h = _get_screen_size()
        else:
            self.screen_w, self.screen_h = int(screen_size[0]), int(screen_size[1])
        self._target_provider = target_provider or ScreenDotTargetProvider()
        self.targets = self._target_provider.get_targets(self.screen_w, self.screen_h)
        self.num_targets = len(self.targets)
        # After calibrate(): mapping from (gaze_x_norm, gaze_y_norm) -> (screen_x, screen_y)
        self.map_matrix = None  # 2x3 affine or (2,2) + (2,) for linear
        self.dead_zone = None   # scalar or (dx, dy); don't move cursor if change smaller
        self.calibrated = False
        # Per-dot: list of (gaze_x, gaze_y) means; std; sample counts
        self._gaze_means = []
        self._gaze_stds_x = []
        self._gaze_stds_y = []
        self._sample_counts = []
        # Polynomial mapping (2nd order, standard in eye tracking) when 6+ points
        self._poly_coeffs_x = None  # (6,) for 1, gx, gy, gx^2, gx*gy, gy^2
        self._poly_coeffs_y = None
        self._use_polynomial = False  # set True when polynomial is stable; use affine until then
        self._samples_per_dot = []   # list of lists of (gx,gy) for dead zone computation
        self.dead_px = 12.0          # dead zone in screen pixels; computed after calibration
        self._pose_means = []
        self._pose_stds = []
        self.W_pose = None
        self.W_map = None
        self.feature_mu = None
        self.feature_std = None
        self.binoc_mu = None
        self.binoc_std = None
        self.binoc_W = None
        self.src_mu = None
        self.src_std = None
        self.ml = None
        self.training_data_dir = training_data_dir
        self.training_mode = training_mode
        self.training_data_name = _normalize_training_name(training_data_name, self.training_mode)
        self.training_data_path = self._resolve_training_path(self.training_data_name, training_data_dir)
        # Expose effective paths for fallback / reuse (see current_training_data_path, current_roi_signature)
        self.fullscreen = bool(fullscreen)
        self.window_name = window_name
        if self.training_data_path is not None:
            os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
            self._maybe_rename_unsuffixed()
            self._migrate_training_file()

    @property
    def current_training_data_path(self):
        """Path to the CSV file in use (override or default). Exposed for fallback/reuse."""
        return self._training_path_override if self._training_path_override else self.training_data_path

    @property
    def current_roi_signature(self):
        """ROI signature for the current training set (8-char hex), or from override path. None if not ROI-based."""
        if self._training_path_override:
            base = os.path.basename(self._training_path_override)
            if "_roi_" in base and base.endswith(".csv"):
                sig = base.split("_roi_")[-1].replace(".csv", "").strip()
                if sig:
                    return sig
        return getattr(self, "_roi_signature", None)

    def set_training_override(self, csv_path: str | None):
        """Use a specific training CSV (and its matching npz) for load/save. Pass None to clear. For fallback/reuse."""
        self._training_path_override = csv_path

    def _migrate_training_file(self):
        if self.training_data_path is None or not os.path.exists(self.training_data_path):
            return
        try:
            with open(self.training_data_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                rows = list(reader)
        except Exception:
            return
        needs_norm = not all(col in fieldnames for col in self.NORM_TARGET_COLUMNS)
        needs_header_update = not all(col in fieldnames for col in self.training_columns)
        if not needs_norm and not needs_header_update:
            return
        legacy_path = self.training_data_path.replace(".csv", "_legacy.csv")
        try:
            if not os.path.exists(legacy_path):
                os.replace(self.training_data_path, legacy_path)
            else:
                os.remove(self.training_data_path)
        except Exception:
            return
        migrated = []
        for row in rows:
            try:
                sx = float(row.get("screen_x", ""))
                sy = float(row.get("screen_y", ""))
            except Exception:
                continue
            screen_w = float(row.get("screen_w", self.screen_w))
            screen_h = float(row.get("screen_h", self.screen_h))
            if screen_w <= 0 or screen_h <= 0:
                screen_w = float(self.screen_w)
                screen_h = float(self.screen_h)
            row["screen_w"] = screen_w
            row["screen_h"] = screen_h
            row["screen_x_norm"] = float(sx) / float(screen_w)
            row["screen_y_norm"] = float(sy) / float(screen_h)
            for col in self.training_columns:
                if col not in row:
                    row[col] = ""
            migrated.append(row)
        if migrated:
            try:
                with open(self.training_data_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.training_columns)
                    writer.writeheader()
                    writer.writerows(migrated)
            except Exception:
                return


    def _make_dots(self):
        """Deprecated: use ScreenDotTargetProvider to supply targets."""
        return ScreenDotTargetProvider().get_targets(self.screen_w, self.screen_h)

    def _resolve_training_path(self, name, base_dir):
        if not name:
            return None
        if getattr(self, "_roi_signature", None) and getattr(self, "training_mode", None) == "glass_frame":
            name = f"{name}_roi_{self._roi_signature}"
        suffix = _training_suffix(self.training_mode)
        base_dir_path = base_dir
        if base_dir_path and not os.path.isabs(base_dir_path):
            base_dir_path = os.path.join(PROJECT_ROOT, base_dir_path)
        if os.path.isabs(name) or os.sep in name:
            root, ext = os.path.splitext(name)
            if suffix and not root.endswith(f"_{suffix}"):
                root = f"{root}_{suffix}"
            return root + (ext or ".csv")
        return os.path.join(base_dir_path, f"{name}.csv")

    def _maybe_rename_unsuffixed(self):
        suffix = _training_suffix(self.training_mode)
        if not suffix or not self.training_data_path:
            return
        if os.path.exists(self.training_data_path):
            return
        base_name = self.training_data_name or ""
        suffix_tag = f"_{suffix}"
        if base_name.endswith(suffix_tag):
            base_name = base_name[: -len(suffix_tag)]
        if not base_name:
            return
        base_dir_path = self.training_data_dir
        if base_dir_path and not os.path.isabs(base_dir_path):
            base_dir_path = os.path.join(PROJECT_ROOT, base_dir_path)
        unsuffixed_path = os.path.join(base_dir_path, f"{base_name}.csv")
        if os.path.exists(unsuffixed_path):
            try:
                os.replace(unsuffixed_path, self.training_data_path)
            except Exception:
                pass

    def _ml_model_path(self):
        # If override CSV path contains _roi_<sig>, load npz for that ROI set (fallback/reuse)
        if getattr(self, "_training_path_override", None):
            base = os.path.basename(self._training_path_override)
            if "_roi_" in base and base.endswith(".csv"):
                sig = base.split("_roi_")[-1].replace(".csv", "").strip()
                if sig:
                    return os.path.join(PROJECT_ROOT, f"gaze_ml_glassframe_roi_{sig}.npz")
        suffix = _training_suffix(self.training_mode)
        if suffix:
            roi_sig = getattr(self, "_roi_signature", None)
            if roi_sig and self.training_mode == "glass_frame":
                return os.path.join(PROJECT_ROOT, f"gaze_ml_{suffix}_roi_{roi_sig}.npz")
            filename = f"gaze_ml_{suffix}.npz"
            target = os.path.join(PROJECT_ROOT, filename)
            legacy = os.path.join(PROJECT_ROOT, "gaze_ml.npz")
            if not os.path.exists(target) and os.path.exists(legacy):
                try:
                    os.replace(legacy, target)
                except Exception:
                    pass
            return target
        return os.path.join(PROJECT_ROOT, "gaze_ml.npz")

    def load_saved_ml(self):
        """Load saved PolyRidgeRegressor (or legacy GazeML) and optionally binocular ridge from npz."""
        path = self._ml_model_path()
        if not path or not os.path.isfile(path):
            return False
        try:
            data = np.load(path)
            if "degree" in data:
                self.ml = PolyRidgeRegressor(lam=5e-3, degree=int(data.get("degree", ML_POLY_DEGREE)))
                self.ml.load(path)
            else:
                self.ml = GazeML(lam=1e-1)
                self.ml.mu = data["mu"]
                self.ml.std = data["std"]
                self.ml.W = data["W"]
                self.ml.input_dim = len(self.ml.mu)
            if "binoc_mu" in data and "binoc_std" in data and "binoc_W" in data and data["binoc_mu"].size:
                self.binoc_mu = data["binoc_mu"]
                self.binoc_std = data["binoc_std"]
                self.binoc_W = data["binoc_W"]
            else:
                self.binoc_mu = None
                self.binoc_std = None
                self.binoc_W = None
            self.calibrated = True
            return True
        except Exception as e:
            print(f"Could not load saved ML from {path}: {e}")
            return False

    def _load_training_data(self):
        path = self.current_training_data_path
        if path is None or not os.path.exists(path):
            return [], []
        feature_vecs = []
        screen_targets = []
        screen_size_checked = False
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        q = float(row.get("quality", 1.0) or 1.0)
                    except Exception:
                        q = 1.0
                    if q < 0.7:
                        continue
                    try:
                        gxL = float(row.get("gxL", 0.0) or 0.0)
                        gyL = float(row.get("gyL", 0.0) or 0.0)
                        gxR = float(row.get("gxR", 0.0) or 0.0)
                        gyR = float(row.get("gyR", 0.0) or 0.0)
                        yaw = float(row.get("yaw", 0.0) or 0.0)
                        pitch = float(row.get("pitch", 0.0) or 0.0)
                        iod = float(row.get("iod", 0.0) or 0.0)
                        roll = float(row.get("roll", 0.0) or 0.0)
                        lid_hL = float(row.get("lid_hL", 0.0) or 0.0)
                        lid_hR = float(row.get("lid_hR", 0.0) or 0.0)
                        face_w = float(row.get("face_w", 0.0) or 0.0)
                        face_h = float(row.get("face_h", 0.0) or 0.0)
                        nose_x = float(row.get("nose_x", 0.5) or 0.5)
                        nose_y = float(row.get("nose_y", 0.5) or 0.5)
                        wL = float(row.get("wL", 0.5) or 0.5)
                        wR = float(row.get("wR", 0.5) or 0.5)
                        sx = float(row.get("screen_x_norm", ""))
                        sy = float(row.get("screen_y_norm", ""))
                    except Exception:
                        continue
                    iod_safe = max(iod, 1e-4)
                    nx = (nose_x - 0.5) / iod_safe
                    ny = (nose_y - 0.5) / iod_safe
                    yaw_c = float(np.clip(yaw, -HEAD_YAW_CLIP, HEAD_YAW_CLIP))
                    pitch_c = float(np.clip(pitch, -HEAD_PITCH_CLIP, HEAD_PITCH_CLIP))
                    roll_c = float(np.clip(roll, -HEAD_ROLL_CLIP, HEAD_ROLL_CLIP))
                    feats = np.array([
                        gxL, gyL, gxR, gyR,
                        yaw_c, pitch_c, iod_safe, roll_c,
                        lid_hL, lid_hR, face_w, face_h,
                        nx, ny, wL, wR,
                    ], dtype=np.float64)
                    if not np.isfinite(feats).all() or not np.isfinite([sx, sy]).all():
                        continue
                    try:
                        conf = float(row.get("conf", 1.0))
                    except Exception:
                        conf = 1.0
                    try:
                        eye_w = float(row.get("eye_w", 0.0))
                    except Exception:
                        eye_w = 0.0
                    lid_min = min(lid_hL, lid_hR) if (lid_hL > 0 or lid_hR > 0) else 0.0
                    if conf < TRAIN_MIN_CONF:
                        continue
                    if lid_min and lid_min < TRAIN_MIN_LID_H:
                        continue
                    if eye_w and eye_w < TRAIN_MIN_EYE_W:
                        continue
                    if not screen_size_checked and getattr(self, "screen_w", None) and getattr(self, "screen_h", None):
                        try:
                            rw = float(row.get("screen_w", 0) or 0)
                            rh = float(row.get("screen_h", 0) or 0)
                            if rw > 0 and rh > 0 and (abs(rw - self.screen_w) > 1 or abs(rh - self.screen_h) > 1):
                                print(f"[calibration] Warning: training CSV screen size ({rw:.0f}x{rh:.0f}) differs from current ({self.screen_w:.0f}x{self.screen_h:.0f}). Norms still valid; consider retraining on this display.")
                        except Exception:
                            pass
                        screen_size_checked = True
                    feature_vecs.append(feats)
                    screen_targets.append((sx, sy))
        except Exception:
            return [], []
        return feature_vecs, screen_targets

    def _append_training_rows(self, rows):
        """Append training rows to the calibration CSV. Only called from calibration/training flow (e.g. GlassFrameTraining), never from the test script or hover app."""
        path = self.current_training_data_path
        if path is None or not rows:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            file_exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.training_columns)
                if not file_exists or f.tell() == 0:
                    writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            print(f"Could not write training data: {e}")

    def get_dot(self, dot_index):
        """Return (x, y) screen position of dot number (0..14)."""
        if 0 <= dot_index < self.num_targets:
            return self.targets[dot_index]
        return None

    def _dot_jitter_threshold(self, dot_xy):
        """Return (tx, ty) max std for this dot; corners get +20% allowance."""
        x, y = dot_xy
        near_edge = (
            x < 0.12 * self.screen_w or x > 0.88 * self.screen_w
            or y < 0.12 * self.screen_h or y > 0.88 * self.screen_h
        )
        scale = 1.2 if near_edge else 1.0
        return MAX_STD_X_FOR_DOT * scale, MAX_STD_Y_FOR_DOT * scale

    def _neighbor_pairs(self):
        """Pairs (i, j) of neighboring dot indices in 5x4 grid (horizontal and vertical)."""
        return self._target_provider.get_neighbor_pairs(self.num_targets)

    def _refine_with_neighbors(self, M, gaze_means):
        """
        Refine affine M using neighbor consistency: for each neighboring dot pair (i,j),
        screen_delta should match M @ gaze_delta. Down-weight or drop dots that disagree.
        Re-fit with remaining points for a more consistent mapping.
        """
        if M is None or len(gaze_means) != self.num_targets:
            return M
        if self.src_mu is None or self.src_std is None:
            return M
        gaze_arr = np.array(gaze_means, dtype=np.float32)
        dots_arr = np.array(self.targets, dtype=np.float32)
        mu = self.src_mu.astype(np.float32)
        std = self.src_std.astype(np.float32)
        gaze_arr_z = (gaze_arr - mu) / std
        # Per-dot residual: how well do this dot's neighbor relationships match under M?
        residuals = np.zeros(self.num_targets)
        for i, j in self._neighbor_pairs():
            g_i = gaze_arr_z[i]
            g_j = gaze_arr_z[j]
            if np.isnan(g_i[0]) or np.isnan(g_j[0]):
                continue
            screen_delta = dots_arr[j] - dots_arr[i]
            # M @ [g, gy, 1]; delta in screen = M @ [dg, dgy, 0]
            dg = np.array([g_j[0] - g_i[0], g_j[1] - g_i[1], 0.0], dtype=np.float32)
            predicted_delta = M @ dg
            err = np.linalg.norm(predicted_delta - screen_delta)
            residuals[i] += err
            residuals[j] += err
        # Use points with lower residual (consistent with neighbors); re-fit without worst outliers
        valid_ix = [
            idx for idx in range(self.num_targets)
            if not (np.isnan(gaze_arr_z[idx, 0]) or np.isnan(gaze_arr_z[idx, 1]))
        ]
        if len(valid_ix) < 3:
            return M
        valid_ix.sort(key=lambda idx: residuals[idx])
        keep = valid_ix[: max(3, len(valid_ix) - 2)]  # drop at most 2 worst
        src = gaze_arr_z[keep]
        dst = dots_arr[keep]
        M2, _ = cv2.estimateAffine2D(src, dst)
        return M2 if M2 is not None else M

    def _show_fullscreen(self, name="cal"):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(name, self.screen_w, self.screen_h)
        return name

    def _draw_dots(self, img, highlight_index=None, cursor_xy=None, dots_visible=True):
        """Draw all dots (if dots_visible); optionally highlight one; draw cursor if cursor_xy given."""
        if dots_visible:
            for i, (x, y) in enumerate(self.targets):
                r = DOT_RADIUS_CALIB
                color = (0, 255, 255) if i == highlight_index else (180, 180, 180)
                cv2.circle(img, (x, y), r, color, 2)
        if cursor_xy is not None:
            cx, cy = int(cursor_xy[0]), int(cursor_xy[1])
            cv2.circle(img, (cx, cy), CURSOR_RADIUS, (0, 255, 0), 2)
        return img

    def calibrate(self, tracker):
        """
        Run calibration: show each of 20 dots; user looks at dot for 1s without blinking.
        Record (pupil_x/crop_w, pupil_y/crop_h) only when not blinking. Compute mean per dot
        and dead zone from std of fixation samples. Fit gaze -> screen mapping.
        """
        name = self._show_fullscreen(self.window_name)
        gaze_means = []
        gaze_stds_x = []
        gaze_stds_y = []
        pose_means = []
        target_fps = 60
        frame_dt = 1.0 / target_fps
        samples_per_dot = []
        feature_means = []
        feature_vecs = []
        screen_targets = []
        self._raw_training_rows = []

        if hasattr(tracker, "calibrate_ear_threshold"):
            tracker.calibrate_ear_threshold()

        # Brief intro: encourage head-pose diversity for better accuracy when moving head.
        intro_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        intro_img[:] = (40, 40, 40)
        cv2.putText(intro_img, "Calibration", (self.screen_w // 2 - 80, self.screen_h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(intro_img, "Look at each dot. Vary head pose slightly between dots",
                    (self.screen_w // 2 - 240, self.screen_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(intro_img, "(left/right, up/down) so the model works when you move.",
                    (self.screen_w // 2 - 240, self.screen_h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(intro_img, "Press any key to start", (self.screen_w // 2 - 120, self.screen_h // 2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.imshow(name, intro_img)
        cv2.waitKey(0)

        for dot_i in range(self.num_targets):
            success = False
            mean_x = mean_y = np.nan
            std_x = std_y = 0.02
            mean_yaw = mean_pitch = 0.0
            std_yaw = std_pitch = 0.0
            redo_notice = None
            while not success:
                samples = []
                feature_samples = []
                gaze_buf = deque(maxlen=MEDIAN_FILTER_LEN)
                blink_frames = 0
                total_frames = 0
                miss_feats = 0
                ok = 0
                t_start = time.perf_counter()
                last_draw = 0
                while True:
                    now = time.perf_counter()
                    if now - last_draw >= frame_dt:
                        img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                        img[:] = (30, 30, 30)
                        self._draw_dots(img, highlight_index=dot_i)
                        msg = f"Look at dot {dot_i + 1}/{self.num_targets} - hold {FIXATION_SEC:.1f}s, don't blink"
                        if redo_notice:
                            msg += f" ({redo_notice})"
                        cv2.putText(
                            img, msg, (self.screen_w // 2 - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                        )
                        cv2.putText(
                            img, "Vary head pose slightly between dots for better accuracy.",
                            (self.screen_w // 2 - 220, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                        )
                        cv2.imshow(name, img)
                        last_draw = now

                    if hasattr(tracker, "process_frame_binocular"):
                        gx, gy, yaw, pitch, fv, conf, is_blink = tracker.process_frame_binocular()
                    else:
                        gx, gy, is_blink = tracker.process_frame()
                        yaw = getattr(tracker, "last_yaw", 0.0)
                        pitch = getattr(tracker, "last_pitch", 0.0)
                        fv = None
                        conf = 0.5
                    if CALIB_IGNORE_BLINK:
                        is_blink = False
                    total_frames += 1
                    if is_blink:
                        blink_frames += 1
                    elapsed = time.perf_counter() - t_start
                    if elapsed < SETTLE_SEC:
                        continue
                    if (not is_blink) and gx is not None and gy is not None:
                        def _safe_float(v, default=0.0):
                            try:
                                return float(v)
                            except Exception:
                                return float(default)
                        gaze_buf.append((gx, gy))
                        gx_m = float(np.median([p[0] for p in gaze_buf]))
                        gy_m = float(np.median([p[1] for p in gaze_buf]))
                        if len(gaze_buf) >= 2:
                            pgx, pgy = gaze_buf[-2]
                            if abs(gx_m - pgx) > MAX_JUMP or abs(gy_m - pgy) > MAX_JUMP:
                                continue
                        samples.append((gx_m, gy_m, yaw, pitch))
                        ok += 1
                        if fv is None:
                            gxL = getattr(tracker, "last_left_gx", None)
                            gyL = getattr(tracker, "last_left_gy", None)
                            gxR = getattr(tracker, "last_right_gx", None)
                            gyR = getattr(tracker, "last_right_gy", None)
                            if gxL is None or gyL is None:
                                gxL, gyL = gx, gy
                            if gxR is None or gyR is None:
                                gxR, gyR = gx, gy
                            iod = getattr(tracker, "last_iod", 0.0)
                            roll = getattr(tracker, "last_roll", 0.0)
                            lid_hL = getattr(tracker, "last_left_lid_h", 0.0)
                            lid_hR = getattr(tracker, "last_right_lid_h", 0.0)
                            face_w = getattr(tracker, "last_face_w", 0.0)
                            face_h = getattr(tracker, "last_face_h", 0.0)
                            nose_x = getattr(tracker, "last_nose_x", 0.5)
                            nose_y = getattr(tracker, "last_nose_y", 0.5)
                            iod_safe = max(_safe_float(iod), 1e-4)
                            nx = (_safe_float(nose_x) - 0.5) / iod_safe
                            ny = (_safe_float(nose_y) - 0.5) / iod_safe
                            yaw_c = float(np.clip(_safe_float(yaw), -HEAD_YAW_CLIP, HEAD_YAW_CLIP))
                            pitch_c = float(np.clip(_safe_float(pitch), -HEAD_PITCH_CLIP, HEAD_PITCH_CLIP))
                            roll_c = float(np.clip(_safe_float(getattr(tracker, "last_roll", 0.0)), -HEAD_ROLL_CLIP, HEAD_ROLL_CLIP))
                            wL = float(getattr(tracker, "last_w_left", 0.5))
                            wR = float(getattr(tracker, "last_w_right", 0.5))
                            fv = np.array(
                                [
                                    gxL, gyL, gxR, gyR,
                                    yaw_c, pitch_c, iod_safe, roll_c,
                                    _safe_float(lid_hL), _safe_float(lid_hR),
                                    _safe_float(face_w), _safe_float(face_h),
                                    nx, ny, wL, wR,
                                ],
                                dtype=np.float64,
                            )
                        feature_samples.append(fv)
                        iod = getattr(tracker, "last_iod", 0.0)
                        roll = getattr(tracker, "last_roll", 0.0)
                        eye_w = max(
                            v for v in (
                                getattr(tracker, "last_left_eye_w", None),
                                getattr(tracker, "last_right_eye_w", None),
                            )
                            if v is not None
                        ) if (getattr(tracker, "last_left_eye_w", None) is not None or getattr(tracker, "last_right_eye_w", None) is not None) else None
                        lid_h = max(
                            v for v in (
                                getattr(tracker, "last_left_lid_h", None),
                                getattr(tracker, "last_right_lid_h", None),
                            )
                            if v is not None
                        ) if (getattr(tracker, "last_left_lid_h", None) is not None or getattr(tracker, "last_right_lid_h", None) is not None) else None
                        screen_x = self.targets[dot_i][0]
                        screen_y = self.targets[dot_i][1]
                        screen_x_norm = float(screen_x) / float(self.screen_w)
                        screen_y_norm = float(screen_y) / float(self.screen_h)
                        self._raw_training_rows.append(
                            {
                                "timestamp": time.time(),
                                "dot_index": dot_i,
                                "stage": "calib25",
                                "screen_x": screen_x,
                                "screen_y": screen_y,
                                "screen_x_norm": screen_x_norm,
                                "screen_y_norm": screen_y_norm,
                                "screen_w": self.screen_w,
                                "screen_h": self.screen_h,
                                "gx": gx,
                                "gy": gy,
                                "yaw": yaw,
                                "pitch": pitch,
                                "gxL": float(fv[0]),
                                "gyL": float(fv[1]),
                                "gxR": float(fv[2]),
                                "gyR": float(fv[3]),
                                "iod": _safe_float(iod),
                                "roll": _safe_float(roll),
                                "lid_hL": _safe_float(getattr(tracker, "last_left_lid_h", None)),
                                "lid_hR": _safe_float(getattr(tracker, "last_right_lid_h", None)),
                                "face_w": _safe_float(getattr(tracker, "last_face_w", None)),
                                "face_h": _safe_float(getattr(tracker, "last_face_h", None)),
                                "nose_x": _safe_float(getattr(tracker, "last_nose_x", None)),
                                "nose_y": _safe_float(getattr(tracker, "last_nose_y", None)),
                                "wL": float(getattr(tracker, "last_w_left", 0.5)),
                                "wR": float(getattr(tracker, "last_w_right", 0.5)),
                                "quality": _training_row_quality(
                                    gx, gy,
                                    getattr(tracker, "last_left_lid_h", None),
                                    getattr(tracker, "last_right_lid_h", None),
                                    eye_w,
                                    conf=conf,
                                    iod_std=0.0,
                                    roll_std=0.0,
                                ),
                                "eye_w": eye_w if eye_w is not None else "",
                                "lid_h": lid_h if lid_h is not None else "",
                                "conf": conf,
                                "bbox_x": "",
                                "bbox_y": "",
                                "bbox_w": "",
                                "bbox_h": "",
                                "label_x": "",
                                "label_y": "",
                                "label_x_norm": "",
                                "label_y_norm": "",
                            }
                        )
                    else:
                        miss_feats += 1
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cv2.destroyWindow(name)
                        return "quit"
                    if key == ord('n'):
                        cv2.destroyWindow(name)
                        return "next"
                    # Ensure we collect enough samples even at low FPS.
                    if elapsed >= FIXATION_SEC:
                        # Dynamic requirement based on observed sample rate.
                        effective_fps = len(samples) / max(1e-6, (elapsed - SETTLE_SEC))
                        needed = max(MIN_SAMPLES_FALLBACK, int(effective_fps * FIXATION_SEC * 0.4))
                        needed = min(needed, MIN_SAMPLES_PER_DOT)
                        if len(samples) >= needed:
                            break
                        if len(samples) == 0:
                            redo_notice = "redo: too few valid samples"
                    # If blink dominates, relax earlier so we can move on.
                    if elapsed >= 1.0 and total_frames > 0:
                        blink_ratio = blink_frames / max(1, total_frames)
                        if blink_ratio > 0.8 and redo_notice is None:
                            redo_notice = "redo: too many blinks"
                    # Hard stop if we still can't gather enough; avoid endless loop.
                    if elapsed >= FIXATION_SEC * 3.0 and len(samples) >= MIN_SAMPLES_FALLBACK:
                        break

                if len(samples) < MIN_SAMPLES_FALLBACK:
                    print(f"[dot {dot_i}] ok={ok} miss_feats={miss_feats} blink_frames={blink_frames} total={total_frames}")
                    print(f"Dot {dot_i + 1}: only {len(samples)} valid samples (need {MIN_SAMPLES_FALLBACK}). Redo this dot.")
                    redo_notice = f"redo: need {MIN_SAMPLES_FALLBACK}+ samples"
                    continue
                print(f"[dot {dot_i}] ok={ok} miss_feats={miss_feats} blink_frames={blink_frames} total={total_frames}")
                arr = np.array(samples, dtype=np.float64)
                mean_x, mean_y = np.median(arr[:, 0]), np.median(arr[:, 1])
                mean_yaw, mean_pitch = np.median(arr[:, 2]), np.median(arr[:, 3])
                std_x = float(np.std(arr[:, 0]))
                std_y = float(np.std(arr[:, 1]))
                std_yaw = float(np.std(arr[:, 2]))
                std_pitch = float(np.std(arr[:, 3]))
                if feature_samples and USE_HEAD_COMP:
                    farr = np.array(feature_samples, dtype=np.float64)
                    std_iod = float(np.std(farr[:, 6]))
                    std_roll = float(np.std(farr[:, 7]))
                else:
                    std_iod = 0.0
                    std_roll = 0.0
                if np.isnan(std_x) or std_x < 1e-6:
                    std_x = 0.02
                if np.isnan(std_y) or std_y < 1e-6:
                    std_y = 0.02
                tx, ty = self._dot_jitter_threshold(self.targets[dot_i])
                if std_x > tx or std_y > ty:
                    print(
                        f"Dot {dot_i + 1}: jitter too high (std_x={std_x:.3f}, std_y={std_y:.3f}). Redo this dot."
                    )
                    redo_notice = f"redo: jitter {std_x:.3f},{std_y:.3f}"
                    continue
                if USE_HEAD_COMP and not ALLOW_HEAD_MOTION_TRAINING and (std_iod > MAX_STD_IOD or std_roll > MAX_STD_ROLL):
                    print(
                        f"Dot {dot_i + 1}: head motion too high (std_iod={std_iod:.4f}, std_roll={std_roll:.4f}). Redo this dot."
                    )
                    redo_notice = "redo: head moved"
                    continue
                if USE_POSE_MODEL and not ALLOW_HEAD_MOTION_TRAINING and (std_yaw > MAX_STD_YAW or std_pitch > MAX_STD_PITCH):
                    print(
                        f"Dot {dot_i + 1}: pose drift too high (yaw={std_yaw:.3f}, pitch={std_pitch:.3f}). Redo this dot."
                    )
                    redo_notice = f"redo: pose {std_yaw:.3f},{std_pitch:.3f}"
                    continue
                success = True
                redo_notice = None
            samples_per_dot.append(samples)
            self._sample_counts.append(len(samples))
            if not success or len(samples) < 2:
                # Too few samples: don't use this dot in fit (would skew mapping)
                mean_x, mean_y = np.nan, np.nan
                std_x, std_y = 0.02, 0.02
                mean_yaw = mean_pitch = 0.0
                std_yaw = std_pitch = 0.0
            gaze_means.append((mean_x, mean_y))
            gaze_stds_x.append(std_x)
            gaze_stds_y.append(std_y)
            pose_means.append((mean_yaw, mean_pitch))
            if feature_samples:
                dot_screen_x, dot_screen_y = self.targets[dot_i][0], self.targets[dot_i][1]
                dot_sx_norm = float(dot_screen_x) / float(self.screen_w)
                dot_sy_norm = float(dot_screen_y) / float(self.screen_h)
                farr = np.array(feature_samples, dtype=np.float64)
                fv_mean = np.mean(farr, axis=0)
                feature_means.append(fv_mean)
                for fv in feature_samples:
                    feature_vecs.append(fv)
                    screen_targets.append((dot_sx_norm, dot_sy_norm))
            else:
                fallback = np.array(
                    [mean_x, mean_y, mean_x, mean_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    dtype=np.float64,
                )
                feature_means.append(fallback)

            # Before next dot: show arrow for 1s so user can move eyes; no data recorded
            if dot_i < self.num_targets - 1:
                next_xy = self.get_dot(dot_i + 1)
                t_trans = time.perf_counter()
                while (time.perf_counter() - t_trans) < NEXT_DOT_TRANSITION_SEC:
                    img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    img[:] = (30, 30, 30)
                    self._draw_dots(img, highlight_index=None)
                    # Arrow pointing toward next dot
                    cx, cy = self.screen_w // 2, self.screen_h // 2
                    if next_xy is not None:
                        nx, ny = next_xy
                        cv2.arrowedLine(img, (cx, cy), (nx, ny), (0, 255, 255), 4, tipLength=0.2)
                    cv2.putText(img, "Look at next dot", (self.screen_w // 2 - 120, cy - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img, "Vary head pose slightly for better accuracy when moving.",
                                (self.screen_w // 2 - 200, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.imshow(name, img)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        cv2.destroyWindow(name)
                        return "quit"
                    if key == ord('n'):
                        cv2.destroyWindow(name)
                        return "next"
                    # Do not call tracker.process_frame() for recording during transition

        # Dead zone: cap so cursor can move with small eye movements
        dx = np.mean(gaze_stds_x) * DEAD_ZONE_MULTIPLIER
        dy = np.mean(gaze_stds_y) * DEAD_ZONE_MULTIPLIER
        self._dead_zone_x = float(np.clip(dx, DEAD_ZONE_MIN, DEAD_ZONE_MAX))
        self._dead_zone_y = float(np.clip(dy, DEAD_ZONE_MIN, DEAD_ZONE_MAX))
        self.dead_zone = (self._dead_zone_x, self._dead_zone_y)
        self._gaze_means = gaze_means
        self._gaze_stds_x = gaze_stds_x
        self._gaze_stds_y = gaze_stds_y
        self._pose_means = list(pose_means)

        # Fit mapping: gaze = (pupil_x/crop_w, pupil_y/crop_h) -> screen. Need at least 1 valid point.
        valid_ix = [i for i in range(self.num_targets) if not (np.isnan(gaze_means[i][0]) or np.isnan(gaze_means[i][1]))]
        valid = [(gaze_means[i], self.targets[i]) for i in valid_ix]
        cv2.destroyWindow(name)
        print(f"Valid dots used in fit: {len(valid)} / {self.num_targets}")

        # Prefer binocular ridge if we have enough feature vectors (can succeed with fewer dots).
        hist_vecs, hist_targets = self._load_training_data()
        if hist_vecs:
            feature_vecs = list(feature_vecs) + list(hist_vecs)
            screen_targets = list(screen_targets) + list(hist_targets)

        if USE_BINOCULAR_RIDGE and len(feature_vecs) >= MIN_BINOC_DOTS:
            mu, std, W = _fit_binocular_ridge(feature_vecs, screen_targets, lam=1e-1)
            self.binoc_mu = mu
            self.binoc_std = std
            self.binoc_W = W
        else:
            self.binoc_mu = None
            self.binoc_std = None
            self.binoc_W = None
        if feature_vecs and screen_targets:
            self.ml = PolyRidgeRegressor(lam=5e-3, degree=ML_POLY_DEGREE)
            ok = self.ml.fit(feature_vecs, screen_targets)
            if not ok:
                print("ML fit failed: not enough data.")
            path = self._ml_model_path()
            if path and self.ml.W is not None:
                self.ml.save(path, extra={
                    "binoc_mu": self.binoc_mu if self.binoc_mu is not None else np.array([]),
                    "binoc_std": self.binoc_std if self.binoc_std is not None else np.array([]),
                    "binoc_W": self.binoc_W if self.binoc_W is not None else np.array([]),
                })
        self._append_training_rows(self._raw_training_rows)

        if len(valid) == 0 and self.binoc_W is None:
            print("Calibration failed: no valid gaze points (try to look at each dot and avoid blinking).")
            return
        # Coverage check: need enough spread so affine/polynomial don't explode
        if isinstance(self._target_provider, ScreenDotTargetProvider):
            grid_count = self._target_provider.rows * self._target_provider.cols
            rows = set(i // self._target_provider.cols for i in valid_ix if i < grid_count)
            cols = set(i % self._target_provider.cols for i in valid_ix if i < grid_count)
        else:
            rows = set()
            cols = set()
        if self.binoc_W is None and (len(valid) < MIN_VALID_DOTS or len(rows) < MIN_ROWS_COLS or len(cols) < MIN_ROWS_COLS):
            print(f"Calibration rejected: need at least {MIN_VALID_DOTS} valid dots and {MIN_ROWS_COLS} rows & columns. "
                  f"Got {len(valid)} dots, {len(rows)} rows, {len(cols)} cols. Redo calibration and cover the screen.")
            return
        src_pts = np.array([g for g, _ in valid], dtype=np.float64)
        dst_pts = np.array([s for _, s in valid], dtype=np.float64)
        if np.std(src_pts[:, 0]) < 1e-3 or np.std(src_pts[:, 1]) < 1e-3:
            print("Calibration failed: gaze features degenerate (std too small). Check iris/landmarks.")
            return
        mu = src_pts.mean(axis=0)
        std = src_pts.std(axis=0)
        std = np.maximum(std, 0.03)
        self.src_mu = mu
        self.src_std = std
        src_pts = (src_pts - mu) / std
        M = None
        if len(src_pts) >= 3:
            stds = []
            for i in valid_ix:
                sx = self._gaze_stds_x[i] if i < len(self._gaze_stds_x) else 0.02
                sy = self._gaze_stds_y[i] if i < len(self._gaze_stds_y) else 0.02
                stds.append((sx, sy))
            stds = np.array(stds, dtype=np.float64)
            base_w = 1.0 / np.maximum(stds[:, 0] ** 2 + stds[:, 1] ** 2, 1e-6)
            M = _fit_affine_irls_huber(src_pts, dst_pts, base_w=base_w, iters=10, delta=40.0, lam=1e-3)
        # binocular ridge already handled above
        if len(src_pts) >= 8 and USE_POLYNOMIAL and not FORCE_DISABLE_POLY:
            self._poly_coeffs_x, self._poly_coeffs_y = self._fit_poly_ridge(src_pts, dst_pts, lam=1e-2)
            rmse = self._poly_rmse(src_pts, dst_pts)
            if rmse <= 70:
                self._use_polynomial = True
            else:
                self._use_polynomial = False
                self._poly_coeffs_x = None
                self._poly_coeffs_y = None
        elif FORCE_DISABLE_POLY:
            self._use_polynomial = False
            self._poly_coeffs_x = None
            self._poly_coeffs_y = None
        if M is None and len(src_pts) >= 2:
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        if M is None:
            # Fallback: scale+offset from gaze bbox to screen bbox (works for 1 or 2+ points)
            gmin, gmax = src_pts.min(axis=0), src_pts.max(axis=0)
            smin, smax = dst_pts.min(axis=0), dst_pts.max(axis=0)
            gr = np.maximum(gmax - gmin, 1e-6)
            sx = (smax[0] - smin[0]) / gr[0]
            sy = (smax[1] - smin[1]) / gr[1]
            M = np.array([[sx, 0, smin[0] - sx * gmin[0]], [0, sy, smin[1] - sy * gmin[1]]], dtype=np.float32)
        if M is not None:
            center_target = (int(self.screen_w // 2), int(self.screen_h // 2))
            center_idx = None
            for i, (sx, sy) in enumerate(self.targets):
                if (sx, sy) == center_target:
                    center_idx = i
                    break
            if center_idx is not None and not (np.isnan(gaze_means[center_idx][0]) or np.isnan(gaze_means[center_idx][1])):
                cgx, cgy = gaze_means[center_idx]
                if self.src_mu is not None and self.src_std is not None:
                    cgx = (cgx - float(self.src_mu[0])) / float(self.src_std[0])
                    cgy = (cgy - float(self.src_mu[1])) / float(self.src_std[1])
                mapped = M @ np.array([cgx, cgy, 1.0], dtype=np.float64)
                dx = center_target[0] - mapped[0]
                dy = center_target[1] - mapped[1]
                M[0, 2] += dx
                M[1, 2] += dy
        # Keep the mapping as-is; neighbor refinement can over-constrain noisy gaze space.
        if M is not None:
            self.map_matrix = M
        if not self._use_polynomial:
            self._use_polynomial = False
        if USE_POSE_MODEL and self._pose_means:
            F = []
            Y = []
            for i in valid_ix:
                gx, gy = gaze_means[i]
                if np.isnan(gx) or np.isnan(gy):
                    continue
                if self.src_mu is not None and self.src_std is not None:
                    gx = (gx - float(self.src_mu[0])) / float(self.src_std[0])
                    gy = (gy - float(self.src_mu[1])) / float(self.src_std[1])
                yaw, pitch = self._pose_means[i]
                F.append([gx, gy, yaw, pitch, 1.0])
                Y.append([self.targets[i][0], self.targets[i][1]])
            if len(F) >= 6:
                F = np.array(F, dtype=np.float64)
                Y = np.array(Y, dtype=np.float64)
                self.W_pose = self._fit_ridge_linear(F, Y, lam=1e-2)
        self._valid_dot_count = len(valid) if len(valid) > 0 else len(feature_vecs)
        self.calibrated = True

        # Accuracy metric: per-dot screen error after fit
        errs_px = []
        errs_by_dot = []
        for i in valid_ix:
            gx, gy = gaze_means[i]
            if np.isnan(gx) or np.isnan(gy):
                continue
            mapped = self.gaze_to_screen(gx, gy)
            if mapped is None:
                continue
            tx, ty = self.targets[i][0], self.targets[i][1]
            ex = mapped[0] - tx
            ey = mapped[1] - ty
            e_px = float(np.sqrt(ex * ex + ey * ey))
            errs_px.append(e_px)
            errs_by_dot.append((i, tx, ty, mapped[0], mapped[1], e_px))
        if errs_px:
            errs_px = np.array(errs_px)
            print(f"[calibration] mean_error_px={np.mean(errs_px):.1f} median_error_px={np.median(errs_px):.1f} n_dots={len(errs_px)}")
            errs_by_dot.sort(key=lambda t: t[5], reverse=True)
            print("[calibration] worst 5 dots (dot_index target_x target_y mapped_x mapped_y error_px):")
            for t in errs_by_dot[:5]:
                print(f"  dot {t[0]}: target=({t[1]:.0f},{t[2]:.0f}) mapped=({t[3]:.1f},{t[4]:.1f}) err={t[5]:.1f}px")

        self._samples_per_dot = list(samples_per_dot)
        self.dead_px = self._compute_dead_zone_pixels(self._samples_per_dot)
        print(f"dead_px = {self.dead_px:.1f}")
        if DISABLE_DEADZONE_DURING_TUNING:
            self.dead_zone = (0.0, 0.0)
            self.dead_px = 0.0
        self._export_calibration_data()
        # Sanity check: where does true center dot map to?
        center_target = (int(self.screen_w // 2), int(self.screen_h // 2))
        center_idx = None
        for i, (sx, sy) in enumerate(self.targets):
            if (sx, sy) == center_target:
                center_idx = i
                break
        if center_idx is not None and center_idx < len(self._gaze_means):
            gx0, gy0 = self._gaze_means[center_idx]
            if not (np.isnan(gx0) or np.isnan(gy0)):
                mapped = self.gaze_to_screen(gx0, gy0)
                sc = (self.screen_w / 2, self.screen_h / 2)
                print("mapped center:", mapped, "screen center:", sc)

    def _poly_terms(self, gx, gy):
        """Second-order polynomial terms: 1, gx, gy, gx^2, gx*gy, gy^2."""
        return np.array([1.0, gx, gy, gx * gx, gx * gy, gy * gy], dtype=np.float64)

    def _fit_polynomial_2d(self, src_pts, dst_pts):
        """Fit screen = poly(gaze) with 2nd-order polynomial (6 coeffs per axis)."""
        n = src_pts.shape[0]
        X = np.zeros((n, 6), dtype=np.float64)
        for i in range(n):
            X[i] = self._poly_terms(src_pts[i, 0], src_pts[i, 1])
        screen_x = dst_pts[:, 0].astype(np.float64)
        screen_y = dst_pts[:, 1].astype(np.float64)
        self._poly_coeffs_x, _, _, _ = np.linalg.lstsq(X, screen_x, rcond=None)
        self._poly_coeffs_y, _, _, _ = np.linalg.lstsq(X, screen_y, rcond=None)

    def _fit_poly_ridge(self, src_pts, dst_pts, lam=1e-2):
        n = src_pts.shape[0]
        X = np.zeros((n, 6), dtype=np.float64)
        for i in range(n):
            X[i] = self._poly_terms(src_pts[i, 0], src_pts[i, 1])
        Yx = dst_pts[:, 0].astype(np.float64)
        Yy = dst_pts[:, 1].astype(np.float64)
        XtX = X.T @ X
        reg = lam * np.eye(X.shape[1])
        beta_x = np.linalg.solve(XtX + reg, X.T @ Yx)
        beta_y = np.linalg.solve(XtX + reg, X.T @ Yy)
        return beta_x, beta_y

    def _poly_map(self, gx, gy):
        t = self._poly_terms(gx, gy)
        x = float(np.dot(t, self._poly_coeffs_x))
        y = float(np.dot(t, self._poly_coeffs_y))
        return x, y

    def _poly_rmse(self, src_pts, dst_pts):
        X = np.stack([self._poly_terms(gx, gy) for gx, gy in src_pts], axis=0)
        pred_x = X @ self._poly_coeffs_x
        pred_y = X @ self._poly_coeffs_y
        pred = np.stack([pred_x, pred_y], axis=1)
        return float(np.sqrt(np.mean(np.sum((pred - dst_pts) ** 2, axis=1))))

    def _fit_ridge_linear(self, F, Y, lam=1e-2):
        FtF = F.T @ F
        reg = lam * np.eye(F.shape[1])
        W = np.linalg.solve(FtF + reg, F.T @ Y)
        return W

    def _prune_by_residual(self, src_pts, dst_pts, max_drop=3):
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=15)
        if M is None:
            return src_pts, dst_pts
        ones = np.ones((len(src_pts), 1), dtype=np.float32)
        pred = (M @ np.hstack([src_pts.astype(np.float32), ones]).T).T
        err = np.linalg.norm(pred - dst_pts, axis=1)
        keep = np.argsort(err)[: max(3, len(err) - max_drop)]
        return src_pts[keep], dst_pts[keep]

    def _compute_dead_zone_pixels(self, samples_per_dot, k=2.5, percentile=75):
        """Compute dead zone in screen pixels from mapped fixation jitter (robust MAD)."""
        radii = []
        for samples in samples_per_dot:
            if len(samples) < 10:
                continue
            pts = []
            for sample in samples:
                gx = sample[0]
                gy = sample[1]
                yaw = sample[2] if len(sample) > 2 else None
                pitch = sample[3] if len(sample) > 3 else None
                sp = self.gaze_to_screen(gx, gy, yaw=yaw, pitch=pitch)
                if sp is not None:
                    pts.append(sp)
            if len(pts) < 10:
                continue
            pts = np.array(pts, dtype=np.float64)
            mx, my = np.median(pts, axis=0)
            dx = np.abs(pts[:, 0] - mx)
            dy = np.abs(pts[:, 1] - my)
            madx = np.median(dx)
            mady = np.median(dy)
            sigx = 1.4826 * madx
            sigy = 1.4826 * mady
            r = float(np.sqrt(sigx * sigx + sigy * sigy))
            radii.append(r)
        if not radii:
            return 12.0
        base = np.percentile(radii, percentile)
        dead_px_min = 6.0 * (self.screen_w / 1920.0) if getattr(self, "screen_w", None) else 6.0
        dead_px_max_scaled = DEAD_PX_MAX * (self.screen_w / 1920.0) if getattr(self, "screen_w", None) else DEAD_PX_MAX
        dead_px = float(np.clip(k * base, dead_px_min, dead_px_max_scaled))
        return dead_px

    def gaze_to_screen(self, gaze_x_norm, gaze_y_norm, yaw=None, pitch=None, feature_vec=None):
        """Map normalized gaze (0-1) to screen (x, y). Uses polynomial if fitted else affine.
        Gaze X is already flipped in EyeTracker.process_frame() for front-facing camera."""
        if not self.calibrated:
            return None
        gx = float(gaze_x_norm)
        gy = float(gaze_y_norm)
        if self.src_mu is not None and self.src_std is not None:
            gx = (gx - float(self.src_mu[0])) / float(self.src_std[0])
            gy = (gy - float(self.src_mu[1])) / float(self.src_std[1])
        if self.W_map is not None and feature_vec is not None:
            fv = np.asarray(feature_vec, dtype=np.float64)
            if self.feature_mu is not None and self.feature_std is not None:
                fv = (fv - self.feature_mu) / self.feature_std
            f = np.hstack([fv, 1.0])
            out = f @ self.W_map
            x, y = float(out[0]), float(out[1])
        elif self.W_pose is not None and yaw is not None and pitch is not None:
            f = np.array([gx, gy, float(yaw), float(pitch), 1.0], dtype=np.float64)
            out = f @ self.W_pose
            x, y = float(out[0]), float(out[1])
        elif self._use_polynomial and self._poly_coeffs_x is not None and self._poly_coeffs_y is not None:
            x, y = self._poly_map(gx, gy)
            if (x < 0 or x > self.screen_w or y < 0 or y > self.screen_h) and self.map_matrix is not None:
                pt = np.array([gx, gy, 1.0], dtype=np.float32)
                out = self.map_matrix @ pt
                x, y = float(out[0]), float(out[1])
        elif self.map_matrix is not None:
            pt = np.array([gx, gy, 1.0], dtype=np.float32)
            out = self.map_matrix @ pt
            x, y = float(out[0]), float(out[1])
        else:
            return None
        # Return raw; clamp after smoothing in runtime to reduce sticky edges.
        self._last_map_raw = (x, y)
        return (x, y)

    def map_binocular(self, feature_vec):
        if self.binoc_W is None or self.binoc_mu is None or self.binoc_std is None:
            return None
        fv = np.asarray(feature_vec, dtype=np.float64)
        if len(fv) != len(self.binoc_mu):
            fv = fv[:len(self.binoc_mu)]
        fvz = (fv - self.binoc_mu) / self.binoc_std
        xyb = np.hstack([fvz, 1.0]) @ self.binoc_W
        self._last_map_raw = (float(xyb[0]), float(xyb[1]))
        return float(xyb[0]), float(xyb[1])

    def _export_calibration_data(self, filepath=None):
        """Write calibration data to calibrate.txt. Gaze = pupil_x/crop_width, pupil_y/crop_height (normalized)."""
        if filepath is None:
            filepath = CALIBRATE_TXT
        try:
            with open(filepath, "w") as f:
                f.write("# Calibration export (gaze = iris-based eye-local features)\n")
                f.write(f"screen_w={self.screen_w} screen_h={self.screen_h}\n")
                f.write(f"dead_zone_x={getattr(self, '_dead_zone_x', None)} dead_zone_y={getattr(self, '_dead_zone_y', None)}\n")
                f.write(f"dead_px={getattr(self, 'dead_px', None)}\n")
                f.write(f"cursor_smooth={CURSOR_SMOOTH}\n")
                f.write(f"use_polynomial={self._use_polynomial}\n")
                f.write(f"src_mu={getattr(self, 'src_mu', None)} src_std={getattr(self, 'src_std', None)}\n")
                f.write(f"feature_mu={getattr(self, 'feature_mu', None)} feature_std={getattr(self, 'feature_std', None)}\n")
                f.write(f"head_comp_enabled={USE_HEAD_COMP}\n")
                f.write(f"max_std_iod={MAX_STD_IOD} max_std_roll={MAX_STD_ROLL}\n")
                f.write(f"# dot_index screen_x screen_y gaze_mean_x gaze_mean_y std_x std_y sample_count\n")
                for i in range(self.num_targets):
                    sx, sy = self.targets[i]
                    g = self._gaze_means[i] if i < len(self._gaze_means) else (np.nan, np.nan)
                    gx, gy = g[0], g[1]
                    stdx = self._gaze_stds_x[i] if i < len(self._gaze_stds_x) else np.nan
                    stdy = self._gaze_stds_y[i] if i < len(self._gaze_stds_y) else np.nan
                    cnt = self._sample_counts[i] if i < len(self._sample_counts) else 0
                    f.write(f"{i} {sx} {sy} {gx} {gy} {stdx} {stdy} {cnt}\n")
                f.write("# Affine 2x3 map_matrix (row-major)\n")
                if self.map_matrix is not None:
                    for row in self.map_matrix:
                        f.write(" ".join(str(x) for x in row) + "\n")
                f.write("# Polynomial coeffs_x (1, gx, gy, gx^2, gx*gy, gy^2)\n")
                if self._poly_coeffs_x is not None:
                    f.write(" ".join(str(x) for x in self._poly_coeffs_x) + "\n")
                f.write("# Polynomial coeffs_y\n")
                if self._poly_coeffs_y is not None:
                    f.write(" ".join(str(x) for x in self._poly_coeffs_y) + "\n")
                f.write("# Pose model weights (gx, gy, yaw, pitch, 1) -> x,y\n")
                if self.W_pose is not None:
                    for row in self.W_pose:
                        f.write(" ".join(str(x) for x in row) + "\n")
            print(f"Calibration data written to {filepath}")
        except Exception as e:
            print(f"Could not write {filepath}: {e}")

    def export_profile(self, filepath="calibration_profile.json"):
        """
        Export a machine-readable calibration profile for external software.
        Includes mapping params, normalization, and per-dot stats.
        """
        profile = {
            "version": 1,
            "screen": {"width": self.screen_w, "height": self.screen_h},
            "mapping": {
                "use_polynomial": bool(self._use_polynomial),
                "map_matrix": self.map_matrix.tolist() if self.map_matrix is not None else None,
                "poly_coeffs_x": self._poly_coeffs_x.tolist() if self._poly_coeffs_x is not None else None,
                "poly_coeffs_y": self._poly_coeffs_y.tolist() if self._poly_coeffs_y is not None else None,
                "pose_weights": self.W_pose.tolist() if self.W_pose is not None else None,
                "binocular_weights": self.binoc_W.tolist() if self.binoc_W is not None else None,
                "src_mu": self.src_mu.tolist() if self.src_mu is not None else None,
                "src_std": self.src_std.tolist() if self.src_std is not None else None,
                "feature_mu": self.feature_mu.tolist() if self.feature_mu is not None else None,
                "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
                "head_comp_enabled": USE_HEAD_COMP,
                "max_std_iod": MAX_STD_IOD,
                "max_std_roll": MAX_STD_ROLL,
            },
            "dead_zone": {
                "norm": [getattr(self, "_dead_zone_x", None), getattr(self, "_dead_zone_y", None)],
                "pixels": getattr(self, "dead_px", None),
            },
            "dots": [],
        }
        for i in range(self.num_targets):
            sx, sy = self.targets[i]
            g = self._gaze_means[i] if i < len(self._gaze_means) else (np.nan, np.nan)
            stdx = self._gaze_stds_x[i] if i < len(self._gaze_stds_x) else np.nan
            stdy = self._gaze_stds_y[i] if i < len(self._gaze_stds_y) else np.nan
            cnt = self._sample_counts[i] if i < len(self._sample_counts) else 0
            profile["dots"].append(
                {
                    "index": i,
                    "screen_x": int(sx),
                    "screen_y": int(sy),
                    "gaze_mean_x": None if np.isnan(g[0]) else float(g[0]),
                    "gaze_mean_y": None if np.isnan(g[1]) else float(g[1]),
                    "std_x": None if np.isnan(stdx) else float(stdx),
                    "std_y": None if np.isnan(stdy) else float(stdy),
                    "sample_count": int(cnt),
                }
            )
        try:
            with open(filepath, "w") as f:
                json.dump(profile, f, indent=2)
            print(f"Calibration profile written to {os.path.abspath(filepath)}")
        except Exception as e:
            print(f"Could not write {filepath}: {e}")

    def run_eye_control(self, tracker):
        """
        Fullscreen: one green dot moved by your eyes using calibrated mapping. Press 'q' to quit.
        """
        if not self.calibrated and self.binoc_W is None:
            print("No mapping; cannot run eye control.")
            return
        print("Starting eye control - you should see a fullscreen window with a green dot. Press Q to quit.")
        name = "eye_control"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cursor_x, cursor_y = float(self.screen_w // 2), float(self.screen_h // 2)
        vx, vy = 0.0, 0.0
        alpha = CURSOR_SMOOTH  # kept for reference; spring-damper used below
        one_euro = OneEuroFilter2D(min_cutoff=6.0, beta=0.20, d_cutoff=1.0)
        stable_xy = None
        last_mapped = None
        frame_count = 0
        fps_t0 = time.perf_counter()
        fps_last = 0.0
        DEBUG_FRAMES = 150  # write gx, gy, mapped to file for first ~5 sec to diagnose
        gaze_buf = deque(maxlen=MEDIAN_FILTER_LEN)
        _prev_gaze = getattr(self, "_prev_gaze", None)
        _dead_gaze = getattr(self, "_dead_gaze", None)
        _ema_gaze = getattr(self, "_ema_gaze", None)
        last_fv = None
        last_good = {"t": 0.0, "xy": None}
        last_t = time.perf_counter()
        blink_hold_until = 0.0
        blink_open_count = 0
        debug_file = None
        diagnosis_file = None
        weight_file = None
        debug_path = os.path.abspath(DEBUG_OUTPUT_FILE)
        print("Runtime debug path:", debug_path)
        try:
            debug_file = open(DEBUG_OUTPUT_FILE, "w")
            debug_file.write("# Runtime debug: gx gy mapped_x mapped_y (first {} frames)\n".format(DEBUG_FRAMES))
        except Exception as e:
            print(f"Could not open {DEBUG_OUTPUT_FILE} for debug: {e}")
        try:
            weight_file = open(os.path.join(LOG_DIR, "eye_weights_debug.txt"), "w")
            weight_file.write("# gxL gyL gxR gyR eye_wL lid_hL eye_wR lid_hR yaw wL wR conf\n")
        except Exception as e:
            print(f"Could not open eye_weights_debug.txt: {e}")
        try:
            diagnosis_file = open(DIAGNOSIS_OUTPUT_FILE, "w")
            valid_count = getattr(self, "_valid_dot_count", None)
            if valid_count is None:
                valid_count = sum(1 for i in range(self.num_targets) if i < len(self._gaze_means) and not (np.isnan(self._gaze_means[i][0]) or np.isnan(self._gaze_means[i][1])))
            diagnosis_file.write("Valid dots used in fit: {} / {}\n".format(valid_count, self.num_targets))
            diagnosis_file.write("map_matrix (affine 2x3 row-major):\n")
            if self.map_matrix is not None:
                for row in self.map_matrix:
                    diagnosis_file.write("  " + " ".join(str(x) for x in row) + "\n")
            if self._poly_coeffs_x is not None and self._poly_coeffs_y is not None:
                diagnosis_file.write("poly_coeffs_x: " + " ".join(str(x) for x in self._poly_coeffs_x) + "\n")
                diagnosis_file.write("poly_coeffs_y: " + " ".join(str(x) for x in self._poly_coeffs_y) + "\n")
            diagnosis_file.write("use_polynomial: {}\n".format(self._use_polynomial))
            diagnosis_file.write("\nRuntime debug (first {} lines):\n".format(DEBUG_LINES_FOR_DIAGNOSIS))
            diagnosis_file.flush()
        except Exception as e:
            print(f"Could not open {DIAGNOSIS_OUTPUT_FILE} for diagnosis: {e}")

        while True:
            now = time.perf_counter()
            dt = max(1e-3, min(0.05, now - last_t))
            last_t = now
            try:
                gx, gy, yaw, pitch, fv, conf, is_blink = tracker.process_frame_binocular()
            except Exception as e:
                print("process_frame error:", e)
                gx, gy, yaw, pitch, fv, conf, is_blink = None, None, None, None, None, 0.0, True

            lid_hL = getattr(tracker, "last_left_lid_h", None)
            lid_hR = getattr(tracker, "last_right_lid_h", None)
            eye_wL = getattr(tracker, "last_left_eye_w", None)
            eye_wR = getattr(tracker, "last_right_eye_w", None)
            lid_min = min(v for v in (lid_hL, lid_hR) if v is not None) if (lid_hL is not None or lid_hR is not None) else None
            eye_w_min = min(v for v in (eye_wL, eye_wR) if v is not None) if (eye_wL is not None or eye_wR is not None) else None
            ratios = []
            if eye_wL and lid_hL:
                ratios.append(lid_hL / max(eye_wL, 1e-6))
            if eye_wR and lid_hR:
                ratios.append(lid_hR / max(eye_wR, 1e-6))
            lid_ratio_min = min(ratios) if ratios else None
            blink_gate = (
                is_blink
                or (lid_min is not None and lid_min < BLINK_LID_H_MIN)
                or (eye_w_min is not None and eye_w_min < BLINK_EYE_W_MIN)
                or (lid_ratio_min is not None and lid_ratio_min < BLINK_LID_RATIO)
            )

            if blink_gate:
                blink_hold_until = max(blink_hold_until, now + BLINK_HOLD_SEC)
                blink_open_count = 0
            elif now < blink_hold_until:
                blink_open_count = 0
            else:
                blink_open_count += 1

            if blink_gate or now < blink_hold_until or blink_open_count < BLINK_RESUME_FRAMES:
                frame_count += 1
                img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                img[:] = (40, 40, 40)
                cx, cy = int(cursor_x), int(cursor_y)
                cv2.circle(img, (cx, cy), CURSOR_RADIUS, (0, 255, 0), 2)
                cv2.putText(img, "Eye cursor - move with your eyes (press Q to quit)", (self.screen_w // 2 - 220, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if fps_last > 0.0:
                    cv2.putText(
                        img, f"FPS {fps_last:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2
                    )
                cv2.imshow(name, img)
                key = cv2.waitKey(1)
                if key != -1 and (key & 0xFF) == ord('q'):
                    break
                continue

            if gx is not None and gy is not None and fv is not None:
                _dead_gaze = (gx, gy)
                dfv = float(np.linalg.norm(fv - last_fv)) if last_fv is not None else 0.0
                last_fv = fv
                motion_penalty = float(np.exp(-(dfv * dfv) / (0.20 * 0.20)))
                track_conf = float(conf) * motion_penalty  # for ML/binoc gating
                raw_sum = float(getattr(tracker, "last_raw_sum", 0.0))
                smooth_conf = float(np.tanh(raw_sum))
                smooth_conf = max(0.2, min(1.0, smooth_conf))  # for filter aggressiveness only
                screen_pt = None
                ml_input_dim = getattr(self.ml, "input_dim", None) if self.ml is not None else None
                fv_dim_ok = (ml_input_dim is not None and len(fv) == ml_input_dim)
                if not fv_dim_ok and self.ml is not None and ml_input_dim is not None:
                    if not getattr(self, "_fv_dim_mismatch_logged", False):
                        print(f"[run_eye_control] ML skipped: fv len={len(fv)} != ml.input_dim={ml_input_dim}. Fall back; retrain with current fv.")
                        self._fv_dim_mismatch_logged = True
                use_ml = (USE_ML_ONLY and self.ml is not None and getattr(self.ml, "W", None) is not None and track_conf >= 0.25 and fv_dim_ok)
                if use_ml:
                    fv_in = fv[:self.ml.input_dim]
                    screen_pt = self.ml.predict(fv_in)
                    if screen_pt is not None and (0.0 <= screen_pt[0] <= 1.0 and 0.0 <= screen_pt[1] <= 1.0):
                        screen_pt = (screen_pt[0] * self.screen_w, screen_pt[1] * self.screen_h)
                        last_good["t"] = now
                        last_good["xy"] = screen_pt
                    else:
                        screen_pt = None
                if screen_pt is None and self.binoc_W is not None:
                    screen_pt = self.map_binocular(fv)
                    if screen_pt is not None and (0.0 <= screen_pt[0] <= 1.0 and 0.0 <= screen_pt[1] <= 1.0):
                        screen_pt = (screen_pt[0] * self.screen_w, screen_pt[1] * self.screen_h)
                        last_good["t"] = now
                        last_good["xy"] = screen_pt
                    else:
                        screen_pt = None
                if screen_pt is None:
                    screen_pt = self.gaze_to_screen(gx, gy, yaw=yaw, pitch=pitch, feature_vec=fv)
                if screen_pt is None and last_good["xy"] is not None and (now - last_good["t"]) < 0.15:
                    screen_pt = last_good["xy"]
                if screen_pt is not None:
                    sx, sy = screen_pt
                    sx = float(np.clip(sx, 0, self.screen_w))
                    sy = float(np.clip(sy, 0, self.screen_h))
                    last_mapped = (sx, sy)
                    raw = getattr(self, "_last_map_raw", None)
                    if frame_count < DEBUG_FRAMES and debug_file is not None:
                        if raw is not None:
                            line = (
                                f"gx={gx:.4f} gy={gy:.4f} "
                                f"raw=({raw[0]:.1f},{raw[1]:.1f}) "
                                f"mapped=({sx:.1f},{sy:.1f})\n"
                            )
                        else:
                            line = f"gx={gx:.4f} gy={gy:.4f} mapped=({sx:.1f},{sy:.1f})\n"
                        debug_file.write(line)
                        debug_file.flush()
                        if frame_count < DEBUG_LINES_FOR_DIAGNOSIS and diagnosis_file is not None:
                            diagnosis_file.write(line)
                            diagnosis_file.flush()
                    dead_px = float(np.clip(getattr(self, "dead_px", 6.0), 0.0, DEAD_PX_MAX))
                    if stable_xy is None:
                        stable_xy = (sx, sy)
                    else:
                        dxs = sx - stable_xy[0]
                        dys = sy - stable_xy[1]
                        if dxs * dxs + dys * dys >= dead_px * dead_px:
                            stable_xy = (sx, sy)
                    sx, sy = stable_xy

                    one_euro.fx.min_cutoff = 4.0 + 6.0 * smooth_conf
                    one_euro.fy.min_cutoff = 4.0 + 6.0 * smooth_conf
                    sx_f, sy_f = one_euro(sx, sy, t=now)
                    cursor_x, cursor_y = sx_f, sy_f
                    cursor_x = float(np.clip(cursor_x, 0, self.screen_w))
                    cursor_y = float(np.clip(cursor_y, 0, self.screen_h))
                    vx = 0.0
                    vy = 0.0
                    if weight_file is not None and frame_count < DEBUG_FRAMES:
                        gxL = getattr(tracker, "last_left_gx", None)
                        gyL = getattr(tracker, "last_left_gy", None)
                        gxR = getattr(tracker, "last_right_gx", None)
                        gyR = getattr(tracker, "last_right_gy", None)
                        ewL = getattr(tracker, "last_left_eye_w", None)
                        lhL = getattr(tracker, "last_left_lid_h", None)
                        ewR = getattr(tracker, "last_right_eye_w", None)
                        lhR = getattr(tracker, "last_right_lid_h", None)
                        wL = getattr(tracker, "last_w_left", None)
                        wR = getattr(tracker, "last_w_right", None)
                        yaw_val = getattr(tracker, "last_yaw", None)
                        line = f"{gxL} {gyL} {gxR} {gyR} {ewL} {lhL} {ewR} {lhR} {yaw_val} {wL} {wR} {track_conf}\n"
                        weight_file.write(line)
            frame_count += 1
            if frame_count % 60 == 0:
                elapsed = max(1e-6, time.perf_counter() - fps_t0)
                fps_last = frame_count / elapsed
                print(f"[eye_control] FPS={fps_last:.1f}")

            img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            img[:] = (40, 40, 40)
            cx, cy = int(cursor_x), int(cursor_y)
            cv2.circle(img, (cx, cy), CURSOR_RADIUS, (0, 255, 0), 2)
            cv2.putText(img, "Eye cursor - move with your eyes (press Q to quit)", (self.screen_w // 2 - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if fps_last > 0.0:
                cv2.putText(
                    img, f"FPS {fps_last:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2
                )
            cv2.imshow(name, img)
            key = cv2.waitKey(1)
            if key != -1 and (key & 0xFF) == ord('q'):
                break
        if debug_file is not None:
            try:
                debug_file.close()
            except Exception:
                pass
            print(f"Runtime debug written to {DEBUG_OUTPUT_FILE} (first {DEBUG_FRAMES} frames).")
        if diagnosis_file is not None:
            try:
                diagnosis_file.close()
            except Exception:
                pass
            print("Diagnosis file (valid count, map_matrix, first {} debug lines): {}".format(
                DEBUG_LINES_FOR_DIAGNOSIS, os.path.abspath(DIAGNOSIS_OUTPUT_FILE)))
        if weight_file is not None:
            try:
                weight_file.close()
            except Exception:
                pass
        self._prev_gaze = _prev_gaze
        self._dead_gaze = _dead_gaze
        self._ema_gaze = _ema_gaze
        cv2.destroyWindow(name)
        print("Eye control ended.")
