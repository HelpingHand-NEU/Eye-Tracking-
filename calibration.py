"""
Calibration and eye-controlled cursor for left-eye gaze.
20 dots on screen; calibrate by looking at each for 1s (no blink); then cursor follows gaze.
"""
import os
import sys
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
USE_POLYNOMIAL = False  # keep affine only until gaze features are stable
USE_BINOCULAR_RIDGE = True
USE_HEAD_COMP = True
USE_POSE_MODEL = False
DISABLE_DEADZONE_DURING_TUNING = False
CALIB_IGNORE_BLINK = False


def _get_screen_size():
    """Real screen size; avoid hardcoded 1920x1080 on other displays."""
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
CURSOR_SMOOTH = 0.25   # 0=instant, 1=no move; lower = faster follow (0.25 = responsive)
CURSOR_RADIUS = 20
CALIBRATE_TXT = "calibrate.txt"  # export path for calibration data
DEBUG_OUTPUT_FILE = "eye_control_debug.txt"  # runtime gx, gy, mapped for first N frames
DIAGNOSIS_OUTPUT_FILE = "eye_control_diagnosis.txt"  # valid count, map_matrix, first ~20 debug lines
DEBUG_LINES_FOR_DIAGNOSIS = 20
MAX_JUMP = 0.12  # normalized; reject frame if gaze jump > this (outlier rejection)
MEDIAN_FILTER_LEN = 5  # number of frames for median filter on gaze
GAZE_EMA_NEW_WEIGHT = 0.35  # EMA new-sample weight (0=sticky, 1=raw)
MAX_STD_X_FOR_DOT = 0.035
MAX_STD_Y_FOR_DOT = 0.050
MAX_STD_IOD = 0.010
MAX_STD_ROLL = 0.025
MAX_STD_YAW = 0.06
MAX_STD_PITCH = 0.06
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


class Calibration:
    """
    20 dots at fixed positions on screen. Fullscreen. Calibrate by 1s fixation per dot
    (no blink); record pupil normalized by crop size; compute dead zone from fixation std.
    """

    def __init__(self, target_provider=None):
        self.screen_w, self.screen_h = _get_screen_size()
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

    def _make_dots(self):
        """Deprecated: use ScreenDotTargetProvider to supply targets."""
        return ScreenDotTargetProvider().get_targets(self.screen_w, self.screen_h)

    def get_dot(self, dot_index):
        """Return (x, y) screen position of dot number (0..14)."""
        if 0 <= dot_index < self.num_targets:
            return self.targets[dot_index]
        return None

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
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
        name = self._show_fullscreen("calibration")
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

        if hasattr(tracker, "calibrate_ear_threshold"):
            tracker.calibrate_ear_threshold()

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
                allow_blink_samples = False
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
                    if (not is_blink or allow_blink_samples) and gx is not None and gy is not None:
                        samples.append((gx, gy, yaw, pitch))
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
                            fv = np.array([gxL, gyL, gxR, gyR, float(yaw), float(pitch), float(iod), float(roll)], dtype=np.float64)
                        feature_samples.append(fv)
                    else:
                        miss_feats += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyWindow(name)
                        return False
                    # Ensure we collect enough samples even at low FPS.
                    if elapsed >= FIXATION_SEC:
                        # Dynamic requirement based on observed sample rate.
                        effective_fps = len(samples) / max(1e-6, (elapsed - SETTLE_SEC))
                        needed = max(MIN_SAMPLES_FALLBACK, int(effective_fps * FIXATION_SEC * 0.4))
                        needed = min(needed, MIN_SAMPLES_PER_DOT)
                        if len(samples) >= needed:
                            break
                        if len(samples) == 0:
                            allow_blink_samples = True
                            redo_notice = "redo: blink gate relaxed"
                    # If blink dominates, relax earlier so we can move on.
                    if elapsed >= 1.0 and total_frames > 0:
                        blink_ratio = blink_frames / max(1, total_frames)
                        if blink_ratio > 0.8:
                            allow_blink_samples = True
                            if redo_notice is None:
                                redo_notice = "redo: blink gate relaxed"
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
                mean_x, mean_y, mean_yaw, mean_pitch = np.mean(arr, axis=0)
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
                if std_x > MAX_STD_X_FOR_DOT or std_y > MAX_STD_Y_FOR_DOT:
                    print(
                        f"Dot {dot_i + 1}: jitter too high (std_x={std_x:.3f}, std_y={std_y:.3f}). Redo this dot."
                    )
                    redo_notice = f"redo: jitter {std_x:.3f},{std_y:.3f}"
                    continue
                if USE_HEAD_COMP and (std_iod > MAX_STD_IOD or std_roll > MAX_STD_ROLL):
                    print(
                        f"Dot {dot_i + 1}: head motion too high (std_iod={std_iod:.4f}, std_roll={std_roll:.4f}). Redo this dot."
                    )
                    redo_notice = "redo: head moved"
                    continue
                if USE_POSE_MODEL and (std_yaw > MAX_STD_YAW or std_pitch > MAX_STD_PITCH):
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
                fv_mean = np.mean(np.array(feature_samples), axis=0)
                feature_means.append(fv_mean)
                feature_vecs.append(fv_mean)
                screen_targets.append(self.targets[dot_i])
            else:
                fallback = np.array([mean_x, mean_y, mean_x, mean_y, 0.0, 0.0], dtype=np.float64)
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
                    cv2.imshow(name, img)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        cv2.destroyWindow(name)
                        return False
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
        if USE_BINOCULAR_RIDGE and len(feature_vecs) >= MIN_BINOC_DOTS:
            mu, std, W = _fit_binocular_ridge(feature_vecs, screen_targets, lam=1e-1)
            self.binoc_mu = mu
            self.binoc_std = std
            self.binoc_W = W
        else:
            self.binoc_mu = None
            self.binoc_std = None
            self.binoc_W = None

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
        std = np.maximum(std, 1e-4)
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
            M = _fit_affine_irls_huber(src_pts, dst_pts, base_w=base_w, iters=12, delta=30.0)
        # binocular ridge already handled above
        if len(src_pts) >= 8:
            self._poly_coeffs_x, self._poly_coeffs_y = self._fit_poly_ridge(src_pts, dst_pts, lam=1e-2)
            rmse = self._poly_rmse(src_pts, dst_pts)
            if rmse <= 110:
                self._use_polynomial = True
            else:
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
        self._samples_per_dot = list(samples_per_dot)
        self.dead_px = self._compute_dead_zone_pixels(self._samples_per_dot)
        print(f"dead_px = {self.dead_px:.1f}")
        if DISABLE_DEADZONE_DURING_TUNING:
            self.dead_zone = (0.0, 0.0)
            self.dead_px = 0.0
        self._export_calibration_data()
        # Sanity check: where does "center gaze" map to?
        scx, scy = self.screen_w / 2, self.screen_h / 2
        d = [(i, (x - scx) ** 2 + (y - scy) ** 2) for i, (x, y) in enumerate(self.targets)]
        d.sort(key=lambda t: t[1])
        mid = [i for i, _ in d[:4]]
        gxs = [self._gaze_means[i][0] for i in mid if i < len(self._gaze_means) and not np.isnan(self._gaze_means[i][0])]
        gys = [self._gaze_means[i][1] for i in mid if i < len(self._gaze_means) and not np.isnan(self._gaze_means[i][1])]
        if gxs and gys:
            gx0, gy0 = float(np.mean(gxs)), float(np.mean(gys))
            mapped = self.gaze_to_screen(gx0, gy0)
            sc = (self.screen_w / 2, self.screen_h / 2)
            print("mapped center-ish:", mapped, "screen center:", sc)

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
        dead_px = float(np.clip(k * base, 6.0, DEAD_PX_MAX))
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
        last_t = time.perf_counter()
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
            weight_file = open("eye_weights_debug.txt", "w")
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

            if not is_blink and gx is not None and gy is not None and fv is not None:
                _dead_gaze = (gx, gy)
                if last_fv is not None:
                    dfv = float(np.linalg.norm(fv - last_fv))
                    if dfv > 0.35:
                        frame_count += 1
                        continue
                last_fv = fv
                if self.binoc_W is not None:
                    screen_pt = self.map_binocular(fv)
                else:
                    screen_pt = self.gaze_to_screen(gx, gy, yaw=yaw, pitch=pitch, feature_vec=fv)
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

                    raw_sum = float(getattr(tracker, "last_raw_sum", 0.0))
                    conf = float(np.tanh(raw_sum))
                    conf = max(0.2, min(1.0, conf))
                    one_euro.fx.min_cutoff = 4.0 + 6.0 * conf
                    one_euro.fy.min_cutoff = 4.0 + 6.0 * conf
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
                        line = f"{gxL} {gyL} {gxR} {gyR} {ewL} {lhL} {ewR} {lhR} {yaw_val} {wL} {wR} {conf}\n"
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
