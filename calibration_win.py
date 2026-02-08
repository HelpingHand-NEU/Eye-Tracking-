"""
Calibration and eye-controlled cursor for left-eye gaze.
20 dots on screen; calibrate by looking at each for 1s (no blink); then cursor follows gaze.
"""
import os
import sys
import cv2
import numpy as np
import time
from collections import deque

# Minimum valid samples per dot (pupil_x/crop_w, pupil_y/crop_h) so fit isn't degenerate
MIN_SAMPLES_PER_DOT = 20
MAX_DOT_ATTEMPTS = 5  # allow more retries per dot if pupil lock is unstable


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


NUM_DOTS = 20
FIXATION_SEC = 1.3  # slightly longer for stable pupil lock at edges
NEXT_DOT_TRANSITION_SEC = 1.0  # show arrow for this long between dots; no data recorded
MIN_VALID_DOTS = 12  # require at least this many valid dots for fit
MIN_ROWS_COLS = 3    # require at least 3 distinct rows and 3 distinct columns (coverage)
# Dead zone: cap so cursor stays responsive (large dead zone = dot barely moves)
DEAD_ZONE_MULTIPLIER = 2.0
DEAD_ZONE_MAX = 0.04   # max movement threshold in normalized gaze; larger = stickier cursor
DEAD_ZONE_MIN = 0.008  # min to avoid jitter
CURSOR_SMOOTH = 0.25   # 0=instant, 1=no move; lower = faster follow (0.25 = responsive)
CURSOR_RADIUS = 20
CALIBRATE_TXT = "calibrate.txt"  # export path for calibration data
DEBUG_OUTPUT_FILE = "eye_control_debug.txt"  # runtime gx, gy, mapped for first N frames
DIAGNOSIS_OUTPUT_FILE = "eye_control_diagnosis.txt"  # valid count, map_matrix, first ~20 debug lines
DEBUG_LINES_FOR_DIAGNOSIS = 20
MAX_JUMP = 0.06  # normalized; reject frame if gaze jump > this (outlier rejection)
MEDIAN_FILTER_LEN = 5  # number of frames for median filter on gaze
MAX_STD_FOR_DOT = 0.035  # reject calibration dot if std_x or std_y > this
DOT_RADIUS_CALIB = 15
# Flip is applied in EyeTracker.process_frame() so calibration and runtime use same gaze coords.
FLIP_GAZE_X = True  # kept for reference; actual flip done in eye_tracker_win


class Calibration:
    """
    20 dots at fixed positions on screen. Fullscreen. Calibrate by 1s fixation per dot
    (no blink); record pupil normalized by crop size; compute dead zone from fixation std.
    """

    def __init__(self):
        self.screen_w, self.screen_h = _get_screen_size()
        self.dots = self._make_dots()
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

    def _make_dots(self):
        """Place 20 dots in 5x4 grid with margins."""
        margin_x = self.screen_w * 0.12
        margin_y = self.screen_h * 0.12
        usable_w = self.screen_w - 2 * margin_x
        usable_h = self.screen_h - 2 * margin_y
        positions = []
        for row in range(4):
            for col in range(5):
                x = margin_x + (col + 0.5) * (usable_w / 5)
                y = margin_y + (row + 0.5) * (usable_h / 4)
                positions.append((int(x), int(y)))
        return positions

    def get_dot(self, dot_index):
        """Return (x, y) screen position of dot number (0..14)."""
        if 0 <= dot_index < NUM_DOTS:
            return self.dots[dot_index]
        return None

    def _neighbor_pairs(self):
        """Pairs (i, j) of neighboring dot indices in 5x4 grid (horizontal and vertical)."""
        pairs = []
        for i in range(NUM_DOTS):
            row, col = i // 5, i % 5
            if col < 4:
                pairs.append((i, i + 1))
            if row < 3:
                pairs.append((i, i + 5))
        return pairs

    def _refine_with_neighbors(self, M, gaze_means):
        """
        Refine affine M using neighbor consistency: for each neighboring dot pair (i,j),
        screen_delta should match M @ gaze_delta. Down-weight or drop dots that disagree.
        Re-fit with remaining points for a more consistent mapping.
        """
        if M is None or len(gaze_means) != NUM_DOTS:
            return M
        gaze_arr = np.array(gaze_means, dtype=np.float32)
        dots_arr = np.array(self.dots, dtype=np.float32)
        # Per-dot residual: how well do this dot's neighbor relationships match under M?
        residuals = np.zeros(NUM_DOTS)
        for i, j in self._neighbor_pairs():
            g_i = gaze_arr[i]
            g_j = gaze_arr[j]
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
        valid_ix = [idx for idx in range(NUM_DOTS) if not (np.isnan(gaze_arr[idx, 0]) or np.isnan(gaze_arr[idx, 1]))]
        if len(valid_ix) < 3:
            return M
        valid_ix.sort(key=lambda idx: residuals[idx])
        keep = valid_ix[: max(3, len(valid_ix) - 2)]  # drop at most 2 worst
        src = gaze_arr[keep]
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
            for i, (x, y) in enumerate(self.dots):
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
        target_fps = 30
        frame_dt = 1.0 / target_fps
        samples_per_dot = []

        for dot_i in range(NUM_DOTS):
            samples = []
            for attempt in range(MAX_DOT_ATTEMPTS):
                t_start = time.perf_counter()
                last_draw = 0
                while (time.perf_counter() - t_start) < FIXATION_SEC:
                    now = time.perf_counter()
                    if now - last_draw >= frame_dt:
                        img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                        img[:] = (30, 30, 30)
                        self._draw_dots(img, highlight_index=dot_i)
                        msg = f"Look at dot {dot_i + 1}/{NUM_DOTS} - hold 1s, don't blink"
                        if attempt > 0:
                            msg += f" (redo: need {MIN_SAMPLES_PER_DOT}+ samples)"
                        cv2.putText(
                            img, msg, (self.screen_w // 2 - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                        )
                        cv2.imshow(name, img)
                        last_draw = now

                    gx, gy, is_blink = tracker.process_frame()
                    # gx, gy are already pupil_x/crop_width, pupil_y/crop_height (normalized)
                    if not is_blink and gx is not None and gy is not None:
                        samples.append((gx, gy))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyWindow(name)
                        return False

                if len(samples) >= MIN_SAMPLES_PER_DOT:
                    break
                print(f"Dot {dot_i + 1}: only {len(samples)} valid samples (need {MIN_SAMPLES_PER_DOT}). Redo this dot.")
            samples_per_dot.append(samples)
            self._sample_counts.append(len(samples))
            if len(samples) < 2:
                mean_x, mean_y = np.nan, np.nan
                std_x, std_y = 0.02, 0.02
            elif len(samples) < MIN_SAMPLES_PER_DOT:
                # Too few samples: don't use this dot in fit (would skew mapping)
                mean_x, mean_y = np.nan, np.nan
                std_x, std_y = 0.02, 0.02
            else:
                arr = np.array(samples)
                mean_x, mean_y = np.mean(arr, axis=0)
                std_x, std_y = np.std(arr, axis=0) if len(samples) >= 2 else (0.02, 0.02)
                if np.isnan(std_x) or std_x < 1e-6:
                    std_x = 0.02
                if np.isnan(std_y) or std_y < 1e-6:
                    std_y = 0.02
                # Drop bad calibration points: high std = unstable pupil lock (eyelash/shadow)
                if std_x > MAX_STD_FOR_DOT or std_y > MAX_STD_FOR_DOT:
                    mean_x, mean_y = np.nan, np.nan
            gaze_means.append((mean_x, mean_y))
            gaze_stds_x.append(std_x)
            gaze_stds_y.append(std_y)

            # Before next dot: show arrow for 1s so user can move eyes; no data recorded
            if dot_i < NUM_DOTS - 1:
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

        # Fit mapping: gaze = (pupil_x/crop_w, pupil_y/crop_h) -> screen. Need at least 1 valid point.
        valid_ix = [i for i in range(NUM_DOTS) if not (np.isnan(gaze_means[i][0]) or np.isnan(gaze_means[i][1]))]
        valid = [(gaze_means[i], self.dots[i]) for i in valid_ix]
        cv2.destroyWindow(name)
        print(f"Valid dots used in fit: {len(valid)} / {NUM_DOTS}")
        if len(valid) == 0:
            print("Calibration failed: no valid gaze points (try to look at each dot and avoid blinking).")
            return
        # Coverage check: need enough spread so affine/polynomial don't explode
        rows = set(i // 5 for i in valid_ix)
        cols = set(i % 5 for i in valid_ix)
        if len(valid) < MIN_VALID_DOTS or len(rows) < MIN_ROWS_COLS or len(cols) < MIN_ROWS_COLS:
            print(f"Calibration rejected: need at least {MIN_VALID_DOTS} valid dots and {MIN_ROWS_COLS} rows & columns. "
                  f"Got {len(valid)} dots, {len(rows)} rows, {len(cols)} cols. Redo calibration and cover the screen.")
            return
        src_pts = np.array([g for g, _ in valid], dtype=np.float32)
        dst_pts = np.array([s for _, s in valid], dtype=np.float32)
        M = None
        if len(valid) >= 3:
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        if M is None and len(valid) >= 2:
            src = np.array([g for g, _ in valid], dtype=np.float32)
            dst = np.array([s for _, s in valid], dtype=np.float32)
            M, _ = cv2.estimateAffine2D(src, dst)
        if M is None:
            # Fallback: scale+offset from gaze bbox to screen bbox (works for 1 or 2+ points)
            gmin, gmax = src_pts.min(axis=0), src_pts.max(axis=0)
            smin, smax = dst_pts.min(axis=0), dst_pts.max(axis=0)
            gr = np.maximum(gmax - gmin, 1e-6)
            sx = (smax[0] - smin[0]) / gr[0]
            sy = (smax[1] - smin[1]) / gr[1]
            M = np.array([[sx, 0, smin[0] - sx * gmin[0]], [0, sy, smin[1] - sy * gmin[1]]], dtype=np.float32)
        # Refine using neighbor consistency; never drop the mapping we have
        M_refined = self._refine_with_neighbors(M, gaze_means)
        self.map_matrix = M_refined if M_refined is not None else M
        # 2nd-order polynomial: more stable than affine with sparse/uneven points when 6+ valid
        if len(valid) >= 6:
            self._fit_polynomial_2d(src_pts, dst_pts)
            self._use_polynomial = True  # use polynomial to avoid affine blow-up
        else:
            self._use_polynomial = False
        self._valid_dot_count = len(valid)
        self.calibrated = True
        self._samples_per_dot = list(samples_per_dot)
        self.dead_px = self._compute_dead_zone_pixels(self._samples_per_dot)
        print(f"dead_px = {self.dead_px:.1f}")
        self._export_calibration_data()
        # Sanity check: where does "center gaze" map to?
        mid = [7, 8, 12, 13]  # middle-ish dots in 5x4 grid
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

    def _compute_dead_zone_pixels(self, samples_per_dot, k=2.5, percentile=75):
        """Compute dead zone in screen pixels from mapped fixation jitter (robust MAD)."""
        radii = []
        for samples in samples_per_dot:
            if len(samples) < 10:
                continue
            pts = []
            for gx, gy in samples:
                sp = self.gaze_to_screen(gx, gy)
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
        dead_px = float(np.clip(k * base, 6.0, 40.0))
        return dead_px

    def gaze_to_screen(self, gaze_x_norm, gaze_y_norm):
        """Map normalized gaze (0-1) to screen (x, y). Uses polynomial if fitted else affine.
        Gaze X is already flipped in EyeTracker.process_frame() for front-facing camera."""
        if not self.calibrated:
            return None
        gx = float(gaze_x_norm)
        gy = float(gaze_y_norm)
        if self._use_polynomial and self._poly_coeffs_x is not None and self._poly_coeffs_y is not None:
            t = self._poly_terms(gx, gy)
            x = float(np.dot(t, self._poly_coeffs_x))
            y = float(np.dot(t, self._poly_coeffs_y))
        elif self.map_matrix is not None:
            pt = np.array([gx, gy, 1.0], dtype=np.float32)
            out = self.map_matrix @ pt
            x, y = float(out[0]), float(out[1])
        else:
            return None
        # Reject out-of-bounds instead of clamping (clamping caused cursor stuck at 0/960)
        if x < 0 or x > self.screen_w or y < 0 or y > self.screen_h:
            return None
        return (x, y)

    def _export_calibration_data(self, filepath=None):
        """Write calibration data to calibrate.txt. Gaze = pupil_x/crop_width, pupil_y/crop_height (normalized)."""
        if filepath is None:
            filepath = CALIBRATE_TXT
        try:
            with open(filepath, "w") as f:
                f.write(f"# Calibration export (gaze = pupil_x/crop_width, pupil_y/crop_height)\n")
                f.write(f"screen_w={self.screen_w} screen_h={self.screen_h}\n")
                f.write(f"dead_zone_x={getattr(self, '_dead_zone_x', None)} dead_zone_y={getattr(self, '_dead_zone_y', None)}\n")
                f.write(f"dead_px={getattr(self, 'dead_px', None)}\n")
                f.write(f"cursor_smooth={CURSOR_SMOOTH}\n")
                f.write(f"use_polynomial={self._use_polynomial}\n")
                f.write(f"# dot_index screen_x screen_y gaze_mean_x gaze_mean_y std_x std_y sample_count\n")
                for i in range(NUM_DOTS):
                    sx, sy = self.dots[i]
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
            print(f"Calibration data written to {filepath}")
        except Exception as e:
            print(f"Could not write {filepath}: {e}")

    def run_eye_control(self, tracker):
        """
        Fullscreen: one green dot moved by your eyes using calibrated mapping. Press 'q' to quit.
        """
        if not self.calibrated or (self.map_matrix is None and not self._use_polynomial):
            print("No mapping; cannot run eye control.")
            return
        print("Starting eye control - you should see a fullscreen window with a green dot. Press Q to quit.")
        name = "eye_control"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cursor_x, cursor_y = float(self.screen_w // 2), float(self.screen_h // 2)
        alpha = CURSOR_SMOOTH  # smooth: cursor = alpha*cursor + (1-alpha)*mapped
        frame_count = 0
        DEBUG_FRAMES = 150  # write gx, gy, mapped to file for first ~5 sec to diagnose
        gaze_buf = deque(maxlen=MEDIAN_FILTER_LEN)
        _prev_gaze = getattr(self, "_prev_gaze", None)
        debug_file = None
        diagnosis_file = None
        debug_path = os.path.abspath(DEBUG_OUTPUT_FILE)
        print("Runtime debug path:", debug_path)
        try:
            debug_file = open(DEBUG_OUTPUT_FILE, "w")
            debug_file.write("# Runtime debug: gx gy mapped_x mapped_y (first {} frames)\n".format(DEBUG_FRAMES))
        except Exception as e:
            print(f"Could not open {DEBUG_OUTPUT_FILE} for debug: {e}")
        try:
            diagnosis_file = open(DIAGNOSIS_OUTPUT_FILE, "w")
            valid_count = getattr(self, "_valid_dot_count", None)
            if valid_count is None:
                valid_count = sum(1 for i in range(NUM_DOTS) if i < len(self._gaze_means) and not (np.isnan(self._gaze_means[i][0]) or np.isnan(self._gaze_means[i][1])))
            diagnosis_file.write("Valid dots used in fit: {} / {}\n".format(valid_count, NUM_DOTS))
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
            try:
                gx, gy, is_blink = tracker.process_frame()
            except Exception as e:
                print("process_frame error:", e)
                gx, gy, is_blink = None, None, True
            if not is_blink and gx is not None and gy is not None:
                # Reject impossible gaze jumps (outlier spikes)
                if _prev_gaze is not None:
                    pgx, pgy = _prev_gaze
                    if abs(gx - pgx) > MAX_JUMP or abs(gy - pgy) > MAX_JUMP:
                        gx, gy = None, None
                if gx is not None:
                    _prev_gaze = (gx, gy)
                    gaze_buf.append((gx, gy))
                    # Median filter: robust to one bad frame
                    gx = float(np.median([p[0] for p in gaze_buf]))
                    gy = float(np.median([p[1] for p in gaze_buf]))
                else:
                    _prev_gaze = None
            else:
                _prev_gaze = None

            if not is_blink and gx is not None and gy is not None:
                screen_pt = self.gaze_to_screen(gx, gy)
                if screen_pt is not None:
                    sx, sy = screen_pt
                    if frame_count < DEBUG_FRAMES and debug_file is not None:
                        line = f"gx={gx:.4f} gy={gy:.4f} mapped=({sx:.1f},{sy:.1f})\n"
                        debug_file.write(line)
                        debug_file.flush()
                        if frame_count < DEBUG_LINES_FOR_DIAGNOSIS and diagnosis_file is not None:
                            diagnosis_file.write(line)
                            diagnosis_file.flush()
                    # Apply dead zone in screen pixels: only move if beyond jitter
                    dead_px = getattr(self, 'dead_px', 12.0)
                    dx = sx - cursor_x
                    dy = sy - cursor_y
                    if dx * dx + dy * dy >= dead_px * dead_px:
                        cursor_x = alpha * cursor_x + (1.0 - alpha) * sx
                        cursor_y = alpha * cursor_y + (1.0 - alpha) * sy
            frame_count += 1

            img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            img[:] = (40, 40, 40)
            cx, cy = int(cursor_x), int(cursor_y)
            cv2.circle(img, (cx, cy), CURSOR_RADIUS, (0, 255, 0), 2)
            cv2.putText(img, "Eye cursor - move with your eyes (press Q to quit)", (self.screen_w // 2 - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(name, img)
            key = cv2.waitKey(30)
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
        self._prev_gaze = _prev_gaze
        cv2.destroyWindow(name)
        print("Eye control ended.")
