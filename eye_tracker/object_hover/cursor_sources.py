import time
import os
from ..calibration import OneEuroFilter2D

try:
    import pyautogui
except Exception:  # pragma: no cover - optional dependency
    pyautogui = None


class CursorSource:
    space = "frame"

    def get_xy(self):
        raise NotImplementedError


class MouseCursor(CursorSource):
    space = "screen"

    def get_xy(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for MouseCursor.")
        x, y = pyautogui.position()
        return int(x), int(y)


class WindowMouseCursor(CursorSource):
    space = "window"

    def __init__(self, window_name):
        self.window_name = window_name
        self._xy = None

    def handle_event(self, event, x, y, flags, userdata):
        self._xy = (int(x), int(y))

    def get_xy(self):
        return self._xy


class EyeTrackerCursor(CursorSource):
    space = "screen"

    STATS_INTERVAL = 200

    def __init__(self, tracker, calibration, min_cutoff=1.2, beta=0.12, gaze_jump_thresh=0.08):
        self.tracker = tracker
        self.calibration = calibration
        # Configurable smoothing: lower min_cutoff = smoother (less jitter), higher beta = less lag
        self._gaze_filter = OneEuroFilter2D(min_cutoff=float(min_cutoff), beta=float(beta), d_cutoff=1.0)
        self._last_gaze_before_filter = None  # for jump rejection before OneEuro
        self._gaze_jump_thresh = float(gaze_jump_thresh)  # reject frame if normalized gaze jumps more than this
        self._log_start = time.perf_counter()
        self._log_file = None
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "eye_gaze_raw_log.txt")
        try:
            self._log_file = open(self._log_path, "w")
            self._log_file.write("# t gx gy eye_w lid_h blink\n")
        except Exception:
            self._log_file = None
        self._stats = {"ml": 0, "binoc": 0, "fallback": 0, "drop": 0}
        self._frame_count = 0
        self._last_good = {"t": 0.0, "xy": None}
        self._fv_dim_mismatch_logged = False

    def _log_sample(self, gx, gy, eye_w, lid_h, is_blink):
        if self._log_file is None:
            return
        t = time.perf_counter() - self._log_start
        if t > 10.0:
            return
        try:
            self._log_file.write(f"{t:.3f} {gx:.5f} {gy:.5f} {eye_w:.5f} {lid_h:.5f} {int(is_blink)}\n")
            self._log_file.flush()
        except Exception:
            pass

    def _get_eye_stats(self):
        ewL = getattr(self.tracker, "last_left_eye_w", None)
        ewR = getattr(self.tracker, "last_right_eye_w", None)
        lhL = getattr(self.tracker, "last_left_lid_h", None)
        lhR = getattr(self.tracker, "last_right_lid_h", None)
        eye_w = max([v for v in (ewL, ewR) if v is not None], default=None)
        lid_h = max([v for v in (lhL, lhR) if v is not None], default=None)
        return eye_w, lid_h

    def get_xy(self):
        if self.tracker is None or self.calibration is None:
            return None
        try:
            if hasattr(self.tracker, "process_frame_binocular"):
                gx, gy, yaw, pitch, fv, conf, is_blink = self.tracker.process_frame_binocular()
            else:
                gx, gy, is_blink = self.tracker.process_frame()
                yaw = getattr(self.tracker, "last_yaw", None)
                pitch = getattr(self.tracker, "last_pitch", None)
                fv = None
                conf = 1.0
            self._frame_count += 1

            if gx is None or gy is None:
                self._stats["drop"] += 1
                self._maybe_print_stats()
                return None

            eye_w, lid_h = self._get_eye_stats()
            if eye_w is None or lid_h is None:
                eye_w, lid_h = 0.0, 0.0
            self._log_sample(gx, gy, eye_w, lid_h, is_blink)

            if is_blink:
                self._stats["drop"] += 1
                self._maybe_print_stats()
                return None
            # Reject large jumps (spikes) before smoothing so cursor doesn't jump
            if self._last_gaze_before_filter is not None:
                lgx, lgy = self._last_gaze_before_filter
                if abs(gx - lgx) > self._gaze_jump_thresh or abs(gy - lgy) > self._gaze_jump_thresh:
                    gx, gy = lgx, lgy
            if conf < 0.2:
                self._stats["drop"] += 1
                self._maybe_print_stats()
                return None
            lid_ratio = (lid_h / eye_w) if eye_w > 1e-6 else 0.0
            if lid_h < 0.008 or eye_w < 0.015 or lid_ratio < 0.15:
                self._stats["drop"] += 1
                self._maybe_print_stats()
                return None
            if abs(gx) > 0.6 or abs(gy) > 0.6:
                self._stats["drop"] += 1
                self._maybe_print_stats()
                return None

            gaze_before_filter = (gx, gy)
            now = time.perf_counter()
            gx, gy = self._gaze_filter(gx, gy, t=now)

            used_ml = False
            used_binoc = False
            screen_pt = None
            ml = getattr(self.calibration, "ml", None)
            input_dim = getattr(ml, "input_dim", None) if ml is not None else None
            fv_dim_ok = (fv is not None and input_dim is not None and len(fv) == input_dim)
            if fv is not None and input_dim is not None and len(fv) != input_dim and not getattr(self, "_fv_dim_mismatch_logged", False):
                print(f"[cursor] ML skipped: fv len={len(fv)} != ml.input_dim={input_dim}. Fall back; retrain with current fv.")
                self._fv_dim_mismatch_logged = True
            fv_in = (fv[:input_dim] if fv_dim_ok and fv is not None else None)
            if fv_dim_ok and ml is not None and getattr(ml, "W", None) is not None and fv_in is not None:
                screen_pt = self.calibration.ml.predict(fv_in)
                if screen_pt is not None:
                    if not (0.0 <= screen_pt[0] <= 1.0 and 0.0 <= screen_pt[1] <= 1.0):
                        screen_pt = None
                    else:
                        screen_pt = (
                            screen_pt[0] * self.calibration.screen_w,
                            screen_pt[1] * self.calibration.screen_h,
                        )
                        used_ml = True
                        self._last_good["t"] = now
                        self._last_good["xy"] = screen_pt
            if not used_ml and fv is not None and len(fv) >= 14 and getattr(self.calibration, "binoc_W", None) is not None:
                screen_pt = self.calibration.map_binocular(fv)
                if screen_pt is not None:
                    if not (0.0 <= screen_pt[0] <= 1.0 and 0.0 <= screen_pt[1] <= 1.0):
                        screen_pt = None
                    else:
                        screen_pt = (
                            screen_pt[0] * self.calibration.screen_w,
                            screen_pt[1] * self.calibration.screen_h,
                        )
                        used_binoc = True
                        self._last_good["t"] = now
                        self._last_good["xy"] = screen_pt
            if screen_pt is None:
                screen_pt = self.calibration.gaze_to_screen(gx, gy, yaw=yaw, pitch=pitch, feature_vec=fv)
            if screen_pt is None:
                screen_pt = self.calibration.gaze_to_screen(gx, gy, yaw=yaw, pitch=pitch, feature_vec=fv)
            if screen_pt is None and self._last_good["xy"] is not None and (now - self._last_good["t"]) < 0.15:
                screen_pt = self._last_good["xy"]
            if screen_pt is None:
                self._stats["drop"] += 1
                self._maybe_print_stats()
                return None
            self._last_gaze_before_filter = gaze_before_filter
            if used_ml:
                self._stats["ml"] += 1
            elif used_binoc:
                self._stats["binoc"] += 1
            else:
                self._stats["fallback"] += 1
            self._maybe_print_stats()
            return int(screen_pt[0]), int(screen_pt[1])
        except Exception:
            self._stats["drop"] += 1
            self._maybe_print_stats()
            return None

    def _maybe_print_stats(self):
        if self._frame_count % self.STATS_INTERVAL != 0:
            return
        total = self._stats["ml"] + self._stats["binoc"] + self._stats["fallback"] + self._stats["drop"]
        if total == 0:
            return
        ml_pct = 100.0 * self._stats["ml"] / total
        binoc_pct = 100.0 * self._stats["binoc"] / total
        fallback_pct = 100.0 * self._stats["fallback"] / total
        drop_pct = 100.0 * self._stats["drop"] / total
        print(
            f"[cursor] ML={ml_pct:.1f}% binoc={binoc_pct:.1f}% fallback={fallback_pct:.1f}% drop={drop_pct:.1f}% "
            f"(n={total})"
        )

    def release(self):
        if self.tracker is None:
            return
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
        try:
            if getattr(self.tracker, "cap", None) is not None:
                self.tracker.cap.release()
        except Exception:
            pass
        try:
            if getattr(self.tracker, "back_cap", None) is not None:
                self.tracker.back_cap.release()
        except Exception:
            pass
