import numpy as np


class HoverSelector:
    def __init__(self, max_px=80):
        self.max_px = float(max_px)

    def select(self, cursor_xy, detections):
        if cursor_xy is None or not detections:
            return None
        cx, cy = cursor_xy
        best = None
        best_d = None
        for det in detections:
            x, y, w, h = det["bbox"]
            px = x + w / 2.0
            py = y + h / 2.0
            d = np.hypot(px - cx, py - cy)
            if best is None or d < best_d:
                best = det
                best_d = d
        if best is None or best_d > self.max_px:
            return None
        return best
