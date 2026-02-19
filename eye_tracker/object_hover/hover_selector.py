import numpy as np


class HoverSelector:
    def __init__(self, max_px=80):
        self.max_px = float(max_px)

    @staticmethod
    def _dist_point_to_rect(px, py, x, y, w, h):
        rx0, ry0 = x, y
        rx1, ry1 = x + w, y + h
        dx = 0.0
        if px < rx0:
            dx = rx0 - px
        elif px > rx1:
            dx = px - rx1
        dy = 0.0
        if py < ry0:
            dy = ry0 - py
        elif py > ry1:
            dy = py - ry1
        return float(np.hypot(dx, dy))

    def select(self, cursor_xy, detections):
        if cursor_xy is None or not detections:
            return None
        cx, cy = cursor_xy
        best = None
        best_d = 1e18
        for det in detections:
            x, y, w, h = det["bbox"]
            d = self._dist_point_to_rect(cx, cy, x, y, w, h)
            if d <= self.max_px and d < best_d:
                best = det
                best_d = d
        if best is None:
            return None
        return best
