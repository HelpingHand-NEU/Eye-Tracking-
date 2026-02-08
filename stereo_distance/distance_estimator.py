import numpy as np


class StereoDistanceEstimator:
    """
    Computes distance using stereo disparity.
    Z = (focal_px * baseline_m) / disparity_px
    """

    def __init__(self, focal_px, baseline_m, min_disp=1.0, max_disp=200.0):
        self.focal_px = float(focal_px)
        self.baseline_m = float(baseline_m)
        self.min_disp = float(min_disp)
        self.max_disp = float(max_disp)

    def estimate(self, left_pt, right_pt):
        """
        left_pt/right_pt: (x, y) pixel coordinates for the same object in left/right images.
        Returns distance in meters or None if invalid.
        """
        if left_pt is None or right_pt is None:
            return None
        lx, _ = left_pt
        rx, _ = right_pt
        disp = float(lx - rx)
        if not np.isfinite(disp):
            return None
        if abs(disp) < self.min_disp or abs(disp) > self.max_disp:
            return None
        z = (self.focal_px * self.baseline_m) / abs(disp)
        return float(z)
