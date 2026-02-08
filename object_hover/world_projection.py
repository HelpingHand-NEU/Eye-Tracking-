import numpy as np


class WorldProjector:
    """
    Optional image->world mapping using a homography (ground plane).
    Provide a 3x3 homography H where world = H * [x, y, 1].
    """

    def __init__(self, homography=None):
        self.H = homography

    def image_to_world(self, xy):
        if self.H is None or xy is None:
            return None
        x, y = xy
        pt = np.array([x, y, 1.0], dtype=np.float64)
        out = self.H @ pt
        if abs(out[2]) < 1e-9:
            return None
        wx = out[0] / out[2]
        wy = out[1] / out[2]
        return float(wx), float(wy)
