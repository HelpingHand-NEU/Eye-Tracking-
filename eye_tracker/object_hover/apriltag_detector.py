"""
AprilTag detector with the same interface as YOLODetector.
detect(frame) -> list of {"bbox": (x, y, w, h), "score": float, "label": str}
"""
import cv2
import numpy as np
try:
    from pupil_apriltags import Detector
except Exception as exc:
    Detector = None
    _APRILTAG_IMPORT_ERROR = exc


class AprilTagDetector:
    """
    Wrapper for pupil-apriltags.
    detect(frame) -> list of detections:
      {"bbox": (x, y, w, h), "score": float, "label": str}
    """

    def __init__(self, families="tag36h11", quad_decimate=1.0, quad_sigma=0.0, refine_edges=True, **kwargs):
        if Detector is None:
            raise ImportError(
                "pupil-apriltags is not available. Install dependencies for AprilTag mode "
                "(e.g. `pip install pupil-apriltags`)."
            ) from _APRILTAG_IMPORT_ERROR
        self.detector = Detector(
            families=families,
            nthreads=1,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            **kwargs,
        )

    def detect(self, frame):
        if frame is None:
            return []
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        results = self.detector.detect(gray)
        dets = []
        for r in results:
            corners = np.array(r.corners, dtype=np.int32)
            x_min = int(corners[:, 0].min())
            y_min = int(corners[:, 1].min())
            x_max = int(corners[:, 0].max())
            y_max = int(corners[:, 1].max())
            w = x_max - x_min
            h = y_max - y_min
            score = float(getattr(r, "decision_margin", 0.0) or 0.0)
            label = f"tag_{r.tag_id}"
            dets.append(
                {
                    "bbox": (x_min, y_min, w, h),
                    "score": score,
                    "label": label,
                    "tag_id": r.tag_id,
                }
            )
        return dets
