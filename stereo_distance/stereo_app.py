import cv2
import time
from detector_interface import DummyCenterDetector
from distance_estimator import StereoDistanceEstimator


class StereoCameraPair:
    def __init__(self, left_id=0, right_id=1, width=640, height=480):
        self.left_id = left_id
        self.right_id = right_id
        self.width = width
        self.height = height
        self.left = cv2.VideoCapture(left_id)
        self.right = cv2.VideoCapture(right_id)
        self._configure(self.left)
        self._configure(self.right)

    def _configure(self, cap):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self):
        ret_l, frame_l = self.left.read()
        ret_r, frame_r = self.right.read()
        if not ret_l or not ret_r:
            return None, None
        return frame_l, frame_r

    def release(self):
        if self.left:
            self.left.release()
        if self.right:
            self.right.release()


def bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)


def pick_best_detection(detections):
    if not detections:
        return None
    return max(detections, key=lambda d: d.get("score", 0.0))


def main():
    # Replace DummyCenterDetector with your real detector later.
    detector = DummyCenterDetector()
    # These must be calibrated for your camera rig.
    focal_px = 700.0
    baseline_m = 0.06
    estimator = StereoDistanceEstimator(focal_px=focal_px, baseline_m=baseline_m)

    cams = StereoCameraPair(left_id=0, right_id=1, width=640, height=480)
    try:
        while True:
            frame_l, frame_r = cams.read()
            if frame_l is None or frame_r is None:
                continue

            det_l = pick_best_detection(detector.detect(frame_l))
            det_r = pick_best_detection(detector.detect(frame_r))

            left_pt = bbox_center(det_l["bbox"]) if det_l else None
            right_pt = bbox_center(det_r["bbox"]) if det_r else None
            z_m = estimator.estimate(left_pt, right_pt)

            # Draw detections
            if det_l:
                x, y, w, h = det_l["bbox"]
                cv2.rectangle(frame_l, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if det_r:
                x, y, w, h = det_r["bbox"]
                cv2.rectangle(frame_r, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if z_m is not None:
                cv2.putText(frame_l, f"Z={z_m:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame_r, f"Z={z_m:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("left", frame_l)
            cv2.imshow("right", frame_r)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
    finally:
        cams.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
