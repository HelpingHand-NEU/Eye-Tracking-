class ObjectDetector:
    """
    Interface for object detection.
    Implement detect(frame) -> list of detections, each with:
      - "bbox": (x, y, w, h) in pixel coords
      - "score": float
      - "label": str
    """

    def detect(self, frame):
        raise NotImplementedError


class DummyCenterDetector(ObjectDetector):
    """
    Fallback detector that returns the frame center as a single detection.
    Useful for testing the stereo distance pipeline without a real detector.
    """

    def detect(self, frame):
        h, w = frame.shape[:2]
        bbox_w = max(10, w // 10)
        bbox_h = max(10, h // 10)
        x = w // 2 - bbox_w // 2
        y = h // 2 - bbox_h // 2
        return [
            {"bbox": (x, y, bbox_w, bbox_h), "score": 1.0, "label": "center"}
        ]
