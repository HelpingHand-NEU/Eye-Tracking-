from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper for Ultralytics YOLO.
    detect(frame) -> list of detections:
      {"bbox": (x, y, w, h), "score": float, "label": str}
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.25):
        self.model = YOLO(model_path)
        self.conf = float(conf)

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        if not results:
            return []
        dets = []
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            score = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            dets.append(
                {
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "score": score,
                    "label": label,
                }
            )
        return dets
