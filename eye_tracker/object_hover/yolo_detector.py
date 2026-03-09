import os
import ssl
try:
    import certifi
except Exception:
    certifi = None

if certifi is not None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    _YOLO_IMPORT_ERROR = exc


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class YOLODetector:
    """
    Wrapper for Ultralytics YOLO.
    detect(frame) -> list of detections:
      {"bbox": (x, y, w, h), "score": float, "label": str}
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.25, imgsz=None, iou=0.6):
        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO is not available. Install dependencies for this mode "
                "(e.g. `pip install ultralytics`) and ensure torch is installed."
            ) from _YOLO_IMPORT_ERROR
        if model_path is None:
            model_path = "yolov8n.pt"
        if not os.path.isabs(model_path):
            model_path = os.path.join(_project_root(), model_path)
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz) if imgsz is not None else None
        self.iou = float(iou) if iou is not None else None

    def detect(self, frame):
        kwargs = {"source": frame, "conf": self.conf, "verbose": False}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.iou is not None:
            kwargs["iou"] = self.iou
        results = self.model.predict(**kwargs)
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
