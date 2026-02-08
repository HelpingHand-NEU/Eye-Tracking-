import cv2
import numpy as np
from input_source import CameraSource, ImageSource
from yolo_detector import YOLODetector
from cursor_sources import MouseCursor
from hover_selector import HoverSelector
from world_projection import WorldProjector


class HoverAppConfig:
    def __init__(
        self,
        source_type="camera",
        image_path=None,
        camera_id=0,
        model_path="yolov8n.pt",
        conf=0.25,
        hover_max_px=80,
        homography=None,
    ):
        self.source_type = source_type
        self.image_path = image_path
        self.camera_id = camera_id
        self.model_path = model_path
        self.conf = conf
        self.hover_max_px = hover_max_px
        self.homography = homography


class HoverApp:
    def __init__(self, config):
        self.config = config
        if config.source_type == "camera":
            self.source = CameraSource(camera_id=config.camera_id)
        elif config.source_type == "image":
            if not config.image_path:
                raise ValueError("image_path is required when source_type='image'")
            self.source = ImageSource(config.image_path)
        else:
            raise ValueError("source_type must be 'camera' or 'image'")

        self.detector = YOLODetector(model_path=config.model_path, conf=config.conf)
        self.cursor = MouseCursor()
        self.selector = HoverSelector(max_px=config.hover_max_px)
        self.projector = WorldProjector(homography=config.homography)

    def _draw_detection(self, frame, det, color=(0, 255, 0), thickness=2):
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        label = f'{det["label"]} {det["score"]:.2f}'
        cv2.putText(
            frame, label, (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    def run(self):
        while True:
            frame = self.source.read()
            if frame is None:
                break

            detections = self.detector.detect(frame)
            cursor_xy = self.cursor.get_xy()
            selected = self.selector.select(cursor_xy, detections)

            # Draw cursor
            cv2.circle(frame, cursor_xy, 6, (255, 255, 0), -1)

            # Draw all detections in light color
            for det in detections:
                self._draw_detection(frame, det, color=(80, 180, 80), thickness=1)

            # Highlight selected detection
            if selected:
                self._draw_detection(frame, selected, color=(0, 255, 255), thickness=3)
                cx = selected["bbox"][0] + selected["bbox"][2] // 2
                cy = selected["bbox"][1] + selected["bbox"][3] // 2
                world_xy = self.projector.image_to_world((cx, cy))
                if world_xy is not None:
                    cv2.putText(
                        frame, f"world=({world_xy[0]:.2f},{world_xy[1]:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                    )

            cv2.imshow("object_hover", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

        self.source.release()
        cv2.destroyAllWindows()


def main():
    # Example: camera source
    cfg = HoverAppConfig(
        source_type="camera",
        camera_id=0,
        model_path="yolov8n.pt",
        conf=0.35,
        hover_max_px=90,
        homography=None,
    )
    app = HoverApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
