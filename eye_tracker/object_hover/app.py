import cv2
from .input_source import CameraSource, ImageSource
from .yolo_detector import YOLODetector
from .apriltag_detector import AprilTagDetector
from .cursor_sources import MouseCursor, WindowMouseCursor, EyeTrackerCursor
from .hover_selector import HoverSelector
from .world_projection import WorldProjector


class HoverAppConfig:
    def __init__(
        self,
        source_type="camera",
        image_path=None,
        camera_id=0,
        cursor_type="mouse_window",
        eye_tracker=None,
        calibration=None,
        fullscreen=False,
        training_mode="webcam",
        detector_type="yolo",
        model_path="yolov8n.pt",
        conf=0.25,
        imgsz=None,
        iou=0.6,
        apriltag_families="tag36h11",
        hover_max_px=80,
        homography=None,
        calibrate=True,
    ):
        self.source_type = source_type
        self.image_path = image_path
        self.camera_id = camera_id
        self.cursor_type = cursor_type
        self.eye_tracker = eye_tracker
        self.calibration = calibration
        self.fullscreen = bool(fullscreen)
        self.training_mode = training_mode
        self.detector_type = detector_type
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.apriltag_families = apriltag_families
        self.hover_max_px = hover_max_px
        self.homography = homography
        self.calibrate = bool(calibrate)


class HoverApp:
    def __init__(self, config):
        self.config = config
        self.window_name = "object_hover"
        self.screen_w = None
        self.screen_h = None
        if config.source_type == "camera":
            self.source = CameraSource(camera_id=config.camera_id)
        elif config.source_type == "image":
            if not config.image_path:
                raise ValueError("image_path is required when source_type='image'")
            self.source = ImageSource(config.image_path)
        else:
            raise ValueError("source_type must be 'camera' or 'image'")

        if config.detector_type == "apriltag":
            self.detector = AprilTagDetector(families=config.apriltag_families)
        else:
            self.detector = YOLODetector(
                model_path=config.model_path,
                conf=config.conf,
                imgsz=config.imgsz,
                iou=config.iou,
            )
        if config.cursor_type == "mouse_window":
            self.cursor = WindowMouseCursor(self.window_name)
            self.cursor_space = "window"
        elif config.cursor_type == "mouse_screen":
            self.cursor = MouseCursor()
            self.cursor_space = "screen"
        elif config.cursor_type == "eye_tracking":
            if config.training_mode != "webcam":
                raise ValueError("eye_tracking cursor requires training_mode='webcam' for now.")
            if config.eye_tracker is None or config.calibration is None:
                raise ValueError("eye_tracker and calibration are required for cursor_type='eye_tracking'")
            self.cursor = EyeTrackerCursor(config.eye_tracker, config.calibration)
            self.cursor_space = "screen"
            self.screen_w = int(config.calibration.screen_w)
            self.screen_h = int(config.calibration.screen_h)
        else:
            raise ValueError("cursor_type must be 'mouse_window', 'mouse_screen', or 'eye_tracking'")
        self.selector = HoverSelector(max_px=config.hover_max_px)
        self.projector = WorldProjector(homography=config.homography)
        self._last_window_size = None
        self._focused = None  # Current object/tag under gaze (for voice/robot integration)

    def _get_window_rect(self):
        try:
            x, y, w, h = cv2.getWindowImageRect(self.window_name)
            if w > 0 and h > 0:
                return int(x), int(y), int(w), int(h)
        except Exception:
            pass
        return None

    def _compute_display_rect(self, frame_w, frame_h, win_w, win_h):
        if frame_w <= 0 or frame_h <= 0 or win_w <= 0 or win_h <= 0:
            return None
        scale = min(win_w / frame_w, win_h / frame_h)
        disp_w = max(1, int(frame_w * scale))
        disp_h = max(1, int(frame_h * scale))
        x0 = (win_w - disp_w) // 2
        y0 = (win_h - disp_h) // 2
        return x0, y0, disp_w, disp_h, scale

    def _map_cursor_to_frame(self, cursor_xy, frame, win_rect):
        if cursor_xy is None:
            return None
        if self.cursor_space == "frame":
            return cursor_xy
        frame_h, frame_w = frame.shape[:2]
        if self.cursor_space == "screen" and self.screen_w and self.screen_h:
            win_x = win_y = 0
            win_w, win_h = self.screen_w, self.screen_h
        elif win_rect is None:
            win_x = win_y = 0
            win_w, win_h = frame_w, frame_h
        else:
            win_x, win_y, win_w, win_h = win_rect
        if win_w <= 0 or win_h <= 0:
            return None
        if self.cursor_space == "screen":
            cursor_xy = (cursor_xy[0] - win_x, cursor_xy[1] - win_y)
        disp = self._compute_display_rect(frame_w, frame_h, win_w, win_h)
        if disp is None:
            return None
        dx0, dy0, disp_w, disp_h, scale = disp
        cx, cy = cursor_xy
        if not (dx0 <= cx < dx0 + disp_w and dy0 <= cy < dy0 + disp_h):
            return None
        x = int((cx - dx0) / scale)
        y = int((cy - dy0) / scale)
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        return (x, y)

    def get_focused_object(self):
        """
        Return the object or AprilTag the user is currently looking at (under the gaze cursor).
        For voice command / robotic arm: call this to get what to act on.

        Returns:
            None if nothing is under the cursor, else a dict:
            - label: str, e.g. "tag_3" (AprilTag) or "bottle" (YOLO)
            - tag_id: int or None. AprilTag ID (1-10 etc.) if detector is apriltag, else None
            - bbox: (x, y, w, h) in image pixels
            - center: (cx, cy) center of the detection
        """
        if self._focused is None:
            return None
        x, y, w, h = self._focused["bbox"]
        out = {
            "label": self._focused["label"],
            "tag_id": self._focused.get("tag_id"),
            "bbox": (x, y, w, h),
            "center": (x + w // 2, y + h // 2),
        }
        return out

    def _draw_detection(self, frame, det, color=(0, 255, 0), thickness=2):
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        label = f'{det["label"]} {det["score"]:.2f}'
        cv2.putText(
            frame, label, (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.screen_w and self.screen_h:
            cv2.resizeWindow(self.window_name, self.screen_w, self.screen_h)
        if self.config.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if isinstance(self.cursor, WindowMouseCursor):
            cv2.setMouseCallback(self.window_name, self.cursor.handle_event)
        while True:
            frame = self.source.read()
            if frame is None:
                break

            win_rect = self._get_window_rect()
            if isinstance(self.source, ImageSource):
                if not hasattr(self, "_cached_dets"):
                    self._cached_dets = self.detector.detect(frame)
                detections = self._cached_dets
            else:
                detections = self.detector.detect(frame)
            raw_cursor_xy = self.cursor.get_xy()
            cursor_xy = self._map_cursor_to_frame(raw_cursor_xy, frame, win_rect)
            selected = self.selector.select(cursor_xy, detections)
            self._focused = selected  # Exposed via get_focused_object() for voice/robot

            # Draw cursor: green when AprilTag selected, yellow otherwise
            is_apriltag = selected and selected.get("label", "").startswith("tag_")
            cursor_color = (0, 255, 0) if is_apriltag else (255, 255, 0)
            if cursor_xy is not None:
                cv2.circle(frame, cursor_xy, 6, cursor_color, -1)
                if is_apriltag:
                    cv2.circle(frame, cursor_xy, 8, cursor_color, 2)

            # Highlight selected detection (rectangle around tag/object)
            if selected:
                sel_color = (0, 255, 0) if is_apriltag else (0, 255, 255)
                self._draw_detection(frame, selected, color=sel_color, thickness=3)
                cx = selected["bbox"][0] + selected["bbox"][2] // 2
                cy = selected["bbox"][1] + selected["bbox"][3] // 2
                world_xy = self.projector.image_to_world((cx, cy))
                if world_xy is not None:
                    cv2.putText(
                        frame, f"world=({world_xy[0]:.2f},{world_xy[1]:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                    )

            frame_h, frame_w = frame.shape[:2]
            if self.screen_w and self.screen_h:
                win_w, win_h = self.screen_w, self.screen_h
            elif win_rect is not None:
                _, _, win_w, win_h = win_rect
            else:
                win_w, win_h = frame_w, frame_h
            disp = self._compute_display_rect(frame_w, frame_h, win_w, win_h)
            if disp is None or (win_w == frame_w and win_h == frame_h):
                display = frame
            else:
                dx0, dy0, disp_w, disp_h, _ = disp
                canvas = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                display = cv2.copyMakeBorder(
                    canvas,
                    top=dy0,
                    bottom=max(0, win_h - dy0 - disp_h),
                    left=dx0,
                    right=max(0, win_w - dx0 - disp_w),
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

        self.source.release()
        if hasattr(self.cursor, "release"):
            try:
                self.cursor.release()
            except Exception:
                pass
        cv2.destroyAllWindows()


def main():
    # Example: camera source
    cfg = HoverAppConfig(
        source_type="image",
        image_path="path/to/your/image.jpg",
        cursor_type="mouse_window",
        model_path="yolov8n.pt",
        conf=0.35,
        hover_max_px=90,
        homography=None,
    )
    app = HoverApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
