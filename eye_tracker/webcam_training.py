import cv2
import numpy as np
import os

from .calibration import Calibration, _get_screen_size
from .eye_tracker import EyeTracker
from .object_hover.yolo_detector import YOLODetector


class EyeTrackingTrainingConfig:
    def __init__(
        self,
        screen_size=(1512, 982),
        image_path=None,
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        model_path="yolov8l.pt",
        conf=0.15,
        imgsz=960,
        iou=0.6,
        random_points=5,
        dwell_frames=15,
        fullscreen=True,
        training_mode="webcam",
    ):
        self.screen_size = screen_size
        self.image_path = image_path
        self.training_data_name = training_data_name
        self.training_data_dir = training_data_dir
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.random_points = int(random_points)
        self.dwell_frames = int(dwell_frames)
        self.fullscreen = bool(fullscreen)
        self.training_mode = training_mode


class EyeTrackingTraining:
    def __init__(self, config):
        if config.training_mode != "webcam":
            raise ValueError("EyeTrackingTraining supports training_mode='webcam' only.")
        self.config = config
        self.window_name = "eye_tracking_training"
        if config.screen_size is None:
            self.screen_w, self.screen_h = _get_screen_size()
        else:
            self.screen_w = int(config.screen_size[0])
            self.screen_h = int(config.screen_size[1])
        self.detector = YOLODetector(
            model_path=config.model_path,
            conf=config.conf,
            imgsz=config.imgsz,
            iou=config.iou,
        )

    def _init_tracker(self):
        # Use same index as main.py CAM_WEBCAM (1 = webcam when iPhone at 0)
        return EyeTracker(front_camera_id=1, reset=True, training_mode="webcam")

    def _ensure_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.config.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, self.screen_w, self.screen_h)

    def _draw_arrow(self, img, target_xy):
        cx, cy = self.screen_w // 2, self.screen_h // 2
        tx, ty = target_xy
        cv2.arrowedLine(img, (cx, cy), (tx, ty), (0, 255, 255), 4, tipLength=0.2)

    def _draw_dot(self, img, xy, color=(0, 255, 255), radius=15):
        cv2.circle(img, (int(xy[0]), int(xy[1])), radius, color, 2)

    def _transition(self, target_xy, seconds=1.0):
        t0 = cv2.getTickCount()
        freq = cv2.getTickFrequency()
        while (cv2.getTickCount() - t0) / freq < seconds:
            img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            img[:] = (30, 30, 30)
            self._draw_arrow(img, target_xy)
            cv2.putText(
                img,
                "Look at the next dot",
                (self.screen_w // 2 - 140, self.screen_h // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("n"):
                return False
        return True

    def _collect_feature(self, tracker):
        gx, gy, yaw, pitch, fv, conf, is_blink = tracker.process_frame_binocular()
        if is_blink or gx is None or gy is None or fv is None:
            return None
        return fv, conf

    def _append_rows(self, calib, rows):
        if rows:
            calib._append_training_rows(rows)

    def run(self):
        tracker = self._init_tracker()
        calib = Calibration(
            training_data_name=self.config.training_data_name,
            training_data_dir=self.config.training_data_dir,
            screen_size=self.config.screen_size,
            fullscreen=self.config.fullscreen,
            window_name=self.window_name,
            training_mode="webcam",
        )

        self._ensure_window()
        result = calib.calibrate(tracker)
        if result == "quit":
            cv2.destroyAllWindows()
            return
        if result == "next":
            pass
        self._ensure_window()

        self._run_random_points(tracker, calib)
        if self.config.image_path:
            self._run_object_training(tracker, calib, self.config.image_path)

        cv2.destroyAllWindows()

    def _run_random_points(self, tracker, calib):
        if self.config.random_points <= 0:
            return
        self._ensure_window()
        points = []
        margin_x = int(self.screen_w * 0.08)
        margin_y = int(self.screen_h * 0.08)
        for _ in range(self.config.random_points):
            x = np.random.randint(margin_x, self.screen_w - margin_x)
            y = np.random.randint(margin_y, self.screen_h - margin_y)
            points.append((x, y))

        rows = []
        for i, pt in enumerate(points):
            if not self._transition(pt, seconds=1.0):
                break
            samples = []
            t0 = cv2.getTickCount()
            freq = cv2.getTickFrequency()
            while (cv2.getTickCount() - t0) / freq < 1.5:
                fv_conf = self._collect_feature(tracker)
                if fv_conf is None:
                    continue
                fv, conf = fv_conf
                samples.append((fv, conf))
                img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                img[:] = (30, 30, 30)
                self._draw_dot(img, pt)
                cv2.putText(
                    img,
                    f"Random point {i + 1}/{len(points)}",
                    (self.screen_w // 2 - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(self.window_name, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("n"):
                    return
            if not samples:
                continue
            fv_stack = np.stack([s[0] for s in samples], axis=0)
            fv_mean = np.mean(fv_stack, axis=0)
            conf = float(np.mean([s[1] for s in samples]))
            row = self._make_training_row(
                tracker=tracker,
                fv=fv_mean,
                conf=conf,
                screen_x=pt[0],
                screen_y=pt[1],
                stage="random5",
            )
            rows.append(row)

        self._append_rows(calib, rows)

    def _run_object_training(self, tracker, calib, image_path):
        self._ensure_window()
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image: {image_path}")
            return
        detections = self.detector.detect(frame)
        if not detections:
            print("No detections found for object training.")
            return

        rows = []
        for idx, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            if not self._transition((self.screen_w // 2, self.screen_h // 2), seconds=0.5):
                break

            buf = []
            while True:
                fv_conf = self._collect_feature(tracker)
                if fv_conf is not None:
                    buf.append(fv_conf)
                    if len(buf) > self.config.dwell_frames:
                        buf.pop(0)

                display = self._render_image(frame, highlight=det)
                cv2.putText(
                    display,
                    f"Object {idx + 1}/{len(detections)} - look at box and press SPACE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("n"):
                    return
                if key == ord(" "):
                    if not buf:
                        continue
                    fv_stack = np.stack([b[0] for b in buf], axis=0)
                    fv_mean = np.mean(fv_stack, axis=0)
                    conf = float(np.mean([b[1] for b in buf]))
                    row = self._make_training_row(
                        tracker=tracker,
                        fv=fv_mean,
                        conf=conf,
                        screen_x=cx,
                        screen_y=cy,
                        stage="object",
                        bbox=(x, y, w, h),
                        label_xy=(cx, cy),
                    )
                    rows.append(row)
                    break

        self._append_rows(calib, rows)

    def _render_image(self, frame, highlight=None):
        frame_h, frame_w = frame.shape[:2]
        scale = min(self.screen_w / frame_w, self.screen_h / frame_h)
        disp_w = max(1, int(frame_w * scale))
        disp_h = max(1, int(frame_h * scale))
        dx0 = (self.screen_w - disp_w) // 2
        dy0 = (self.screen_h - disp_h) // 2
        resized = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        if highlight is not None:
            x, y, w, h = highlight["bbox"]
            x = int(x * scale)
            y = int(y * scale)
            w = int(w * scale)
            h = int(h * scale)
            cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 255), 3)
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        canvas[:] = (0, 0, 0)
        canvas[dy0 : dy0 + disp_h, dx0 : dx0 + disp_w] = resized
        return canvas

    def _make_training_row(
        self,
        tracker,
        fv,
        conf,
        screen_x,
        screen_y,
        stage,
        bbox=None,
        label_xy=None,
    ):
        screen_x_norm = float(screen_x) / float(self.screen_w)
        screen_y_norm = float(screen_y) / float(self.screen_h)
        row = {
            "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
            "dot_index": "",
            "stage": stage,
            "screen_x": screen_x,
            "screen_y": screen_y,
            "screen_x_norm": screen_x_norm,
            "screen_y_norm": screen_y_norm,
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "gx": "",
            "gy": "",
            "yaw": getattr(tracker, "last_yaw", 0.0),
            "pitch": getattr(tracker, "last_pitch", 0.0),
            "gxL": float(fv[0]),
            "gyL": float(fv[1]),
            "gxR": float(fv[2]),
            "gyR": float(fv[3]),
            "iod": float(getattr(tracker, "last_iod", 0.0) or 0.0),
            "roll": float(getattr(tracker, "last_roll", 0.0) or 0.0),
            "lid_hL": float(getattr(tracker, "last_left_lid_h", 0.0) or 0.0),
            "lid_hR": float(getattr(tracker, "last_right_lid_h", 0.0) or 0.0),
            "face_w": float(getattr(tracker, "last_face_w", 0.0) or 0.0),
            "face_h": float(getattr(tracker, "last_face_h", 0.0) or 0.0),
            "nose_x": float(getattr(tracker, "last_nose_x", 0.0) or 0.0),
            "nose_y": float(getattr(tracker, "last_nose_y", 0.0) or 0.0),
            "eye_w": "",
            "lid_h": "",
            "conf": conf,
            "bbox_x": "",
            "bbox_y": "",
            "bbox_w": "",
            "bbox_h": "",
            "label_x": "",
            "label_y": "",
            "label_x_norm": "",
            "label_y_norm": "",
        }
        if bbox is not None:
            row["bbox_x"] = bbox[0]
            row["bbox_y"] = bbox[1]
            row["bbox_w"] = bbox[2]
            row["bbox_h"] = bbox[3]
        if label_xy is not None:
            row["label_x"] = label_xy[0]
            row["label_y"] = label_xy[1]
            row["label_x_norm"] = float(label_xy[0]) / float(self.screen_w)
            row["label_y_norm"] = float(label_xy[1]) / float(self.screen_h)
        return row


def main():
    default_image = "/Users/peterwang/Library/CloudStorage/OneDrive-Personal/文档/Capstone/Table3.jpg"
    image_path = default_image if os.path.exists(default_image) else None
    cfg = EyeTrackingTrainingConfig(
        screen_size=(1512, 982),
        image_path=image_path,
        training_data_name="peter",
    )
    trainer = EyeTrackingTraining(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
