#!/usr/bin/env python3
"""Switch between calibration, object detection (YOLO), and AprilTag detection."""
import os
import sys
import cv2
from eye_tracker.calibration import Calibration
from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.object_hover.app import HoverApp, HoverAppConfig

# --- Only indices 0 and 1 exist. 1=webcam (eye). For interface use 0=USB (disconnect iPhone so USB is 0). ---
CAM_WEBCAM = 1
CAM_EXTERNAL = 0


def run_mode(mode: str):
    mode = mode.strip().lower().replace(" ", "_")
    if mode not in ("calibrate", "object_detection", "april_tags"):
        print(f'Unknown mode "{mode}". Use "calibrate", "object_detection", or "april_tags".')
        sys.exit(1)
    if mode == "calibrate":
        from eye_tracker.eye_tracking_training import main as training_main
        training_main()
        return
    if mode == "april_tags":
        run_hover(detector_type="apriltag", interface_camera=True)
        return
    run_hover(detector_type="yolo", interface_camera=True)


def run_hover(detector_type="yolo", interface_camera=True, image_path=None, calibrate_first=False):
    """Run hover app: eye tracking (webcam) + interface (external camera or image)."""
    training_mode = "webcam"
    eye_cam = CAM_WEBCAM
    tracker = EyeTracker(front_camera_id=eye_cam, reset=True, training_mode=training_mode)
    calibration = Calibration(
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        screen_size=None,
        fullscreen=True,
        window_name="calibration",
        training_mode=training_mode,
    )
    if calibrate_first:
        if calibration.calibrate(tracker) == "quit":
            cv2.destroyAllWindows()
            sys.exit(0)
    else:
        if not calibration.load_saved_ml():
            print("No saved calibration (gaze_ml_webcam.npz). Run with mode 'calibrate' first.")
            cv2.destroyAllWindows()
            sys.exit(1)

    if interface_camera:
        source_type, camera_id, image_path = "camera", CAM_EXTERNAL, None
    else:
        source_type, camera_id = "image", 0
        if not image_path:
            raise ValueError("image_path required when interface_camera=False")

    cfg = HoverAppConfig(
        source_type=source_type,
        image_path=image_path,
        camera_id=camera_id,
        cursor_type="eye_tracking",
        eye_tracker=tracker,
        calibration=calibration,
        training_mode=training_mode,
        detector_type=detector_type,
        model_path="yolov8n.pt",
        conf=0.35,
        apriltag_families="tag36h11",
        hover_max_px=90,
        calibrate=calibrate_first,
        fullscreen=True,
    )
    HoverApp(cfg).run()


if __name__ == "__main__":
    run_mode(sys.argv[1] if len(sys.argv) > 1 else "object_detection")
