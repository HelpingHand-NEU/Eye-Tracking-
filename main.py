#!/usr/bin/env python3
"""Switch between calibration, object detection (YOLO), and AprilTag detection."""
import os
import sys
import cv2
from eye_tracker.calibration import Calibration
from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.object_hover.app import HoverApp, HoverAppConfig
from eye_tracker.glass_frame_training import GlassFrameTraining, GlassFrameTrainingConfig

# --- Only indices 0 and 1 exist. 1=webcam (eye). For interface use 0=USB (disconnect iPhone so USB is 0). ---
CAM_WEBCAM = 1
CAM_EXTERNAL = 0
# Eye tracking source:
# - "webcam": face-landmark + iris (MediaPipe)
# - "glass_frame": pupil-only tracking (no face landmarks)
EYE_TRACKING_MODE = "glass_frame"
# Camera used by the eye-tracking source above.
EYE_TRACKING_CAMERA_ID = CAM_WEBCAM
# Outward-facing camera used for AprilTag board/tag capture in glass-frame calibration.
TAG_CAMERA_ID = CAM_EXTERNAL
# For glass-frame mode:
GLASS_CALIBRATION_MODE = "automated_calibration"  # or "manual_calibration"
APRILTAG_LENGTH_M = 0.04
GLASS_QUALITY_PROFILE = "max_accuracy"  # fast | balanced | max_accuracy


def run_mode(mode: str):
    mode = mode.strip().lower().replace(" ", "_")
    if mode not in ("calibrate", "object_detection", "april_tags"):
        print(f'Unknown mode "{mode}". Use "calibrate", "object_detection", or "april_tags".')
        sys.exit(1)
    if mode == "calibrate":
        run_calibration()
        return
    if mode == "april_tags":
        run_hover(detector_type="apriltag", interface_camera=True)
        return
    run_hover(detector_type="yolo", interface_camera=True)


def run_hover(detector_type="yolo", interface_camera=True, image_path=None, calibrate_first=False):
    """Run hover app: eye tracking + interface (external camera or image)."""
    training_mode = EYE_TRACKING_MODE
    eye_cam = EYE_TRACKING_CAMERA_ID
    tracker = EyeTracker(
        front_camera_id=eye_cam,
        reset=True,
        training_mode=training_mode,
        glass_quality_profile=GLASS_QUALITY_PROFILE,
    )
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
            model_name = "gaze_ml_glassframe.npz" if training_mode == "glass_frame" else "gaze_ml_webcam.npz"
            print(f"No saved calibration ({model_name}). Run with mode 'calibrate' first.")
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


def run_calibration():
    """Run calibration for the currently selected eye-tracking source."""
    training_mode = EYE_TRACKING_MODE
    if training_mode == "glass_frame":
        cfg = GlassFrameTrainingConfig(
            calibration_mode=GLASS_CALIBRATION_MODE,
            training_data_name="peter",
            training_data_dir="eyetracking_ml",
            eye_camera_id=EYE_TRACKING_CAMERA_ID,
            tag_camera_id=TAG_CAMERA_ID,
            apriltag_length=APRILTAG_LENGTH_M,
            tag_ids=(1, 2, 3, 4, 5),
            training_mode="glass_frame",
            glass_quality_profile=GLASS_QUALITY_PROFILE,
        )
        GlassFrameTraining(cfg).run()
        return
    tracker = EyeTracker(
        front_camera_id=EYE_TRACKING_CAMERA_ID,
        reset=True,
        training_mode=training_mode,
        glass_quality_profile=GLASS_QUALITY_PROFILE,
    )
    calibration = Calibration(
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        screen_size=None,
        fullscreen=True,
        window_name="calibration",
        training_mode=training_mode,
    )
    result = calibration.calibrate(tracker)
    cv2.destroyAllWindows()
    if result == "quit":
        sys.exit(0)


if __name__ == "__main__":
    run_mode(sys.argv[1] if len(sys.argv) > 1 else "object_detection")
