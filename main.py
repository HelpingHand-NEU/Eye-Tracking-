#!/usr/bin/env python3
"""Eye tracking + AprilTag selection (default), calibration, or YOLO object hover.

Default — gaze cursor on the **back camera** window; look at an AprilTag to highlight it
(uses saved glass-frame / webcam calibration NPZ + CSV).

  python main.py                    # eye tracking on back camera + AprilTags
  python main.py calibrate          # collect data & train mapping
  python main.py object_detection   # YOLO objects instead of tags
  python main.py april_tags         # same as default (explicit)

When EYE_TRACKING_MODE is "glass_frame", the pupil pipeline uses JEO 3DTracker (CSI ROIs).

Optional: calibration_config.json — cameras, cursor smoothing, gaze_inference.mode
(poly_ridge | binocular_ridge | ensemble | affine | auto).
"""
import os
import sys
import json

# On Jetson, prefer system OpenCV (built with GStreamer) so CSI nvarguscamerasrc works.
# Pip OpenCV is often built without GStreamer and falls back to V4L2 probing.
if os.path.exists("/etc/nv_tegra_release") and os.path.exists("/usr/lib/python3/dist-packages"):
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

import cv2
from eye_tracker.calibration import Calibration
from eye_tracker.object_hover.app import HoverApp, HoverAppConfig
from eye_tracker.glass_frame_training import GlassFrameTraining, GlassFrameTrainingConfig
from eye_tracker.glass_frame_jeo_tracker import (
    JEOGlassFrameTracker,
    jeo_tracker_available,
    load_rois_and_signature,
)

# --- Camera mapping (Jetson Nano) ---
# Back camera (AprilTag / scene) = cam0 = 24-pin CSI = sensor-id 0.
# Eye camera (pupil / 3DTracker) = cam1 = 15-pin CSI = sensor-id 1.
# Glass-frame uses JEOresearch/EyeTracker 3DTracker (vendor); see CREDITS.md.
CAM_WEBCAM = 1   # Eye camera index when using USB (non-Jetson)
CAM_EXTERNAL = 0 # Fallback AprilTag camera (USB) when not using CSI for back
EYE_TRACKING_MODE = "glass_frame"  # "webcam" | "glass_frame" — glass_frame = CSI + 3DTracker
EYE_TRACKING_CAMERA_ID = CAM_WEBCAM
# Back camera: 24-pin CSI (cam0) = sensor 0. None = use TAG_CAMERA_ID as USB.
TAG_CAMERA_CSI_SENSOR_ID = 0  # cam0 / 24-pin CSI for AprilTag (back) camera
# Eye camera: 15-pin CSI (cam1) = sensor 1 when back uses cam0.
GLASS_EYE_SENSOR_ID = 1  # cam1 for pupil/eye (3DTracker); use 0 if only one CSI camera
TAG_CAMERA_ID = CAM_EXTERNAL  # used only when TAG_CAMERA_CSI_SENSOR_ID is None
GLASS_CALIBRATION_MODE = "automated_calibration"  # or "manual_calibration"
APRILTAG_LENGTH_M = 0.05
GLASS_QUALITY_PROFILE = "max_accuracy"  # kept for config API; glass_frame uses JEO 3DTracker only


def _project_root():
    return os.path.dirname(os.path.abspath(__file__))


def load_calibration_config():
    """Load calibration_config.json from project root if present. Returns dict or None."""
    path = os.path.join(_project_root(), "calibration_config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _print_usage():
    print(
        "Usage: python main.py [MODE]\n"
        "  (default)   Eye tracking: gaze cursor on back camera, AprilTag hover/select\n"
        "  calibrate   Run calibration / training (glass_frame or webcam)\n"
        "  april_tags  Same as default\n"
        "  object_detection | yolo   YOLO object hover instead of AprilTags\n"
        "  help        Show this message\n"
        "See calibration_config.json for cameras and gaze_inference.mode."
    )


def run_mode(mode: str):
    mode = mode.strip().lower().replace(" ", "_")
    aliases = {"yolo": "object_detection", "tags": "april_tags", "gaze": "eye_tracking"}
    mode = aliases.get(mode, mode)
    if mode in ("help", "-h", "--help"):
        _print_usage()
        return
    if mode not in ("calibrate", "eye_tracking", "object_detection", "april_tags"):
        print(f'Unknown mode "{mode}".')
        _print_usage()
        sys.exit(1)
    if mode == "calibrate":
        run_calibration()
        return
    if mode == "eye_tracking":
        run_eye_tracking_default()
        return
    if mode == "april_tags":
        run_eye_tracking_default()
        return
    run_hover(detector_type="yolo", interface_camera=True)


def run_eye_tracking_default():
    """Pupil → screen gaze → cursor on live back camera; AprilTag under cursor is selected."""
    print("Starting eye tracking (back camera + AprilTags). Press Q in the video window to quit.")
    run_hover(detector_type="apriltag", interface_camera=True)


def run_hover(detector_type="yolo", interface_camera=True, image_path=None, calibrate_first=False,
              training_file_override=None, roi_file_override=None):
    """Run hover app: eye tracking + interface (external camera or image).
    training_file_override: use this CSV (and its matching npz) for calibration (fallback to previous ROI set).
    roi_file_override: use this glass_frame_rois.json for tracker ROIs (fallback to previous ROI draw).
    Env: GLASS_FRAME_TRAINING_FILE, GLASS_FRAME_ROI_FILE for overrides without code change."""
    cfg_json = load_calibration_config()
    cam = (cfg_json.get("camera") or {}) if cfg_json else {}
    training_file_override = training_file_override or os.environ.get("GLASS_FRAME_TRAINING_FILE")
    roi_file_override = roi_file_override or os.environ.get("GLASS_FRAME_ROI_FILE")
    training_mode = (cfg_json.get("eye_tracking_mode") or EYE_TRACKING_MODE) if cfg_json else EYE_TRACKING_MODE
    rois_path = roi_file_override
    roi_signature = None
    if training_mode == "glass_frame":
        if not jeo_tracker_available():
            print("Glass-frame mode requires JEO 3DTracker. Clone: git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker")
            sys.exit(1)
        _, _path, roi_signature = load_rois_and_signature(roi_file_override)
        if rois_path is None:
            rois_path = _path
        tracker = JEOGlassFrameTracker(
            # For CSI path, keep eye_camera_id=0 (same as working dual-CSI test script).
            # eye_sensor_id selects cam0/cam1; eye_camera_id is only a fallback selector in tracker.
            eye_camera_id=0 if cam.get("glass_eye_sensor_id", GLASS_EYE_SENSOR_ID) is not None else cam.get("eye_tracking_camera_id", EYE_TRACKING_CAMERA_ID),
            eye_sensor_id=cam.get("glass_eye_sensor_id", GLASS_EYE_SENSOR_ID),
            rois_path=rois_path,
        )
        if not tracker.is_opened():
            print("Could not open glass-frame eye camera (CSI or USB). Check GLASS_EYE_SENSOR_ID and EYE_TRACKING_CAMERA_ID.")
            sys.exit(1)
    else:
        from eye_tracker.eye_tracker import EyeTracker
        tracker = EyeTracker(
            front_camera_id=EYE_TRACKING_CAMERA_ID,
            reset=True,
            training_mode=training_mode,
            glass_quality_profile=GLASS_QUALITY_PROFILE,
        )
    # When using a training file override, we don't need roi_signature for path resolution (override path has sig in name)
    calib_roi_sig = None if training_file_override else roi_signature
    calibration = Calibration(
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        screen_size=None,
        fullscreen=True,
        window_name="calibration",
        training_mode=training_mode,
        roi_signature=calib_roi_sig,
    )
    if training_file_override:
        calibration.set_training_override(training_file_override)
    # Expose which training file and ROI are in use (for fallback/reuse)
    if training_mode == "glass_frame":
        print(f"Training data file: {calibration.current_training_data_path}")
        if calibration.current_roi_signature:
            print(f"ROI signature: {calibration.current_roi_signature}")
        if getattr(tracker, "get_roi_info", None):
            roi_info = tracker.get_roi_info()
            if roi_info:
                print(f"ROI (normalized): left={roi_info.get('left_eye_roi')} right={roi_info.get('right_eye_roi')}")
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
        source_type, camera_id, image_path = "camera", cam.get("tag_camera_id", CAM_EXTERNAL), None
    else:
        source_type, camera_id = "image", 0
        if not image_path:
            raise ValueError("image_path required when interface_camera=False")

    csi_sensor_id = (cam.get("tag_camera_csi_sensor_id", TAG_CAMERA_CSI_SENSOR_ID) if (training_mode == "glass_frame" and interface_camera) else None)
    smooth = (cfg_json.get("cursor_smoothing") or {}) if cfg_json else {}
    gaze_inf = (cfg_json.get("gaze_inference") or {}) if cfg_json else {}
    gaze_mode = gaze_inf.get("mode", "auto")
    win_title = gaze_inf.get("window_title")
    if not win_title:
        win_title = (
            "Eye tracking — AprilTags (Q quit)"
            if detector_type == "apriltag"
            else "Object hover (Q quit)"
        )

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
        csi_sensor_id=csi_sensor_id,
        gaze_smooth_min_cutoff=smooth.get("min_cutoff"),
        gaze_smooth_beta=smooth.get("beta"),
        gaze_jump_thresh=smooth.get("gaze_jump_thresh"),
        window_name=win_title,
        gaze_inference_mode=gaze_mode,
    )
    HoverApp(cfg).run()


def run_calibration():
    """Run calibration from main. With glass_frame mode uses CSI eye camera + JEO 3DTracker."""
    cfg_json = load_calibration_config()
    training_mode = (cfg_json.get("eye_tracking_mode") or EYE_TRACKING_MODE) if cfg_json else EYE_TRACKING_MODE
    cal = (cfg_json.get("calibration") or {}) if cfg_json else {}
    cam = (cfg_json.get("camera") or {}) if cfg_json else {}
    if training_mode == "glass_frame":
        if not jeo_tracker_available():
            print("Glass-frame calibration requires JEO 3DTracker. Clone: git clone https://github.com/JEOresearch/EyeTracker.git vendor/EyeTracker")
            sys.exit(1)
        print("Running glass-frame calibration (CSI eye camera + JEO 3DTracker)...")
        roi_file = os.environ.get("GLASS_FRAME_ROI_FILE")
        default_tag_dir = os.path.expanduser(
            "~/Downloads/apriltags_36h11_ids0-10_50mm/print_50mm_300dpi"
        )
        tag_image_dir = cal.get("tag_image_dir")
        if tag_image_dir:
            tag_image_dir = os.path.expanduser(str(tag_image_dir))
        elif os.path.isdir(default_tag_dir):
            tag_image_dir = default_tag_dir
        else:
            tag_image_dir = None
        # Optional "tag_ids" in JSON = skip live camera discovery (fixed list). Otherwise tag count comes
        # from unique AprilTags the back camera sees during the discovery screen.
        tag_calibration_order = None
        if "tag_ids" in cal and cal["tag_ids"] is not None:
            tag_calibration_order = tuple(int(x) for x in cal["tag_ids"])
        tag_ids_tuple = (
            tag_calibration_order
            if tag_calibration_order is not None
            else (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        )
        cfg = GlassFrameTrainingConfig(
            calibration_mode=cal.get("mode") or GLASS_CALIBRATION_MODE,
            training_data_name="peter",
            training_data_dir="eyetracking_ml",
            # Keep CSI calibration path consistent with dual-CSI test script.
            eye_camera_id=0 if cam.get("glass_eye_sensor_id", GLASS_EYE_SENSOR_ID) is not None else cam.get("eye_tracking_camera_id", EYE_TRACKING_CAMERA_ID),
            eye_sensor_id=cam.get("glass_eye_sensor_id", GLASS_EYE_SENSOR_ID),
            tag_camera_id=cam.get("tag_camera_id", TAG_CAMERA_ID),
            tag_camera_csi_sensor_id=cam.get("tag_camera_csi_sensor_id", TAG_CAMERA_CSI_SENSOR_ID),
            apriltag_length=APRILTAG_LENGTH_M,
            tag_image_dir=tag_image_dir,
            tag_ids=tag_ids_tuple,
            tag_calibration_order=tag_calibration_order,
            num_automated_positions=cal.get("num_automated_positions", 9),
            training_mode="glass_frame",
            glass_quality_profile=GLASS_QUALITY_PROFILE,
            rois_path=roi_file if roi_file else None,
            validation_fraction=cal.get("validation_fraction", 0.15),
            outlier_mad_multiplier=cal.get("outlier_mad_multiplier", 2.5),
            min_train_samples=cal.get("min_train_samples", 10),
            tag_camera_buffer_drain=cal.get("tag_camera_buffer_drain", 2),
            overlap_tag_eye_detection=cal.get("overlap_tag_eye_detection", True),
            tag_quad_decimate=float(cal.get("tag_quad_decimate", 1.0)),
            tag_detect_scale=float(cal.get("tag_detect_scale", 1.0)),
            record_every_n_frames=int(cal.get("record_every_n_frames", 2)),
            stability_window_frames=int(cal.get("stability_window_frames", 5)),
            stability_std_threshold=float(cal.get("stability_std_threshold", 0.025)),
            aggregate_group_size=int(cal.get("aggregate_group_size", 4)),
        )
        GlassFrameTraining(cfg).run()
        return
    from eye_tracker.eye_tracker import EyeTracker
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
    run_mode(sys.argv[1] if len(sys.argv) > 1 else "eye_tracking")
