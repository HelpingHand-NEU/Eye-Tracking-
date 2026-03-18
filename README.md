# Eye Tracking Application

A terminal-based eye tracking application that accesses both front and back cameras, tracks your eyes using MediaPipe, and overlays the eye tracking visualization on the forward-facing camera feed.

## Features

- **Dual Camera Support**: Automatically detects and uses both front and back cameras
- **Real-time Eye Tracking**: Uses MediaPipe Face Mesh for accurate eye and iris tracking
- **Visual Overlay**: Displays eye landmarks, eye centers, and iris (pupil) positions
- **Camera Switching**: Switch between front and back camera views with a keypress
- **Terminal-based**: Runs from the command line

## Requirements

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Windows: One-time environment setup

A dedicated virtual environment is set up in `eye_tracker_venv` with all dependencies installed. To use it:

- **Option A — Double-click:** Run `run_eye_tracker_win.bat` to start the Windows eye tracker (it uses the venv automatically).
- **Option B — Terminal:** From the `Eye-Tracking-` folder run:
  ```bat
  eye_tracker_venv\Scripts\activate
  python eye_tracker_win.py
  ```

## Usage

Run the application from the terminal:

```bash
python eye_tracker.py
```

### Command Line Arguments

- `--front-camera ID`: Specify the camera ID for the front camera (default: 0)
- `--back-camera ID`: Specify the camera ID for the back camera (default: 1)

Example:
```bash
python eye_tracker.py --front-camera 0 --back-camera 1
```

### Controls

- **'q'**: Quit the application
- **'s'**: Switch between front and back camera views
- **'f'**: Toggle fullscreen mode

## Visual Indicators

- **Green circles**: Eye centers
- **Yellow dots**: Eye landmark points
- **Blue circles**: Iris (pupil) centers with outer ring
- **Yellow line**: Connection between both eyes

## Notes

- The application will automatically try to find available cameras if the default IDs don't work
- If only one camera is available, the application will still run but camera switching will be disabled
- Make sure you have proper camera permissions on your system

---

## Modes and Running the App

| Mode | Description |
|------|-------------|
| **calibrate** | Eye-tracking calibration (follow dots). Run once; saves model for other modes. |
| **object_detection** | Live camera + YOLO; gaze selects objects. |
| **april_tags** | Live camera + AprilTags; gaze selects tags. |

```bash
python main.py                    # default: object_detection
python main.py calibrate
python main.py object_detection
python main.py april_tags
```

## Declaring Instances (Programmatic Use)

### Eye tracking: webcam vs glass frame

- **Webcam:** `EyeTracker(front_camera_id=CAM_WEBCAM, training_mode="webcam")`. Uses built-in webcam. Calibration: `gaze_ml_webcam.npz`, CSVs like `peter_webcam.csv`.
- **Glass frame:** Use `training_mode="glass_frame"` and the glass-frame training entry point.

```python
from eye_tracker.eye_tracker import EyeTracker

tracker = EyeTracker(front_camera_id=1, reset=True, training_mode="webcam")
```

### Calibration

```python
from eye_tracker.eye_tracking_training import main as calibration_main
calibration_main()  # Run calibration UI (webcam)
```

### Different users (initializing ML per user)

Each user should run **calibration once** with their own name. Set **`training_data_name`** and use the same **`training_data_dir`** so that training CSVs and the saved model are organized per user.

- **`training_data_name`**: e.g. `"peter"`, `"alice"`. Training CSV will be `{training_data_dir}/{name}_webcam.csv` (e.g. `eyetracking_ml/peter_webcam.csv`).
- **`training_data_dir`**: folder for calibration data (default `"eyetracking_ml"`).
- The saved ML model used at runtime is **`gaze_ml_webcam.npz`** in the project root (one file per machine; last calibration overwrites it). To support multiple users on one machine with separate models, you would need to save/load npz per user (e.g. `gaze_ml_webcam_{name}.npz`); current code uses a single shared npz.

**Example: calibrate for a new user**

```python
from eye_tracker.calibration import Calibration
from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.eye_tracking_training import EyeTrackingTraining, EyeTrackingTrainingConfig

config = EyeTrackingTrainingConfig(
    training_data_name="alice",   # different user
    training_data_dir="eyetracking_ml",
    fullscreen=True,
    training_mode="webcam",
)
trainer = EyeTrackingTraining(config)
trainer.run()  # follow dots; writes eyetracking_ml/alice_webcam.csv and updates gaze_ml_webcam.npz
```

Then for hover app / object detection, use the same **`training_data_name`** and **`training_data_dir`** when creating **Calibration**, and call **`calibration.load_saved_ml()`** so it loads the same model.

### Hover app: AprilTags or object detection

```python
from eye_tracker.calibration import Calibration
from eye_tracker.eye_tracker import EyeTracker
from eye_tracker.object_hover.app import HoverApp, HoverAppConfig

tracker = EyeTracker(front_camera_id=1, reset=True, training_mode="webcam")
calibration = Calibration(
    training_data_name="peter",
    training_data_dir="eyetracking_ml",
    screen_size=None, fullscreen=True, window_name="calibration", training_mode="webcam",
)
calibration.load_saved_ml()

# AprilTags
cfg = HoverAppConfig(
    source_type="camera", camera_id=0,
    cursor_type="eye_tracking", eye_tracker=tracker, calibration=calibration,
    training_mode="webcam", detector_type="apriltag", apriltag_families="tag36h11",
    hover_max_px=90, fullscreen=True,
)
app = HoverApp(cfg)
app.run()

# Object detection (YOLO): same but detector_type="yolo", model_path="yolov8n.pt", conf=0.35
```

## Voice Command / Robotic Arm: Get the Focused Object

The app highlights the object or AprilTag under the user’s gaze. **Voice command or robot code** can read it via the same `HoverApp` instance.

### API

```python
obj = app.get_focused_object()
```

- **Returns `None`** if nothing is under the cursor.
- **Otherwise** a dict:
  - **`label`**: str — e.g. `"tag_3"` (AprilTag) or `"bottle"` (YOLO)
  - **`tag_id`**: int or None — AprilTag ID (1–10, etc.) if AprilTags, else None
  - **`bbox`**: (x, y, w, h) — in image pixels
  - **`center`**: (cx, cy) — center of the detection
 
Use **`tag_id`** for AprilTags or **`label`** for YOLO when sending the target to the robotic arm.

### Integration pattern

`app.run()` is blocking. Run it in a thread and poll from your voice/robot logic:

```python
import threading

app = HoverApp(cfg)
thread = threading.Thread(target=app.run, daemon=True)
thread.start()

# When voice command needs “what is the user looking at?”:
focused = app.get_focused_object()
if focused:
    # AprilTags: focused["tag_id"]; YOLO: focused["label"]
    send_to_robotic_arm(focused["tag_id"] or focused["label"])
```

## Cameras

In `main.py`: **CAM_WEBCAM** (eye tracking), **CAM_EXTERNAL** (interface feed). Only indices 0 and 1 on some Macs; disconnect iPhone so USB can be 0.

## Jetson Nano Quick Start

1. Install JetPack and verify camera access first (`v4l2-ctl --list-devices`).
2. Create a venv and install Jetson dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-jetson.txt
```

3. If YOLO mode fails with torch errors, install Jetson-compatible PyTorch first, then reinstall `ultralytics`.
4. If MediaPipe is unavailable from pip on your Jetson image, install a Jetson-compatible MediaPipe build (wheel/source), then rerun.

### Linux/Jetson compatibility behavior in this repo

- Detector imports are lazy: missing YOLO/AprilTag packages no longer break unrelated modes.
- Errors now point to the exact missing package for the selected mode.

---

## Glass-frame (camera on glasses) and ML data

For the **camera mounted on the glass frame** (CSI on Jetson), all eye-tracking and **machine learning data** come from the vendored **JEOresearch/EyeTracker** 3DTracker. We do not use a separate in-tree implementation.

- **Training/calibration:** `glass_frame_training.py` uses `JEOGlassFrameTracker`, which runs the vendor’s `process_frame()` on each frame and reads `gaze_vector.txt` (origin + direction) to build the 16-D feature vector for ML. Run calibration with `python main.py calibrate` (with `EYE_TRACKING_MODE = "glass_frame"`).
- **Live tracking:** `main.py` and the hover app use `JEOGlassFrameTracker` for glass-frame mode; the same tracker feeds `map_binocular()` for gaze→screen mapping.
- **Standalone test:** `python test_csi_glass_frame_pupil.py` (default `--sensor-id 0` on Jetson).
- **ROI (better detection when camera isn’t close):** Run `python calibrate_glass_frame_rois.py [--sensor-id 0]`, press **1** and drag a rectangle around your **left** eye, press **2** and drag around your **right** eye, then **s** to save to `glass_frame_rois.json`. The next time you start the eye tracker (test, calibration, or main app), it will use these cropped regions for the 3D tracker and convert pupil coordinates back to full-frame (glass-frame) coordinates.

- **ROI redraw → new training file:** If you redraw the ROIs, a **new** training CSV (and matching `.npz` model) is created automatically so old data is not mixed with the new crop. Files are named with an ROI signature (e.g. `peter_glassframe_roi_a1b2c3d4.csv` and `gaze_ml_glassframe_roi_a1b2c3d4.npz`). A sidecar `*_roi_meta.json` next to each CSV stores the ROI coordinates and signature for that file.

- **Exposed for fallback/reuse:** You can switch back to a previous ROI set and its training data:
  - **Calibration:** `calibration.current_training_data_path` (which CSV is in use), `calibration.current_roi_signature` (8-char ROI id), `calibration.set_training_override(csv_path)` to use a specific CSV/npz pair.
  - **Tracker:** `tracker.get_roi_info(frame_w, frame_h)` returns normalized `left_eye_roi` / `right_eye_roi`, optional pixel rects `left_rect_px` / `right_rect_px`, and `roi_signature`.
  - **Env / args:** Set `GLASS_FRAME_TRAINING_FILE` or pass `training_file_override` to use an old CSV; set `GLASS_FRAME_ROI_FILE` or pass `roi_file_override` to use an old ROI file. Example: `GLASS_FRAME_TRAINING_FILE=eyetracking_ml/peter_glassframe_roi_abc12345.csv python main.py object_detection`.

See **CREDITS.md** and `vendor/EyeTracker/3DTracker/README_CSI.md`.

## Acknowledgments / Third-party open source

This project uses **someone else's open source repository** for 3D eye tracking on the glass-frame (CSI) camera:

- **[JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker)** — 3DTracker (Orlosky 3D eye tracker). The algorithm (pupil detection, ellipse fitting, ray intersection, 3D gaze vector) is from that repo. We vendor it under `vendor/EyeTracker` and use it for all glass-frame tracking and ML; we do not claim authorship of that code. See `vendor/EyeTracker/LICENSE` and their [3DTracker readme](https://github.com/JEOresearch/EyeTracker/tree/main/3DTracker).


