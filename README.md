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


