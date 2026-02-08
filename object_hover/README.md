# Object Hover (YOLO + Cursor)

Standalone module that:
- runs YOLO object detection
- uses a cursor (mouse now, gaze later)
- draws a hover box on the selected object
- optionally projects object center to world XY

## Install

```
pip install ultralytics opencv-python pyautogui
```

## Run (camera)

```
python3 object_hover/app.py
```

Press `q` to quit.

## Run (image)

Edit `HoverAppConfig` in `object_hover/app.py`:

```
cfg = HoverAppConfig(
    source_type="image",
    image_path="/absolute/path/to/image.jpg",
)
```

## Swap cursor source

Later, replace `MouseCursor()` with your eye-tracker cursor:

```
self.cursor = EyeTrackerCursor(...)
```

## World XY

If you provide a homography matrix, the app will print world XY for the selected object.
Set `homography` in `HoverAppConfig`.

