# Stereo Distance Prototype

This is a standalone, modular prototype that uses **two cameras** to estimate
distance to a detected object. It is intentionally separated into:

- `detector_interface.py` — object detection abstraction
- `distance_estimator.py` — stereo disparity distance math
- `stereo_app.py` — wiring + visualization

## How it works

1. Detect an object in the **left** and **right** frames.
2. Use the **pixel disparity** between left/right detections.
3. Estimate distance with:

```
Z = (focal_px * baseline_m) / disparity_px
```

## Quick start

```
python3 stereo_distance/stereo_app.py
```

Press `q` to quit.

## Replace the detector

Implement `ObjectDetector.detect(frame)` in `detector_interface.py`.
Return a list of detections:

```
{"bbox": (x, y, w, h), "score": 0.92, "label": "target"}
```

## Calibration needed

You must set the correct `focal_px` and `baseline_m` in `stereo_app.py` to get
accurate distances.
