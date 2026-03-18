# Calibration data and manual-mode cameras

This document confirms (1) what data is recorded for training and whether it is sufficient for an accurate model, and (2) how manual mode uses the two cameras (back = AprilTag, front = pupil with cropped frames).

---

## 1. Data recorded for training

### 1.1 What is recorded (glass-frame CSV)

Each training row contains:

**Targets (where the user was looking):**

| Column            | Meaning |
|-------------------|--------|
| `screen_x`, `screen_y` | Target position in “screen” space (pixels). |
| `screen_x_norm`, `screen_y_norm` | Same target in normalized [0, 1] (used for training). |
| `screen_w`, `screen_h` | Width/height of the “screen” (display or tag-camera image). |

**16-D feature vector (inputs to the model):**

| Column   | Source | Meaning |
|----------|--------|--------|
| `gxL`, `gyL` | Eye tracker (left crop) | Left-eye gaze direction (2D from 3D). |
| `gxR`, `gyR` | Eye tracker (right crop) | Right-eye gaze direction. |
| `yaw`, `pitch` | Placeholder for glass-frame | 0 (JEO tracker does not expose head pose here). |
| `iod`, `roll` | Placeholder | 0.01, 0. |
| `lid_hL`, `lid_hR` | Placeholder | 0.02 each. |
| `face_w`, `face_h` | Placeholder | 1.0 each. |
| `nose_x`, `nose_y` | Placeholder | 0.5 each. |
| `wL`, `wR` | Placeholder | 0.5 each. |

So the **effective inputs** for the current glass-frame pipeline are the four gaze components **gxL, gyL, gxR, gyR**; the rest are fixed placeholders. The model learns **gaze (L/R) → screen position (norm)**.

**Extra columns (for analysis / future algorithms):**

| Column | Meaning |
|--------|--------|
| `pupil_L_x_norm`, `pupil_L_y_norm` | Left pupil in left-crop coords, normalized [0,1]. |
| `pupil_R_x_norm`, `pupil_R_y_norm` | Right pupil in right-crop coords, normalized [0,1]. |
| `pupil_L_x_full`, `pupil_L_y_full` | Left pupil in main (full) frame pixels. |
| `pupil_R_x_full`, `pupil_R_y_full` | Right pupil in main frame pixels. |
| `tag_id` | AprilTag ID. |
| `tag_cam_x`, `tag_cam_y` | Tag center in tag-camera image (back camera). |
| `tag_center_x`, `tag_center_y` | Tag center from corner mean (more stable under perspective). |
| `tag_c0_x` … `tag_c3_y` | Four corner coordinates of the tag in the image (for homography/refinement). |
| `label_x_norm`, `label_y_norm` | Same as tag position normalized by tag frame size. |
| `board_distance_m` | Rough distance to tag (from tag size in image). |
| `conf`, `quality` | Tracker confidence / row quality. |

So we **do** record enough **types** of data for training and for later analysis:

- **For the current model:** 16-D feature (gaze L/R + placeholders) and `screen_x_norm`, `screen_y_norm`. This is the correct type and format used by `_load_training_data()` and the ridge/poly models.
- **For your own algorithm:** You also have per-eye pupil positions (crop-normalized and full-frame), tag positions in the back camera, and optional distance. That is enough to try alternative mappings (e.g. pupil position → tag position, or different feature combinations).

### 1.2 Sufficiency for an accurate model

- **Minimum:** Training requires ≥10 samples; binocular ridge needs ≥8. With 5–13 automated targets or many tags in manual mode, you can easily exceed this.
- **Quality:** Rows are dropped if `quality < 0.7`, or invalid/finite checks fail. So enough **good** rows matter more than raw count.
- **Coverage:** More targets (e.g. 9 or 13 in automated, or many tags in manual) improve spatial coverage and usually accuracy.
- **Conclusion:** The **types** of data are correct and sufficient; accuracy depends on **amount**, **quality**, and **spatial coverage** of the collected rows. Use the exploration script (below) to inspect relationships and residuals after you have data.

### 1.3 Using graphs to explore and develop your own algorithm

Once you have a CSV (e.g. after a manual or automated run), you can:

1. **Load the CSV** and filter valid rows (e.g. `quality >= 0.7`, finite values).
2. **Plot relationships**, for example:
   - **Gaze vs target:** `screen_x_norm` vs `gxL` (or `(gxL+gxR)/2`), and same for y. You should see approximate monotonic relationships; spread indicates noise or need for more features.
   - **Left vs right:** `gxL` vs `gxR`, `gyL` vs `gyR` (agreement between eyes).
   - **Pupil vs target:** `pupil_L_x_norm` vs `screen_x_norm` (and y) to see if raw pupil position alone correlates with look-at position.
   - **Residuals:** After fitting the current model (or a simple linear fit), plot `predicted - target` vs `target` or vs `gxL` to see bias/variance and consider a custom mapping.
3. **Develop your own algorithm:** e.g. different features (pupil norm/full, gaze L/R, tag distance), different model (linear, polynomial, or small MLP), or different normalization. The same CSV columns support this.

A small **exploration script** is provided:

```bash
python scripts/explore_glass_frame_training_data.py [path_to.csv]
```

If no path is given, it looks for `eyetracking_ml/*glassframe*.csv`. It produces plots in `eyetracking_ml/explore_plots/`: gaze vs screen, left vs right gaze, target coverage, and (if present) pupil vs screen. Use these to explore relationships and to develop your own algorithm once you have enough data.

---

## 2. Manual mode: two cameras and their roles

Manual mode **does** use two cameras as intended.

### 2.1 Back camera (AprilTag / “tag” camera)

- **Config:** `tag_camera_id` → in `main.py` set via `TAG_CAMERA_ID = CAM_EXTERNAL` (e.g. USB camera index 0).
- **Role:** Captures the scene (e.g. board with AprilTags). Each frame is passed to the AprilTag detector; when a tag is detected, its **center (cx, cy) in the tag-camera image** is used as the **target** for that row.
- **Recorded:**  
  `screen_xy = (cx, cy)`, `screen_size = (tag_frame width, tag_frame height)`  
  so `screen_x_norm = cx / tag_frame_width`, `screen_y_norm = cy / tag_frame_height`.  
  Also stored: `tag_cam_x`, `tag_cam_y`, `label_x_norm`, `label_y_norm`, `tag_id`, `tag_bbox_*`, `board_distance_m`.
- **Summary:** The **back camera** provides the **coordinates of the AprilTag(s)** (and thus the “look-at” target in tag-camera image space).

### 2.2 Front-facing camera (eye / pupil camera) and cropped frames

- **Config:** `eye_camera_id` and `eye_sensor_id` → in `main.py` passed as `EYE_TRACKING_CAMERA_ID` and `GLASS_EYE_SENSOR_ID`. For Jetson, `GLASS_EYE_SENSOR_ID = 0` uses the **CSI** camera (glass-frame camera facing the user’s eyes).
- **Role:** Captures the user’s face/eyes. The **cropped-frame strategy** is used:
  - Left/right eye ROIs from `glass_frame_rois.json` are applied to the full frame.
  - Each ROI is resized to the size expected by the JEO 3DTracker and processed separately.
  - The tracker returns gaze directions and pupil positions; pupil positions are converted to **crop-normalized** and **full-frame** coordinates.
- **Recorded:**  
  - 16-D feature vector (including `gxL`, `gyL`, `gxR`, `gyR`) from `tracker.process_frame_binocular()`.  
  - `pupil_L_x_norm`, `pupil_L_y_norm`, `pupil_R_x_norm`, `pupil_R_y_norm` (in crop).  
  - `pupil_L_x_full`, `pupil_L_y_full`, `pupil_R_x_full`, `pupil_R_y_full` (in main frame).  
  - `conf` / `quality`.
- **Summary:** The **front-facing (eye) camera** provides **pupil and gaze data** using the **cropped left/right ROI** strategy; all of this is in relation to the **main (full) frame** where needed (e.g. full-frame pupil coordinates).

### 2.3 Data flow in manual mode

1. **Tag camera** (back): `tag_cap.read()` → detect AprilTags → for each detected tag, (cx, cy) in tag image → `screen_xy`, `screen_x_norm`, `screen_y_norm`.
2. **Eye camera** (front): `tracker.process_frame_binocular()` uses ROIs to crop left/right eyes, runs JEO 3DTracker on each crop, returns gaze (gxL, gyL, gxR, gyR) and pupil positions (crop and full-frame).
3. When **Space** is pressed (recording on), and both a tag and valid gaze are available, one row is appended: **target from back camera**, **features and pupil from front camera (cropped-frame pipeline)**.

So: **back camera = AprilTag coordinates; front camera = pupil (and gaze) in relation to the main frame via the cropped-frame strategy.** This matches the intended design.

---

## 3. Tag location: center vs corners (accuracy)

We record **both**:

- **Center:** `tag_center_x`, `tag_center_y` from the **corner mean** (not bbox center). Research indicates the tag’s geometric center from the four detected corners is more stable under perspective and gives better sub-pixel behavior than the axis-aligned bbox center.
- **Four corners:** `tag_c0_x` … `tag_c3_y` for each corner. Corner-based refinement (e.g. sub-pixel or homography) can improve localization; the pipeline uses center for the gaze target and stores corners for optional later use.

So training uses the **center** (from corners) for the look-at target; corners are available for higher-accuracy or homography-based methods.

---

## 4. Training process improvements (from research)

- **More targets / coverage:** Use 9 or 13 automated positions, or many tags in manual mode, to improve spatial coverage and reduce interpolation error.
- **Tag placement:** Reduce large **camera yaw** to the tag; accuracy drops with oblique viewing. Keep tags roughly facing the camera where possible.
- **Quality filtering:** We already filter by `quality` and `conf`; keeping a high bar (e.g. only high-confidence, non-blink frames) improves model quality.
- **Optional validation split:** Reserve 10–20% of rows for validation and report validation RMSE to detect overfitting (not yet implemented in the script).
- **AprilTag refinement:** If you need sub-pixel accuracy, consider corner refinement (e.g. avoid heavy bilinear upscaling before detection; research suggests nearest-neighbor preserves corner accuracy better).

---

See **docs/CALIBRATION_CHECKLIST.md** for a step-by-step checklist (ROIs, config, run calibration, check report, reduce jitter).

---

## Summary

| Question | Answer |
|----------|--------|
| Enough and correct data types for training? | Yes. 16-D feature + `screen_x_norm`/`screen_y_norm` are correct; extra columns support analysis and custom algorithms. |
| Use graphs to explore and develop our own algorithm? | Yes. Load CSV, plot gaze/pupil vs target and residuals; script provided in `scripts/explore_glass_frame_training_data.py`. |
| Manual mode: two cameras? | Yes. Back camera = AprilTag coordinates. Front camera = pupil/gaze with cropped (left/right) frames, in relation to main frame. |
