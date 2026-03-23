# Calibration checklist (glass-frame / CSI)

Use this checklist so calibration is reproducible and you get **good, less noisy data** for smooth, less jittered eye tracking.

---

## Before first run

- [ ] **Dual CSI smoke test (optional):** `python3 test_dual_csi_glass_apriltag.py` — eye (cam1) + AprilTag (cam0). Defaults are tuned for smooth UI; use `--full-tag-quality` if tags are hard to detect. Calibration CSV already stores tag pixel coords from the back camera (`tag_cam_x/y`, `tag_center_*`, corners, `label_x_norm`/`label_y_norm`, `board_distance_m`) — see `docs/CALIBRATION_DATA_AND_CAMERAS.md`.
- [ ] **ROIs:** Run `python calibrate_glass_frame_rois.py`, draw left/right eye regions, save `glass_frame_rois.json` (e.g. in project root). Same ROIs must be used for calibration and hover.
- [ ] **Cameras:** Back camera (AprilTag) on cam0 (24-pin CSI); eye camera on cam1 (15-pin CSI). Or set in `calibration_config.json` (copy from `calibration_config.json.example`).
- [ ] **Smooth dual-CSI calibration (default):** Glass-frame calibration overlaps AprilTag detection with JEO eye processing and drains the tag-camera buffer (`overlap_tag_eye_detection`, `tag_camera_buffer_drain` in `calibration` JSON). Set `tag_quad_decimate` to `2.0` only if you need more speed and accept slightly coarser tag corners.
- [ ] **AprilTag images (automated only):** Tag images for IDs 1–5 in `~/Downloads/apriltags_tag36h11` (or set `tag_image_dir` in config).

---

## Optional: config file for reproducibility

- [ ] Copy `calibration_config.json.example` to `calibration_config.json`.
- [ ] Set `camera` (sensor IDs, camera IDs), `calibration` (mode, validation_fraction, outlier_mad_multiplier, num_automated_positions), and `cursor_smoothing` (min_cutoff, beta, gaze_jump_thresh) as needed. This makes every run reproducible.
- [ ] Jetson smooth/noise defaults: keep `record_every_n_frames=2`, `stability_window_frames=5`, `stability_std_threshold=0.025`, `aggregate_group_size=4` to reduce noisy rows while keeping enough training data.

---

## Run calibration

- [ ] **Automated:** `python main.py calibrate` with `calibration.mode` = `"automated_calibration"`. Look at each target for the full dwell time; vary head pose slightly between targets.
- [ ] **Manual:** Set `calibration.mode` = `"manual_calibration"`. Place AprilTags 0, 1, 2, … on the board. SPACE = start (tag 0) → SPACE = pause → SPACE = next tag (1) and record → repeat. Q = quit. Ensure front (3 windows) and back (1 window) show correctly.
- [ ] After run: check console for **Train RMSE** and **Validation RMSE**. Validation RMSE should be in a similar range to train (large gap may indicate overfitting).
- [ ] Check **report file:** `eyetracking_ml/calibration_report_*.txt` or next to the saved `.npz`. It lists samples, outliers removed, train/val RMSE, and approx. pixel/angular error.

---

## After calibration: check data quality

- [ ] Run `python scripts/explore_glass_frame_training_data.py` (optionally with path to your CSV). Inspect plots in `eyetracking_ml/explore_plots/`: gaze vs screen, left vs right gaze, target coverage. Remove or re-record if coverage is poor or relationships are broken.
- [ ] If validation RMSE is high or cursor is jittery: increase smoothing in `calibration_config.json` (`cursor_smoothing.min_cutoff` lower, e.g. 1.0 or 0.8) and/or increase `outlier_mad_multiplier` (e.g. 2.0) to drop more outliers and retrain.

---

## Run hover (object detection)

- [ ] `python main.py object_detection` or `python main.py april_tags`. Cursor should follow gaze; object under cursor is highlighted.
- [ ] If cursor is **jittery:** lower `cursor_smoothing.min_cutoff` in `calibration_config.json` (e.g. 1.0) and/or lower `gaze_jump_thresh` (e.g. 0.06), then restart.
- [ ] If cursor is **sluggish:** raise `min_cutoff` (e.g. 1.4) or `beta` (e.g. 0.15).

---

## Summary

| Goal | Action |
|------|--------|
| Less noisy training data | Use `outlier_mad_multiplier` (e.g. 2.5), collect with stable fixations |
| Generalization check | Use `validation_fraction` (e.g. 0.15) and read Validation RMSE in report |
| Reproducible runs | Use `calibration_config.json` |
| Smoother, less jittered cursor | Lower `cursor_smoothing.min_cutoff` (e.g. 1.0) |
