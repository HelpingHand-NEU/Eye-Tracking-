# Pupil detection accuracy (glass-frame and face-based)

## Two modes

1. **Eye-only (glass-frame)**  
   Camera is mounted on the glass frame and **cannot capture the entire face**; only the eye region is visible.  
   - **No face detection:** The glass-frame path does **not** use MediaPipe Face Mesh or any face/eye landmarks.  
   - **Custom landmarks (optional):** If the glass frame blocks between the eyes, you can define your own left/right eye regions. Run `python3 calibrate_glass_frame_rois.py`, draw rectangles around each eye, press **s** to save. This creates `glass_frame_rois.json`; pupil detection then runs only inside these regions.  
   - **Lower-half crop:** If no custom ROIs file exists, the lower 50% of the frame is used by default. Disable with `--no-crop` or `tracker.glass_eye_crop_y_start = 0`.  
   - **Pupil-only:** CLAHE, glint suppression, contour + ray refinement, PuRe when installed; no face or landmarks.

2. **Face visible (e.g. front webcam)**  
   When the front camera sees the full face, `estimate_gaze_from_frame()` uses Face Mesh. Not used for the glass-frame camera.  
   - Uses **MediaPipe Face Mesh** with eye and iris landmarks (Google’s pipeline).  
   - **Eye ROI:** Pupil detection runs only inside left/right eye ROIs from landmarks; iris used when available, else averaged pupil from both eyes.

---

## What to install for best accuracy

### For eye-only (glass-frame) – no face

| Package            | Install                    | Role                          |
|--------------------|----------------------------|-------------------------------|
| **pupil-detectors** | `pip install pupil-detectors` | PuRe algorithm (recommended). |

- Already listed in `requirements.txt`.  
- If you don’t install it, the built-in contour + ellipse method is used (less robust).

### For face + iris (when face is in frame)

| Package       | Install                | Role                                      |
|---------------|------------------------|-------------------------------------------|
| **mediapipe** | `pip install mediapipe` | Face Mesh + iris landmarks (already used). |

- No extra download: the model is bundled with the package.  
- For the **new Face Landmarker API** (optional, separate from current code):  
  - Install: `pip install mediapipe` (same).  
  - Download the task model from [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) and pass its path if you switch code to the new API.

---

## Summary

- **Glass-frame, eye-only:**  
  Install **pupil-detectors** for best accuracy; no face or landmarks required.

- **When the face is visible:**  
  **mediapipe** is already used for Face Mesh + iris; no extra download needed for the current code.

- **Landmarks** are only used when a face is detected (MediaPipe iris). For pure eye-only frames, the code does **not** use face or landmarks; it uses PuRe or the contour+ellipse fallback.
