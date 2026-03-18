# Credits / Third-party open source

This project uses **someone else's open source repository** for 3D eye tracking on the glass-frame (CSI) camera:

- **[JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker)** — 3DTracker (Orlosky 3D eye tracker). The pupil detection, ellipse fitting, ray intersection, and 3D gaze vector logic are from that repo. We vendor it under `vendor/EyeTracker` and invoke it via `run_csi_jeo_3dtracker.py`. We do **not** claim authorship of that code. See `vendor/EyeTracker/LICENSE` and their [3DTracker readme](https://github.com/JEOresearch/EyeTracker/tree/main/3DTracker).

All other code in this repository is developed as part of this project unless otherwise noted in file headers or documentation.
