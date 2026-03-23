"""
Gaze → screen mapping for eye-controlled cursor.

Uses weights saved from glass-frame / webcam calibration (CSV + NPZ):
  - **poly_ridge** — polynomial ridge regression on the full feature vector (primary for glass-frame).
  - **binocular_ridge** — linear ridge on normalized binocular features (auxiliary head in same NPZ).
  - **ensemble** — average of poly_ridge and binocular_ridge when both are valid in [0,1].
  - **affine** — legacy gaze_to_screen (affine / polynomial dot calibration), no ML file required for this path.
  - **auto** — try poly_ridge, then binocular_ridge, then affine (same as original cursor behavior).

Re-run ``python main.py calibrate`` after collecting more data to improve poly_ridge / binocular fits.
"""

from __future__ import annotations


def _norm_to_pixels(calibration, nx: float, ny: float):
    if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
        return None
    return float(nx * calibration.screen_w), float(ny * calibration.screen_h)


def _predict_poly_ridge(calibration, fv) -> tuple[float, float] | None:
    """Return normalized (x, y) in [0,1] or None."""
    if fv is None:
        return None
    ml = getattr(calibration, "ml", None)
    input_dim = getattr(ml, "input_dim", None) if ml is not None else None
    if ml is None or input_dim is None or getattr(ml, "W", None) is None:
        return None
    if len(fv) != input_dim:
        return None
    fv_in = fv[:input_dim]
    out = calibration.ml.predict(fv_in)
    if out is None:
        return None
    nx, ny = float(out[0]), float(out[1])
    if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
        return None
    return nx, ny


def _predict_binocular_ridge(calibration, fv) -> tuple[float, float] | None:
    if fv is None or len(fv) < 14:
        return None
    if getattr(calibration, "binoc_W", None) is None:
        return None
    out = calibration.map_binocular(fv)
    if out is None:
        return None
    nx, ny = float(out[0]), float(out[1])
    if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
        return None
    return nx, ny


def _predict_affine(calibration, gx, gy, yaw, pitch, fv):
    return calibration.gaze_to_screen(gx, gy, yaw=yaw, pitch=pitch, feature_vec=fv)


def predict_gaze_screen_xy(
    calibration,
    fv,
    gx,
    gy,
    yaw,
    pitch,
    mode: str = "auto",
):
    """
    Map current frame to screen pixel coordinates.

    Returns:
        ((sx, sy), source) where source is one of
        ``poly_ridge``, ``binocular_ridge``, ``ensemble``, ``affine``, or (None, None).
    """
    mode = (mode or "auto").strip().lower().replace("-", "_")

    if mode == "poly_ridge":
        n = _predict_poly_ridge(calibration, fv)
        if n is None:
            return None, None
        pt = _norm_to_pixels(calibration, n[0], n[1])
        return (pt, "poly_ridge") if pt is not None else (None, None)

    if mode == "binocular_ridge":
        n = _predict_binocular_ridge(calibration, fv)
        if n is None:
            return None, None
        pt = _norm_to_pixels(calibration, n[0], n[1])
        return (pt, "binocular_ridge") if pt is not None else (None, None)

    if mode == "ensemble":
        a = _predict_poly_ridge(calibration, fv)
        b = _predict_binocular_ridge(calibration, fv)
        if a is not None and b is not None:
            nx = 0.5 * (a[0] + b[0])
            ny = 0.5 * (a[1] + b[1])
            pt = _norm_to_pixels(calibration, nx, ny)
            return (pt, "ensemble") if pt is not None else (None, None)
        if a is not None:
            pt = _norm_to_pixels(calibration, a[0], a[1])
            return (pt, "poly_ridge") if pt is not None else (None, None)
        if b is not None:
            pt = _norm_to_pixels(calibration, b[0], b[1])
            return (pt, "binocular_ridge") if pt is not None else (None, None)
        pt = _predict_affine(calibration, gx, gy, yaw, pitch, fv)
        return (pt, "affine") if pt is not None else (None, None)

    if mode == "affine":
        pt = _predict_affine(calibration, gx, gy, yaw, pitch, fv)
        return (pt, "affine") if pt is not None else (None, None)

    # auto (default): poly → binoc → affine
    n = _predict_poly_ridge(calibration, fv)
    if n is not None:
        pt = _norm_to_pixels(calibration, n[0], n[1])
        if pt is not None:
            return pt, "poly_ridge"
    n = _predict_binocular_ridge(calibration, fv)
    if n is not None:
        pt = _norm_to_pixels(calibration, n[0], n[1])
        if pt is not None:
            return pt, "binocular_ridge"
    pt = _predict_affine(calibration, gx, gy, yaw, pitch, fv)
    return (pt, "affine") if pt is not None else (None, None)


def valid_inference_modes():
    return ("auto", "poly_ridge", "binocular_ridge", "ensemble", "affine")
