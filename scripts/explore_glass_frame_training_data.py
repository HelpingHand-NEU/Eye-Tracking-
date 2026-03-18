#!/usr/bin/env python3
"""
Explore glass-frame calibration CSV to see relationships between gaze/pupil and screen target.
Use this to sanity-check data and to develop your own mapping algorithm once you have enough data.

Usage:
  python scripts/explore_glass_frame_training_data.py [path_to.csv]
  If no path given, looks for eyetracking_ml/*glassframe*.csv in project root.

Requires: matplotlib, pandas (or use only csv + numpy and avoid pandas).
"""

from __future__ import annotations

import csv
import os
import sys

try:
    import numpy as np
except ImportError:
    print("numpy required: pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required for plots: pip install matplotlib")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_glassframe_csv(path_arg: str | None) -> str | None:
    if path_arg and os.path.isfile(path_arg):
        return path_arg
    base = os.path.join(PROJECT_ROOT, "eyetracking_ml")
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base)):
        if "glassframe" in name.lower() and name.endswith(".csv"):
            return os.path.join(base, name)
    return None


def load_csv(path: str) -> tuple[list[dict], list[str]]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    return rows, fieldnames


def numeric(row: dict, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, default)
        if v == "" or v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def filter_valid(rows: list[dict], min_quality: float = 0.7) -> list[dict]:
    out = []
    for r in rows:
        q = numeric(r, "quality", 1.0)
        if q < min_quality:
            continue
        sxn = numeric(r, "screen_x_norm", float("nan"))
        syn = numeric(r, "screen_y_norm", float("nan"))
        gxL = numeric(r, "gxL", float("nan"))
        gyL = numeric(r, "gyL", float("nan"))
        gxR = numeric(r, "gxR", float("nan"))
        gyR = numeric(r, "gyR", float("nan"))
        if not (abs(sxn) <= 2 and abs(syn) <= 2):
            continue
        try:
            if not all(np.isfinite([sxn, syn, gxL, gyL, gxR, gyR])):
                continue
        except Exception:
            continue
        out.append(r)
    return out


def extract_arrays(rows: list[dict]) -> dict:
    sxn = np.array([numeric(r, "screen_x_norm") for r in rows])
    syn = np.array([numeric(r, "screen_y_norm") for r in rows])
    gxL = np.array([numeric(r, "gxL") for r in rows])
    gyL = np.array([numeric(r, "gyL") for r in rows])
    gxR = np.array([numeric(r, "gxR") for r in rows])
    gyR = np.array([numeric(r, "gyR") for r in rows])
    gx_avg = 0.5 * (gxL + gxR)
    gy_avg = 0.5 * (gyL + gyR)
    out = {
        "screen_x_norm": sxn, "screen_y_norm": syn,
        "gxL": gxL, "gyL": gyL, "gxR": gxR, "gyR": gyR,
        "gx_avg": gx_avg, "gy_avg": gy_avg,
    }
    # Optional pupil columns (may be empty in CSV)
    pLx = [numeric(r, "pupil_L_x_norm", float("nan")) for r in rows]
    pLy = [numeric(r, "pupil_L_y_norm", float("nan")) for r in rows]
    pRx = [numeric(r, "pupil_R_x_norm", float("nan")) for r in rows]
    pRy = [numeric(r, "pupil_R_y_norm", float("nan")) for r in rows]
    if any(np.isfinite(pLx)):
        out["pupil_L_x_norm"] = np.array(pLx)
        out["pupil_L_y_norm"] = np.array(pLy)
        out["pupil_R_x_norm"] = np.array(pRx)
        out["pupil_R_y_norm"] = np.array(pRy)
    return out


def plot_relationships(data: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Gaze (left) vs screen target
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(data["gxL"], data["screen_x_norm"], alpha=0.5, s=8)
    axes[0].set_xlabel("gxL (left gaze x)")
    axes[0].set_ylabel("screen_x_norm")
    axes[0].set_title("Left gaze x vs screen x")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(data["gyL"], data["screen_y_norm"], alpha=0.5, s=8)
    axes[1].set_xlabel("gyL (left gaze y)")
    axes[1].set_ylabel("screen_y_norm")
    axes[1].set_title("Left gaze y vs screen y")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gaze_left_vs_screen.png"), dpi=120)
    plt.close()

    # 2) Average gaze vs screen
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(data["gx_avg"], data["screen_x_norm"], alpha=0.5, s=8)
    axes[0].set_xlabel("(gxL+gxR)/2")
    axes[0].set_ylabel("screen_x_norm")
    axes[0].set_title("Avg gaze x vs screen x")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(data["gy_avg"], data["screen_y_norm"], alpha=0.5, s=8)
    axes[1].set_xlabel("(gyL+gyR)/2")
    axes[1].set_ylabel("screen_y_norm")
    axes[1].set_title("Avg gaze y vs screen y")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gaze_avg_vs_screen.png"), dpi=120)
    plt.close()

    # 3) Left vs right gaze (agreement)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(data["gxL"], data["gxR"], alpha=0.5, s=8)
    axes[0].set_xlabel("gxL")
    axes[0].set_ylabel("gxR")
    axes[0].set_title("Left vs right gaze x")
    axes[0].plot([-0.5, 0.5], [-0.5, 0.5], "k--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(data["gyL"], data["gyR"], alpha=0.5, s=8)
    axes[1].set_xlabel("gyL")
    axes[1].set_ylabel("gyR")
    axes[1].set_title("Left vs right gaze y")
    axes[1].plot([-0.5, 0.5], [-0.5, 0.5], "k--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gaze_left_vs_right.png"), dpi=120)
    plt.close()

    # 4) Target coverage (screen_x_norm, screen_y_norm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data["screen_x_norm"], data["screen_y_norm"], alpha=0.5, s=10)
    ax.set_xlabel("screen_x_norm")
    ax.set_ylabel("screen_y_norm")
    ax.set_title("Target coverage (normalized screen)")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "target_coverage.png"), dpi=120)
    plt.close()

    # 5) Pupil vs screen (if available)
    if "pupil_L_x_norm" in data:
        ok = np.isfinite(data["pupil_L_x_norm"]) & np.isfinite(data["screen_x_norm"])
        if np.sum(ok) > 5:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].scatter(data["pupil_L_x_norm"][ok], data["screen_x_norm"][ok], alpha=0.5, s=8)
            axes[0].set_xlabel("pupil_L_x_norm")
            axes[0].set_ylabel("screen_x_norm")
            axes[0].set_title("Left pupil x vs screen x")
            axes[0].grid(True, alpha=0.3)
            axes[1].scatter(data["pupil_L_y_norm"][ok], data["screen_y_norm"][ok], alpha=0.5, s=8)
            axes[1].set_xlabel("pupil_L_y_norm")
            axes[1].set_ylabel("screen_y_norm")
            axes[1].set_title("Left pupil y vs screen y")
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "pupil_left_vs_screen.png"), dpi=120)
            plt.close()

    print(f"Plots saved under: {out_dir}")


def main() -> None:
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = find_glassframe_csv(path_arg)
    if not path:
        print("No glass-frame CSV found. Run calibration first or pass path: python scripts/explore_glass_frame_training_data.py <path_to.csv>")
        sys.exit(1)
    print(f"Loading: {path}")
    rows, fieldnames = load_csv(path)
    print(f"Total rows: {len(rows)}, columns: {len(fieldnames)}")
    valid = filter_valid(rows)
    print(f"Valid rows (quality>=0.7, finite): {len(valid)}")
    if len(valid) < 5:
        print("Too few valid rows to plot. Collect more calibration data.")
        sys.exit(0)
    data = extract_arrays(valid)
    out_dir = os.path.join(PROJECT_ROOT, "eyetracking_ml", "explore_plots")
    plot_relationships(data, out_dir)


if __name__ == "__main__":
    main()
