#!/usr/bin/env python3
"""
Create larger, printable versions of AprilTag images.
Uses nearest-neighbor scaling to keep edges sharp for reliable detection.
"""
import os
import cv2
import numpy as np

TAG_DIR = os.path.expanduser("~/Downloads/apriltags_tag36h11")
OUT_DIR = os.path.expanduser("~/Downloads/apriltags_printable")
TAG_IDS = range(1, 11)

# Printable size: 300x300 px per tag (good for ~2-3" when printed at 150 DPI)
TAG_SIZE_PX = 300
BORDER_PX = 20  # White border around each tag


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isdir(TAG_DIR):
        print(f"Source folder not found: {TAG_DIR}")
        print("Run download_apriltags.py first.")
        return

    print(f"Creating printable tags ({TAG_SIZE_PX}x{TAG_SIZE_PX} px) in: {OUT_DIR}")

    for tag_id in TAG_IDS:
        filename = f"tag36_11_{tag_id:05d}.png"
        src_path = os.path.join(TAG_DIR, filename)
        out_path = os.path.join(OUT_DIR, f"tag36_11_{tag_id:05d}_print.png")

        if not os.path.isfile(src_path):
            print(f"  Skip {filename}: not found")
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"  Skip {filename}: could not read")
            continue

        # Scale up with nearest-neighbor (preserves sharp edges)
        scaled = cv2.resize(img, (TAG_SIZE_PX, TAG_SIZE_PX), interpolation=cv2.INTER_NEAREST)

        # Add white border
        bordered = cv2.copyMakeBorder(
            scaled,
            BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        # Add ID label below the tag
        h, w = bordered.shape[:2]
        label_img = np.ones((40, w, 3), dtype=np.uint8) * 255
        cv2.putText(
            label_img, f"ID {tag_id}",
            (w // 2 - 30, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )
        result = np.vstack([bordered, label_img])

        cv2.imwrite(out_path, result)
        print(f"  Created: tag36_11_{tag_id:05d}_print.png (ID {tag_id})")

    print(f"\nDone. Printable tags in: {OUT_DIR}")
    print(f"Size: {TAG_SIZE_PX + 2 * BORDER_PX}x{TAG_SIZE_PX + 2 * BORDER_PX + 40} px each")
    print("Print at 100-150 DPI for ~2-4 inch tags.")


if __name__ == "__main__":
    main()
