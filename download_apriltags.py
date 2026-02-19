#!/usr/bin/env python3
"""
Download AprilTag images (tag36h11 family, IDs 1-10) to the Downloads folder.
Source: https://github.com/AprilRobotics/apriltag-imgs
"""
import os
import ssl
import urllib.request
import urllib.error

# tag36h11 family - IDs 1-10
BASE_URL = "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11"
TAG_IDS = range(1, 11)  # 1 through 10

def get_downloads_folder():
    return os.path.expanduser("~/Downloads")

def main():
    downloads = get_downloads_folder()
    out_dir = os.path.join(downloads, "apriltags_tag36h11")
    os.makedirs(out_dir, exist_ok=True)

    ctx = ssl.create_default_context()
    # Fallback for macOS SSL cert issues (known GitHub URL only)
    _ctx_unverified = None
    def _get_ctx():
        nonlocal _ctx_unverified
        if _ctx_unverified is None:
            _ctx_unverified = ssl.create_default_context()
            _ctx_unverified.check_hostname = False
            _ctx_unverified.verify_mode = ssl.CERT_NONE
        return _ctx_unverified

    print(f"Downloading AprilTags (tag36h11) ID 1-10 to: {out_dir}")

    for tag_id in TAG_IDS:
        filename = f"tag36_11_{tag_id:05d}.png"
        url = f"{BASE_URL}/{filename}"
        out_path = os.path.join(out_dir, filename)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
                    with open(out_path, "wb") as f:
                        f.write(r.read())
            except (ssl.SSLError, urllib.error.URLError):
                # Retry with unverified SSL (macOS cert issues)
                with urllib.request.urlopen(req, context=_get_ctx(), timeout=30) as r:
                    with open(out_path, "wb") as f:
                        f.write(r.read())
            print(f"  Downloaded: {filename} (ID {tag_id})")
        except Exception as e:
            print(f"  Failed {filename}: {e}")

    print(f"\nDone. Tags saved in: {out_dir}")
    print("Print these images and place them in your scene for detection.")

if __name__ == "__main__":
    main()
