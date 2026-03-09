#!/usr/bin/env python3
"""
Troubleshoot USB camera not detected on macOS.
Run this to diagnose why your external camera may not appear.
"""
import subprocess
import sys

def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    except Exception as e:
        return type('R', (), {'returncode': -1, 'stdout': '', 'stderr': str(e)})()

def main():
    print("=" * 60)
    print("USB Camera Troubleshooting for macOS")
    print("=" * 60)

    # 1. Check if USB device is seen at system level
    print("\n1. USB devices (look for 'Camera', 'Webcam', 'UVC', 'Video'):")
    print("-" * 40)
    r = run("system_profiler SPUSBDataType 2>/dev/null")
    if r.returncode == 0 and r.stdout:
        lines = [l for l in r.stdout.splitlines() if any(k in l.lower() for k in ('camera', 'webcam', 'uvc', 'video', 'imaging'))]
        if lines:
            for l in lines:
                print("  ", l.strip())
        else:
            # Show USB tree (abbreviated)
            for line in r.stdout.splitlines()[:50]:
                if line.strip() and not line.startswith("    " * 3):
                    print("  ", line[:70])
    else:
        print("  (system_profiler failed or empty)")

    # 2. Camera permissions
    print("\n2. macOS Camera permissions:")
    print("-" * 40)
    print("  Go to: System Settings → Privacy & Security → Camera")
    print("  Ensure these are enabled:")
    print("    • Terminal (if running: python3 main.py)")
    print("    • Cursor (if running from Cursor IDE)")
    print("    • Python")
    print("  If your app isn't listed, run the script from Terminal once")
    print("  to trigger the permission prompt.")

    # 3. Reset permissions (optional)
    print("\n3. If camera was denied and you need to reset:")
    print("-" * 40)
    print("  Run in Terminal (this resets all camera permissions):")
    print("    tccutil reset Camera com.apple.Terminal")
    print("  Or for Cursor:")
    print("    tccutil reset Camera com.tutorial.cursor")
    print("  Then restart the app and accept the permission prompt.")

    # 4. Test with OpenCV
    print("\n4. OpenCV camera test:")
    print("-" * 40)
    try:
        import cv2
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                print(f"  Camera {i}: OK ({w}x{h})")
            else:
                print(f"  Camera {i}: NOT available")
    except Exception as e:
        print(f"  Error: {e}")
        print("  (Often caused by missing camera permission)")

    # 5. Additional tips
    print("\n5. Other things to try:")
    print("-" * 40)
    print("  • Unplug and replug the USB camera")
    print("  • Try a different USB port (prefer USB-A, not hub)")
    print("  • Close Photo Booth, Zoom, FaceTime - they may hold the camera")
    print("  • Test in Photo Booth first - if it works there, it's a permissions issue")
    print("  • Restart your Mac if the camera was just connected")
    print("  • Some USB3 Vision cameras need vendor drivers (not standard UVC)")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
