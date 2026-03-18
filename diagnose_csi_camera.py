#!/usr/bin/env python3
"""
Diagnose why CSI camera (IMX219 on cam1) is not detected on Jetson Nano.
Run on the Jetson: python3 diagnose_csi_camera.py
"""

import os
import subprocess
import sys


def run(cmd, check=False):
    """Run command, return (stdout, stderr, returncode)."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (r.stdout or "", r.stderr or "", r.returncode)
    except Exception as e:
        return ("", str(e), -1)


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("CSI Camera Diagnostic (IMX219 on cam1 / sensor-id=1)")
    print("Run this script ON the Jetson Nano.")

    # -------------------------------------------------------------------------
    section("1. Platform")
    # -------------------------------------------------------------------------
    if os.path.exists("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release", "r") as f:
            rel = f.read().strip()
        print(f"  Jetson L4T: {rel}")
    else:
        print("  NOT a Jetson (no /etc/nv_tegra_release). CSI is only on Jetson.")
        print("  Run this script on the Jetson Nano.")

    # -------------------------------------------------------------------------
    section("2. Video devices (/dev/video*)")
    # -------------------------------------------------------------------------
    for d in ["/dev/video0", "/dev/video1", "/dev/video2"]:
        exists = os.path.exists(d)
        print(f"  {d}: {'exists' if exists else 'MISSING'}")
    out, err, code = run("ls -la /dev/video* 2>/dev/null")
    if out.strip():
        print(out)

    # -------------------------------------------------------------------------
    section("2b. Is camera in use? (USB /dev/video*)")
    # -------------------------------------------------------------------------
    print("  Commands to see which process uses a device:")
    print("    lsof /dev/video0          # list open files on device")
    print("    fuser -v /dev/video0      # verbose: process using device")
    print("  Checking now:")
    for dev in ["/dev/video0", "/dev/video1"]:
        if not os.path.exists(dev):
            continue
        out, err, code = run(["lsof", dev])
        if out.strip():
            print(f"  {dev} in use:")
            for line in out.strip().splitlines():
                print(f"    {line}")
        else:
            print(f"  {dev}: no process found (lsof)")
    print("  Note: CSI (nvarguscamerasrc) does not use /dev/video*.")
    print("  If CSI fails with 'CaptureSession', close: Jetson camera demo, other GStreamer apps, this script.")

    # -------------------------------------------------------------------------
    section("3. V4L2 device info (if v4l2-ctl available)")
    # -------------------------------------------------------------------------
    out, err, code = run("v4l2-ctl --list-devices 2>/dev/null")
    if code == 0 and out.strip():
        print(out)
    else:
        print("  v4l2-ctl not found or no devices. Install: sudo apt install v4l-utils")

    # -------------------------------------------------------------------------
    section("4. OpenCV and GStreamer")
    # -------------------------------------------------------------------------
    try:
        import cv2
        build = cv2.getBuildInformation()
        has_gst = "GStreamer" in build and "YES" in build.split("GStreamer")[-1].split("\n")[0]
        print(f"  OpenCV GStreamer support: {'YES' if has_gst else 'NO (CSI will not work with this OpenCV)'}")
        if not has_gst:
            print("  Fix: use JetPack OpenCV (sudo apt install python3-opencv) or build OpenCV with GStreamer.")
    except Exception as e:
        print(f"  OpenCV import error: {e}")

    # -------------------------------------------------------------------------
    section("5. GStreamer nvarguscamerasrc")
    # -------------------------------------------------------------------------
    out, err, code = run("gst-inspect-1.0 nvarguscamerasrc 2>&1")
    if code == 0:
        print("  nvarguscamerasrc: available")
        if "sensor-id" in out or "sensor_id" in out:
            print("  (sensor-id property present)")
    else:
        print("  nvarguscamerasrc: NOT found")
        print("  Install JetPack / GStreamer plugins for Jetson.")

    # -------------------------------------------------------------------------
    section("6. Try opening CSI sensor-id=0 and sensor-id=1 (GStreamer)")
    # -------------------------------------------------------------------------
    for sid in [0, 1]:
        pipeline = (
            f"nvarguscamerasrc sensor-id={sid} ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! fakesink"
        )
        out, err, code = run(["gst-launch-1.0", "-e", pipeline])
        status = "OK" if code == 0 else "FAILED"
        print(f"  sensor-id={sid} (cam{sid}): {status}")
        if code != 0 and err:
            for line in err.strip().split("\n")[-3:]:
                print(f"    {line}")

    # -------------------------------------------------------------------------
    section("7. Kernel / dmesg (camera, imx219, vi, nvcsi)")
    # -------------------------------------------------------------------------
    out, err, code = run("dmesg 2>/dev/null | grep -iE 'imx219|vi_|nvcsi|camera|tegra_cam' | tail -25")
    if out.strip():
        print(out)
    else:
        print("  No matching kernel messages (or dmesg not readable; run with sudo).")
        out2, _, _ = run("dmesg 2>/dev/null | tail -5")
        if out2.strip():
            print("  Last 5 dmesg lines:")
            print(out2)

    # -------------------------------------------------------------------------
    section("8. Device tree / config (Jetson-IO)")
    # -------------------------------------------------------------------------
    if os.path.exists("/opt/nvidia/jetson-io/jetson-io.py"):
        print("  Jetson-IO present: /opt/nvidia/jetson-io/jetson-io.py")
        print("  To configure CSI connector: sudo /opt/nvidia/jetson-io/jetson-io.py")
        print("  -> Configure NVIDIA Jetson CSI Connector -> compatible hardware -> IMX219")
    else:
        print("  Jetson-IO path not found (may differ by JetPack version).")
    dtb = run("ls /boot/dtb/*.dtb 2>/dev/null | head -3")
    if dtb[0].strip():
        print("  Device tree: (check that camera overlay is applied)")

    # -------------------------------------------------------------------------
    section("9. Summary and next steps")
    # -------------------------------------------------------------------------
    print("""
  If sensor-id=1 fails in step 6:
    - Ensure IMX219 is on the CAM1 connector (second CSI port).
    - Run: sudo /opt/nvidia/jetson-io/jetson-io.py
      and configure BOTH CSI connectors for compatible camera if needed.
    - Reboot after saving pin changes.
  If OpenCV has no GStreamer:
    - Use: sudo apt install python3-opencv  (JetPack OpenCV)
    - Or uninstall pip opencv so the system one is used.
  If no /dev/video* for CSI:
    - Configure CSI with Jetson-IO and reboot.
    - Check ribbon cable (blue side away from board, firmly seated).
""")


if __name__ == "__main__":
    main()
    sys.exit(0)
