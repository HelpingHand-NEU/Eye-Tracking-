#!/bin/bash
# Fix CSI camera not detected on Jetson Nano (IMX219 on cam1).
# Run on the Jetson:  bash fix_csi_camera.sh
# You will be prompted for sudo password.

set -e
echo "=== CSI camera fix for Jetson ==="

echo ""
echo "Step 1: Installing V4L2 utils and GStreamer..."
sudo apt-get update -qq
sudo apt-get install -y v4l-utils gstreamer1.0-tools gstreamer1.0-plugins-good

echo ""
echo "Step 2: Installing NVIDIA L4T GStreamer and Camera packages..."
sudo apt-get install -y nvidia-l4t-gstreamer nvidia-l4t-camera 2>/dev/null || {
    echo "  (nvidia-l4t-* may already be installed or use different package names on your JetPack)"
}

echo ""
echo "Step 3: Checking for nvarguscamerasrc..."
if gst-inspect-1.0 nvarguscamerasrc &>/dev/null; then
    echo "  nvarguscamerasrc: OK"
else
    echo "  nvarguscamerasrc: still not found. Try: sudo apt install nvidia-jetpack (or reinstall JetPack multimedia)"
fi

echo ""
echo "Step 4: Configure CSI connector (Jetson-IO)"
echo "  You must run this interactively to enable cam1 (IMX219):"
echo "    sudo /opt/nvidia/jetson-io/jetson-io.py"
echo "  Then: Configure NVIDIA Jetson CSI Connector -> compatible hardware -> select IMX219"
echo "  Save and reboot."
echo ""
if [ -t 0 ]; then
    read -p "Run Jetson-IO now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[yY]$ ]]; then
        sudo /opt/nvidia/jetson-io/jetson-io.py
    fi
else
    echo "  (Run Jetson-IO manually when ready: sudo /opt/nvidia/jetson-io/jetson-io.py)"
fi

echo ""
echo "Step 5: Re-run diagnostic"
echo "  After reboot, run:  python3 diagnose_csi_camera.py"
echo ""
echo "Done."
