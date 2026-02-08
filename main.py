from eye_tracker_win import EyeTracker, UsbCameraDetector
from calibration_win import Calibration

if __name__ == "__main__":
    usb_cam = UsbCameraDetector()
    usb_cam.detect()
    if usb_cam.available:
        tracker = EyeTracker(front_camera_id=usb_cam.camera_id, reset=True)
    else:
        tracker = EyeTracker()

    if tracker.reset:
        calib = Calibration()
        calib.calibrate(tracker)
        tracker.reset = False
        # After calibration, always show the eye-controlled dot (green dot follows your eyes)
        if calib.calibrated:
            try:
                calib.run_eye_control(tracker)
            except Exception as e:
                print("Eye control error:", e)
                import traceback
                traceback.print_exc()
        else:
            print("Calibration had too few valid points; running debug view.")
            tracker.crop_left_eye_region()
    else:
        tracker.crop_left_eye_region()
