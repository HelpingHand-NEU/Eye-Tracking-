import os
import cv2
import numpy as np
import time
import dlib

# dlib 68-face indices: left eye full contour 36-41 (per landmarks_points.png)
LEFT_EYE_INDICES = (36, 37, 38, 39, 40, 41)
# Right eye 42-45 (for later)
RIGHT_EYE_INDICES = (42, 43, 44, 45)
# Left eye + eyebrow (17-21, 36-41) for future use when a larger ROI is needed
LEFT_EYE_REGION_INDICES = (17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41)

# Default threshold for binary eye ROI (for pupil detection); tune if needed
DEFAULT_EYE_THRESH = 50
# Pupil filter: gray -> Gaussian blur -> THRESH_BINARY_INV (pupil = white)
PUPIL_THRESH = 55
GAUSSIAN_KERNEL_PUPIL = (7, 7)
# Contour area limits for pupil (reject tiny noise and huge blobs)
PUPIL_MIN_AREA = 3
PUPIL_MAX_AREA_RATIO = 0.6  # max contour area / crop area

# Eye Aspect Ratio: below this the eye is considered closed (blink)
EAR_CLOSED_THRESH = 0.22

# Front-facing camera: pupil left in image = user looking right; flip X so calibration and runtime match.
FLIP_GAZE_X = True

# Dynamic crop resize from face closeness (eye region size = lw + lh)
REFERENCE_EYE_SPAN = 80
TARGET_CROP_W_BASE = 100
TARGET_CROP_H_BASE = 40
TARGET_CROP_W_MIN, TARGET_CROP_W_MAX = 30, 200
TARGET_CROP_H_MIN, TARGET_CROP_H_MAX = 15, 80
DISPLAY_HEIGHT_BASE = 350
DISPLAY_HEIGHT_MAX = 600


class UsbCameraDetector:
    """
    Detect the USB camera by location (index), check if it can be used.
    Use .camera_id after .detect() to get the camera number for EyeTracker.
    """

    def __init__(self, max_try=4):
        self.max_try = max_try
        self.camera_id = None
        self.available = False

    def detect(self):
        """
        Find USB camera: try indices 1, 2, 3 first (typical USB), then 0.
        Set self.camera_id and self.available if a usable camera is found.
        """
        order = list(range(1, self.max_try)) + [0]
        for i in order:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.camera_id = i
                    self.available = True
                    return
        self.camera_id = 0
        self.available = False


class EyeTracker:
    def __init__(self, front_camera_id=0, back_camera_id=1, reset=False):
        self.front_camera_id = front_camera_id
        self.back_camera_id = back_camera_id
        self.cap = cv2.VideoCapture(self.front_camera_id)
        # Back camera is optional; use it only if available
        self.back_cap = cv2.VideoCapture(self.back_camera_id)
        if not self.back_cap.isOpened():
            self.back_cap = None  # OK if back camera not found for now
        self.detector = dlib.get_frontal_face_detector()
        self._left_eye_was_closed = False  # for blink detection
        # Left eye ROI in frame: updated each frame when face detected (for pupil coord normalization)
        self.left_eye_crop_x = None
        self.left_eye_crop_y = None
        self.left_eye_crop_width = None
        self.left_eye_crop_height = None
        # Pupil position in crop (pixels) and normalized [0,1]; None when not detected
        self.left_eye_pupil_x = None
        self.left_eye_pupil_y = None
        self.left_eye_pupil_x_norm = None
        self.left_eye_pupil_y_norm = None
        self._last_gx = None
        self._last_gy = None  # for spike rejection (reject if jump > 0.12 in one frame)
        self.reset = reset  # when True, calibration should run before eye control
        script_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(predictor_path)

    def _face_rects_to_locations(self, faces):
        """Convert dlib rects to list of (x, y, w, h) for each face."""
        return [
            (f.left(), f.top(), f.right() - f.left(), f.bottom() - f.top())
            for f in faces
        ]

    def _print_face_locations(self, faces):
        """Print the location of each detected face in the frame."""
        locations = self._face_rects_to_locations(faces)
        for i, (x, y, w, h) in enumerate(locations, 1):
            print(f"Face {i}: x={x}, y={y}, w={w}, h={h}")

    def get_landmarks(self, gray, face):
        """Return dlib shape (68 landmarks) for one face rect."""
        return self.predictor(gray, face)

    def get_left_eye_region(self, landmarks):
        """
        Left eye full contour (landmarks 36-41). Returns (x, y, w, h) bounding box with padding.
        """
        xs = [landmarks.part(i).x for i in LEFT_EYE_INDICES]
        ys = [landmarks.part(i).y for i in LEFT_EYE_INDICES]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        pad = 3
        x = max(0, x1 - pad)
        y = max(0, y1 - pad)
        w = (x2 - x1) + 2 * pad
        h = (y2 - y1) + 2 * pad
        return (x, y, w, h)

    def get_left_eye_region_with_eyebrow(self, landmarks):
        """
        Left eye plus eyebrow (LEFT_EYE_REGION_INDICES: 17-21, 36-41). Returns (x, y, w, h).
        Use when a larger ROI is needed (e.g. reference-style crop). Kept for future use.
        """
        xs = [landmarks.part(i).x for i in LEFT_EYE_REGION_INDICES]
        ys = [landmarks.part(i).y for i in LEFT_EYE_REGION_INDICES]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        pad = 5
        x = max(0, x1 - pad)
        y = max(0, y1 - pad)
        w = (x2 - x1) + 2 * pad
        h = (y2 - y1) + 2 * pad
        return (x, y, w, h)

    def _left_eye_ear(self, landmarks):
        """
        Eye Aspect Ratio for left eye (dlib 36-41). Open eye ~0.25-0.35, closed < 0.2.
        EAR = (|p37-p41| + |p38-p40|) / (2 * |p36-p39|).
        """
        p36 = np.array([landmarks.part(36).x, landmarks.part(36).y])
        p37 = np.array([landmarks.part(37).x, landmarks.part(37).y])
        p38 = np.array([landmarks.part(38).x, landmarks.part(38).y])
        p39 = np.array([landmarks.part(39).x, landmarks.part(39).y])
        p40 = np.array([landmarks.part(40).x, landmarks.part(40).y])
        p41 = np.array([landmarks.part(41).x, landmarks.part(41).y])
        v1 = np.linalg.norm(p37 - p41)
        v2 = np.linalg.norm(p38 - p40)
        h = np.linalg.norm(p36 - p39)
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def check_left_eye_blink(self, landmarks):
        """
        Update blink state from left eye EAR; print "blink" when a blink completes (closed -> open).
        """
        ear = self._left_eye_ear(landmarks)
        if ear < EAR_CLOSED_THRESH:
            self._left_eye_was_closed = True
        else:
            if self._left_eye_was_closed:
                print("blink")
            self._left_eye_was_closed = False

    def _dynamic_crop_size(self, lw, lh):
        """
        From eye region size (lw, lh), compute closeness as eye_span = lw + lh.
        Return (target_w, target_h, display_max_height) for resizing the crop and window.
        """
        eye_span = lw + lh
        scale = eye_span / REFERENCE_EYE_SPAN
        target_w = max(TARGET_CROP_W_MIN, min(TARGET_CROP_W_MAX, int(TARGET_CROP_W_BASE * scale)))
        target_h = max(TARGET_CROP_H_MIN, min(TARGET_CROP_H_MAX, int(TARGET_CROP_H_BASE * scale)))
        display_max_h = min(DISPLAY_HEIGHT_MAX, max(100, int(DISPLAY_HEIGHT_BASE * scale)))
        return target_w, target_h, display_max_h

    def _fit_to_frame_ratio(self, img, frame_w, frame_h, max_height=400):
        """
        Return an image with the same aspect ratio as the frame (frame_w x frame_h).
        img is scaled to fit and centered (letterboxed). max_height caps the output height.
        """
        out_h = min(max_height, frame_h)
        out_w = int(out_h * frame_w / frame_h)
        if out_w < 1 or out_h < 1:
            out_w, out_h = max(1, out_w), max(1, out_h)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if img.size == 0:
            return canvas
        in_h, in_w = img.shape[0], img.shape[1]
        scale = min(out_w / in_w, out_h / in_h)
        new_w, new_h = max(1, int(in_w * scale)), max(1, int(in_h * scale))
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        y0 = (out_h - new_h) // 2
        x0 = (out_w - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = scaled
        return canvas

    def roi_gray_threshold(self, roi_bgr, thresh_value=None):
        """
        Convert ROI to gray and apply binary threshold. Returns (gray_roi, binary_roi).
        """
        if thresh_value is None:
            thresh_value = DEFAULT_EYE_THRESH
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_roi = cv2.threshold(gray_roi, thresh_value, 255, cv2.THRESH_BINARY)
        return gray_roi, binary_roi

    def roi_pupil_binary(self, roi_bgr, thresh_value=None, use_otsu=True):
        """
        Filter for pupil detection: gray -> Gaussian blur -> threshold (Otsu or fixed) -> morphology.
        Pupil (dark) becomes white. Returns (gray_roi, binary_roi).
        Otsu adapts to lighting; morphology cleans noise so contour locks onto pupil not eyelash/shadow.
        """
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, GAUSSIAN_KERNEL_PUPIL, 0)
        if use_otsu:
            _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        else:
            if thresh_value is None:
                thresh_value = PUPIL_THRESH
            _, binary_roi = cv2.threshold(gray_roi, thresh_value, 255, cv2.THRESH_BINARY_INV)
        return gray_roi, binary_roi

    def get_pupil_center(self, binary_roi, crop_w, crop_h):
        """
        Find pupil center from binary image (pupil = white). Returns (cx, cy) in crop
        coords or None. Picks largest contour within area limits.
        """
        if binary_roi is None or binary_roi.size == 0:
            return None
        contours, _ = cv2.findContours(
            binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        crop_area = crop_w * crop_h
        max_area = crop_area * PUPIL_MAX_AREA_RATIO
        valid = [c for c in contours if PUPIL_MIN_AREA <= cv2.contourArea(c) <= max_area]
        if not valid:
            return None
        best = max(valid, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def process_frame(self):
        """
        Process one frame: update pupil and crop state. No display.
        Returns (pupil_x_norm, pupil_y_norm, is_blink).
        Norm = ratio: pupil_x_norm = pupil_x / crop_width, pupil_y_norm = pupil_y / crop_height.
        is_blink True when no face or eye closed; no pupil data recorded then.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            self.left_eye_crop_x = self.left_eye_crop_y = None
            self.left_eye_crop_width = self.left_eye_crop_height = None
            self.left_eye_pupil_x = self.left_eye_pupil_y = None
            self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
            return None, None, True
        landmarks = self.get_landmarks(gray, faces[0])
        self.check_left_eye_blink(landmarks)
        ear = self._left_eye_ear(landmarks)
        lx, ly, lw, lh = self.get_left_eye_region(landmarks)
        target_w, target_h, _ = self._dynamic_crop_size(lw, lh)
        left_eye_crop_raw = frame[ly : ly + lh, lx : lx + lw].copy()
        left_eye_crop = cv2.resize(left_eye_crop_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)
        self.left_eye_crop_x = lx
        self.left_eye_crop_y = ly
        self.left_eye_crop_width = target_w
        self.left_eye_crop_height = target_h
        _, left_eye_binary = self.roi_pupil_binary(left_eye_crop)
        if ear < EAR_CLOSED_THRESH:
            self.left_eye_pupil_x = self.left_eye_pupil_y = None
            self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
            return None, None, True
        pupil_center = self.get_pupil_center(left_eye_binary, target_w, target_h)
        if pupil_center is None:
            self.left_eye_pupil_x = self.left_eye_pupil_y = None
            self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
            return None, None, False
        cx, cy = pupil_center
        self.left_eye_pupil_x = cx
        self.left_eye_pupil_y = cy
        # Record ratio: pupil position / cropped rectangle size (consistent across distance)
        gx = cx / target_w
        gy = cy / target_h
        if FLIP_GAZE_X:
            gx = 1.0 - gx
        # Clamp to [0,1]; reject out-of-range gaze
        if gx < 0 or gx > 1 or gy < 0 or gy > 1:
            return None, None, False
        # Reject sudden spikes (e.g. >0.12 in one frame) to avoid random cursor jumps
        if self._last_gx is not None and self._last_gy is not None:
            if abs(gx - self._last_gx) > 0.12 or abs(gy - self._last_gy) > 0.12:
                return None, None, False
        self._last_gx, self._last_gy = gx, gy
        self.left_eye_pupil_x_norm = gx
        self.left_eye_pupil_y_norm = gy
        return gx, gy, False

    def crop_left_eye_region(self):
        """
        Run camera loop and display a separate window with the cropped left eye region.
        Uses get_left_eye_region for the bounding box. Press 'q' to quit.
        """
        if not self.cap.isOpened():
            print(f"Error: Could not open camera (ID: {self.front_camera_id})")
            raise ValueError(f"Could not open camera (ID: {self.front_camera_id})")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(frame)
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            if len(faces) > 0:
                landmarks = self.get_landmarks(gray, faces[0])
                self.check_left_eye_blink(landmarks)
                ear = self._left_eye_ear(landmarks)
                lx, ly, lw, lh = self.get_left_eye_region(landmarks)
                target_w, target_h, display_max_h = self._dynamic_crop_size(lw, lh)
                left_eye_crop_raw = frame[ly : ly + lh, lx : lx + lw].copy()
                left_eye_crop = cv2.resize(left_eye_crop_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)
                self.left_eye_crop_x = lx
                self.left_eye_crop_y = ly
                self.left_eye_crop_width = target_w
                self.left_eye_crop_height = target_h
                _, left_eye_binary = self.roi_pupil_binary(left_eye_crop)
                if ear >= EAR_CLOSED_THRESH:
                    pupil_center = self.get_pupil_center(left_eye_binary, target_w, target_h)
                    if pupil_center is not None:
                        cx, cy = pupil_center
                        self.left_eye_pupil_x = cx
                        self.left_eye_pupil_y = cy
                        self.left_eye_pupil_x_norm = cx / target_w
                        self.left_eye_pupil_y_norm = cy / target_h
                        print(f"pupil position: ({cx}, {cy})  norm: ({self.left_eye_pupil_x_norm:.3f}, {self.left_eye_pupil_y_norm:.3f})")
                        cv2.circle(left_eye_crop, (cx, cy), 3, (0, 255, 0), 1)
                    else:
                        self.left_eye_pupil_x = self.left_eye_pupil_y = None
                        self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
                else:
                    self.left_eye_pupil_x = self.left_eye_pupil_y = None
                    self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
                display_crop = self._fit_to_frame_ratio(left_eye_crop, frame_w, frame_h, max_height=display_max_h)
                cv2.imshow("Left Eye Region", display_crop)
                display_thresh = self._fit_to_frame_ratio(left_eye_binary, frame_w, frame_h, max_height=display_max_h)
                cv2.imshow("Left Eye Threshold", display_thresh)
            else:
                self.left_eye_crop_x = self.left_eye_crop_y = None
                self.left_eye_crop_width = self.left_eye_crop_height = None
                self.left_eye_pupil_x = self.left_eye_pupil_y = None
                self.left_eye_pupil_x_norm = self.left_eye_pupil_y_norm = None
                display_crop = self._fit_to_frame_ratio(np.zeros((1, 1, 3), dtype=np.uint8), frame_w, frame_h)
                h, w = display_crop.shape[0], display_crop.shape[1]
                cv2.putText(display_crop, "No face detected", (w // 2 - 70, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Left Eye Region", display_crop)
                display_thresh = self._fit_to_frame_ratio(np.zeros((1, 1), dtype=np.uint8), frame_w, frame_h)
                cv2.imshow("Left Eye Threshold", display_thresh)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        if self.back_cap is not None:
            self.back_cap.release()
        cv2.destroyAllWindows()

    def capture_frames(self):
        if not self.cap.isOpened():
            print(f"Error: Could not open camera (ID: {self.front_camera_id})")
            raise ValueError(f"Could not open camera (ID: {self.front_camera_id})")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(frame)
            self._print_face_locations(faces)
            frame = self.draw_faces(frame, faces)
            for face in faces:
                landmarks = self.get_landmarks(gray, face)
                lx, ly, lw, lh = self.get_left_eye_region(landmarks)
                cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
                cv2.putText(frame, "L", (lx, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        if self.back_cap is not None:
            self.back_cap.release()
        cv2.destroyAllWindows()
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces
    
    def draw_faces(self, frame, faces):
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
    
    def draw_landmarks(self, frame, landmarks):
        """Draw all 68 landmarks (dlib shape)."""
        for i in range(landmarks.num_parts):
            pt = landmarks.part(i)
            cv2.circle(frame, (pt.x, pt.y), 2, (0, 0, 255), -1)
        return frame
    
    def check_eye_blink(self, landmarks):
        """Check if the eye is closed by comparing eye landmarks."""
        left_eye_region = self.get_left_eye_region(landmarks)
        right_eye_region = self.get_left_eye_region(landmarks)
        if left_eye_region.area() < 100 or right_eye_region.area() < 100:
            return True
        return False
