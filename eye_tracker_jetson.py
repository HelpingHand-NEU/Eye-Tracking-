#!/usr/bin/env python3
"""
Eye Tracking Application
Accesses both front and back cameras, tracks eyes, and overlays tracking on forward-facing camera.
"""

# Import numpy before cv2 to avoid ABI errors on Jetson. MediaPipe is lazy-imported
# so CSI pupil-only test (test_csi_glass_frame_pupil.py) does not pull in NumPy 1.x deps.
import json
import numpy as np
import cv2
import sys
import argparse
import time
import os
from collections import deque

# Default path for user-defined glass-frame eye regions (your own "landmarks").
# When this file exists, pupil detection runs only inside these ROIs; no face detection.
GLASS_FRAME_ROIS_FILENAME = "glass_frame_rois.json"


class EyeTracker:
    def __init__(self, front_camera_id=0, back_camera_id=1,
                 camera_width=640, camera_height=480, camera_fps=30,
                 front_device=None, back_device=None,
                 front_sensor_id=None, back_sensor_id=None):
        """
        Initialize the Eye Tracker with camera IDs.
        
        Args:
            front_camera_id: Camera index for forward-facing camera (default: 0)
            back_camera_id: Camera index for back camera (default: 1)
        """
        self.front_camera_id = front_camera_id
        self.back_camera_id = back_camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.front_device = front_device
        self.back_device = back_device
        self.front_sensor_id = front_sensor_id
        self.back_sensor_id = back_sensor_id
        
        # MediaPipe is lazy-loaded in _ensure_mediapipe() to avoid import errors when
        # only using CSI pupil detection (e.g. test_csi_glass_frame_pupil.py) with NumPy 2.x.
        self._mediapipe_initialized = False
        self.mp_face_mesh = None
        # PuRe (pupil-detectors) for robust glass-frame pupil detection; lazy-loaded.
        self._pupil_detector_2d = None
        self._pupil_detector_initialized = False
        # Glass-frame pupil refinement (match webcam algorithm: ray refinement + CLAHE)
        self._glass_ray_count = 24
        self._glass_ray_step = 2
        self._glass_edge_grad_thresh = 7.0
        self._glass_edge_min_bright = 45.0
        # Kalman for single-eye glass-frame output (smooth pupil like webcam gaze)
        self._glass_pupil_kalman = None
        self._glass_pupil_last = None
        # Crop to lower part of frame where eye sits (reduces false positives from ceiling/sky).
        # 0.5 = lower half; 0.4 = lower 60%. Set to 0 to disable crop.
        self.glass_eye_crop_y_start = 0.5
        # User-defined eye regions ("custom landmarks") when glass frame blocks between eyes.
        # List of (x, y, w, h) in normalized [0,1]; loaded from glass_frame_rois.json if present.
        self._custom_glass_rois = None
        self._custom_glass_rois_path = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.face_mesh = None
        
        # Eye landmark indices (MediaPipe Face Mesh)
        # Left eye landmarks
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris landmarks (for more precise eye tracking)
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
        self.front_cap = None
        self.back_cap = None
        
        # Calibration data
        self.calibrated = False
        self.calibration_points = []  # Screen positions for calibration
        self.calibration_eye_data = []  # Eye data at each calibration point
        self.calibration_matrix = None  # Transformation matrix
        self.calibration_poly_features = False  # Whether to use polynomial features
        
        # Heatmap for gaze visualization
        self.heatmap = None
        self.heatmap_decay = 0.98  # Decay factor (higher = slower decay)
        self.heatmap_intensity = 3.0  # Intensity multiplier
        self.show_heatmap = True  # Whether to show heatmap
        
        # Kalman filter for gaze prediction
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        self._init_kalman()
        
        # Gaze tracking state
        self.last_gaze_point = None
        self.gaze_buffer = deque(maxlen=5)  # Temporal smoothing buffer
        self.baseline_eye_distance = None  # For scale compensation
        
    def _init_kalman(self):
        """Initialize Kalman filter parameters."""
        # State transition matrix (constant velocity model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 1],  # y' = y + vy
            [0, 0, 1, 0],  # vx' = vx
            [0, 0, 0, 1]   # vy' = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], dtype=np.float32)
        
        # Process noise covariance (how much we trust the model) - reduced for smoother tracking
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        
        # Measurement noise covariance (how much we trust measurements) - slightly increased
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.15
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initial state
        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def _ensure_mediapipe(self):
        """Lazy-init MediaPipe so CSI pupil-only scripts avoid NumPy/matplotlib ABI issues."""
        if self._mediapipe_initialized:
            return
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mediapipe_initialized = True

    def _is_jetson(self):
        """Detect NVIDIA Jetson platforms."""
        return os.path.exists("/etc/nv_tegra_release")

    def _gstreamer_available(self):
        """Check if OpenCV was built with GStreamer support."""
        try:
            build_info = cv2.getBuildInformation()
        except Exception:
            return False
        return "GStreamer" in build_info and "YES" in build_info.split("GStreamer")[1].splitlines()[0]

    def _build_csi_gstreamer_pipeline(self, sensor_id, width=None, height=None, fps=None):
        """Build a GStreamer pipeline for Jetson CSI cameras (IMX219 etc.)."""
        w = width if width is not None else self.camera_width
        h = height if height is not None else self.camera_height
        f = fps if fps is not None else self.camera_fps
        # IMX219 supported modes; prefer 1280x720@30 for compatibility (60 may fail on some stacks)
        supported_modes = {
            (3280, 2464, 21),
            (3280, 1848, 28),
            (1920, 1080, 30),
            (1640, 1232, 30),
            (1280, 720, 60),
            (1280, 720, 30),
        }
        if (w, h, f) not in supported_modes:
            w, h, f = 1280, 720, 30
        # Explicit caps and nvvidconv output size improve compatibility with OpenCV appsink
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
            f"format=(string)NV12, framerate=(fraction){f}/1 ! "
            f"nvvidconv ! video/x-raw, format=(string)BGRx, width=(int){w}, height=(int){h} ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
        )

    def _build_v4l2_gstreamer_pipeline(self, device_path):
        """Build a GStreamer pipeline for V4L2 (USB) cameras."""
        return (
            f"v4l2src device={device_path} ! "
            f"video/x-raw, width={self.camera_width}, height={self.camera_height}, "
            f"framerate={self.camera_fps}/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )

    def _open_camera(self, camera_id, prefer_csi=False, device_path=None, sensor_id=None):
        """Open a camera with Jetson-friendly fallbacks."""
        if self._is_jetson():
            if not self._gstreamer_available():
                print("Error: OpenCV was built without GStreamer support. "
                      "Jetson CSI cameras require GStreamer. "
                      "Install an OpenCV build with GStreamer enabled.")
                return cv2.VideoCapture(), None, False

            if sensor_id is not None:
                pipeline = self._build_csi_gstreamer_pipeline(sensor_id)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap, f"CSI sensor-id {sensor_id}", True
                return cap, None, False

            if device_path:
                if os.path.exists(device_path):
                    pipeline = self._build_v4l2_gstreamer_pipeline(device_path)
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if cap.isOpened():
                        return cap, f"V4L2 {device_path}", True
                else:
                    return cv2.VideoCapture(), None, False

            if prefer_csi:
                pipeline = self._build_csi_gstreamer_pipeline(camera_id)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap, f"CSI sensor-id {camera_id}", True

            device_path = f"/dev/video{camera_id}"
            if os.path.exists(device_path):
                pipeline = self._build_v4l2_gstreamer_pipeline(device_path)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap, f"V4L2 {device_path}", True

        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            return cap, f"OpenCV ID {camera_id}", False
        return cap, None, False
    
    def predict_gaze_kalman(self, measurement):
        """Use Kalman filter to predict and smooth gaze point."""
        if measurement is None:
            # If no measurement, just predict based on current state
            prediction = self.kalman.predict()
            if prediction is not None and len(prediction) >= 2:
                return (float(prediction[0]), float(prediction[1]))
            return None
        
        # Convert measurement to numpy array
        measurement_array = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        
        # Predict next state
        prediction = self.kalman.predict()
        
        # Update with measurement
        self.kalman.correct(measurement_array)
        
        # Return predicted position
        if prediction is not None and len(prediction) >= 2:
            return (float(prediction[0]), float(prediction[1]))
        return None
    
    def get_head_pose(self, landmarks, image_shape):
        """Estimate head pose (pitch, yaw, roll) to compensate for head movement."""
        h, w = image_shape[:2]
        
        try:
            # Key facial landmarks for head pose estimation
            image_points = np.array([
                [int(landmarks.landmark[1].x * w), int(landmarks.landmark[1].y * h)],  # Nose tip
                [int(landmarks.landmark[175].x * w), int(landmarks.landmark[175].y * h)],  # Chin
                [int(landmarks.landmark[33].x * w), int(landmarks.landmark[33].y * h)],  # Left eye corner
                [int(landmarks.landmark[263].x * w), int(landmarks.landmark[263].y * h)],  # Right eye corner
                [int(landmarks.landmark[61].x * w), int(landmarks.landmark[61].y * h)],  # Left mouth
                [int(landmarks.landmark[291].x * w), int(landmarks.landmark[291].y * h)],  # Right mouth
            ], dtype=np.float32)
            
            # 3D model points (approximate, in mm)
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye corner
                (225.0, 170.0, -135.0),  # Right eye corner
                (-150.0, -150.0, -125.0),  # Left mouth
                (150.0, -150.0, -125.0),  # Right mouth
            ], dtype=np.float32)
            
            # Camera parameters (approximate)
            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Extract Euler angles
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                pitch = np.arcsin(-rotation_matrix[2, 0])
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                
                return np.array([pitch, yaw, roll])
        except:
            pass
        return None
    
    def get_face_scale(self, landmarks):
        """Estimate face scale (distance from camera) based on eye distance."""
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_INDICES)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_INDICES)
        
        if left_eye_center is None or right_eye_center is None:
            return 1.0
        
        # Distance between eyes (normalized)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        # Set baseline during calibration if not set
        if self.baseline_eye_distance is None:
            self.baseline_eye_distance = eye_distance
        
        # Normalize to baseline
        if self.baseline_eye_distance > 0:
            scale_factor = self.baseline_eye_distance / max(eye_distance, 0.05)
            return np.clip(scale_factor, 0.5, 2.0)  # Limit scale range
        
        return 1.0
    
    def get_raw_eye_features(self, landmarks):
        """Extract raw eye features for calibration with quality checks."""
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_INDICES)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_INDICES)
        left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS_INDICES)
        right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS_INDICES)
        
        if (left_eye_center is None or right_eye_center is None or 
            left_iris_center is None or right_iris_center is None):
            return None
        
        # Get eye dimensions
        left_eye_points = self.get_eye_landmarks(landmarks, self.LEFT_EYE_INDICES)
        right_eye_points = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES)
        
        left_eye_width = np.max(left_eye_points[:, 0]) - np.min(left_eye_points[:, 0])
        left_eye_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
        right_eye_width = np.max(right_eye_points[:, 0]) - np.min(right_eye_points[:, 0])
        right_eye_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
        
        left_eye_size = max(left_eye_width, left_eye_height, 0.01)
        right_eye_size = max(right_eye_width, right_eye_height, 0.01)
        
        # Quality checks
        # Check eye size is reasonable
        if left_eye_size < 0.01 or right_eye_size < 0.01:
            return None  # Eyes too small
        
        # Calculate normalized pupil positions
        left_pupil_normalized = (left_iris_center - left_eye_center) / left_eye_size
        right_pupil_normalized = (right_iris_center - right_eye_center) / right_eye_size
        
        # Quality check: pupil should be within reasonable bounds of eye
        if (abs(left_pupil_normalized[0]) > 0.5 or abs(left_pupil_normalized[1]) > 0.5 or
            abs(right_pupil_normalized[0]) > 0.5 or abs(right_pupil_normalized[1]) > 0.5):
            return None  # Pupil too far from center (likely error)
        
        # Return feature vector: [left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, face_center_x, face_center_y]
        face_center = (left_eye_center + right_eye_center) / 2.0
        
        # Quality check: face center should be within reasonable bounds
        if (face_center[0] < 0.1 or face_center[0] > 0.9 or 
            face_center[1] < 0.1 or face_center[1] > 0.9):
            return None  # Face too close to edge
        
        return np.array([
            left_pupil_normalized[0],
            left_pupil_normalized[1],
            right_pupil_normalized[0],
            right_pupil_normalized[1],
            face_center[0],
            face_center[1]
        ])
    
    def calibrate(self, back_cap, front_cap):
        """Perform calibration by asking user to look at specific points."""
        print("\n" + "="*60)
        print("CALIBRATION")
        print("="*60)
        print("Please look at each point on the screen when it appears.")
        print("Keep your head still and only move your eyes.")
        print("Press SPACE when you're looking at the point.")
        print("="*60 + "\n")
        
        # Define calibration points (9-point grid for better coverage)
        calibration_points = [
            (0.1, 0.1),  # Top-left
            (0.5, 0.1),  # Top-center
            (0.9, 0.1),  # Top-right
            (0.1, 0.5),  # Middle-left
            (0.5, 0.5),  # Center
            (0.9, 0.5),  # Middle-right
            (0.1, 0.9),  # Bottom-left
            (0.5, 0.9),  # Bottom-center
            (0.9, 0.9),  # Bottom-right
        ]
        
        point_names = ["Top-Left", "Top-Center", "Top-Right", 
                      "Middle-Left", "Center", "Middle-Right",
                      "Bottom-Left", "Bottom-Center", "Bottom-Right"]
        
        self.calibration_points = []
        self.calibration_eye_data = []
        
        # Get frame dimensions
        ret, test_frame = back_cap.read()
        if not ret:
            print("Error: Could not read from back camera for calibration")
            return False
        
        h, w = test_frame.shape[:2]
        
        for i, (target_x, target_y) in enumerate(calibration_points):
            print(f"\nLook at {point_names[i]} point...")
            time.sleep(1)
            
            # Collect samples for this calibration point (increased for better accuracy)
            samples = []
            sample_count = 0
            max_samples = 100  # Calibration sample count
            
            while sample_count < max_samples:
                ret, frame = back_cap.read()
                if not ret:
                    continue
                
                # Process frame
                self._ensure_mediapipe()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        eye_features = self.get_raw_eye_features(face_landmarks)
                        if eye_features is not None:
                            samples.append(eye_features)
                            sample_count += 1
                
                # Draw calibration UI
                display_frame = frame.copy()
                pixel_x = int(target_x * w)
                pixel_y = int(target_y * h)
                
                # Draw target point
                cv2.circle(display_frame, (pixel_x, pixel_y), 20, (0, 255, 0), -1)
                cv2.circle(display_frame, (pixel_x, pixel_y), 25, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Look here: {point_names[i]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Collecting samples: {sample_count}/{max_samples}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Calibration", display_frame)
                cv2.waitKey(1)
            
            # Average the samples for this point with outlier rejection
            if samples:
                samples_array = np.array(samples)
                
                # Remove outliers using IQR method
                Q1 = np.percentile(samples_array, 25, axis=0)
                Q3 = np.percentile(samples_array, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Filter outliers
                mask = np.all((samples_array >= lower_bound) & (samples_array <= upper_bound), axis=1)
                filtered_samples = samples_array[mask]
                
                if len(filtered_samples) > max_samples * 0.7:  # Keep if >70% valid
                    avg_features = np.mean(filtered_samples, axis=0)
                    self.calibration_eye_data.append(avg_features)
                    self.calibration_points.append((target_x, target_y))
                    print(f"✓ Calibrated {point_names[i]} point ({len(filtered_samples)}/{len(samples)} valid samples)")
                else:
                    print(f"✗ Skipped {point_names[i]} - too many outliers ({len(filtered_samples)}/{len(samples)} valid)")
        
        cv2.destroyWindow("Calibration")
        
        # Set baseline eye distance for scale compensation
        if len(self.calibration_eye_data) > 0:
            # Use average eye distance from calibration
            eye_distances = []
            for eye_data in self.calibration_eye_data:
                # Extract face center and estimate eye distance
                left_eye_x = eye_data[0] * 0.1 + eye_data[4]  # Approximate
                right_eye_x = eye_data[2] * 0.1 + eye_data[4]  # Approximate
                eye_distances.append(abs(right_eye_x - left_eye_x))
            if eye_distances:
                self.baseline_eye_distance = np.mean(eye_distances)
        
        # Build calibration matrix using least squares
        if len(self.calibration_eye_data) >= 5:
            self._build_calibration_matrix()
            self.calibrated = True
            print("\n✓ Calibration complete!")
            return True
        else:
            print("\n✗ Calibration failed: Not enough points")
            return False
    
    def _build_calibration_matrix(self):
        """Build transformation matrix from eye features to screen coordinates."""
        # Convert to numpy arrays
        X = np.array(self.calibration_eye_data)  # Eye features (N x 6)
        y = np.array(self.calibration_points)  # Screen coordinates (N x 2)
        
        # Add polynomial features for better mapping (quadratic terms)
        # This helps with non-linear eye-to-screen mapping
        X_poly = np.hstack([
            X,  # Linear terms
            X[:, :4] ** 2,  # Quadratic terms for pupil positions
            np.ones((X.shape[0], 1))  # Bias term
        ])
        
        # Solve for transformation matrix using least squares with regularization
        # y = X_poly * W, where W is (11 x 2)
        # Add small regularization to prevent overfitting
        lambda_reg = 0.01
        W = np.linalg.lstsq(
            X_poly.T @ X_poly + lambda_reg * np.eye(X_poly.shape[1]),
            X_poly.T @ y,
            rcond=None
        )[0]
        
        self.calibration_matrix = W
        self.calibration_poly_features = True
        print(f"Calibration matrix shape: {W.shape}")
    
    def estimate_gaze_point_calibrated(self, landmarks, image_shape):
        """Estimate gaze point using calibration data with head pose and scale compensation."""
        eye_features = self.get_raw_eye_features(landmarks)
        if eye_features is None or self.calibration_matrix is None:
            return None
        
        # Get head pose for compensation
        head_pose = self.get_head_pose(landmarks, image_shape)
        
        # Get face scale for distance compensation
        face_scale = self.get_face_scale(landmarks)
        
        # Build polynomial features (matching calibration)
        if self.calibration_poly_features:
            features_poly = np.hstack([
                eye_features,  # Linear terms
                eye_features[:4] ** 2,  # Quadratic terms for pupil positions
                1.0  # Bias term
            ])
        else:
            features_poly = np.hstack([eye_features, 1.0])
        
        # Transform to screen coordinates
        gaze_point = features_poly @ self.calibration_matrix
        
        # Apply head pose compensation (if available)
        if head_pose is not None:
            # Compensate for yaw (left/right head rotation)
            yaw_compensation = head_pose[1] * 0.3  # Scale factor
            gaze_point[0] += yaw_compensation
            
            # Compensate for pitch (up/down head rotation)
            pitch_compensation = head_pose[0] * 0.3
            gaze_point[1] += pitch_compensation
        
        # Apply scale compensation
        if face_scale != 1.0:
            # Adjust gaze point based on distance from camera
            face_center = eye_features[4:6]  # Last two features are face center
            gaze_point = face_center + (gaze_point - face_center) * face_scale
        
        # Clamp to image bounds
        gaze_point[0] = np.clip(gaze_point[0], 0.0, 1.0)
        gaze_point[1] = np.clip(gaze_point[1], 0.0, 1.0)
        
        return gaze_point
    
    def initialize_cameras(self):
        """Initialize both front and back cameras."""
        print("Initializing cameras...")
        is_jetson = self._is_jetson()
        if is_jetson:
            print("Jetson platform detected; using GStreamer pipelines when possible.")
            if "DISPLAY" not in os.environ:
                print("Warning: DISPLAY is not set; GUI windows may not appear.")
            if self.front_device is None and self.front_sensor_id is None:
                self.front_sensor_id = self.front_camera_id
            if self.back_device is None and self.back_sensor_id is None:
                self.back_sensor_id = self.back_camera_id

        # Try to open front camera
        self.front_cap, front_desc, front_is_gst = self._open_camera(
            self.front_camera_id, prefer_csi=is_jetson,
            device_path=self.front_device, sensor_id=self.front_sensor_id
        )
        if not self.front_cap.isOpened():
            print(f"Warning: Could not open front camera (ID: {self.front_camera_id})")
            print("Trying alternative camera IDs...")
            # Try common camera IDs
            for cam_id in [0, 1, 2]:
                self.front_cap, front_desc, front_is_gst = self._open_camera(
                    cam_id, prefer_csi=is_jetson
                )
                if self.front_cap.isOpened():
                    self.front_camera_id = cam_id
                    print(f"Found front camera at ID: {cam_id}")
                    break
            else:
                print("Error: Could not open any camera for front view")
                return False
        if front_desc:
            print(f"Front camera opened using {front_desc}")

        # Try to open back camera
        self.back_cap = None
        back_desc = None
        back_is_gst = False
        if self.back_camera_id != self.front_camera_id:
            self.back_cap, back_desc, back_is_gst = self._open_camera(
                self.back_camera_id, prefer_csi=is_jetson,
                device_path=self.back_device, sensor_id=self.back_sensor_id
            )
            if not self.back_cap.isOpened():
                print(f"Warning: Could not open back camera (ID: {self.back_camera_id})")
                # Try to find another camera
                for cam_id in [0, 1, 2]:
                    if cam_id != self.front_camera_id:
                        self.back_cap, back_desc, back_is_gst = self._open_camera(
                            cam_id, prefer_csi=is_jetson
                        )
                        if self.back_cap.isOpened():
                            self.back_camera_id = cam_id
                            print(f"Found back camera at ID: {cam_id}")
                            break
                else:
                    print("Warning: Only one camera available. Using front camera only.")
                    self.back_cap = None
        else:
            print("Note: Front and back camera IDs are the same; using front camera only.")

        if self.back_cap and back_desc:
            print(f"Back camera opened using {back_desc}")

        # Set camera properties for better performance (only for non-GStreamer)
        if self.front_cap and not front_is_gst:
            self.front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        if self.back_cap and not back_is_gst:
            self.back_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.back_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        
        print("Cameras initialized successfully!")
        return True
    
    def get_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmark coordinates."""
        eye_points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            eye_points.append([landmark.x, landmark.y])
        return np.array(eye_points)
    
    def get_eye_center(self, landmarks, eye_indices):
        """Calculate the center point of an eye."""
        eye_points = self.get_eye_landmarks(landmarks, eye_indices)
        if len(eye_points) > 0:
            center = np.mean(eye_points, axis=0)
            return center
        return None
    
    def get_iris_center(self, landmarks, iris_indices):
        """Calculate the center point of the iris (pupil)."""
        iris_points = []
        for idx in iris_indices:
            landmark = landmarks.landmark[idx]
            iris_points.append([landmark.x, landmark.y])
        if len(iris_points) > 0:
            center = np.mean(iris_points, axis=0)
            return center
        return None

    def _get_eye_roi_rects(self, landmarks, h, w, padding_ratio=0.4):
        """
        Get left and right eye bounding rects (x0, y0, w_roi, h_roi) in pixel coords
        from face landmarks. Returns ([left_rect], [right_rect]); each rect may be None.
        """
        def rect_for_eye(eye_points_norm):
            if eye_points_norm is None or len(eye_points_norm) == 0:
                return None
            px = (eye_points_norm[:, 0] * w).astype(np.float32)
            py = (eye_points_norm[:, 1] * h).astype(np.float32)
            x_min, x_max = float(np.min(px)), float(np.max(px))
            y_min, y_max = float(np.min(py)), float(np.max(py))
            bw, bh = x_max - x_min, y_max - y_min
            pad_w = max(8, bw * padding_ratio)
            pad_h = max(8, bh * padding_ratio)
            x0 = max(0, int(x_min - pad_w))
            y0 = max(0, int(y_min - pad_h))
            x1 = min(w, int(x_max + pad_w))
            y1 = min(h, int(y_max + pad_h))
            rw, rh = x1 - x0, y1 - y0
            if rw < 30 or rh < 30:
                return None
            return (x0, y0, rw, rh)

        left_pts = self.get_eye_landmarks(landmarks, self.LEFT_EYE_INDICES)
        right_pts = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES)
        left_rect = rect_for_eye(left_pts)
        right_rect = rect_for_eye(right_pts)
        return left_rect, right_rect

    def draw_eye_tracking(self, image, landmarks, image_shape):
        """Draw eye tracking overlay on the image."""
        h, w = image_shape[:2]
        
        # Get eye centers
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_INDICES)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_INDICES)
        
        # Get iris centers (more precise)
        left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS_INDICES)
        right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS_INDICES)
        
        # Draw left eye
        if left_eye_center is not None:
            left_eye_pixel = (int(left_eye_center[0] * w), int(left_eye_center[1] * h))
            cv2.circle(image, left_eye_pixel, 5, (0, 255, 0), 2)
            
            # Draw eye outline
            left_eye_points = self.get_eye_landmarks(landmarks, self.LEFT_EYE_INDICES)
            for point in left_eye_points:
                pixel = (int(point[0] * w), int(point[1] * h))
                cv2.circle(image, pixel, 2, (0, 255, 255), 1)
        
        # Draw right eye
        if right_eye_center is not None:
            right_eye_pixel = (int(right_eye_center[0] * w), int(right_eye_center[1] * h))
            cv2.circle(image, right_eye_pixel, 5, (0, 255, 0), 2)
            
            # Draw eye outline
            right_eye_points = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES)
            for point in right_eye_points:
                pixel = (int(point[0] * w), int(point[1] * h))
                cv2.circle(image, pixel, 2, (0, 255, 255), 1)
        
        # Draw iris centers (pupil tracking)
        if left_iris_center is not None:
            left_iris_pixel = (int(left_iris_center[0] * w), int(left_iris_center[1] * h))
            cv2.circle(image, left_iris_pixel, 3, (255, 0, 0), -1)
            cv2.circle(image, left_iris_pixel, 8, (255, 0, 0), 1)
        
        if right_iris_center is not None:
            right_iris_pixel = (int(right_iris_center[0] * w), int(right_iris_center[1] * h))
            cv2.circle(image, right_iris_pixel, 3, (255, 0, 0), -1)
            cv2.circle(image, right_iris_pixel, 8, (255, 0, 0), 1)
        
        # Draw line connecting eyes
        if left_eye_center is not None and right_eye_center is not None:
            left_pixel = (int(left_eye_center[0] * w), int(left_eye_center[1] * h))
            right_pixel = (int(right_eye_center[0] * w), int(right_eye_center[1] * h))
            cv2.line(image, left_pixel, right_pixel, (255, 255, 0), 1)
        
        return image
    
    def estimate_gaze_point(self, landmarks, image_shape):
        """Estimate gaze point - uses calibration if available, otherwise uses default method."""
        if self.calibrated:
            return self.estimate_gaze_point_calibrated(landmarks, image_shape)
        else:
            return self.estimate_gaze_point_default(landmarks, image_shape)
    
    def estimate_gaze_point_default(self, landmarks, image_shape):
        """
        Estimate where the user is looking based on pupil/iris position relative to eye socket.
        Returns gaze point in normalized coordinates (0-1).
        """
        h, w = image_shape[:2]
        
        # Get eye centers (eye socket centers) and iris/pupil centers
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_INDICES)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_INDICES)
        left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS_INDICES)
        right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS_INDICES)
        
        if (left_eye_center is None or right_eye_center is None or 
            left_iris_center is None or right_iris_center is None):
            return None
        
        # Get eye boundary points to calculate eye dimensions
        left_eye_points = self.get_eye_landmarks(landmarks, self.LEFT_EYE_INDICES)
        right_eye_points = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_INDICES)
        
        # Calculate eye dimensions (width and height) for normalization
        left_eye_width = np.max(left_eye_points[:, 0]) - np.min(left_eye_points[:, 0])
        left_eye_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
        right_eye_width = np.max(right_eye_points[:, 0]) - np.min(right_eye_points[:, 0])
        right_eye_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
        
        # Use individual eye dimensions for more accurate normalization
        left_eye_size = max(left_eye_width, left_eye_height, 0.01)
        right_eye_size = max(right_eye_width, right_eye_height, 0.01)
        
        # Calculate normalized pupil position within each eye (0-1 range, center = 0.5)
        # This gives us the relative position of the pupil in the eye socket
        left_pupil_normalized = (left_iris_center - left_eye_center) / left_eye_size
        right_pupil_normalized = (right_iris_center - right_eye_center) / right_eye_size
        
        # Average both eyes for stability
        pupil_normalized = (left_pupil_normalized + right_pupil_normalized) / 2.0
        
        # Get the center point between the two eyes (face center reference)
        face_center = (left_eye_center + right_eye_center) / 2.0
        
        # Map pupil position to screen coordinates
        # Use a more conservative sensitivity for stability
        # The normalized pupil position ranges roughly from -0.3 to 0.3 for normal eye movement
        gaze_sensitivity = 2.5  # Adjusted for more stable tracking
        
        # Calculate gaze point: start from face center and offset by pupil movement
        gaze_point = face_center + pupil_normalized * gaze_sensitivity
        
        # Clamp to image bounds
        gaze_point[0] = np.clip(gaze_point[0], 0.0, 1.0)
        gaze_point[1] = np.clip(gaze_point[1], 0.0, 1.0)
        
        return gaze_point
    
    def smooth_gaze_point(self, gaze_point):
        """Apply multiple smoothing techniques: temporal buffer + Kalman filtering."""
        if gaze_point is None:
            # If no measurement, predict based on current state
            if self.last_gaze_point is not None:
                prediction = self.predict_gaze_kalman(None)
                if prediction is not None:
                    self.last_gaze_point = np.array(prediction)
                return self.last_gaze_point
            return None
        
        # Convert gaze_point to numpy array if it isn't already
        if isinstance(gaze_point, (list, tuple)):
            gaze_point = np.array(gaze_point)
        
        # Add to temporal buffer
        self.gaze_buffer.append(gaze_point)
        
        # Use median of buffer to reduce outliers (if we have enough samples)
        if len(self.gaze_buffer) >= 3:
            buffer_array = np.array(list(self.gaze_buffer))
            median_gaze = np.median(buffer_array, axis=0)
        else:
            median_gaze = gaze_point
        
        # Then apply Kalman filter for smooth prediction
        smoothed = self.predict_gaze_kalman(median_gaze)
        if smoothed is not None:
            self.last_gaze_point = np.array(smoothed)
            return self.last_gaze_point
        else:
            # Fallback to median if Kalman fails
            self.last_gaze_point = median_gaze
            return median_gaze
    
    def update_heatmap(self, gaze_point, image_shape):
        """Update the heatmap with a new gaze point."""
        if gaze_point is None:
            return
        
        h, w = image_shape[:2]
        
        # Initialize heatmap if needed
        if self.heatmap is None or self.heatmap.shape[:2] != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Decay existing heatmap
        self.heatmap *= self.heatmap_decay
        
        # Convert gaze point to pixel coordinates (clamped to valid bounds)
        try:
            gx = float(gaze_point[0])
            gy = float(gaze_point[1])
        except Exception:
            return
        x = int(np.clip(gx, 0.0, 1.0) * (w - 1))
        y = int(np.clip(gy, 0.0, 1.0) * (h - 1))
        
        # Add heat at gaze point (Gaussian-like distribution)
        radius = 40
        y_coords, x_coords = np.ogrid[:h, :w]
        mask = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius ** 2
        distance = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
        heat_value = np.exp(-distance ** 2 / (2 * (radius / 3) ** 2))
        self.heatmap[mask] += heat_value[mask] * self.heatmap_intensity
        
        # Normalize heatmap
        self.heatmap = np.clip(self.heatmap, 0, 1)
    
    def draw_heatmap(self, image):
        """Draw the heatmap overlay on the image."""
        if self.heatmap is None:
            return image
        
        h, w = image.shape[:2]
        
        # Create colored heatmap (blue to green to yellow to red)
        heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Normalize heatmap for better visualization
        heatmap_normalized = self.heatmap / (np.max(self.heatmap) + 1e-6)
        
        # Blue channel (low values)
        heatmap_colored[:, :, 0] = (np.clip(heatmap_normalized * 2, 0, 1) * 255).astype(np.uint8)
        
        # Green channel (medium values)
        heatmap_colored[:, :, 1] = (np.clip((heatmap_normalized - 0.3) * 2, 0, 1) * 255).astype(np.uint8)
        
        # Red channel (high values)
        heatmap_colored[:, :, 2] = (np.clip((heatmap_normalized - 0.6) * 2.5, 0, 1) * 255).astype(np.uint8)
        
        # Blend heatmap with original image
        alpha = 0.5  # Transparency of heatmap
        image = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return image
    
    def draw_gaze_indicator(self, image, gaze_point):
        """Draw a simple green dot showing where the user is currently looking."""
        if gaze_point is None:
            return image
        
        h, w = image.shape[:2]
        # Gaze point is expected to be normalized (0..1). Be defensive and clamp
        # to valid pixel coordinates so the dot never disappears off-screen.
        try:
            gx = float(gaze_point[0])
            gy = float(gaze_point[1])
        except Exception:
            return image
        
        x = int(np.clip(gx, 0.0, 1.0) * (w - 1))
        y = int(np.clip(gy, 0.0, 1.0) * (h - 1))
        
        # Draw a simple green dot
        cv2.circle(image, (x, y), 8, (0, 255, 0), -1)  # Green filled circle
        
        return image
    
    def process_frame(self, frame, show_eye_tracking=True):
        """Process a single frame for eye tracking visualization."""
        self._ensure_mediapipe()
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Draw eye tracking if enabled
        if show_eye_tracking and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame = self.draw_eye_tracking(frame, face_landmarks, frame.shape)
        
        return frame
    
    def estimate_gaze_from_frame(self, frame):
        """Estimate gaze point from a frame without drawing, with smoothing."""
        self._ensure_mediapipe()
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Estimate gaze point if face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                gaze_point = self.estimate_gaze_point(face_landmarks, frame.shape)
                if gaze_point is not None:
                    # Apply smoothing to reduce jitter
                    smoothed_gaze = self.smooth_gaze_point(gaze_point)
                    return smoothed_gaze
        
        # Fallback: eye-only frame (glass-frame style) when no face landmarks
        gaze_point = self.estimate_gaze_from_eye_frame(frame)
        if gaze_point is not None:
            smoothed_gaze = self.smooth_gaze_point(gaze_point)
            return smoothed_gaze
        return None

    def _ensure_pupil_detector(self):
        """Lazy-init PuRe (pupil-detectors) for glass-frame. No-op if not installed."""
        if self._pupil_detector_initialized:
            return
        self._pupil_detector_initialized = True
        try:
            from pupil_detectors import Detector2D
            self._pupil_detector_2d = Detector2D()
        except Exception:
            self._pupil_detector_2d = None

    def _suppress_glints(self, gray_roi):
        """Suppress bright specular highlights (glints) via inpainting for better pupil contrast."""
        if gray_roi is None or gray_roi.size == 0:
            return gray_roi
        p = float(np.percentile(gray_roi, 99.3))
        thr = int(max(220, min(250, p)))
        mask = (gray_roi >= thr).astype(np.uint8) * 255
        if cv2.countNonZero(mask) == 0:
            return gray_roi
        return cv2.inpaint(gray_roi, mask, 3, cv2.INPAINT_TELEA)

    def _ellipse_support_score(self, ellipse, points):
        """Return [0,1] score: how many points lie near the ellipse boundary."""
        if ellipse is None or points is None or len(points) == 0:
            return 0.0
        (cx, cy), (ma, mi), angle = ellipse
        axes = (max(1, int(ma * 0.5)), max(1, int(mi * 0.5)))
        poly = cv2.ellipse2Poly((int(cx), int(cy)), axes, int(angle), 0, 360, 6)
        if poly is None or len(poly) < 6:
            return 0.0
        contour = poly.reshape((-1, 1, 2)).astype(np.int32)
        inliers = 0
        for pt in points:
            d = abs(float(cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), True)))
            if d <= 3.0:
                inliers += 1
        return float(inliers) / max(1, len(points))

    def _refine_pupil_center_with_rays(self, gray_blur, seed_xy):
        """Starburst-like refinement: cast rays from center, find pupil boundary, fit ellipse. Returns (cx, cy, quality) or None."""
        if gray_blur is None or gray_blur.size == 0 or seed_xy is None:
            return None
        h, w = gray_blur.shape[:2]
        cx0, cy0 = float(seed_xy[0]), float(seed_xy[1])
        if cx0 < 1 or cy0 < 1 or cx0 >= (w - 1) or cy0 >= (h - 1):
            return None
        max_r = int(max(12, min(w, h) * 0.45))
        pts = []
        angles = np.linspace(0.0, 2.0 * np.pi, self._glass_ray_count, endpoint=False)
        for ang in angles:
            dx, dy = float(np.cos(ang)), float(np.sin(ang))
            px, py = int(round(cx0)), int(round(cy0))
            prev = float(gray_blur[py, px])
            best_pt, best_grad = None, 0.0
            s = self._glass_ray_step
            while s <= max_r:
                x = int(round(cx0 + dx * s))
                y = int(round(cy0 + dy * s))
                if x < 1 or y < 1 or x >= (w - 1) or y >= (h - 1):
                    break
                val = float(gray_blur[y, x])
                grad = val - prev
                if grad > self._glass_edge_grad_thresh and val > self._glass_edge_min_bright and grad > best_grad:
                    best_grad = grad
                    best_pt = (x, y)
                prev = val
                s += self._glass_ray_step
            if best_pt is not None:
                pts.append(best_pt)
        if len(pts) < 8:
            return None
        cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        try:
            ellipse = cv2.fitEllipse(cnt)
        except Exception:
            return None
        (ecx, ecy), (ma, mi), _ = ellipse
        a, b = max(float(ma), float(mi)), max(1e-6, min(float(ma), float(mi)))
        aspect = b / a
        if a < 6.0 or b < 3.0 or aspect < 0.18:
            return None
        support = self._ellipse_support_score(ellipse, pts)
        quality = float(np.clip(0.5 * support + 0.5 * np.clip(aspect / 0.6, 0.0, 1.0), 0.0, 1.0))
        return (float(ecx), float(ecy), quality)

    def _detect_pupil_pure(self, gray):
        """Run PuRe on grayscale image. Returns (cx, cy) in image coords or None."""
        if self._pupil_detector_2d is None or gray is None or gray.size == 0:
            return None
        h, w = gray.shape[:2]
        if h < 30 or w < 30:
            return None
        try:
            result = self._pupil_detector_2d.detect(gray)
            if result is None:
                return None
            ellipse = result.get("ellipse") if isinstance(result, dict) else getattr(result, "ellipse", None)
            if ellipse is None:
                return None
            center = ellipse.get("center") if isinstance(ellipse, dict) else getattr(ellipse, "center", None)
            if center is None or len(center) < 2:
                return None
            cx, cy = float(center[0]), float(center[1])
            if cx < 1 or cy < 1 or cx >= (w - 1) or cy >= (h - 1):
                return None
            return (cx, cy)
        except Exception:
            return None

    def _detect_pupil_center(self, frame):
        """Detect pupil center in an eye-only (glass-frame) frame. Returns normalized (x, y) or None.
        Uses same pipeline as webcam: PuRe or contour + CLAHE + glint suppress + ray refinement."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
        h, w = frame.shape[:2]
        no_glint = self._suppress_glints(gray)
        blur = cv2.GaussianBlur(no_glint, (7, 7), 0)

        # 1) Try PuRe first (robust for head-mounted / glass-frame)
        self._ensure_pupil_detector()
        pure_xy = self._detect_pupil_pure(blur)
        if pure_xy is not None:
            # Require dark center (pupil) to avoid locking onto bright blobs
            px, py = int(round(pure_xy[0])), int(round(pure_xy[1]))
            if 3 <= px < w - 3 and 3 <= py < h - 3:
                patch = blur[py - 3:py + 4, px - 3:px + 4]
                if patch.size > 0 and np.mean(patch) < 140:
                    return np.array([pure_xy[0] / w, pure_xy[1] / h], dtype=np.float32)
            elif 1 <= px < w - 1 and 1 <= py < h - 1:
                if blur[py, px] < 130:
                    return np.array([pure_xy[0] / w, pure_xy[1] / h], dtype=np.float32)

        # 2) Fallback: CLAHE + contour + ellipse + ray refinement (match webcam algorithm)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            norm = clahe.apply(blur)
        except Exception:
            norm = cv2.equalizeHist(blur)
        _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            thresh2 = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 5)
            thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
            thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        frame_area = float(h * w)
        best = None
        best_score = -1.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > frame_area * 0.12:
                continue
            if len(cnt) < 5:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if x <= 2 or y <= 2 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.35:
                continue
            aspect = min(bw, bh) / max(float(max(bw, bh)), 1e-6)
            if aspect < 0.45 or aspect > 1.0:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_in = cv2.mean(blur, mask=mask)[0]
            if mean_in > 100:
                continue
            ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            ring = cv2.subtract(ring, mask)
            if np.count_nonzero(ring) == 0:
                continue
            mean_ring = cv2.mean(blur, mask=ring)[0]
            contrast = mean_ring - mean_in
            if contrast < 10:
                continue
            ecc_pen = 1.0
            if len(cnt) >= 5:
                try:
                    (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
                    a, b = max(float(ma), float(mi)), max(1e-6, min(float(ma), float(mi)))
                    ecc = np.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
                    ecc_pen = np.clip(1.0 - 0.5 * ecc, 0.5, 1.0)
                except Exception:
                    pass
            border_pen = 1.0
            m = cv2.moments(cnt)
            if m["m00"] > 1e-6:
                cx_m = m["m10"] / m["m00"]
                cy_m = m["m01"] / m["m00"]
                if cx_m < 6 or cx_m > (w - 6) or cy_m < 6 or cy_m > (h - 6):
                    border_pen = 0.5
            score = (area * 0.3 + circularity * 100 + contrast * 0.5) * ecc_pen * border_pen
            if score > best_score:
                best_score = score
                best = cnt
        if best is None or len(best) < 5 or best_score < 25.0:
            return None
        try:
            ellipse = cv2.fitEllipse(best)
            ((cx, cy), (ma, MA), angle) = ellipse
            cx, cy = float(cx), float(cy)
        except Exception:
            m = cv2.moments(best)
            if m["m00"] == 0:
                return None
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        # Ray refinement (same as webcam): sub-pixel pupil boundary -> better center
        refined = self._refine_pupil_center_with_rays(blur, (cx, cy))
        if refined is not None:
            rcx, rcy, rq = refined
            alpha = float(np.clip(0.25 + 0.55 * rq, 0.25, 0.8))
            cx = (1.0 - alpha) * cx + alpha * rcx
            cy = (1.0 - alpha) * cy + alpha * rcy
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return None
        return np.array([cx / w, cy / h], dtype=np.float32)

    def _pupil_from_face_mesh(self, frame):
        """If frame contains a face, return normalized (x,y) pupil from iris landmarks else None."""
        self._ensure_mediapipe()
        if self.face_mesh is None:
            return None
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
                return None
            landmarks = results.multi_face_landmarks[0]
            left_iris = self.get_iris_center(landmarks, self.LEFT_IRIS_INDICES)
            right_iris = self.get_iris_center(landmarks, self.RIGHT_IRIS_INDICES)
            if left_iris is not None and right_iris is not None:
                cx = (left_iris[0] + right_iris[0]) / 2.0
                cy = (left_iris[1] + right_iris[1]) / 2.0
            elif left_iris is not None:
                cx, cy = left_iris[0], left_iris[1]
            elif right_iris is not None:
                cx, cy = right_iris[0], right_iris[1]
            else:
                return None
            return np.array([cx, cy], dtype=np.float32)
        except Exception:
            return None

    def _pupil_from_eye_landmark_rois(self, frame):
        """
        Detect that a pair of eyes is visible via Face Mesh, then run pupil detection
        only within the left and right eye ROIs. Returns normalized (x,y) gaze or None.
        """
        self._ensure_mediapipe()
        if self.face_mesh is None:
            return None
        try:
            full_h, full_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
                return None
            landmarks = results.multi_face_landmarks[0]
            left_rect, right_rect = self._get_eye_roi_rects(landmarks, full_h, full_w)
            pupils = []
            for rect in (left_rect, right_rect):
                if rect is None:
                    continue
                x0, y0, rw, rh = rect
                roi = frame[y0:y0 + rh, x0:x0 + rw]
                if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                    continue
                pupil_norm = self._detect_pupil_center(roi)
                if pupil_norm is not None:
                    nx, ny = float(pupil_norm[0]), float(pupil_norm[1])
                    full_x = (x0 + nx * rw) / full_w
                    full_y = (y0 + ny * rh) / full_h
                    pupils.append(np.array([full_x, full_y], dtype=np.float32))
            if len(pupils) == 2:
                return (pupils[0] + pupils[1]) * 0.5
            if len(pupils) == 1:
                return pupils[0]
            return None
        except Exception:
            return None

    def _load_custom_glass_rois(self):
        """Load user-defined eye ROIs from glass_frame_rois.json (your own landmarks). Called lazily."""
        if self._custom_glass_rois is not None:
            return
        for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
            path = os.path.join(base, GLASS_FRAME_ROIS_FILENAME)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                rois = []
                for key in ("left_eye_roi", "right_eye_roi"):
                    r = data.get(key)
                    if r and len(r) >= 4:
                        x, y, w, h = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                        if w > 0.02 and h > 0.02 and 0 <= x <= 1 and 0 <= y <= 1:
                            rois.append((x, y, w, h))
                if rois:
                    self._custom_glass_rois = rois
                    self._custom_glass_rois_path = path
            except Exception:
                pass
            break

    def _pupil_from_custom_glass_rois(self, frame):
        """Run pupil detection only inside user-defined eye ROIs. Returns normalized (x,y) or None."""
        self._load_custom_glass_rois()
        if not self._custom_glass_rois:
            return None
        full_h, full_w = frame.shape[:2]
        pupils = []
        for (nx, ny, nw, nh) in self._custom_glass_rois:
            x0 = int(nx * full_w)
            y0 = int(ny * full_h)
            rw = max(20, int(nw * full_w))
            rh = max(20, int(nh * full_h))
            x0 = max(0, min(x0, full_w - rw))
            y0 = max(0, min(y0, full_h - rh))
            roi = frame[y0:y0 + rh, x0:x0 + rw]
            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                continue
            pupil_norm = self._detect_pupil_center(roi)
            if pupil_norm is not None:
                px = (x0 + pupil_norm[0] * rw) / full_w
                py = (y0 + pupil_norm[1] * rh) / full_h
                pupils.append(np.array([px, py], dtype=np.float32))
        if len(pupils) == 2:
            return (pupils[0] + pupils[1]) * 0.5
        if len(pupils) == 1:
            return pupils[0]
        return None

    def _crop_glass_eye_roi(self, frame):
        """Crop to lower part of frame where the eye/pupil is (reduces false positives from ceiling/sky)."""
        if frame is None or frame.size == 0:
            return None, None
        h, w = frame.shape[:2]
        if self.glass_eye_crop_y_start <= 0 or self.glass_eye_crop_y_start >= 1.0:
            return frame, (0, 0, w, h)
        y0 = int(h * self.glass_eye_crop_y_start)
        if y0 >= h - 20:
            return frame, (0, 0, w, h)
        crop = frame[y0:, :].copy()
        return crop, (0, y0, w, h - y0)

    def estimate_gaze_from_eye_frame(self, frame):
        """Estimate gaze from glass-frame camera: no face detection (camera cannot capture full face).
        Uses: custom ROIs (glass_frame_rois.json) if present, else lower-half crop + pupil detection, then Kalman."""
        raw = self._pupil_from_custom_glass_rois(frame)
        if raw is None:
            cropped, crop_rect = self._crop_glass_eye_roi(frame)
            if cropped is not None and crop_rect is not None:
                raw = self._detect_pupil_center(cropped)
                if raw is not None:
                    _, y0, cw, ch = crop_rect
                    full_h, full_w = frame.shape[:2]
                    raw = np.array([
                        raw[0],
                        (y0 + raw[1] * ch) / full_h
                    ], dtype=np.float32)
            if raw is None:
                raw = self._detect_pupil_center(frame)
        if raw is None:
            self._glass_pupil_last = None
            return None
        raw = np.asarray(raw, dtype=np.float32).reshape(2)
        # Kalman smooth (same idea as webcam gaze path)
        if self._glass_pupil_kalman is None:
            self._glass_pupil_kalman = cv2.KalmanFilter(4, 2)
            self._glass_pupil_kalman.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            self._glass_pupil_kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            self._glass_pupil_kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.008
            self._glass_pupil_kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.12
            self._glass_pupil_kalman.errorCovPost = np.eye(4, dtype=np.float32)
            self._glass_pupil_kalman.statePost = np.array([[raw[0]], [raw[1]], [0.0], [0.0]], dtype=np.float32)
        self._glass_pupil_kalman.predict()
        self._glass_pupil_kalman.correct(np.array([[raw[0]], [raw[1]]], dtype=np.float32))
        smoothed = np.array([self._glass_pupil_kalman.statePost[0, 0], self._glass_pupil_kalman.statePost[1, 0]], dtype=np.float32)
        # Reject large jumps (outlier) to avoid glitches
        if self._glass_pupil_last is not None:
            jump = np.sqrt(np.sum((smoothed - self._glass_pupil_last) ** 2))
            if jump > 0.12:
                smoothed = self._glass_pupil_last.copy()
        self._glass_pupil_last = smoothed
        return smoothed

    def run(self):
        """Main loop to run the eye tracking application."""
        if not self.initialize_cameras():
            print("Failed to initialize cameras. Exiting.")
            return
        
        # Perform calibration
        if self.back_cap is None:
            print("Note: Only one camera detected. Using the front camera for gaze estimation.")
            print("      Calibration requires a dedicated back camera; skipping calibration.")
        elif not self.calibrated:
            print("\nStarting calibration process...")
            if not self.calibrate(self.back_cap, self.front_cap):
                print("Warning: Calibration failed. Continuing with default tracking.")
            else:
                print("Calibration successful! Using calibrated gaze tracking.")
        
        print("\n" + "="*60)
        print("Eye Tracking Application Started")
        print("="*60)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'f' to toggle fullscreen")
        print("  - Press 'g' to toggle gaze indicator")
        print("  - Press 'h' to toggle heatmap")
        print("  - Press 'c' to clear heatmap")
        print("  - Press 'r' to recalibrate")
        print("="*60)
        print("Note: Back camera tracks your eyes, gaze is overlaid on front camera")
        print("="*60 + "\n")
        
        # Give cameras a moment to initialize
        import time
        time.sleep(0.5)
        
        fullscreen = False
        show_gaze_indicator = True  # Gaze indicator enabled by default
        consecutive_failures_front = 0
        consecutive_failures_back = 0
        max_failures = 10
        
        try:
            while True:
                # Read from both cameras
                ret_front, frame_front = self.front_cap.read()
                ret_back = False
                frame_back = None
                
                if self.back_cap:
                    ret_back, frame_back = self.back_cap.read()
                
                # Check if we can read from front camera (required)
                if not ret_front:
                    consecutive_failures_front += 1
                    if consecutive_failures_front >= max_failures:
                        print(f"Failed to read from Front Camera after {max_failures} attempts")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures_front = 0
                
                # Estimate gaze point from back camera (where user is looking at screen)
                gaze_point = None
                if show_gaze_indicator:
                    if ret_back and frame_back is not None:
                        # Preferred: use back camera for eye tracking
                        gaze_point = self.estimate_gaze_from_frame(frame_back)
                        consecutive_failures_back = 0
                    else:
                        if self.back_cap:
                            consecutive_failures_back += 1
                            if consecutive_failures_back >= max_failures:
                                print("Warning: Back camera not available; falling back to front camera for gaze estimation.")
                        # Fallback: use front camera if back camera isn't available
                        gaze_point = self.estimate_gaze_from_frame(frame_front)
                
                # Display front camera with gaze overlay
                frame = frame_front.copy()
                
                # Update and draw heatmap if gaze point is available
                if show_gaze_indicator and gaze_point is not None:
                    # Update heatmap
                    if self.show_heatmap:
                        self.update_heatmap(gaze_point, frame.shape)
                        # Draw heatmap
                        frame = self.draw_heatmap(frame)
                    # Draw current gaze indicator
                    frame = self.draw_gaze_indicator(frame, gaze_point)
                
                # Add text overlay
                cv2.putText(frame, "Front Camera (with gaze overlay)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                gaze_status = "ON" if (show_gaze_indicator and gaze_point is not None) else "OFF"
                if ret_back and frame_back is not None:
                    back_cam_status = "Connected"
                elif self.back_cap is None:
                    back_cam_status = "Single Cam Mode"
                else:
                    back_cam_status = "Not Available"
                cv2.putText(frame, f"Gaze Tracking: {gaze_status} | Back Cam: {back_cam_status}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit, 'g' to toggle gaze, 'f' for fullscreen", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display the frame
                window_name = "Eye Tracking"
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nExiting...")
                    break
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(window_name, cv2.WINDOW_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                elif key == ord('g'):
                    show_gaze_indicator = not show_gaze_indicator
                    print(f"Gaze Indicator: {'ON' if show_gaze_indicator else 'OFF'}")
                elif key == ord('r'):
                    print("\nRecalibrating...")
                    self.calibrated = False
                    self.calibration_matrix = None
                    self.calibration_poly_features = False
                    if self.calibrate(self.back_cap, self.front_cap):
                        print("Recalibration successful!")
                    else:
                        print("Recalibration failed. Using default tracking.")
                elif key == ord('h'):
                    self.show_heatmap = not self.show_heatmap
                    print(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}")
                elif key == ord('c'):
                    if self.heatmap is not None:
                        self.heatmap.fill(0)
                        print("Heatmap cleared")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
        
        finally:
            # Cleanup
            if self.front_cap:
                self.front_cap.release()
            if self.back_cap:
                self.back_cap.release()
            cv2.destroyAllWindows()
            print("Cameras released. Goodbye!")


def main():
    parser = argparse.ArgumentParser(description='Eye Tracking Application')
    parser.add_argument('--front-camera', type=int, default=0,
                       help='Camera ID for front camera (default: 0)')
    parser.add_argument('--back-camera', type=int, default=1,
                       help='Camera ID for back camera (default: 1)')
    parser.add_argument('--front-device', type=str, default=None,
                       help='Front camera device path (e.g. /dev/video0)')
    parser.add_argument('--back-device', type=str, default=None,
                       help='Back camera device path (e.g. /dev/video1)')
    parser.add_argument('--front-sensor-id', type=int, default=None,
                       help='Jetson CSI front camera sensor-id (overrides device/id)')
    parser.add_argument('--back-sensor-id', type=int, default=None,
                       help='Jetson CSI back camera sensor-id (overrides device/id)')
    parser.add_argument('--camera-width', type=int, default=640,
                       help='Requested camera width (default: 640)')
    parser.add_argument('--camera-height', type=int, default=480,
                       help='Requested camera height (default: 480)')
    parser.add_argument('--camera-fps', type=int, default=30,
                       help='Requested camera FPS (default: 30)')
    
    args = parser.parse_args()
    
    tracker = EyeTracker(
        front_camera_id=args.front_camera,
        back_camera_id=args.back_camera,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        front_device=args.front_device,
        back_device=args.back_device,
        front_sensor_id=args.front_sensor_id,
        back_sensor_id=args.back_sensor_id
    )
    tracker.run()


if __name__ == "__main__":
    main()

