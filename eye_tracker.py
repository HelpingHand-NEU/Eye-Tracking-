#!/usr/bin/env python3
"""
Eye Tracking Application
Accesses both front and back cameras, tracks eyes, and overlays tracking on forward-facing camera.
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import argparse
import time
from collections import deque


class EyeTracker:
    def __init__(self, front_camera_id=0, back_camera_id=1):
        """
        Initialize the Eye Tracker with camera IDs.
        
        Args:
            front_camera_id: Camera index for forward-facing camera (default: 0)
            back_camera_id: Camera index for back camera (default: 1)
        """
        self.front_camera_id = front_camera_id
        self.back_camera_id = back_camera_id
        
        # Initialize MediaPipe Face Mesh for eye tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh model with refined landmarks for better eye tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
        
        # Try to open front camera
        self.front_cap = cv2.VideoCapture(self.front_camera_id)
        if not self.front_cap.isOpened():
            print(f"Warning: Could not open front camera (ID: {self.front_camera_id})")
            print("Trying alternative camera IDs...")
            # Try common camera IDs
            for cam_id in [0, 1, 2]:
                self.front_cap = cv2.VideoCapture(cam_id)
                if self.front_cap.isOpened():
                    self.front_camera_id = cam_id
                    print(f"Found front camera at ID: {cam_id}")
                    break
            else:
                print("Error: Could not open any camera for front view")
                return False
        
        # Try to open back camera
        self.back_cap = cv2.VideoCapture(self.back_camera_id)
        if not self.back_cap.isOpened():
            print(f"Warning: Could not open back camera (ID: {self.back_camera_id})")
            # Try to find another camera
            for cam_id in [0, 1, 2]:
                if cam_id != self.front_camera_id:
                    self.back_cap = cv2.VideoCapture(cam_id)
                    if self.back_cap.isOpened():
                        self.back_camera_id = cam_id
                        print(f"Found back camera at ID: {cam_id}")
                        break
            else:
                print("Warning: Only one camera available. Using front camera only.")
                self.back_cap = None
        
        # Set camera properties for better performance
        if self.front_cap:
            self.front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if self.back_cap:
            self.back_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.back_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
        
        # If no face detected, return None (smoothing will handle it)
        return None
    
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
    
    args = parser.parse_args()
    
    tracker = EyeTracker(front_camera_id=args.front_camera, 
                        back_camera_id=args.back_camera)
    tracker.run()


if __name__ == "__main__":
    main()

