import time
from .calibration import Calibration


class GlassFrameTrainingConfig:
    def __init__(
        self,
        screen_size=None,
        board_size=None,
        training_data_name="peter",
        training_data_dir="eyetracking_ml",
        distance_m=None,
        distance_source="manual",
        training_mode="glass_frame",
    ):
        self.screen_size = screen_size
        self.board_size = board_size
        self.training_data_name = training_data_name
        self.training_data_dir = training_data_dir
        self.distance_m = distance_m
        self.distance_source = distance_source
        self.training_mode = training_mode


class GlassFrameTraining:
    def __init__(self, config):
        if config.training_mode != "glass_frame":
            raise ValueError("GlassFrameTraining requires training_mode='glass_frame'.")
        self.config = config
        self.calib = Calibration(
            training_data_name=self.config.training_data_name,
            training_data_dir=self.config.training_data_dir,
            screen_size=self.config.board_size or self.config.screen_size,
            fullscreen=False,
            training_mode="glass_frame",
        )

    def _make_training_row(
        self,
        pupil_features,
        conf,
        board_xy,
        board_size,
        tag_bbox=None,
        tag_id=None,
        board_distance_m=None,
    ):
        """
        Glass-frame training uses the same pupil feature columns as webcam.
        board_xy/board_size represent AprilTag position and board size in the tag camera frame.
        """
        if board_xy is None or board_size is None:
            return None
        board_w, board_h = board_size
        bx, by = board_xy
        screen_x_norm = float(bx) / float(board_w)
        screen_y_norm = float(by) / float(board_h)
        row = {
            "timestamp": time.time(),
            "dot_index": "",
            "stage": "glassframe_tag",
            "screen_x": bx,
            "screen_y": by,
            "screen_x_norm": screen_x_norm,
            "screen_y_norm": screen_y_norm,
            "screen_w": board_w,
            "screen_h": board_h,
            "gx": "",
            "gy": "",
            "yaw": 0.0,
            "pitch": 0.0,
            "gxL": float(pupil_features[0]),
            "gyL": float(pupil_features[1]),
            "gxR": float(pupil_features[2]),
            "gyR": float(pupil_features[3]),
            "iod": float(pupil_features[6]) if len(pupil_features) > 6 else 0.0,
            "roll": float(pupil_features[7]) if len(pupil_features) > 7 else 0.0,
            "lid_hL": float(pupil_features[8]) if len(pupil_features) > 8 else 0.0,
            "lid_hR": float(pupil_features[9]) if len(pupil_features) > 9 else 0.0,
            "face_w": float(pupil_features[10]) if len(pupil_features) > 10 else 0.0,
            "face_h": float(pupil_features[11]) if len(pupil_features) > 11 else 0.0,
            "nose_x": float(pupil_features[12]) if len(pupil_features) > 12 else 0.0,
            "nose_y": float(pupil_features[13]) if len(pupil_features) > 13 else 0.0,
            "eye_w": "",
            "lid_h": "",
            "conf": float(conf),
            "bbox_x": "",
            "bbox_y": "",
            "bbox_w": "",
            "bbox_h": "",
            "label_x": "",
            "label_y": "",
            "label_x_norm": "",
            "label_y_norm": "",
            "tag_id": tag_id if tag_id is not None else "",
            "board_distance_m": board_distance_m if board_distance_m is not None else "",
        }
        if tag_bbox is not None:
            row["bbox_x"] = tag_bbox[0]
            row["bbox_y"] = tag_bbox[1]
            row["bbox_w"] = tag_bbox[2]
            row["bbox_h"] = tag_bbox[3]
        if board_xy is not None:
            row["label_x"] = bx
            row["label_y"] = by
            row["label_x_norm"] = screen_x_norm
            row["label_y_norm"] = screen_y_norm
        return row

    def run(self):
        raise NotImplementedError(
            "Glass frame training is not implemented yet. "
            "Use webcam training for now."
        )
