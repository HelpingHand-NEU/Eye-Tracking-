import cv2


def _csi_pipeline(sensor_id: int, width: int = 1280, height: int = 720, fps: int = 30) -> str:
    """GStreamer pipeline for Jetson CSI (e.g. cam0 = sensor 0, 24-pin)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx, width=(int){width}, height=(int){height} ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


class InputSource:
    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class CameraSource(InputSource):
    def __init__(self, camera_id=0, width=1280, height=720, fps=30, csi_sensor_id=None):
        if csi_sensor_id is not None:
            pipeline = _csi_pipeline(csi_sensor_id, width=width, height=height, fps=fps)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()


class ImageSource(InputSource):
    def __init__(self, path):
        self.path = path
        self._frame = cv2.imread(path)
        if self._frame is None:
            raise FileNotFoundError(f"Could not read image: {path}")

    def read(self):
        return self._frame.copy()
