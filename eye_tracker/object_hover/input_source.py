import cv2


class InputSource:
    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class CameraSource(InputSource):
    def __init__(self, camera_id=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

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
