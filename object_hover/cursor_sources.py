import pyautogui


class CursorSource:
    def get_xy(self):
        raise NotImplementedError


class MouseCursor(CursorSource):
    def get_xy(self):
        x, y = pyautogui.position()
        return int(x), int(y)
