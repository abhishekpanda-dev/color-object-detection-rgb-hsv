"""Small OpenCV webcam lifecycle helper for the Streamlit interface."""

import weakref

import cv2
import numpy as np


class WebcamCapture:
    """Own one OpenCV camera and provide RGB frames to the existing pipeline."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        capture = cv2.VideoCapture(device_index)
        self._capture = capture
        self._finalizer = weakref.finalize(self, self._release_capture, capture)

        if not capture.isOpened():
            self.release()
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @staticmethod
    def _release_capture(capture: cv2.VideoCapture) -> None:
        """Release the underlying OpenCV resource exactly once."""
        capture.release()

    def is_opened(self) -> bool:
        """Return whether the camera is currently available."""
        return self._capture is not None and self._capture.isOpened()

    def read_rgb_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read one BGR camera frame and convert it immediately to RGB."""
        if not self.is_opened():
            return False, None

        success, bgr_frame = self._capture.read()
        if not success or bgr_frame is None:
            return False, None

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return True, rgb_frame

    def release(self) -> None:
        """Release the camera explicitly; repeated calls are safe."""
        if self._finalizer.alive:
            self._finalizer()
        self._capture = None
