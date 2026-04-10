from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from proctor_app.config import CameraSettings


class CameraStream:
    def __init__(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._settings.index, backend)
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {self._settings.index}")

        # Keep camera latency low by avoiding deep internal frame queues.
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        requested_fps = int(getattr(self._settings, "fps", 0) or 0)
        if requested_fps > 0 and hasattr(cv2, "CAP_PROP_FPS"):
            self._cap.set(cv2.CAP_PROP_FPS, requested_fps)

        if hasattr(cv2, "CAP_PROP_FOURCC"):
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
