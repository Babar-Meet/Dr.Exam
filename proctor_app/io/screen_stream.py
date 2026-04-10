from __future__ import annotations

from typing import Optional

import cv2
import mss
import numpy as np


class ScreenStream:
    def __init__(self, monitor_index: int = 1) -> None:
        self._requested_monitor_index = monitor_index
        self._sct: Optional[mss.mss] = None
        self._monitor = None

    def open(self) -> None:
        self._sct = mss.mss()
        monitors = self._sct.monitors

        if len(monitors) <= 1:
            raise RuntimeError("No monitor found for screen capture")

        if 1 <= self._requested_monitor_index < len(monitors):
            self._monitor = monitors[self._requested_monitor_index]
        else:
            self._monitor = monitors[1]

    def capture(self) -> Optional[np.ndarray]:
        if not self._sct or not self._monitor:
            return None

        shot = self._sct.grab(self._monitor)
        frame = np.array(shot)

        if frame.size == 0:
            return None

        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return bgr

    def close(self) -> None:
        try:
            if self._sct:
                self._sct.close()
        finally:
            self._sct = None
            self._monitor = None
