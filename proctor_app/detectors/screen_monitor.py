from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from proctor_app.config import ScreenRules


@dataclass
class ScreenAssessment:
    suspicious_switch: bool = False
    message: str = ""
    baseline_ready: bool = False
    baseline_distance: int = 0
    transition_distance: int = 0
    diff_ratio: float = 0.0


class ScreenActivityMonitor:
    """Detects likely tab/app switching by persistent visual scene changes."""

    def __init__(self, rules: ScreenRules) -> None:
        self._rules = rules

        self._calibration_started_at: Optional[float] = None
        self._hash_samples: List[np.ndarray] = []

        self._baseline_hash: Optional[np.ndarray] = None
        self._prev_hash: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None

        self._sustained_change_started_at: Optional[float] = None
        self._rapid_spikes: Deque[float] = deque()

        self._exclusion_rect: Optional[Tuple[int, int, int, int]] = None
        self._last_assessment = ScreenAssessment()

    @property
    def last_assessment(self) -> ScreenAssessment:
        return self._last_assessment

    def set_exclusion_rect(self, rect: Optional[Tuple[int, int, int, int]]) -> None:
        self._exclusion_rect = rect

    def analyze(self, frame_bgr: Optional[np.ndarray], now: float) -> ScreenAssessment:
        if frame_bgr is None:
            return self._last_assessment

        prepared = self._prepare_frame(frame_bgr)
        current_hash = self._phash(prepared)

        if self._calibration_started_at is None:
            self._calibration_started_at = now

        if self._baseline_hash is None:
            self._hash_samples.append(current_hash)
            self._prev_hash = current_hash
            self._prev_gray = prepared

            elapsed = now - self._calibration_started_at
            if elapsed < self._rules.warmup_seconds:
                self._last_assessment = ScreenAssessment(
                    suspicious_switch=False,
                    message="Screen monitor calibrating",
                    baseline_ready=False,
                )
                return self._last_assessment

            self._baseline_hash = self._majority_hash(self._hash_samples)
            self._hash_samples.clear()
            self._last_assessment = ScreenAssessment(
                suspicious_switch=False,
                message="Screen baseline ready",
                baseline_ready=True,
            )
            return self._last_assessment

        baseline_distance = self._hamming(current_hash, self._baseline_hash)
        transition_distance = self._hamming(current_hash, self._prev_hash) if self._prev_hash is not None else 0
        diff_ratio = self._diff_ratio(prepared, self._prev_gray)

        major_change = (
            baseline_distance >= self._rules.baseline_hamming_threshold
            and diff_ratio >= self._rules.diff_ratio_threshold
        )

        if major_change:
            if self._sustained_change_started_at is None:
                self._sustained_change_started_at = now
        else:
            self._sustained_change_started_at = None

        sustained_violation = (
            self._sustained_change_started_at is not None
            and (now - self._sustained_change_started_at) >= self._rules.persistent_switch_seconds
        )

        if (
            transition_distance >= self._rules.rapid_hamming_threshold
            and diff_ratio >= (self._rules.diff_ratio_threshold * 0.8)
        ):
            self._rapid_spikes.append(now)

        while self._rapid_spikes and now - self._rapid_spikes[0] > self._rules.rapid_switch_window_seconds:
            self._rapid_spikes.popleft()

        rapid_violation = len(self._rapid_spikes) >= self._rules.rapid_switch_count
        suspicious = sustained_violation or rapid_violation

        if suspicious:
            if sustained_violation:
                msg = "Screen changed away from baseline (possible tab/app switch)"
            else:
                msg = "Rapid screen switching pattern detected"
        else:
            msg = ""

        self._prev_hash = current_hash
        self._prev_gray = prepared

        self._last_assessment = ScreenAssessment(
            suspicious_switch=suspicious,
            message=msg,
            baseline_ready=True,
            baseline_distance=baseline_distance,
            transition_distance=transition_distance,
            diff_ratio=diff_ratio,
        )

        return self._last_assessment

    def _prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = frame_bgr.copy()
        frame = self._apply_exclusion(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (256, 144), interpolation=cv2.INTER_AREA)
        return small

    def _apply_exclusion(self, frame: np.ndarray) -> np.ndarray:
        if not self._exclusion_rect:
            return frame

        x, y, w, h = self._exclusion_rect
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = 0

        return frame

    @staticmethod
    def _phash(gray: np.ndarray) -> np.ndarray:
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        low = dct[:8, :8]
        median = float(np.median(low[1:, :]))
        return (low > median).astype(np.uint8).flatten()

    @staticmethod
    def _majority_hash(samples: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(samples, axis=0)
        return (np.mean(stacked, axis=0) >= 0.5).astype(np.uint8)

    @staticmethod
    def _hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.count_nonzero(a != b))

    @staticmethod
    def _diff_ratio(curr: np.ndarray, prev: Optional[np.ndarray]) -> float:
        if prev is None or prev.shape != curr.shape:
            return 0.0

        abs_diff = cv2.absdiff(curr, prev)
        changed = np.count_nonzero(abs_diff > 25)
        total = curr.shape[0] * curr.shape[1]
        if total == 0:
            return 0.0

        return float(changed / total)
