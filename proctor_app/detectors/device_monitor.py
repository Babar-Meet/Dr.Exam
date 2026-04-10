 
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from proctor_app.config import DeviceRules


@dataclass
class DeviceAssessment:
    unauthorized_detected: bool = False
    labels: List[str] = field(default_factory=list)
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    message: str = ""
    model_ready: bool = False


class DeviceMonitor:
    """Detects phone or unauthorized objects using YOLO."""

    def __init__(self, rules: DeviceRules) -> None:
        self._rules = rules
        self._frame_index = 0
        self._active_since: Optional[float] = None

        self._unauthorized_tokens = self._normalize_tokens(rules.unauthorized_labels)
        self._allowed_background_tokens = self._normalize_tokens(rules.allowed_background_labels)

        self._model = None
        self._model_ready = False
        self._last_assessment = DeviceAssessment(model_ready=False)

        if not rules.enabled:
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(rules.model_name)
            self._model_ready = True
            self._last_assessment = DeviceAssessment(model_ready=True)
        except Exception:
            self._model = None
            self._model_ready = False
            self._last_assessment = DeviceAssessment(model_ready=False)

    @staticmethod
    def _normalize_label_token(token: str) -> str:
        normalized = (token or "").lower().strip().replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    @classmethod
    def _normalize_tokens(cls, labels: Iterable[str]) -> Set[str]:
        tokens: Set[str] = set()
        for label in labels:
            normalized = cls._normalize_label_token(label)
            if normalized:
                tokens.add(normalized)
        return tokens

    @staticmethod
    def _matches_tokens(label_norm: str, tokens: Set[str]) -> bool:
        if not label_norm or not tokens:
            return False

        words = set(label_norm.split())
        for token in tokens:
            if token == label_norm:
                return True
            if token in words:
                return True
            if len(token) >= 5 and token in label_norm:
                return True
            if len(label_norm) >= 5 and label_norm in token:
                return True
        return False

    @property
    def model_ready(self) -> bool:
        return self._model_ready

    def analyze(self, frame_bgr: np.ndarray, now: float) -> DeviceAssessment:
        if not self._rules.enabled:
            return DeviceAssessment(model_ready=False)

        if not self._model_ready or self._model is None:
            return self._last_assessment

        self._frame_index += 1

        if self._frame_index % max(1, self._rules.analyze_stride) != 0:
            return self._last_assessment

        detections = self._predict(frame_bgr)
        if detections:
            if self._active_since is None:
                self._active_since = now
            active = (now - self._active_since) >= self._rules.persistence_seconds
        else:
            self._active_since = None
            active = False

        labels = [label for _, label in detections]
        boxes = [box for box, _ in detections]

        msg = ""
        if active and labels:
            unique_labels = sorted(set(labels))
            msg = f"Unauthorized device detected: {', '.join(unique_labels)}"

        self._last_assessment = DeviceAssessment(
            unauthorized_detected=active,
            labels=labels,
            boxes=boxes,
            message=msg,
            model_ready=True,
        )
        return self._last_assessment

    def _predict(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str]]:
        if self._model is None:
            return []

        inference_size = max(160, int(getattr(self._rules, "inference_size", 640)))
        results = self._model.predict(
            source=frame_bgr,
            conf=self._rules.confidence,
            imgsz=inference_size,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        boxes = result.boxes

        if boxes is None:
            return []

        matches: List[Tuple[Tuple[int, int, int, int], str]] = []

        for item in boxes:
            cls_id = int(item.cls.item())
            label = str(names.get(cls_id, ""))
            label_norm = self._normalize_label_token(label)

            if self._matches_tokens(label_norm, self._allowed_background_tokens):
                continue

            if not self._matches_tokens(label_norm, self._unauthorized_tokens):
                continue

            xyxy = item.xyxy[0].cpu().numpy().astype(np.int32).tolist()
            x1, y1, x2, y2 = xyxy
            matches.append(((x1, y1, x2, y2), label))

        return matches
