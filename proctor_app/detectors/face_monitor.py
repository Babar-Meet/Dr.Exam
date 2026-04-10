from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from proctor_app.config import FaceRules


@dataclass
class FaceAssessment:
    annotated_frame: np.ndarray
    face_count: int
    session_started: bool
    looking_away: bool
    looking_away_message: str
    multiple_faces: bool
    multiple_faces_message: str
    candidate_left_frame: bool
    left_frame_message: str


class FaceMonitor:
    _POSE_LANDMARK_IDS = (1, 152, 33, 263, 61, 291)
    _TASKS_DRAW_LANDMARK_IDS = (1, 10, 33, 61, 133, 152, 263, 291, 362, 468, 473)
    _POSE_MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )

    def __init__(self, rules: FaceRules) -> None:
        self._rules = rules

        self._backend: str = ""
        self._mp_face_mesh = None
        self._drawing = None
        self._drawing_styles = None
        self._mesh = None
        self._landmarker = None

        self._init_backend()
        self._last_tasks_timestamp_ms = 0

        self._session_started = False
        self._last_face_seen_at: Optional[float] = None
        self._away_started_at: Optional[float] = None
        self._multi_started_at: Optional[float] = None

        self._yaw_ema: Optional[float] = None
        self._pitch_ema: Optional[float] = None

    def _init_backend(self) -> None:
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_mesh"):
            self._mp_face_mesh = solutions.face_mesh
            self._drawing = solutions.drawing_utils
            self._drawing_styles = solutions.drawing_styles
            self._mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=self._rules.max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._backend = "solutions"
            return

        self._landmarker = self._create_tasks_landmarker()
        self._backend = "tasks"

    def _create_tasks_landmarker(self):
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:
            raise RuntimeError(
                "MediaPipe installation is missing both solutions and tasks Face Landmarker APIs."
            ) from exc

        model_path = self._resolve_tasks_model_path()
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=self._rules.max_faces,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            min_tracking_confidence=0.2,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _resolve_tasks_model_path() -> Path:
        root_dir = Path(__file__).resolve().parents[2]
        candidates = [
            root_dir / "face_landmarker.task",
            root_dir / "proctor_app" / "face_landmarker.task",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Unable to find Face Landmarker model. Searched: {searched}")

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def analyze(self, frame_bgr: np.ndarray, now: float) -> FaceAssessment:
        annotated = frame_bgr.copy()
        frame_h, frame_w = annotated.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            result = self._mesh.process(rgb)
            faces = result.multi_face_landmarks or []
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(now * 1000.0)
            if timestamp_ms <= self._last_tasks_timestamp_ms:
                timestamp_ms = self._last_tasks_timestamp_ms + 1
            self._last_tasks_timestamp_ms = timestamp_ms

            try:
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                # Keep running if tracker call fails on this frame.
                result = None
            faces = (result.face_landmarks or []) if result is not None else []

        faces_landmarks = [self._landmarks_for_face(face) for face in faces]
        face_boxes = [self._bbox_for_face(landmarks, frame_w, frame_h) for landmarks in faces_landmarks]

        for idx, (face, landmarks, box) in enumerate(zip(faces, faces_landmarks, face_boxes), start=1):
            if self._backend == "solutions":
                self._drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face,
                    connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._drawing_styles.get_default_face_mesh_contours_style(),
                )
            else:
                self._draw_tasks_landmarks(annotated, landmarks, frame_w, frame_h)

            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 120), 2)
            cv2.putText(
                annotated,
                f"Face {idx}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 120),
                2,
                cv2.LINE_AA,
            )

        face_count = len(faces)

        if face_count > 0:
            self._session_started = True
            self._last_face_seen_at = now

        multiple_faces, multiple_faces_message = self._check_multiple_faces(face_count, now)
        looking_away, looking_away_message = self._check_looking_away(faces, face_boxes, frame_w, frame_h, now)
        candidate_left_frame, left_frame_message = self._check_left_frame(face_count, now)

        if face_count == 0:
            cv2.putText(
                annotated,
                "FACE NOT DETECTED",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        return FaceAssessment(
            annotated_frame=annotated,
            face_count=face_count,
            session_started=self._session_started,
            looking_away=looking_away,
            looking_away_message=looking_away_message,
            multiple_faces=multiple_faces,
            multiple_faces_message=multiple_faces_message,
            candidate_left_frame=candidate_left_frame,
            left_frame_message=left_frame_message,
        )

    def _check_multiple_faces(self, face_count: int, now: float) -> Tuple[bool, str]:
        if face_count > 1:
            if self._multi_started_at is None:
                self._multi_started_at = now
            elapsed = now - self._multi_started_at
            if elapsed >= self._rules.multiple_faces_seconds:
                return True, f"Multiple faces detected ({face_count})"
            return False, "Multiple faces observed"

        self._multi_started_at = None
        return False, ""

    def _check_left_frame(self, face_count: int, now: float) -> Tuple[bool, str]:
        if not self._session_started:
            return False, ""

        if face_count > 0 or self._last_face_seen_at is None:
            return False, ""

        elapsed = now - self._last_face_seen_at
        if elapsed >= self._rules.left_frame_seconds:
            return True, f"Candidate left frame for {elapsed:.1f}s"

        return False, "Face temporarily missing"

    def _check_looking_away(
        self,
        faces: Sequence,
        face_boxes: Sequence[Tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
        now: float,
    ) -> Tuple[bool, str]:
        if not faces:
            self._away_started_at = None
            self._yaw_ema = None
            self._pitch_ema = None
            return False, ""

        primary_index = self._largest_face_index(face_boxes)
        primary_landmarks = self._landmarks_for_face(faces[primary_index])

        yaw, pitch = self._head_pose(primary_landmarks, frame_w, frame_h)
        gaze_ratio = self._gaze_ratio(primary_landmarks, frame_w, frame_h)

        # Stabilize pose to avoid triggering on brief micro-movements.
        self._yaw_ema = yaw if self._yaw_ema is None else (0.85 * self._yaw_ema + 0.15 * yaw)
        self._pitch_ema = pitch if self._pitch_ema is None else (0.85 * self._pitch_ema + 0.15 * pitch)

        yaw_threshold = self._rules.yaw_limit_deg + 3.0
        pitch_threshold = self._rules.pitch_limit_deg + 3.0
        gaze_low = max(0.0, self._rules.gaze_low - 0.03)
        gaze_high = min(1.0, self._rules.gaze_high + 0.03)

        reasons: List[str] = []

        if abs(self._yaw_ema) > yaw_threshold:
            side = "left" if self._yaw_ema > 0 else "right"
            reasons.append(f"Head turned {side} ({self._yaw_ema:.1f} deg)")

        if abs(self._pitch_ema) > pitch_threshold:
            vertical = "up" if self._pitch_ema < 0 else "down"
            reasons.append(f"Head tilted {vertical} ({self._pitch_ema:.1f} deg)")

        stable_for_gaze = abs(self._yaw_ema) <= (yaw_threshold * 0.85) and abs(self._pitch_ema) <= pitch_threshold
        if stable_for_gaze and (gaze_ratio < gaze_low or gaze_ratio > gaze_high):
            reasons.append(f"Eyes off screen ({gaze_ratio:.2f})")

        looking_away_now = len(reasons) > 0

        if looking_away_now:
            if self._away_started_at is None:
                self._away_started_at = now
            elapsed = now - self._away_started_at
            if elapsed >= self._rules.look_away_seconds:
                return True, "; ".join(reasons)
            return False, "Potential look-away"

        self._away_started_at = None
        return False, ""

    @staticmethod
    def _largest_face_index(face_boxes: Sequence[Tuple[int, int, int, int]]) -> int:
        areas = []
        for x1, y1, x2, y2 in face_boxes:
            areas.append(max(0, x2 - x1) * max(0, y2 - y1))
        return int(np.argmax(areas)) if areas else 0

    @staticmethod
    def _landmarks_for_face(face: Any) -> Sequence:
        if hasattr(face, "landmark"):
            return face.landmark
        return face

    def _draw_tasks_landmarks(self, frame: np.ndarray, landmarks: Sequence, frame_w: int, frame_h: int) -> None:
        for idx in self._TASKS_DRAW_LANDMARK_IDS:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            px = int(np.clip(lm.x * frame_w, 0, frame_w - 1))
            py = int(np.clip(lm.y * frame_h, 0, frame_h - 1))
            cv2.circle(frame, (px, py), 2, (0, 220, 120), -1)

    @staticmethod
    def _bbox_for_face(landmarks: Sequence, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        xs = [int(np.clip(lm.x * frame_w, 0, frame_w - 1)) for lm in landmarks]
        ys = [int(np.clip(lm.y * frame_h, 0, frame_h - 1)) for lm in landmarks]

        pad_x = int(0.03 * frame_w)
        pad_y = int(0.04 * frame_h)

        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(frame_w - 1, max(xs) + pad_x)
        y2 = min(frame_h - 1, max(ys) + pad_y)

        return x1, y1, x2, y2

    def _head_pose(self, landmarks: Sequence, frame_w: int, frame_h: int) -> Tuple[float, float]:
        image_points = []
        for idx in self._POSE_LANDMARK_IDS:
            lm = landmarks[idx]
            image_points.append((lm.x * frame_w, lm.y * frame_h))

        image_points_np = np.array(image_points, dtype=np.float64)

        focal = float(frame_w)
        cam_matrix = np.array(
            [
                [focal, 0.0, frame_w / 2.0],
                [0.0, focal, frame_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rotation_vec, translation_vec = cv2.solvePnP(
            self._POSE_MODEL_POINTS,
            image_points_np,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        projection = np.hstack((rotation_matrix, translation_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection)

        # OpenCV may return Euler angles as shape (3, 1); normalize to 1D first.
        euler_flat = np.asarray(euler, dtype=np.float64).reshape(-1)
        if euler_flat.size < 2:
            return 0.0, 0.0

        pitch = self._normalize_angle(float(euler_flat[0]))
        yaw = self._normalize_angle(float(euler_flat[1]))

        return yaw, pitch

    @staticmethod
    def _normalize_angle(value: float) -> float:
        normalized = (value + 180.0) % 360.0 - 180.0
        return normalized

    def _gaze_ratio(self, landmarks: Sequence, frame_w: int, frame_h: int) -> float:
        left_ratio = self._projected_ratio(landmarks, iris_idx=468, corner_a=33, corner_b=133, w=frame_w, h=frame_h)
        right_ratio = self._projected_ratio(landmarks, iris_idx=473, corner_a=362, corner_b=263, w=frame_w, h=frame_h)

        if left_ratio is None and right_ratio is None:
            return 0.5
        if left_ratio is None:
            return right_ratio
        if right_ratio is None:
            return left_ratio

        return float((left_ratio + right_ratio) / 2.0)

    @staticmethod
    def _projected_ratio(
        landmarks: Sequence,
        iris_idx: int,
        corner_a: int,
        corner_b: int,
        w: int,
        h: int,
    ) -> Optional[float]:
        if max(iris_idx, corner_a, corner_b) >= len(landmarks):
            return None

        iris = np.array([landmarks[iris_idx].x * w, landmarks[iris_idx].y * h], dtype=np.float64)
        a = np.array([landmarks[corner_a].x * w, landmarks[corner_a].y * h], dtype=np.float64)
        b = np.array([landmarks[corner_b].x * w, landmarks[corner_b].y * h], dtype=np.float64)

        eye_vec = b - a
        denom = float(np.dot(eye_vec, eye_vec))
        if denom < 1e-6:
            return None

        ratio = float(np.dot(iris - a, eye_vec) / denom)
        return float(np.clip(ratio, 0.0, 1.0))
