from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Tuple


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    width: int = 960
    height: int = 540
    fps: int = 30


@dataclass(frozen=True)
class ViewSettings:
    mode: str = "overlay"  # overlay | fullscreen
    overlay_size: Tuple[int, int] = (420, 280)
    window_name: str = "Exam Proctor Preview"


@dataclass(frozen=True)
class FaceRules:
    max_faces: int = 5
    look_away_seconds: float = 1.6
    multiple_faces_seconds: float = 1.0
    left_frame_seconds: float = 2.5
    yaw_limit_deg: float = 30.0
    pitch_limit_deg: float = 22.0
    gaze_low: float = 0.24
    gaze_high: float = 0.76


@dataclass(frozen=True)
class ScreenRules:
    warmup_seconds: float = 3.0
    persistent_switch_seconds: float = 1.8
    capture_fps: float = 3.0
    baseline_hamming_threshold: int = 17
    rapid_hamming_threshold: int = 22
    diff_ratio_threshold: float = 0.23
    rapid_switch_window_seconds: float = 10.0
    rapid_switch_count: int = 4


@dataclass(frozen=True)
class DeviceRules:
    enabled: bool = True
    model_name: str = "yolov8n.pt"
    confidence: float = 0.30
    analyze_stride: int = 3
    inference_size: int = 416
    persistence_seconds: float = 0.6
    unauthorized_labels: Set[str] = field(
        default_factory=lambda: {
            "cell phone",
            "mobile",
            "phone",
            "smartphone",
            "laptop",
            "tablet",
        }
    )
    allowed_background_labels: Set[str] = field(default_factory=lambda: {"tv", "monitor"})


@dataclass(frozen=True)
class ViolationRules:
    global_cooldown_seconds: float = 2.0


@dataclass(frozen=True)
class URLRules:
    enabled: bool = True
    allowed_domains: Tuple[str, ...] = ("darshanums",)
    allowed_ips: Tuple[str, ...] = ("10.255.1.1:8090",)
    allowed_paths: Tuple[str, ...] = ()
    check_interval_seconds: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    evidence_dir: Path
    camera: CameraSettings = CameraSettings()
    view: ViewSettings = ViewSettings()
    face: FaceRules = FaceRules()
    screen: ScreenRules = ScreenRules()
    device: DeviceRules = DeviceRules()
    violation: ViolationRules = ViolationRules()
    url: URLRules = URLRules()


def build_default_config(view_mode: str = "overlay") -> AppConfig:
    root_dir = Path(__file__).resolve().parents[1]
    evidence_dir = root_dir / "evidence"

    mode = view_mode.lower().strip()
    if mode not in {"overlay", "fullscreen"}:
        mode = "overlay"

    view = ViewSettings(mode=mode)

    return AppConfig(root_dir=root_dir, evidence_dir=evidence_dir, view=view)
