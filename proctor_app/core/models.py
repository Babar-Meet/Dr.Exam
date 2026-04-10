from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class ViolationType(str, Enum):
    LOOKING_AWAY = "looking_away"
    MULTIPLE_FACES = "multiple_faces"
    TAB_OR_APP_SWITCH = "tab_or_app_switch"
    UNAUTHORIZED_DEVICE = "unauthorized_device"
    CANDIDATE_LEFT_FRAME = "candidate_left_frame"


@dataclass(frozen=True)
class ViolationSignal:
    active: bool
    message: str


@dataclass(frozen=True)
class ViolationEvent:
    violation_type: ViolationType
    message: str
    timestamp: float
    count: int


@dataclass
class ViolationCounters:
    total: int = 0
    by_type: Dict[ViolationType, int] = field(default_factory=dict)

    def increment(self, violation_type: ViolationType) -> int:
        self.total += 1
        new_value = self.by_type.get(violation_type, 0) + 1
        self.by_type[violation_type] = new_value
        return new_value

    def get(self, violation_type: ViolationType) -> int:
        return self.by_type.get(violation_type, 0)


@dataclass
class ActiveState:
    active_since: Optional[float] = None
    confirmed_active: bool = False
    last_trigger_at: float = 0.0
    last_message: str = ""
