from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from proctor_app.core.focus_state import FocusState


class StateTransitionLogger:
    """Appends focus state transitions as: current_state -> next_state + time."""

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self._log_path = log_path or (self._resolve_downloads_dir() / "proctor_state_transitions.log")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = self._log_path.open("a", encoding="utf-8")
        if self._log_path.stat().st_size == 0:
            self._file.write("timestamp | current_state ----> next_state\n")
            self._file.flush()

        self._last_state: Optional[FocusState] = None
        self._last_key = ""

    @property
    def log_path(self) -> Path:
        return self._log_path

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

    def observe(self, state: FocusState, at: float) -> bool:
        key = state.canonical_key()

        if self._last_state is None:
            self._last_state = state
            self._last_key = key
            return False

        if key == self._last_key:
            return False

        stamp = datetime.fromtimestamp(at).isoformat(timespec="seconds")
        line = f"{stamp} | {self._last_state.value} ----> {state.value}\n"
        self._file.write(line)
        self._file.flush()

        self._last_state = state
        self._last_key = key
        return True

    @staticmethod
    def _resolve_downloads_dir() -> Path:
        home = Path.home()
        default_downloads = home / "Downloads"
        if default_downloads.exists():
            return default_downloads

        profile = os.environ.get("USERPROFILE", "").strip()
        if profile:
            return Path(profile) / "Downloads"

        return default_downloads
