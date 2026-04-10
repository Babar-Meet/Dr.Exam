from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from proctor_app.core.models import ViolationEvent


class EvidenceStore:
    """Stores camera/screen snapshots and a CSV audit trail."""

    def __init__(self, evidence_dir: Path) -> None:
        self._base = evidence_dir
        self._camera_dir = self._base / "camera"
        self._screen_dir = self._base / "screen"
        self._log_path = self._base / "violations.csv"

        self._camera_dir.mkdir(parents=True, exist_ok=True)
        self._screen_dir.mkdir(parents=True, exist_ok=True)
        self._base.mkdir(parents=True, exist_ok=True)

        self._log_file = self._log_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._log_file)

        if self._log_path.stat().st_size == 0:
            self._writer.writerow(
                [
                    "timestamp",
                    "violation_type",
                    "count_for_type",
                    "message",
                    "camera_image",
                    "screen_image",
                ]
            )
            self._log_file.flush()

    def close(self) -> None:
        try:
            self._log_file.close()
        except Exception:
            pass

    def save_event(
        self,
        event: ViolationEvent,
        camera_frame: Optional[np.ndarray],
        screen_frame: Optional[np.ndarray],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        timestamp = datetime.fromtimestamp(event.timestamp)
        stamp = timestamp.strftime("%H%M%S%d%m%Y")
        count_ordinal = self._to_ordinal(event.count)
        context_tag = self._extract_context_tag(event.message)

        name_parts = [self._sanitize_filename_part(event.violation_type.value)]
        if context_tag:
            name_parts.append(context_tag)
        name_parts.extend([count_ordinal, stamp])
        base_name = "_".join(part for part in name_parts if part)

        camera_path = self._save_image(self._camera_dir / f"{base_name}_cam.jpg", camera_frame)
        screen_path = self._save_image(self._screen_dir / f"{base_name}_screen.jpg", screen_frame)

        self._writer.writerow(
            [
                timestamp.isoformat(timespec="milliseconds"),
                event.violation_type.value,
                event.count,
                event.message,
                str(camera_path) if camera_path else "",
                str(screen_path) if screen_path else "",
            ]
        )
        self._log_file.flush()

        return camera_path, screen_path

    @staticmethod
    def _to_ordinal(value: int) -> str:
        if 10 <= (value % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    @classmethod
    def _extract_context_tag(cls, message: str) -> str:
        msg = (message or "").strip()
        if not msg:
            return ""

        app_match = re.search(r"Opened app '([^']+)'", msg, flags=re.IGNORECASE)
        if app_match:
            return "app-" + cls._sanitize_filename_part(app_match.group(1), max_len=48)

        url_match = re.search(r"Blocked URL \(([^)]+)\)", msg, flags=re.IGNORECASE)
        if url_match:
            return "url-" + cls._summarize_url(url_match.group(1))

        tab_match = re.search(r"Opened tab '([^']+)'", msg, flags=re.IGNORECASE)
        if tab_match:
            return "tab-" + cls._sanitize_filename_part(tab_match.group(1), max_len=48)

        device_match = re.search(r"Unauthorized device detected:\s*(.+)$", msg, flags=re.IGNORECASE)
        if device_match:
            return "object-" + cls._sanitize_filename_part(device_match.group(1), max_len=48)

        first_chunk = msg.split("|")[0].split(";")[0]
        return cls._sanitize_filename_part(first_chunk, max_len=48)

    @classmethod
    def _summarize_url(cls, raw_url: str) -> str:
        url = (raw_url or "").strip()
        try:
            parsed = urlparse(url)
        except Exception:
            return cls._sanitize_filename_part(url, max_len=48)

        if parsed.scheme in {"http", "https"}:
            host = parsed.netloc
            path_parts = [part for part in unquote(parsed.path).split("/") if part]
            first_path = path_parts[0] if path_parts else ""
            summary = host if not first_path else f"{host}-{first_path}"
            return cls._sanitize_filename_part(summary, max_len=48)

        if parsed.scheme == "file":
            path_parts = [part for part in unquote(parsed.path).split("/") if part]
            tail = "-".join(path_parts[-2:]) if path_parts else "file"
            return cls._sanitize_filename_part(f"file-{tail}", max_len=48)

        return cls._sanitize_filename_part(url, max_len=48)

    @staticmethod
    def _sanitize_filename_part(value: str, max_len: int = 64) -> str:
        text = (value or "").strip().lower()
        text = text.replace("_", "-")
        text = re.sub(r"[^a-z0-9.-]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-.")
        if not text:
            return ""
        return text[:max_len]

    @staticmethod
    def _save_image(path: Path, frame: Optional[np.ndarray]) -> Optional[Path]:
        if frame is None:
            return None

        ok = cv2.imwrite(str(path), frame)
        return path if ok else None
