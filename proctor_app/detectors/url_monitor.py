from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class URLAssessment:
    unauthorized_url: bool = False
    message: str = ""


class URLMonitor:
    """Monitors active window URL to detect tab/app switching to unauthorized sites."""

    def __init__(
        self,
        allowed_domains: list[str] = None,
        allowed_ips: list[str] = None,
        allowed_paths: list[str] = None,
        check_interval: float = 1.0,
    ) -> None:
        self._allowed_domains = [item.lower().strip() for item in (allowed_domains or ["darshanums"]) if item and item.strip()]
        self._allowed_ips = [item.lower().strip() for item in (allowed_ips or ["10.255.1.1:8090"]) if item and item.strip()]
        self._allowed_paths = [item.lower().strip().lstrip("/") for item in (allowed_paths or []) if item and item.strip()]
        self._check_interval = check_interval
        self._last_check: Optional[float] = None
        self._last_assessment = URLAssessment()

    @property
    def last_assessment(self) -> URLAssessment:
        return self._last_assessment

    def _is_allowed(self, url: str) -> bool:
        if not url:
            return False

        url_lower = url.lower().strip()

        for token in self._allowed_domains:
            if token and token in url_lower:
                return True

        for token in self._allowed_ips:
            if token and token in url_lower:
                return True

        for token in self._allowed_paths:
            if token and token in url_lower:
                return True

        return False

    def analyze(self, active_url: Optional[str], now: float) -> URLAssessment:
        if active_url is None:
            self._last_assessment = URLAssessment(
                unauthorized_url=False,
                message="No active browser detected",
            )
            return self._last_assessment

        if self._is_allowed(active_url):
            self._last_assessment = URLAssessment(
                unauthorized_url=False,
                message="",
            )
        else:
            self._last_assessment = URLAssessment(
                unauthorized_url=True,
                message=f"Unauthorized URL: {active_url[:80]}",
            )

        return self._last_assessment