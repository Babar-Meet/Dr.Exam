"""Detector modules for proctoring signals."""

from proctor_app.detectors.url_monitor import URLAssessment, URLMonitor
from proctor_app.detectors.screen_monitor import ScreenAssessment, ScreenActivityMonitor

__all__ = [
    "URLAssessment",
    "URLMonitor",
    "ScreenAssessment",
    "ScreenActivityMonitor",
]
