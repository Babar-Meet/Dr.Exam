from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from proctor_app.core.models import ViolationType


def configure_preview_window(window_name: str, mode: str, overlay_size: Tuple[int, int]) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if mode == "fullscreen":
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        width, height = overlay_size
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.moveWindow(window_name, 0, 0)

    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def draw_device_boxes(
    frame: np.ndarray,
    boxes: Iterable[Tuple[int, int, int, int]],
    labels: Iterable[str],
) -> np.ndarray:
    output = frame.copy()

    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(output, (x1, y1), (x2, y2), (10, 10, 255), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (10, 10, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_status_overlay(
    frame: np.ndarray,
    mode: str,
    fps: float,
    face_count: int,
    total_violations: int,
    counts: Dict[ViolationType, int],
    active_messages: List[str],
    hint_message: str = "",
) -> np.ndarray:
    output = frame.copy()
    h, w = output.shape[:2]

    panel_h = min(124, h)
    panel = output.copy()
    cv2.rectangle(panel, (0, 0), (w, panel_h), (0, 0, 0), -1)
    output = cv2.addWeighted(panel, 0.45, output, 0.55, 0.0)

    info_line = f"Mode: {mode} | Faces: {face_count} | FPS: {fps:.1f}"
    cv2.putText(output, info_line, (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(
        output,
        f"Total Violations: {total_violations}",
        (12, 56),
        cv2.FONT_HERSHEY_DUPLEX,
        0.68,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    short_counts = (
        f"Away:{counts.get(ViolationType.LOOKING_AWAY, 0)}  "
        f"Multi:{counts.get(ViolationType.MULTIPLE_FACES, 0)}  "
        f"Switch:{counts.get(ViolationType.TAB_OR_APP_SWITCH, 0)}  "
        f"Device:{counts.get(ViolationType.UNAUTHORIZED_DEVICE, 0)}  "
        f"Left:{counts.get(ViolationType.CANDIDATE_LEFT_FRAME, 0)}"
    )
    cv2.putText(output, short_counts, (12, 84), cv2.FONT_HERSHEY_DUPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)

    if active_messages:
        y = min(h - 10, panel_h + 22)
        for message in active_messages[:3]:
            cv2.putText(output, f"VIOLATION: {message}", (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, (20, 20, 255), 2, cv2.LINE_AA)
            y += 24
    elif hint_message:
        cv2.putText(output, hint_message, (12, panel_h + 22), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(output, "Status: SAFE", (12, panel_h + 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 100), 2, cv2.LINE_AA)

    return output
