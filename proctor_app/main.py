from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, List, Optional, Set
from urllib.parse import unquote, urlparse

import cv2

from proctor_app.config import AppConfig, CameraSettings, DeviceRules, URLRules, ViewSettings, build_default_config
from proctor_app.core.evidence_store import EvidenceStore
from proctor_app.core.focus_state import FocusStateResolver, friendly_process_name
from proctor_app.core.models import ViolationSignal, ViolationType
from proctor_app.core.state_transition_logger import StateTransitionLogger
from proctor_app.core.violation_manager import ViolationManager
from proctor_app.detectors.device_monitor import DeviceMonitor
from proctor_app.detectors.face_monitor import FaceMonitor
from proctor_app.detectors.screen_monitor import ScreenActivityMonitor, ScreenAssessment
from proctor_app.detectors.url_monitor import URLMonitor, URLAssessment
from proctor_app.io.camera_stream import CameraStream
from proctor_app.io.focus_context import FocusContext, collect_focus_context
from proctor_app.io.screen_stream import ScreenStream
from proctor_app.ui.renderer import configure_preview_window, draw_device_boxes, draw_status_overlay


def _infer_tab_name(window_title: str, active_url: str) -> str:
    title = (window_title or "").strip()
    title_lower = title.lower()
    url = (active_url or "").strip()
    url_lower = url.lower()
    parsed_url = urlparse(url) if url else None
    host = ((parsed_url.netloc if parsed_url else "") or "").strip().lower()
    path = ((parsed_url.path if parsed_url else "") or "").strip().lower()

    if "chatgpt" in title_lower or "chatgpt" in url_lower:
        return "ChatGPT"

    # Match actual Google search pages; avoid false positives from "- Google Chrome" suffix.
    if "google search" in title_lower or " - google search" in title_lower:
        return "Google Search"
    if host.startswith("www.google.") or host.startswith("google."):
        if path.startswith("/search") or "?q=" in url_lower or "&q=" in url_lower:
            return "Google Search"

    if (parsed_url and (parsed_url.scheme or "").lower() == "file") or url_lower.startswith("file://"):
        decoded_path = unquote((parsed_url.path if parsed_url else "") or "")
        normalized = decoded_path.replace("\\", "/")
        filename = normalized.rsplit("/", 1)[-1].strip()
        if filename:
            return filename

    if "google" in url_lower and ("/search" in url_lower or "?q=" in url_lower):
        return "Google Search"

    browser_suffixes = [
        " - Google Chrome",
        " - Chrome",
        " - Microsoft Edge",
        " - Brave",
        " - Mozilla Firefox",
        " - Opera",
    ]
    for suffix in browser_suffixes:
        if title.endswith(suffix):
            candidate = title[: -len(suffix)].strip()
            if candidate:
                return candidate

    if title:
        return title

    if url:
        try:
            host_fallback = (urlparse(url).netloc or "").strip()
            if host_fallback:
                return host_fallback
        except Exception:
            pass

    return "Browser Tab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exam Proctoring Monitor")
    parser.add_argument("--view-mode", choices=["overlay", "fullscreen"], default="overlay")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--overlay-width", type=int, default=420)
    parser.add_argument("--overlay-height", type=int, default=280)
    parser.add_argument("--disable-device-detector", action="store_true")
    parser.add_argument("--device-label", action="append", default=None, help="Unauthorized label. Repeatable")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> AppConfig:
    base = build_default_config(args.view_mode)

    camera = CameraSettings(
        index=args.camera_index,
        width=base.camera.width,
        height=base.camera.height,
        fps=base.camera.fps,
    )
    view = ViewSettings(
        mode=args.view_mode,
        overlay_size=(args.overlay_width, args.overlay_height),
        window_name=base.view.window_name,
    )

    labels: Set[str]
    if args.device_label:
        labels = {item.strip() for item in args.device_label if item.strip()}
    else:
        labels = set(base.device.unauthorized_labels)

    device = DeviceRules(
        enabled=not args.disable_device_detector,
        model_name=base.device.model_name,
        confidence=base.device.confidence,
        analyze_stride=base.device.analyze_stride,
        inference_size=base.device.inference_size,
        persistence_seconds=base.device.persistence_seconds,
        unauthorized_labels=labels,
        allowed_background_labels=set(base.device.allowed_background_labels),
    )

    return AppConfig(
        root_dir=base.root_dir,
        evidence_dir=base.evidence_dir,
        camera=camera,
        view=view,
        face=base.face,
        screen=base.screen,
        device=device,
        violation=base.violation,
        url=base.url,
    )


def run(config: AppConfig, monitor_index: int) -> None:
    camera = CameraStream(config.camera)
    screen = ScreenStream(monitor_index=monitor_index)

    face_monitor = FaceMonitor(config.face)
    screen_monitor = ScreenActivityMonitor(config.screen)
    device_monitor = DeviceMonitor(config.device)

    url_monitor = None
    if config.url.enabled:
        url_monitor = URLMonitor(
            allowed_domains=list(config.url.allowed_domains),
            allowed_ips=list(config.url.allowed_ips),
            allowed_paths=list(config.url.allowed_paths),
            check_interval=config.url.check_interval_seconds,
        )

    confirm_seconds = {
        ViolationType.LOOKING_AWAY: 0.0,
        ViolationType.MULTIPLE_FACES: 0.0,
        ViolationType.CANDIDATE_LEFT_FRAME: 0.0,
        ViolationType.TAB_OR_APP_SWITCH: 0.0,
        ViolationType.UNAUTHORIZED_DEVICE: 0.0,
    }
    violation_manager = ViolationManager(
        confirm_seconds=confirm_seconds,
        global_cooldown_seconds=config.violation.global_cooldown_seconds,
    )

    evidence = EvidenceStore(config.evidence_dir)
    focus_state_resolver = FocusStateResolver()
    state_transition_logger: Optional[StateTransitionLogger] = None
    try:
        state_transition_logger = StateTransitionLogger()
        print(f"[INFO] State transition log: {state_transition_logger.log_path}")
    except Exception as exc:
        print(f"[WARN] State transition logger disabled: {exc}")

    latest_screen_frame = None
    screen_assessment = ScreenAssessment()
    screen_enabled = True

    url_assessment = URLAssessment()
    url_enabled = config.url.enabled
    focus_sample_interval_seconds = 1.0
    last_focus_check_at = 0.0
    browser_foreground = False
    browser_url_readable = False
    active_url: str = ""
    foreground_process = ""
    foreground_title = ""
    focus_executor = ThreadPoolExecutor(max_workers=1)
    focus_future: Optional[Future[FocusContext]] = None
    app_switch_started_at = None
    browser_url_missing_started_at = None

    allowed_exam_processes = {
        "",
        "python",
        "pythonw",
        "examproctor",
        "powershell",
        "pwsh",
        "cmd",
        "conhost",
        "windowsterminal",
    }

    frame_timestamps: Deque[float] = deque(maxlen=60)
    last_screen_sample_at = 0.0

    try:
        camera.open()

        try:
            screen.open()
        except Exception as exc:
            screen_enabled = False
            print(f"[WARN] Screen capture disabled: {exc}")

        if config.view.mode == "overlay":
            ow, oh = config.view.overlay_size
            screen_monitor.set_exclusion_rect((0, 0, ow, oh))

        configure_preview_window(config.view.window_name, config.view.mode, config.view.overlay_size)

        print("Press Q or ESC to exit")

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[ERROR] Unable to read camera frame")
                break

            now = time.time()
            frame_timestamps.append(now)

            frame = cv2.flip(frame, 1)
            face_assessment = face_monitor.analyze(frame, now)
            display_frame = face_assessment.annotated_frame

            sample_interval = 1.0 / max(0.5, config.screen.capture_fps)
            if screen_enabled and (now - last_screen_sample_at) >= sample_interval:
                latest_screen_frame = screen.capture()
                screen_assessment = screen_monitor.analyze(latest_screen_frame, now)
                last_screen_sample_at = now

            device_assessment = device_monitor.analyze(frame, now)
            if device_assessment.boxes:
                display_frame = draw_device_boxes(display_frame, device_assessment.boxes, device_assessment.labels)

            if focus_future is not None and focus_future.done():
                try:
                    context = focus_future.result()
                except Exception:
                    context = FocusContext()

                foreground_process = context.foreground_process
                foreground_title = context.foreground_title
                browser_foreground = context.browser_foreground
                active_url = context.active_url
                browser_url_readable = context.browser_url_readable

                if url_enabled and url_monitor and browser_foreground and browser_url_readable:
                    url_assessment = url_monitor.analyze(active_url, now)
                else:
                    url_assessment = URLAssessment()

                if state_transition_logger is not None:
                    current_focus_state = focus_state_resolver.resolve(
                        process_name=foreground_process,
                        window_title=foreground_title,
                        browser_foreground=browser_foreground,
                        browser_url_readable=browser_url_readable,
                        active_url=active_url,
                    )
                    state_transition_logger.observe(current_focus_state, now)

                focus_future = None

            if focus_future is None and (now - last_focus_check_at) >= focus_sample_interval_seconds:
                focus_future = focus_executor.submit(collect_focus_context)
                last_focus_check_at = now

            if browser_foreground and not browser_url_readable:
                if browser_url_missing_started_at is None:
                    browser_url_missing_started_at = now
            else:
                browser_url_missing_started_at = None

            non_exam_app_active = (not browser_foreground) and (foreground_process not in allowed_exam_processes)
            if non_exam_app_active:
                if app_switch_started_at is None:
                    app_switch_started_at = now
            else:
                app_switch_started_at = None

            url_switch_active = url_enabled and browser_foreground and browser_url_readable and url_assessment.unauthorized_url
            app_focus_switch_active = app_switch_started_at is not None and (now - app_switch_started_at) >= 0.7
            browser_url_missing_active = (
                url_enabled
                and browser_foreground
                and browser_url_missing_started_at is not None
                and (now - browser_url_missing_started_at) >= 1.0
            )
            screen_switch_active = screen_enabled and (not browser_foreground) and screen_assessment.suspicious_switch

            if app_focus_switch_active:
                switch_message = (
                    f"Action: Opened app '{friendly_process_name(foreground_process)}' | "
                    "Result: Blocked (only browser allowed)"
                )
            elif url_switch_active:
                tab_name = _infer_tab_name(foreground_title, active_url)
                browser_name = friendly_process_name(foreground_process)
                switch_message = (
                    f"Action: Opened tab '{tab_name}' in {browser_name} | "
                    f"Result: Blocked URL ({active_url[:120]})"
                )
            elif browser_url_missing_active:
                tab_name = _infer_tab_name(foreground_title, active_url)
                browser_name = friendly_process_name(foreground_process)
                switch_message = (
                    f"Action: Opened tab '{tab_name}' in {browser_name} | "
                    "Result: Blocked (URL unreadable)"
                )
            elif screen_switch_active:
                switch_message = f"Action: Switched app/window | {screen_assessment.message}"
            else:
                switch_message = ""

            signals = {
                ViolationType.LOOKING_AWAY: ViolationSignal(
                    active=face_assessment.looking_away,
                    message=face_assessment.looking_away_message,
                ),
                ViolationType.MULTIPLE_FACES: ViolationSignal(
                    active=face_assessment.multiple_faces,
                    message=face_assessment.multiple_faces_message,
                ),
                ViolationType.CANDIDATE_LEFT_FRAME: ViolationSignal(
                    active=face_assessment.candidate_left_frame,
                    message=face_assessment.left_frame_message,
                ),
                ViolationType.TAB_OR_APP_SWITCH: ViolationSignal(
                    active=app_focus_switch_active or url_switch_active or browser_url_missing_active or screen_switch_active,
                    message=switch_message,
                ),
                ViolationType.UNAUTHORIZED_DEVICE: ViolationSignal(
                    active=device_assessment.unauthorized_detected,
                    message=device_assessment.message,
                ),
            }

            events = violation_manager.evaluate(signals, now)

            elapsed = frame_timestamps[-1] - frame_timestamps[0] if len(frame_timestamps) > 1 else 0.0
            fps = (len(frame_timestamps) - 1) / elapsed if elapsed > 1e-6 else 0.0

            hints: List[str] = []
            if not face_assessment.session_started:
                hints.append("Waiting for candidate")
            if screen_enabled and not screen_assessment.baseline_ready:
                hints.append("Calibrating screen baseline")
            if browser_foreground and url_enabled and not browser_url_readable:
                hints.append("Browser URL unreadable")
            if config.device.enabled and not device_assessment.model_ready:
                hints.append("Device detector unavailable")

            active_messages = violation_manager.active_messages(signals)
            hint_message = " | ".join(hints)

            # Render first so evidence camera snapshots match what the user sees live.
            display_frame = draw_status_overlay(
                frame=display_frame,
                mode=config.view.mode,
                fps=fps,
                face_count=face_assessment.face_count,
                total_violations=violation_manager.counters.total,
                counts=violation_manager.counters.by_type,
                active_messages=active_messages,
                hint_message=hint_message,
            )

            for event in events:
                event_screen_frame = latest_screen_frame

                # For app/tab switching, grab a fresh desktop frame right at event time.
                if screen_enabled and event.violation_type == ViolationType.TAB_OR_APP_SWITCH:
                    fresh_screen = screen.capture()
                    if fresh_screen is not None:
                        latest_screen_frame = fresh_screen
                        event_screen_frame = fresh_screen

                cam_path, screen_path = evidence.save_event(event, display_frame, event_screen_frame)
                print(
                    f"[VIOLATION] {event.violation_type.value} #{event.count} | "
                    f"cam={cam_path} | screen={screen_path}"
                )

            if config.view.mode == "overlay":
                display_frame = cv2.resize(
                    display_frame,
                    config.view.overlay_size,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(config.view.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            try:
                if cv2.getWindowProperty(config.view.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        focus_executor.shutdown(wait=False)
        if state_transition_logger is not None:
            state_transition_logger.close()
        evidence.close()
        face_monitor.close()
        camera.release()
        screen.close()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    run(config=config, monitor_index=args.monitor_index)


if __name__ == "__main__":
    main()
