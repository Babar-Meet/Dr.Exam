from __future__ import annotations

from dataclasses import dataclass

from proctor_app.core.focus_state import is_browser_process
from proctor_app.io.browser_url import get_active_chrome_context, get_foreground_process_name, get_window_title


@dataclass(frozen=True)
class FocusContext:
    foreground_process: str = ""
    foreground_title: str = ""
    browser_foreground: bool = False
    active_url: str = ""
    browser_url_readable: bool = False


def _correct_process_by_window_title(process_name: str, window_title: str) -> str:
    title_lower = (window_title or "").lower()
    name = (process_name or "").strip().lower()

    if " - google chrome" in title_lower:
        return "chrome"
    if " - microsoft edge" in title_lower:
        return "msedge"
    if " - mozilla firefox" in title_lower:
        return "firefox"
    if " - brave" in title_lower:
        return "brave"
    if " - opera" in title_lower:
        return "opera"

    return name


def collect_focus_context() -> FocusContext:
    foreground_process = (get_foreground_process_name() or "").strip().lower()
    foreground_title = (get_window_title() or "").strip()
    foreground_process = _correct_process_by_window_title(foreground_process, foreground_title)
    browser_foreground = is_browser_process(foreground_process)

    active_url = ""
    browser_url_readable = False

    # URL capture is Chrome-specific in the current implementation.
    if browser_foreground and foreground_process == "chrome":
        _, active_url = get_active_chrome_context()
        browser_url_readable = bool(active_url)

    return FocusContext(
        foreground_process=foreground_process,
        foreground_title=foreground_title,
        browser_foreground=browser_foreground,
        active_url=active_url,
        browser_url_readable=browser_url_readable,
    )
