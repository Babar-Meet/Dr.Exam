from __future__ import annotations

from dataclasses import dataclass


BROWSER_PROCESSES = {"chrome", "msedge", "firefox", "brave", "opera"}


def is_browser_process(process_name: str) -> bool:
    return (process_name or "").strip().lower() in BROWSER_PROCESSES


def friendly_process_name(process_name: str) -> str:
    mapping = {
        "explorer": "File Explorer",
        "powershell": "PowerShell",
        "pwsh": "PowerShell",
        "cmd": "Command Prompt",
        "calculator": "Calculator",
        "calc": "Calculator",
        "chrome": "Chrome",
        "msedge": "Edge",
        "firefox": "Firefox",
        "brave": "Brave",
        "opera": "Opera",
    }
    key = (process_name or "").strip().lower()
    return mapping.get(key, process_name or "Unknown")


@dataclass(frozen=True)
class FocusState:
    kind: str
    value: str

    def canonical_key(self) -> str:
        return f"{self.kind}:{self.value.strip().lower()}"


class FocusStateResolver:
    """Converts process/title/url telemetry into a compact focus state."""

    def resolve(
        self,
        process_name: str,
        window_title: str,
        browser_foreground: bool,
        browser_url_readable: bool,
        active_url: str,
    ) -> FocusState:
        process_key = (process_name or "").strip().lower()
        title = (window_title or "").strip()
        url = (active_url or "").strip()

        if browser_foreground:
            if browser_url_readable and url:
                return FocusState(kind="web", value=url)

            browser_name = friendly_process_name(process_key)
            if title:
                return FocusState(kind="browser", value=f"{browser_name}: {title}")
            return FocusState(kind="browser", value=f"{browser_name} (URL unreadable)")

        if process_key:
            return FocusState(kind="app", value=friendly_process_name(process_key))

        if title:
            return FocusState(kind="window", value=title)

        return FocusState(kind="unknown", value="Unknown")
