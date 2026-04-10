from __future__ import annotations

import ctypes
import json
import os
import subprocess
from typing import Optional
from ctypes import wintypes
from urllib.error import URLError
from urllib.request import urlopen


_USER32 = ctypes.WinDLL("user32", use_last_error=True)
_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _get_foreground_window() -> int:
    hwnd = _USER32.GetForegroundWindow()
    return int(hwnd) if hwnd else 0


def _get_foreground_pid() -> Optional[int]:
    hwnd = _get_foreground_window()
    if not hwnd:
        return None

    pid = wintypes.DWORD(0)
    _USER32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value) if pid.value else None


def _get_process_name_from_pid(pid: int) -> Optional[str]:
    handle = _KERNEL32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None

    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        ok = _KERNEL32.QueryFullProcessImageNameW(
            wintypes.HANDLE(handle),
            wintypes.DWORD(0),
            buffer,
            ctypes.byref(size),
        )
        if not ok:
            return None

        exe_name = os.path.basename(buffer.value).strip().lower()
        if exe_name.endswith(".exe"):
            exe_name = exe_name[:-4]
        return exe_name or None
    finally:
        _KERNEL32.CloseHandle(wintypes.HANDLE(handle))


def _get_foreground_window_title_native() -> Optional[str]:
    hwnd = _get_foreground_window()
    if not hwnd:
        return None

    length = int(_USER32.GetWindowTextLengthW(wintypes.HWND(hwnd)))
    if length <= 0:
        return None

    buffer = ctypes.create_unicode_buffer(length + 1)
    copied = int(_USER32.GetWindowTextW(wintypes.HWND(hwnd), buffer, length + 1))
    if copied <= 0:
        return None

    title = buffer.value.strip()
    return title or None


def _run_powershell(script: str, timeout: int = 3) -> str:
    """Runs a PowerShell snippet and returns stdout (empty on failure)."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except Exception:
        return ""


def get_foreground_process_name() -> Optional[str]:
    """Returns lowercase process name for the current foreground window."""
    pid = _get_foreground_pid()
    if not pid:
        return None

    return _get_process_name_from_pid(pid)


def is_chrome_foreground() -> bool:
    """True when the currently focused app is Google Chrome."""
    return get_foreground_process_name() == "chrome"


def get_active_chrome_context() -> tuple[bool, Optional[str]]:
    """Returns whether Chrome is focused and the best-effort active Chrome URL."""
    if not is_chrome_foreground():
        return False, None

    url = _get_chrome_url_from_uia()
    if url:
        return True, url

    return True, _get_chrome_url_from_devtools()


def get_active_chrome_url() -> Optional[str]:
    """Returns active Chrome tab URL when Chrome is foreground."""
    _, url = get_active_chrome_context()
    return url


def get_active_browser_url() -> Optional[str]:
    """Backwards-compatible alias for current Chrome-only URL capture."""
    return get_active_chrome_url()


def _normalize_http_url(value: str) -> Optional[str]:
    candidate = (value or "").strip()
    if candidate.lower().startswith(("http://", "https://", "file://")):
        return candidate
    return None


def _get_chrome_url_from_uia() -> Optional[str]:
    """Reads the Chrome address-bar value via Windows UI Automation."""
    script = r'''
Add-Type -AssemblyName UIAutomationClient
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
}
"@

$hwnd = [Win32]::GetForegroundWindow()
$pid = 0
[Win32]::GetWindowThreadProcessId($hwnd, [ref]$pid) | Out-Null
$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
if (-not $proc -or $proc.ProcessName.ToLowerInvariant() -ne "chrome") {
    return
}

$root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
if ($null -eq $root) {
    return
}

$condition = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
    [System.Windows.Automation.ControlType]::Edit
)
$edits = $root.FindAll([System.Windows.Automation.TreeScope]::Subtree, $condition)

for ($i = 0; $i -lt $edits.Count; $i++) {
    try {
        $edit = $edits.Item($i)
        $patternObj = $null
        if ($edit.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$patternObj)) {
            $value = $patternObj.Current.Value
            if ($value -and $value -match '^(https?://|file://)') {
                $value
                break
            }
        }
    }
    catch {}
}
'''
    return _normalize_http_url(_run_powershell(script, timeout=3))


def _get_chrome_url_from_devtools() -> Optional[str]:
    """Reads Chrome tabs from local DevTools endpoint when enabled."""
    for port in (9222, 9223, 9229):
        endpoint = f"http://127.0.0.1:{port}/json/list"
        try:
            with urlopen(endpoint, timeout=0.35) as response:
                payload = response.read().decode("utf-8", errors="ignore")
            tabs = json.loads(payload)
        except (OSError, URLError, ValueError, json.JSONDecodeError):
            continue

        if not isinstance(tabs, list):
            continue

        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            if tab.get("type") not in ("page", None, ""):
                continue
            url = _normalize_http_url(str(tab.get("url", "")))
            if url:
                return url

    return None


def _get_chromium_url() -> Optional[str]:
    """Legacy helper kept for compatibility with existing imports."""
    return get_active_chrome_url()


def _get_firefox_url() -> Optional[str]:
    """Firefox URL capture is intentionally disabled in Chrome-only mode."""
    return None


def get_window_title() -> Optional[str]:
    """Get the title of the active window."""
    return _get_foreground_window_title_native()