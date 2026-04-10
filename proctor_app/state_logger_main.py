from __future__ import annotations

import argparse
import time

from proctor_app.core.focus_state import FocusStateResolver
from proctor_app.core.state_transition_logger import StateTransitionLogger
from proctor_app.io.focus_context import collect_focus_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone state transition logger")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval for active state collection (default: 1.0)",
    )
    return parser.parse_args()


def run(interval_seconds: float) -> None:
    interval_seconds = max(0.2, float(interval_seconds))

    resolver = FocusStateResolver()
    logger = StateTransitionLogger()
    print(f"[INFO] State transition log: {logger.log_path}")
    print("[INFO] Monitoring started. Press Ctrl+C to stop.")

    try:
        while True:
            loop_started_at = time.time()
            context = collect_focus_context()
            state = resolver.resolve(
                process_name=context.foreground_process,
                window_title=context.foreground_title,
                browser_foreground=context.browser_foreground,
                browser_url_readable=context.browser_url_readable,
                active_url=context.active_url,
            )
            changed = logger.observe(state, loop_started_at)
            if changed:
                print(f"[STATE] {state.value}")

            elapsed = time.time() - loop_started_at
            sleep_for = max(0.0, interval_seconds - elapsed)
            if sleep_for > 0.0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped.")
    finally:
        logger.close()


def main() -> None:
    args = parse_args()
    run(interval_seconds=args.interval_seconds)


if __name__ == "__main__":
    main()
