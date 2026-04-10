from __future__ import annotations

from typing import Dict, Iterable, List

from proctor_app.core.models import ActiveState, ViolationCounters, ViolationEvent, ViolationSignal, ViolationType


class ViolationManager:
    """Converts detector signals into debounced violation events."""

    def __init__(
        self,
        confirm_seconds: Dict[ViolationType, float],
        global_cooldown_seconds: float,
    ) -> None:
        self._confirm_seconds = dict(confirm_seconds)
        self._global_cooldown_seconds = global_cooldown_seconds
        self._states = {violation_type: ActiveState() for violation_type in ViolationType}
        self._counters = ViolationCounters()

    @property
    def counters(self) -> ViolationCounters:
        return self._counters

    def evaluate(
        self,
        signals: Dict[ViolationType, ViolationSignal],
        now: float,
    ) -> List[ViolationEvent]:
        events: List[ViolationEvent] = []

        for violation_type in ViolationType:
            signal = signals.get(violation_type, ViolationSignal(active=False, message=""))
            state = self._states[violation_type]
            signal_message = (signal.message or "").strip()

            if signal.active:
                if state.active_since is None:
                    state.active_since = now

                required_seconds = self._confirm_seconds.get(violation_type, 0.0)
                active_elapsed = now - state.active_since
                cooldown_elapsed = now - state.last_trigger_at
                message_changed = bool(signal_message) and signal_message != state.last_message
                bypass_cooldown = violation_type == ViolationType.TAB_OR_APP_SWITCH and message_changed

                if (
                    active_elapsed >= required_seconds
                    and (not state.confirmed_active or message_changed)
                    and (cooldown_elapsed >= self._global_cooldown_seconds or bypass_cooldown)
                ):
                    count_for_type = self._counters.increment(violation_type)
                    events.append(
                        ViolationEvent(
                            violation_type=violation_type,
                            message=signal.message,
                            timestamp=now,
                            count=count_for_type,
                        )
                    )
                    state.confirmed_active = True
                    state.last_trigger_at = now
                    state.last_message = signal_message
            else:
                state.active_since = None
                state.confirmed_active = False
                state.last_message = ""

        return events

    def active_messages(
        self,
        signals: Dict[ViolationType, ViolationSignal],
    ) -> List[str]:
        messages: List[str] = []
        for violation_type in ViolationType:
            signal = signals.get(violation_type)
            if signal and signal.active:
                messages.append(signal.message)
        return messages
