"""Task state tracking with loop/stuck detection and auto-recovery."""
from __future__ import annotations

from models import ActionRecord, TaskState
from config import MAX_TASK_STATES

_TASK_STATES: dict[str, TaskState] = {}


class StateTracker:
    """Manages per-task state across steps."""

    @staticmethod
    def get_or_create(task_id: str) -> TaskState:
        if task_id not in _TASK_STATES:
            _TASK_STATES[task_id] = TaskState(task_id=task_id)
        return _TASK_STATES[task_id]

    @staticmethod
    def record_action(
        task_id: str,
        action_type: str,
        selector_value: str | None,
        url: str,
        step_index: int,
        text: str = "",
    ) -> None:
        state = StateTracker.get_or_create(task_id)
        state.history.append(
            ActionRecord(
                action_type=action_type,
                selector_value=selector_value or "",
                url=url,
                step_index=step_index,
                text=text,
            )
        )

    @staticmethod
    def record_filled_field(task_id: str, field_name: str) -> None:
        state = StateTracker.get_or_create(task_id)
        state.filled_fields.add(field_name)

    @staticmethod
    def get_filled_fields(task_id: str) -> set[str]:
        state = _TASK_STATES.get(task_id)
        return state.filled_fields if state else set()

    @staticmethod
    def mark_login_done(task_id: str) -> None:
        state = StateTracker.get_or_create(task_id)
        state.login_done = True

    @staticmethod
    def is_login_done(task_id: str) -> bool:
        state = _TASK_STATES.get(task_id)
        return state.login_done if state else False

    # -----------------------------------------------------------------------
    # Loop detection: same (action_type, selector, url) repeated 2+ times
    # -----------------------------------------------------------------------

    @staticmethod
    def detect_loop(task_id: str, url: str) -> str | None:
        state = _TASK_STATES.get(task_id)
        if not state or len(state.history) < 2:
            return None
        recent = state.history[-1]
        # Scrolls are expected to repeat
        if recent.action_type in ("ScrollAction", "WaitAction"):
            return None
        count = sum(
            1
            for h in state.history
            if h.action_type == recent.action_type
            and h.selector_value == recent.selector_value
            and h.url == url
        )
        if count >= 2:
            return (
                f"LOOP DETECTED: You've done '{recent.action_type}' on "
                f"'{recent.selector_value}' at this URL {count} times. "
                f"Try a DIFFERENT action or element."
            )
        return None

    # -----------------------------------------------------------------------
    # Stuck detection: no meaningful progress for 3+ steps
    # -----------------------------------------------------------------------

    @staticmethod
    def detect_stuck(task_id: str, url: str) -> str | None:
        state = _TASK_STATES.get(task_id)
        if not state or len(state.history) < 3:
            return None
        last_3 = state.history[-3:]
        urls = {h.url for h in last_3}
        selectors = {h.selector_value for h in last_3}
        if len(urls) == 1 and len(selectors) <= 2:
            return (
                "STUCK: No progress for 3+ steps. Try: "
                "1) Scroll down to find new elements, "
                "2) Click a different navigation link, "
                "3) Look for an alternative path."
            )
        return None

    # -----------------------------------------------------------------------
    # History formatting
    # -----------------------------------------------------------------------

    @staticmethod
    def get_recent_history(task_id: str, count: int = 4) -> list[str]:
        state = _TASK_STATES.get(task_id)
        if not state:
            return []
        recent = state.history[-count:]
        lines = []
        for r in recent:
            line = f"Step {r.step_index}: {r.action_type}"
            if r.selector_value:
                line += f" on '{r.selector_value}'"
            if r.text:
                line += f" text='{r.text[:30]}'"
            line += f" at {r.url}"
            lines.append(line)
        return lines

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    @staticmethod
    def auto_cleanup(max_kept: int = MAX_TASK_STATES) -> None:
        while len(_TASK_STATES) > max_kept:
            oldest_key = next(iter(_TASK_STATES))
            del _TASK_STATES[oldest_key]

    @staticmethod
    def cleanup(task_id: str) -> None:
        _TASK_STATES.pop(task_id, None)
