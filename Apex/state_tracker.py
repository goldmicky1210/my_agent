"""Task state tracking with loop/stuck detection, state delta, and memory persistence."""
from __future__ import annotations

from models import ActionRecord, TaskState, Candidate
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
    # Memory / planning persistence
    # -----------------------------------------------------------------------

    @staticmethod
    def update_memory(task_id: str, memory: str, next_goal: str) -> None:
        state = StateTracker.get_or_create(task_id)
        if memory:
            state.memory = memory
        if next_goal:
            state.next_goal = next_goal

    @staticmethod
    def get_memory(task_id: str) -> tuple[str, str]:
        state = _TASK_STATES.get(task_id)
        if not state:
            return "", ""
        return state.memory, state.next_goal

    # -----------------------------------------------------------------------
    # State delta computation
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_state_delta(
        task_id: str,
        url: str,
        page_summary: str,
        candidates: list,
    ) -> str:
        state = StateTracker.get_or_create(task_id)

        prev_url = state.prev_url
        prev_summary = state.prev_summary
        prev_sig_set = set(state.prev_sig_set)

        # Build current signature set from candidates
        cur_sig_set: set[str] = set()
        for c in candidates[:30]:
            sel = getattr(c, 'selector', None)
            text = getattr(c, 'text', '') or ''
            if sel:
                sig = f"{sel.type}:{sel.attribute}:{sel.value}|{text[:80]}"
            else:
                sig = f"|{text[:80]}"
            cur_sig_set.add(sig)

        added = len(cur_sig_set - prev_sig_set) if prev_sig_set else len(cur_sig_set)
        removed = len(prev_sig_set - cur_sig_set) if prev_sig_set else 0
        unchanged = len(cur_sig_set & prev_sig_set) if prev_sig_set else 0

        ps = (prev_summary or "").strip()[:240]
        cs = (page_summary or "").strip()[:240]
        same_summary = bool(ps and cs and ps == cs)

        # Update state
        state.prev_url = url
        state.prev_summary = page_summary
        state.prev_sig_set = list(cur_sig_set)

        parts = [
            f"url_changed={str(prev_url != url).lower()}" if prev_url else "url_changed=unknown",
            f"summary_changed={str(not same_summary).lower()}" if (ps and cs) else "summary_changed=unknown",
            f"candidate_added={added}",
            f"candidate_removed={removed}",
            f"candidate_unchanged={unchanged}",
        ]
        return ", ".join(parts)

    @staticmethod
    def get_prev_sig_set(task_id: str) -> set[str] | None:
        state = _TASK_STATES.get(task_id)
        if not state or not state.prev_sig_set:
            return None
        return set(state.prev_sig_set)

    # -----------------------------------------------------------------------
    # Repeat/stuck detection
    # -----------------------------------------------------------------------

    @staticmethod
    def update_action_sig(task_id: str, url: str, sig: str) -> None:
        state = StateTracker.get_or_create(task_id)
        if sig and sig == state.last_sig and url == state.prev_url:
            state.repeat_count += 1
        else:
            state.repeat_count = 0
        state.last_sig = sig

    @staticmethod
    def get_repeat_count(task_id: str) -> int:
        state = _TASK_STATES.get(task_id)
        return state.repeat_count if state else 0

    # -----------------------------------------------------------------------
    # Loop detection: same (action_type, selector, url) repeated 2+ times
    # -----------------------------------------------------------------------

    @staticmethod
    def detect_loop(task_id: str, url: str) -> str | None:
        state = _TASK_STATES.get(task_id)
        if not state or len(state.history) < 2:
            return None
        recent = state.history[-1]
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
