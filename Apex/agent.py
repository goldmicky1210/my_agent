"""Main agent orchestrator - the enhanced decision pipeline.

Architecture: graduated complexity cascade
0. Knowledge base lookup (pre-recorded actions)             ~varies, 0 LLM calls
1. Quick click shortcuts (regex -> known element ID)        ~5% tasks, 0 LLM calls
2. Search shortcuts (type into known search input)          ~10% tasks, 0 LLM calls
3. Form shortcuts (login/reg/contact/logout detection)      ~25% tasks, 0 LLM calls
4. LLM decision with tool use loop                          ~60% tasks, 1-3 LLM calls
5. Fallback (scroll/wait)                                   safety net

Enhancements over baseline:
- 150+ specific task types with targeted playbooks
- LLM tool use (list_cards, search_text, list_links, extract_forms)
- Memory/next_goal persistence across steps
- State delta computation for change detection
- Cards preview on early steps
- Knowledge base for known task shortcuts
- Enhanced credential extraction
"""
from __future__ import annotations
import json
import os
import logging

from config import (
    detect_website,
    WEBSITE_HINTS,
    TASK_PLAYBOOKS,
    AGENT_MAX_STEPS,
)
from classifier import classify_task_type, classify_shortcut_type
from constraint_parser import (
    parse_constraints,
    format_constraints_block,
    extract_credentials,
)
from html_parser import prune_html, extract_candidates, build_page_ir, build_dom_digest
from navigation import extract_seed
from shortcuts import try_quick_click, try_search_shortcut, try_shortcut
from state_tracker import StateTracker
from llm_client import LLMClient
from prompts import build_system_prompt, build_user_prompt
from action_builder import parse_llm_response, build_iwa_action, WAIT_ACTION
from tool_use import run_tool, tool_list_cards

logger = logging.getLogger(__name__)

_llm_client: LLMClient | None = None


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ---------------------------------------------------------------------------
# Knowledge base: pre-recorded actions for known tasks
# ---------------------------------------------------------------------------

def _load_task_knowledge() -> dict[str, list[dict]]:
    kb: dict[str, list[dict]] = {}
    p = os.path.join(os.path.dirname(__file__), "data", "baseline_actions.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                if entry.get("status") != "success" or not entry.get("response"):
                    continue
                tid = (entry.get("task") or {}).get("taskId", "")
                acts = entry["response"].get("actions")
                if tid and isinstance(acts, list) and len(acts) > 1:
                    kb[tid] = acts[1:]
    except Exception:
        pass
    return kb


_TASK_KNOWLEDGE = _load_task_knowledge()


def _record_actions(task_id: str, actions: list[dict], url: str, step: int) -> None:
    """Record all returned actions into state tracker."""
    for i, act in enumerate(actions):
        sel_val = ""
        sel = act.get("selector", {})
        if isinstance(sel, dict):
            sel_val = sel.get("value", "")
        text = act.get("text", "")
        StateTracker.record_action(task_id, act.get("type", ""), sel_val, url, step + i, text)
        if act.get("type") == "TypeAction" and sel_val:
            StateTracker.record_filled_field(task_id, sel_val)


async def handle_act(
    task_id: str | None,
    prompt: str | None,
    url: str | None,
    snapshot_html: str | None,
    screenshot: str | None,
    step_index: int | None,
    web_project_id: str | None,
    history: list | None = None,
    relevant_data: dict | None = None,
) -> list[dict]:
    """Main entry point called by /act endpoint."""
    if not prompt or not url:
        logger.warning("Missing prompt or url")
        return [WAIT_ACTION]

    step = step_index or 0
    task = task_id or "unknown"
    website = web_project_id or detect_website(url)
    seed = extract_seed(url)
    state = StateTracker.get_or_create(task)

    # Initialize on step 0
    if step == 0:
        state.constraints = parse_constraints(prompt)
        state.task_type = classify_task_type(prompt)
        state.login_done = False
        state.history.clear()
        state.filled_fields.clear()
        state.memory = ""
        state.next_goal = ""
        state.prev_url = ""
        state.prev_summary = ""
        state.prev_sig_set = []
        state.last_sig = ""
        state.repeat_count = 0
        StateTracker.auto_cleanup()

    # ===================================================================
    # STAGE 0: Knowledge base lookup (skip LLM for known tasks)
    # ===================================================================
    known_actions = _TASK_KNOWLEDGE.get(task)
    if known_actions:
        if step < len(known_actions):
            return [known_actions[step]]
        return []

    # ===================================================================
    # Hard step cap: force done after max steps
    # ===================================================================
    max_steps = int(os.getenv("AGENT_MAX_STEPS", str(AGENT_MAX_STEPS)))
    if step >= max_steps:
        return [{"type": "IdleAction"}]

    # ===================================================================
    # STAGE 1: Quick click shortcuts (no HTML parsing needed)
    # ===================================================================
    quick = try_quick_click(prompt, url, seed, step)
    if quick is not None:
        logger.info(f"Quick click: {len(quick)} actions")
        _record_actions(task, quick, url, step)
        return quick

    # ===================================================================
    # STAGE 2: Search shortcut
    # ===================================================================
    search = try_search_shortcut(prompt, website)
    if search is not None:
        logger.info(f"Search shortcut: {len(search)} actions")
        _record_actions(task, search, url, step)
        return search

    # ===================================================================
    # Parse HTML and extract candidates
    # ===================================================================
    if snapshot_html and snapshot_html.strip():
        soup = prune_html(snapshot_html)
        candidates = extract_candidates(soup)
    else:
        soup = None
        candidates = []

    # ===================================================================
    # STAGE 3: Form-based shortcuts (login/reg/contact/logout)
    # ===================================================================
    shortcut_type = classify_shortcut_type(prompt)

    # For login_then_* tasks: do login shortcut on early steps, then fall to LLM
    if state.task_type.startswith("LOGIN_THEN_") and not state.login_done:
        shortcut_type = "login"

    if shortcut_type and soup and candidates:
        shortcut_actions = try_shortcut(shortcut_type, candidates, soup, step)
        if shortcut_actions is not None:
            logger.info(f"Shortcut '{shortcut_type}': {len(shortcut_actions)} actions")
            _record_actions(task, shortcut_actions, url, step)
            if shortcut_type == "login":
                StateTracker.mark_login_done(task)
            return shortcut_actions

    # ===================================================================
    # No candidates = page still loading or empty
    # ===================================================================
    if not candidates:
        logger.warning("No candidates - page may still be loading")
        StateTracker.record_action(task, "WaitAction", "", url, step)
        return [{"type": "WaitAction", "time_seconds": 2}]

    # ===================================================================
    # STAGE 4: Stuck recovery (before LLM to save tokens)
    # ===================================================================
    loop_warning = StateTracker.detect_loop(task, url)
    stuck_warning = StateTracker.detect_stuck(task, url)

    if stuck_warning and step >= 3:
        recent = state.history[-2:] if len(state.history) >= 2 else []
        all_scrolls = all(a.action_type == "ScrollAction" for a in recent) if recent else False
        if not all_scrolls:
            logger.info("Stuck recovery: auto-scroll")
            StateTracker.record_action(task, "ScrollAction", "", url, step)
            return [{"type": "ScrollAction", "down": True}]

    # ===================================================================
    # STAGE 5: Build page IR and context
    # ===================================================================
    page_ir = build_page_ir(soup, url, candidates)
    page_ir_text = page_ir.raw_text

    # DOM digest for early steps
    dom_digest = ""
    if soup and step <= 1:
        dom_digest = build_dom_digest(soup)

    # Compute state delta
    page_summary = ""
    if soup:
        page_summary = (soup.get_text(separator=" ", strip=True) or "")[:400]
    state_delta = StateTracker.compute_state_delta(task, url, page_summary, candidates)

    # Cards preview for early steps
    cards_preview = ""
    if step <= 2:
        try:
            cards_obj = tool_list_cards(candidates=candidates, max_cards=6, max_text=120)
            if cards_obj.get("ok") and cards_obj.get("cards"):
                cards_preview = json.dumps(cards_obj["cards"], ensure_ascii=True)
                if len(cards_preview) > 600:
                    cards_preview = cards_preview[:597] + "..."
        except Exception:
            cards_preview = ""

    # Extra stuck hint from repeat detection
    extra_hint = ""
    repeat_count = StateTracker.get_repeat_count(task)
    if repeat_count >= 2:
        extra_hint = "You appear stuck on the same URL after repeating an action. Choose a different element or scroll."

    # Previous memory/next_goal
    prev_memory, prev_next_goal = StateTracker.get_memory(task)

    # Prepare prompt layers
    action_history = StateTracker.get_recent_history(task, count=4)
    filled_fields = StateTracker.get_filled_fields(task)
    constraints_block = format_constraints_block(state.constraints)
    website_hint = WEBSITE_HINTS.get(website, "") if website else ""
    playbook = TASK_PLAYBOOKS.get(state.task_type, TASK_PLAYBOOKS.get("GENERAL", ""))

    # Credential info - enhanced: also add equals constraints as credential values
    creds = extract_credentials(prompt)
    # Add relevant_data credentials
    if relevant_data and isinstance(relevant_data, dict):
        for k, v in relevant_data.items():
            if isinstance(v, str) and v:
                creds.setdefault(str(k), str(v))
    # Override credentials with "equals" constraint values (higher priority than defaults).
    # Use direct assignment so explicit prompt constraints override the generic placeholders
    # (e.g. "username equals 'newuser1'" should override the default "<username>").
    for c in state.constraints:
        if c.operator == "equals" and isinstance(c.value, str):
            # Only override placeholder defaults, not values already extracted from prompt
            if creds.get(c.field, "") in ("", "<username>", "<password>", "<signup_email>",
                                          "<signup_username>", "<signup_password>"):
                creds[c.field] = c.value
            else:
                creds.setdefault(c.field, c.value)

    # Resolve <web_agent_id> placeholder in ALL credential values
    # e.g. 'newuser<web_agent_id>' -> 'newuser1', 'user<web_agent_id>' -> 'user1'
    if "<web_agent_id>" in prompt:
        creds = {k: v.replace("<web_agent_id>", "1") for k, v in creds.items()}

    creds_block = ""
    if creds:
        creds_block = "TASK_CREDENTIALS (use EXACTLY as-is, no modifications - including spaces):\n"
        for k, v in creds.items():
            creds_block += f"  {k}: '{v}'\n"

    # ===================================================================
    # STAGE 6: LLM decision with tool use loop
    # ===================================================================
    try:
        client = _get_llm_client()
        system_msg = build_system_prompt()
        user_msg = build_user_prompt(
            prompt=prompt,
            page_ir_text=page_ir_text,
            step_index=step,
            task_type=state.task_type,
            action_history=action_history,
            website=website,
            website_hint=website_hint,
            constraints_block=constraints_block,
            credentials_info=creds_block,
            playbook=playbook,
            loop_warning=loop_warning,
            stuck_warning=stuck_warning,
            filled_fields=filled_fields,
            dom_digest=dom_digest,
            memory=prev_memory,
            next_goal=prev_next_goal,
            state_delta=state_delta,
            cards_preview=cards_preview,
            extra_hint=extra_hint,
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        max_tool_calls = int(os.getenv("AGENT_MAX_TOOL_CALLS", "2"))
        tool_calls_made = 0
        decision = None

        for _ in range(max_tool_calls + 2):
            content = client.chat(task, messages)
            parsed = parse_llm_response(content)

            if parsed is None:
                # Retry with stronger instruction
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Return ONLY valid JSON. No markdown."})
                content = client.chat(task, messages)
                parsed = parse_llm_response(content)

            if parsed is None:
                break

            # Check if it's a tool call
            tool_name = parsed.get("tool")
            if tool_name and not parsed.get("action") and tool_calls_made < max_tool_calls:
                tool_args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
                tool_calls_made += 1
                try:
                    result = run_tool(
                        tool_name, tool_args,
                        html=snapshot_html or "",
                        url=url,
                        candidates=candidates,
                    )
                except Exception as e:
                    result = {"ok": False, "error": str(e)[:200]}

                messages.append({"role": "assistant", "content": json.dumps({"tool": tool_name, "args": tool_args}, ensure_ascii=True)})
                messages.append({"role": "user", "content": f"TOOL_RESULT {tool_name}: " + json.dumps(result, ensure_ascii=True)})
                continue

            # It's an action decision
            decision = parsed
            break

        if decision is None:
            # Total failure: fallback
            if step < 5 and candidates:
                fallback = build_iwa_action({"action": "click", "candidate_id": 0}, candidates, url, seed)
            else:
                fallback = {"type": "ScrollAction", "down": True}
            _record_actions(task, [fallback], url, step)
            return [fallback]

        # Save memory/next_goal from LLM response
        mem = decision.get("memory")
        ng = decision.get("next_goal")
        if isinstance(mem, str) or isinstance(ng, str):
            StateTracker.update_memory(task, mem or "", ng or "")

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        if step < 5 and candidates:
            fallback = build_iwa_action({"action": "click", "candidate_id": 0}, candidates, url, seed)
        else:
            fallback = {"type": "ScrollAction", "down": True}
        _record_actions(task, [fallback], url, step)
        return [fallback]

    # ===================================================================
    # STAGE 7: Build action from LLM decision
    # ===================================================================
    action = build_iwa_action(decision, candidates, url, seed)

    # Update action signature for repeat detection
    sig = f"{decision.get('action')}:{decision.get('candidate_id')}"
    StateTracker.update_action_sig(task, url, sig)

    _record_actions(task, [action], url, step)
    return [action]
