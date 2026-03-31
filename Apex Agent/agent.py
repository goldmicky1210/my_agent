"""Main agent orchestrator - the decision pipeline.

Architecture: graduated complexity cascade
1. Quick click shortcuts (regex → known element ID)       ~10% tasks, 0 LLM calls
2. Search shortcuts (type into known search input)        ~10% tasks, 0 LLM calls
3. Form shortcuts (login/reg/contact/logout detection)    ~20% tasks, 0 LLM calls
4. LLM decision with tool support                         ~55% tasks, 1-3 LLM calls
5. Fallback (scroll/click/wait)                           safety net

Key features:
- Tool support (list_cards, visible_text, search_text) for page exploration
- Memory/next_goal tracking across steps for multi-step planning
- State delta for LLM context on what changed
- Login-then-action compound task support
- Adaptive stuck recovery
"""
from __future__ import annotations
import logging
import re

from bs4 import BeautifulSoup

from config import (
    detect_website,
    WEBSITE_HINTS,
    TASK_PLAYBOOKS,
)
from classifier import classify_task_type, classify_shortcut_type
from constraint_parser import (
    parse_constraints,
    format_constraints_block,
    extract_credentials,
)
from html_parser import prune_html, extract_candidates, build_page_ir, build_dom_digest
from navigation import extract_seed
from shortcuts import try_quick_click, try_search_shortcut, try_shortcut, is_already_logged_in
from state_tracker import StateTracker
from llm_client import LLMClient
from prompts import build_system_prompt, build_user_prompt
from action_builder import parse_llm_response, build_iwa_action, WAIT_ACTION

logger = logging.getLogger(__name__)

_llm_client: LLMClient | None = None

# Per-task memory for multi-step planning
_TASK_MEMORY: dict[str, dict] = {}
MAX_TOOL_CALLS = 2


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


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


# ---------------------------------------------------------------------------
# Tool execution (page exploration tools like Vector agent)
# ---------------------------------------------------------------------------

def _execute_tool(tool_name: str, query: str, soup: BeautifulSoup | None, candidates: list) -> str:
    """Execute a page-exploration tool and return result text."""
    if not soup:
        return "No page content available."

    if tool_name == "list_cards":
        # Group candidates by their context (surrounding container text)
        cards: dict[str, list] = {}
        for c in candidates:
            key = c.context[:80] if c.context else f"element_{c.index}"
            if key not in cards:
                cards[key] = []
            cards[key].append(f"[{c.index}] {c.tag} '{c.text}'")
        result_parts = []
        for ctx, items in list(cards.items())[:10]:
            result_parts.append(f"CARD: {ctx}")
            for item in items[:3]:
                result_parts.append(f"  {item}")
        return "\n".join(result_parts)[:1500] if result_parts else "No card groups found."

    if tool_name == "visible_text":
        text = soup.get_text(separator="\n", strip=True)
        return text[:2000]

    if tool_name == "search_text":
        if not query:
            return "No query provided."
        text = soup.get_text(separator="\n", strip=True)
        lines = text.split("\n")
        matches = [line.strip() for line in lines if query.lower() in line.lower()]
        return "\n".join(matches[:20]) if matches else f"No matches for '{query}'."

    return f"Unknown tool: {tool_name}"


async def handle_act(
    task_id: str | None,
    prompt: str | None,
    url: str | None,
    snapshot_html: str | None,
    screenshot: str | None,
    step_index: int | None,
    web_project_id: str | None,
    history: list | None = None,
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
        _TASK_MEMORY.pop(task, None)
        StateTracker.auto_cleanup()

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

    # For login_then_action: do login shortcut on early steps, then fall to LLM
    if state.task_type == "login_then_action" and not state.login_done:
        shortcut_type = "login"
    # Logout tasks may need login first
    if state.task_type == "logout" and not state.login_done and soup and not is_already_logged_in(soup):
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

    dom_digest = ""
    if soup and step <= 1:
        dom_digest = build_dom_digest(soup)

    # Prompt layers
    action_history = StateTracker.get_recent_history(task, count=4)
    filled_fields = StateTracker.get_filled_fields(task)
    constraints_block = format_constraints_block(state.constraints)
    website_hint = WEBSITE_HINTS.get(website, "") if website else ""
    playbook = TASK_PLAYBOOKS.get(state.task_type, TASK_PLAYBOOKS.get("general", ""))

    # Credentials
    creds = extract_credentials(prompt)
    cred_parts = []
    if creds.get("username"):
        cred_parts.append(f"username={creds['username']}")
    if creds.get("password"):
        cred_parts.append(f"password={creds['password']}")
    credentials_info = ", ".join(cred_parts) if cred_parts else ""

    # Memory from previous steps
    mem = _TASK_MEMORY.get(task, {})
    memory = mem.get("memory", "")
    next_goal = mem.get("next_goal", "")

    # ===================================================================
    # STAGE 6: LLM decision with tool loop
    # ===================================================================
    try:
        client = _get_llm_client()
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(
            prompt=prompt,
            page_ir_text=page_ir_text,
            step_index=step,
            task_type=state.task_type,
            action_history=action_history,
            website=website,
            website_hint=website_hint,
            constraints_block=constraints_block,
            credentials_info=credentials_info,
            playbook=playbook,
            loop_warning=loop_warning,
            stuck_warning=stuck_warning,
            filled_fields=filled_fields,
            dom_digest=dom_digest,
            memory=memory,
            next_goal=next_goal,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Tool loop: allow LLM to request tools before deciding on action
        tool_calls = 0
        decision = None

        for _iteration in range(MAX_TOOL_CALLS + 2):
            llm_response = client.chat(task_id=task, messages=messages)
            decision = parse_llm_response(llm_response)

            if decision is None:
                # Bad JSON - retry once
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": "Return ONLY valid JSON. No markdown. Example: {\"action\": \"click\", \"candidate_id\": 0}",
                })
                continue

            # Check if LLM requested a tool
            if decision.get("action") == "tool" and tool_calls < MAX_TOOL_CALLS:
                tool_name = decision.get("tool", "")
                tool_query = decision.get("query", "")
                logger.info(f"Tool request: {tool_name} query={tool_query[:50]}")
                tool_result = _execute_tool(tool_name, tool_query, soup, candidates)
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": f"TOOL RESULT ({tool_name}):\n{tool_result}\n\nNow choose an ACTION (not a tool). Return JSON.",
                })
                tool_calls += 1
                decision = None
                continue

            # Valid action decision
            break

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return [WAIT_ACTION]

    # ===================================================================
    # STAGE 7: Handle result
    # ===================================================================
    if decision is None:
        logger.warning("All parse attempts failed")
        if step <= 3 and candidates:
            fallback = {"type": "ClickAction", "selector": candidates[0].selector.model_dump()}
        else:
            fallback = {"type": "ScrollAction", "down": True}
        _record_actions(task, [fallback], url, step)
        return [fallback]

    # Save memory/next_goal for future steps
    if decision.get("memory") or decision.get("next_goal"):
        _TASK_MEMORY[task] = {
            "memory": str(decision.get("memory", ""))[:200],
            "next_goal": str(decision.get("next_goal", ""))[:200],
        }

    # Build action
    action = build_iwa_action(decision, page_ir.candidates, url, seed)
    action_type = action.get("type", "unknown")

    # Block NavigateAction on step 0 (already on correct page)
    if step == 0 and action_type == "NavigateAction":
        logger.info("Blocked NavigateAction on step 0 -> scroll instead")
        action = {"type": "ScrollAction", "down": True}
        action_type = "ScrollAction"

    # Done signal
    if action_type == "IdleAction":
        logger.info("Task marked done by LLM")
        _record_actions(task, [action], url, step)
        return []

    logger.info(f"LLM action: {action_type}")
    _record_actions(task, [action], url, step)
    return [action]
