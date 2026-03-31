"""Multi-layer prompt construction with tool support and state tracking.

Designed to handle real evaluation tasks requiring:
- Complex constraint matching (find item where X AND Y AND Z)
- Multi-step workflows (login → navigate → find → interact)
- Form filling with exact constraint values
- Page exploration via tools
"""
from __future__ import annotations


def build_system_prompt() -> str:
    return """You are an expert web automation agent. You analyze pages and choose precise actions.

RESPONSE: Return ONLY a JSON object. No markdown, no explanation, no code fences.

ACTIONS:
  {"action": "click", "candidate_id": N}
  {"action": "type", "candidate_id": N, "text": "value"}
  {"action": "select_option", "candidate_id": N, "text": "option text"}
  {"action": "navigate", "url": "http://localhost:PORT/path?seed=X"}
  {"action": "scroll", "direction": "down"}
  {"action": "scroll", "direction": "up"}
  {"action": "done"}

TOOLS (request by returning):
  {"action": "tool", "tool": "list_cards"}
  {"action": "tool", "tool": "visible_text"}
  {"action": "tool", "tool": "search_text", "query": "search string"}

CONSTRAINT RULES:
- EQUALS 'X': type/select EXACTLY 'X'. Find item where field is exactly X.
- NOT EQUALS 'X': choose any value OTHER than X. Find item where field is NOT X.
- CONTAINS 'X': value must include substring X. Find item containing X.
- NOT CONTAINS 'X': value must NOT include X. Avoid items containing X.
- GREATER THAN N: numeric value > N.
- LESS THAN N: numeric value < N.
- For "not" constraints when filling forms: pick a valid alternative value.
- When finding items: scroll through all items checking EVERY constraint.

CRITICAL RULES:
1. candidate_id MUST match [N] from Interactive elements.
2. ALWAYS preserve ?seed= parameter in navigate URLs.
3. Use <username>/<password> for login credentials unless specific values given.
4. DO NOT repeat failed actions. Try different elements or scroll.
5. If task needs login first, login then continue with the actual task.
6. When looking for items with constraints, READ the context text of each candidate.
7. For multi-constraint item selection: ALL constraints must be satisfied.

Return JSON only."""


def build_user_prompt(
    *,
    prompt: str,
    page_ir_text: str,
    step_index: int,
    task_type: str,
    action_history: list[str],
    website: str | None,
    website_hint: str = "",
    constraints_block: str = "",
    credentials_info: str = "",
    playbook: str = "",
    loop_warning: str | None = None,
    stuck_warning: str | None = None,
    filled_fields: set[str] | None = None,
    dom_digest: str = "",
    tool_result: str = "",
    memory: str = "",
    next_goal: str = "",
) -> str:
    parts: list[str] = []

    # --- Core task ---
    parts.append(f"TASK: {prompt}")

    site_line = f"SITE: {website or 'unknown'}"
    if website_hint:
        site_line += f" — {website_hint}"
    parts.append(site_line)

    parts.append(f"TYPE: {task_type}  |  STEP: {step_index} of 12")

    remaining = max(1, 12 - step_index)
    if remaining <= 3:
        parts.append(f"!! ONLY {remaining} STEPS LEFT — take the most direct action NOW.")

    # --- Constraints ---
    if constraints_block:
        parts.append(f"\n{constraints_block}")

    # --- Credentials ---
    if credentials_info:
        parts.append(f"\nCREDENTIALS: {credentials_info}")

    # --- Playbook ---
    if playbook:
        parts.append(f"\nPLAYBOOK: {playbook}")

    # --- Memory from previous steps ---
    if memory:
        parts.append(f"\nMEMORY: {memory}")
    if next_goal:
        parts.append(f"\nNEXT GOAL: {next_goal}")

    # --- Warnings ---
    if loop_warning:
        parts.append(f"\n!! {loop_warning}")
    if stuck_warning:
        parts.append(f"\n!! {stuck_warning}")

    # --- History ---
    if action_history:
        history_text = "\n".join(f"  - {h}" for h in action_history)
    else:
        history_text = "  None yet"
    parts.append(f"\nHISTORY:\n{history_text}")

    # --- Filled fields ---
    if filled_fields:
        parts.append(f"\nALREADY FILLED: {', '.join(sorted(filled_fields))}")

    # --- Tool result ---
    if tool_result:
        parts.append(f"\nTOOL RESULT:\n{tool_result[:2000]}")

    # --- DOM digest (early steps) ---
    if dom_digest and step_index <= 1:
        parts.append(f"\nPAGE STRUCTURE:\n{dom_digest}")

    # --- Page IR ---
    parts.append(f"\nPAGE ELEMENTS:\n{page_ir_text}")

    # --- Final instruction ---
    parts.append("\nChoose ONE action. Return JSON only. Include 'memory' and 'next_goal' fields to track progress.")

    return "\n".join(parts)
