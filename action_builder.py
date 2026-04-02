"""Convert LLM decisions to IWA action dicts.

Handles:
- JSON parsing with multiple fallback strategies
- Candidate validation
- Credential inference
- URL seed preservation
- Same-page navigation → scroll conversion
"""
from __future__ import annotations
import json
import re
import logging
from urllib.parse import urlsplit

from models import Candidate
from navigation import preserve_seed, is_localhost_url

logger = logging.getLogger(__name__)

WAIT_ACTION = {"type": "WaitAction", "time_seconds": 1}


# ---------------------------------------------------------------------------
# JSON parsing (tolerant: raw → fenced → brace extraction)
# ---------------------------------------------------------------------------

def parse_llm_response(content: str) -> dict | None:
    text = content.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown fenced JSON
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Brace extraction
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------------
# Credential inference for type actions
# ---------------------------------------------------------------------------

def _infer_credentials(text: str, candidate: Candidate) -> str:
    """Fill in credential placeholders when LLM leaves text empty."""
    if text:
        return text
    if candidate.input_type == "password":
        return "<password>"
    if candidate.name in {"username", "user", "login"}:
        return "<username>"
    if candidate.input_type == "email" or candidate.name == "email":
        return "<username>"
    return text


# ---------------------------------------------------------------------------
# Build IWA action from LLM decision
# ---------------------------------------------------------------------------

def build_iwa_action(
    decision: dict,
    candidates: list[Candidate],
    current_url: str,
    seed: str | None,
) -> dict:
    action = decision.get("action", "wait")

    # --- Click / Type / Select ---
    if action in ("click", "type", "select_option"):
        cid = decision.get("candidate_id")
        if cid is None or not isinstance(cid, int):
            logger.warning(f"Missing/invalid candidate_id for {action}")
            return WAIT_ACTION
        if cid < 0 or cid >= len(candidates):
            logger.warning(f"candidate_id {cid} out of range (0-{len(candidates) - 1})")
            return WAIT_ACTION

        candidate = candidates[cid]
        sel = candidate.selector.model_dump()

        if action == "click":
            return {"type": "ClickAction", "selector": sel}

        if action == "type":
            text = decision.get("text", decision.get("value", ""))
            text = _infer_credentials(text, candidate)
            return {"type": "TypeAction", "text": text, "selector": sel}

        if action == "select_option":
            text = decision.get("text", "")
            # If no text given but candidate has options, pick first
            if not text and candidate.options:
                text = candidate.options[0]
            return {"type": "SelectDropDownOptionAction", "text": text, "selector": sel}

    # --- Navigate ---
    if action == "navigate":
        url = decision.get("url", "")
        if not url:
            logger.warning("Navigate action missing URL")
            return WAIT_ACTION
        if not is_localhost_url(url):
            logger.warning(f"Blocked non-localhost navigate: {url}")
            return WAIT_ACTION

        final_url = preserve_seed(url, current_url)

        # Same-page check → scroll instead
        cur = urlsplit(current_url)
        fin = urlsplit(final_url)
        if cur.path == fin.path and cur.query == fin.query:
            logger.info("Same-URL navigation → ScrollAction")
            return {"type": "ScrollAction", "down": True}

        return {"type": "NavigateAction", "url": final_url}

    # --- Scroll ---
    if action == "scroll":
        direction = decision.get("direction", "down")
        if direction == "up":
            return {"type": "ScrollAction", "up": True}
        return {"type": "ScrollAction", "down": True}

    # --- Done ---
    if action == "done":
        return {"type": "IdleAction"}

    # --- Unknown → wait ---
    return WAIT_ACTION
