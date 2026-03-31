"""Enhanced constraint extraction from natural-language task prompts.

Combines the best patterns from OJO Agent (300+ patterns) and Onyxdrift
(span-overlap dedup) for robust constraint parsing.
"""
from __future__ import annotations
import re
from models import Constraint

# Field name pattern: word_chars possibly with underscores or "the X" prefix
_FIELD = r"(?:the\s+)?([\w]+(?:_[\w]+)*)"

# Ordered patterns: more specific first, greedy-safe
_PATTERNS: list[tuple[str, str]] = [
    # NOT CONTAIN variants
    (_FIELD + r"(?:\s+that)?\s+does\s+NOT\s+CONTAIN\s+['\"]([^'\"]+)['\"]", "not_contains"),
    (_FIELD + r"(?:\s+that)?\s+does\s+NOT\s+CONTAIN\s+([^\s,'\"\n]+)", "not_contains"),
    (_FIELD + r"\s+not\s+contains?\s+['\"]([^'\"]+)['\"]", "not_contains"),
    (_FIELD + r"\s+not\s+contains?\s+([^\s,'\"\n]+)", "not_contains"),
    # NOT EQUALS variants
    (_FIELD + r"\s+(?:does\s+)?not\s+equals?\s+['\"]([^'\"]+)['\"]", "not_equals"),
    (_FIELD + r"\s+(?:does\s+)?not\s+equals?\s+([^\s,'\"\n]+)", "not_equals"),
    (_FIELD + r"\s+!=\s+['\"]?([^'\"\s,\n\]]+)['\"]?", "not_equals"),
    # CONTAINS variants
    (_FIELD + r"(?:\s+that)?\s+CONTAINS\s+['\"]([^'\"]+)['\"]", "contains"),
    (_FIELD + r"\s+contains?\s+['\"]([^'\"]+)['\"]", "contains"),
    (_FIELD + r"\s+contains?\s+([^\s,'\"\n]+)", "contains"),
    # GREATER/LESS EQUAL (MUST come before equals to prevent partial match)
    (
        _FIELD + r"\s+(?:(?:is|are)\s+)?(?:GREATER\s+THAN\s+OR\s+EQUAL\s+TO|greater\s+(?:than\s+)?or\s+equal\s+to|greater\s+equal(?:\s+to)?|>=)\s+['\"]?([^\s,'\"\n\]]+)['\"]?",
        "greater_equal",
    ),
    (
        _FIELD + r"\s+(?:(?:is|are)\s+)?(?:LESS\s+THAN\s+OR\s+EQUAL\s+TO|less\s+(?:than\s+)?or\s+equal\s+to|less\s+equal(?:\s+to)?|<=)\s+['\"]?([^\s,'\"\n\]]+)['\"]?",
        "less_equal",
    ),
    # EQUALS variants (after greater/less equal to avoid partial match)
    (_FIELD + r"(?:\s+(?:that|field))?\s+equals?\s+['\"]([^'\"]+)['\"]", "equals"),
    (_FIELD + r"(?:\s+(?:that|field))?\s+EQUALS\s+['\"]([^'\"]+)['\"]", "equals"),
    (_FIELD + r"\s+(?:is\s+(?:exactly\s+)?|is\s+)['\"]([^'\"]+)['\"]", "equals"),
    (_FIELD + r"(?:\s+(?:that|field))?\s+equals?\s+([^\s,'\"\n\]]+)", "equals"),
    # GREATER / LESS THAN
    (_FIELD + r"\s+(?:(?:is|are)\s+)?greater\s+than\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "greater_than"),
    (_FIELD + r"\s+(?:(?:is|are)\s+)?less\s+than\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "less_than"),
    (_FIELD + r"\s+BELOW\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "less_than"),
    (_FIELD + r"\s+ABOVE\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "greater_than"),
    (_FIELD + r"\s+AFTER\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "greater_than"),
    (_FIELD + r"\s+BEFORE\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "less_than"),
    # BETWEEN (split into two constraints)
    # handled separately below
]

_NOT_IN_PAT = re.compile(
    r"([\w_]+(?:\s+[\w_]+)?)\s+is\s+not\s+one\s+of\s+\[([^\]]+)\]", re.IGNORECASE
)
_IN_PAT = re.compile(
    r"([\w_]+(?:\s+[\w_]+)?)\s+is\s+one\s+of\s+\[([^\]]+)\]", re.IGNORECASE
)
_BETWEEN_PAT = re.compile(
    _FIELD + r"\s+(?:is\s+)?between\s+['\"]?([^\s,'\"\n\]]+)['\"]?\s+and\s+['\"]?([^\s,'\"\n\]]+)['\"]?",
    re.IGNORECASE,
)


def _spans_overlap(start: int, end: int, used: list[tuple[int, int]]) -> bool:
    return any(start < e and end > s for s, e in used)


def parse_constraints(prompt: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    seen: set[tuple] = set()
    used_spans: list[tuple[int, int]] = []

    # 1. NOT IN / IN (list-value patterns)
    for pat, op in [(_NOT_IN_PAT, "not_in"), (_IN_PAT, "in")]:
        for m in pat.finditer(prompt):
            if _spans_overlap(m.start(), m.end(), used_spans):
                continue
            field = m.group(1).strip().lower().replace(" ", "_")
            vals = [v.strip().strip("'\"") for v in m.group(2).split(",")]
            key = (field, op, str(vals))
            if key not in seen:
                seen.add(key)
                constraints.append(Constraint(field=field, operator=op, value=vals))
                used_spans.append((m.start(), m.end()))

    # 2. BETWEEN (split into >= and <=)
    for m in _BETWEEN_PAT.finditer(prompt):
        if _spans_overlap(m.start(), m.end(), used_spans):
            continue
        field = m.group(1).strip().lower().replace(" ", "_")
        lo, hi = m.group(2).strip(), m.group(3).strip()
        for op, val in [("greater_equal", lo), ("less_equal", hi)]:
            key = (field, op, val)
            if key not in seen:
                seen.add(key)
                constraints.append(Constraint(field=field, operator=op, value=val))
        used_spans.append((m.start(), m.end()))

    # 3. Standard binary patterns
    for pattern, op in _PATTERNS:
        for m in re.finditer(pattern, prompt, re.IGNORECASE):
            if _spans_overlap(m.start(), m.end(), used_spans):
                continue
            field = m.group(1).strip().lower().replace(" ", "_")
            value = m.group(2).strip().strip("'\"").rstrip(".,;:")
            key = (field, op, value)
            if key not in seen:
                seen.add(key)
                constraints.append(Constraint(field=field, operator=op, value=value))
                used_spans.append((m.start(), m.end()))

    return constraints


def format_constraints_block(constraints: list[Constraint]) -> str:
    if not constraints:
        return ""
    lines = ["CONSTRAINTS (use these to find the RIGHT item and fill forms correctly):"]
    for c in constraints:
        op_map = {
            "equals": f"  [{c.field}] MUST EQUAL '{c.value}' exactly",
            "not_equals": f"  [{c.field}] must NOT be '{c.value}' -> choose any other value",
            "contains": f"  [{c.field}] MUST CONTAIN '{c.value}'",
            "not_contains": f"  [{c.field}] must NOT CONTAIN '{c.value}'",
            "greater_than": f"  [{c.field}] must be > {c.value}",
            "less_than": f"  [{c.field}] must be < {c.value}",
            "greater_equal": f"  [{c.field}] must be >= {c.value}",
            "less_equal": f"  [{c.field}] must be <= {c.value}",
            "not_in": f"  [{c.field}] must NOT be any of {c.value}",
            "in": f"  [{c.field}] must be one of {c.value}",
        }
        lines.append(op_map.get(c.operator, f"  [{c.field}] {c.operator} {c.value}"))
    return "\n".join(lines)


def extract_credentials(prompt: str) -> dict[str, str]:
    """Pull literal credential values and well-known placeholders from task text."""
    creds: dict[str, str] = {}
    # Explicit credentials in prompt
    for field in ("username", "password", "email"):
        m = re.search(
            rf"{field}\s*[:=]\s*['\"]?([^'\"\n,]+)['\"]?", prompt, re.IGNORECASE
        )
        if m:
            creds[field] = m.group(1).strip()

    # IWA placeholders
    if "<username>" in prompt:
        creds.setdefault("username", "<username>")
    if "<password>" in prompt:
        creds.setdefault("password", "<password>")
    if "<web_agent_id>" in prompt:
        creds["web_agent_id"] = "1"

    # Defaults for IWA sandbox
    creds.setdefault("username", "<username>")
    creds.setdefault("password", "<password>")
    creds.setdefault("email", "<signup_email>")

    return creds


def extract_search_query(prompt: str) -> str | None:
    """Extract search query from task prompt."""
    constraints = parse_constraints(prompt)
    for c in constraints:
        if c.field == "query":
            return str(c.value)
    m = re.search(r"search\s+(?:for\s+)?.*?['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
    if m:
        return m.group(1)
    return None
