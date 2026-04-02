"""HTML inspection tools for LLM tool-use loop.

The LLM can request these tools to gather more information before deciding on an action.
Each tool returns a dict with ok=True/False and result data.
"""
from __future__ import annotations
import re
import json
from typing import Any
from bs4 import BeautifulSoup


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _safe_truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[:max(0, n - 3)] + "...")


def tool_search_text(*, html: str, query: str, max_matches: int = 20, context_chars: int = 80) -> dict[str, Any]:
    q = str(query or "")
    if not q:
        return {"ok": False, "error": "missing query"}
    try:
        pat = re.compile(re.escape(q), re.IGNORECASE)
    except Exception as e:
        return {"ok": False, "error": f"invalid pattern: {str(e)[:120]}"}

    hay = str(html or "")
    out = []
    for m in pat.finditer(hay):
        if len(out) >= max_matches:
            break
        a = max(0, m.start() - context_chars)
        b = min(len(hay), m.end() + context_chars)
        out.append({
            "start": m.start(),
            "end": m.end(),
            "snippet": _safe_truncate(hay[a:b].replace("\n", " ").replace("\r", " "), 2 * context_chars + 40),
        })
    return {"ok": True, "matches": out, "count": len(out)}


def tool_extract_forms(*, html: str, max_forms: int = 10, max_inputs: int = 25) -> dict[str, Any]:
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    forms = []
    for f in soup.find_all("form")[:max_forms]:
        try:
            f_attrs = {k: str(v) if isinstance(v, str) else " ".join(v) for k, v in (f.attrs or {}).items()}
            inputs = []
            for el in f.find_all(["input", "textarea", "select", "button"])[:max_inputs]:
                try:
                    a = {k: str(v) if isinstance(v, str) else " ".join(v) for k, v in (el.attrs or {}).items()}
                    t = _norm_ws(el.get_text(" ", strip=True))
                    inputs.append({
                        "tag": el.name,
                        "type": (a.get("type") or "").lower(),
                        "id": a.get("id") or "",
                        "name": a.get("name") or "",
                        "placeholder": a.get("placeholder") or "",
                        "text": _safe_truncate(t, 160),
                    })
                except Exception:
                    continue
            forms.append({
                "id": f_attrs.get("id") or "",
                "name": f_attrs.get("name") or "",
                "action": f_attrs.get("action") or "",
                "controls": inputs,
            })
        except Exception:
            continue
    return {"ok": True, "forms": forms, "count": len(forms)}


def tool_list_links(*, html: str, base_url: str, max_links: int = 60) -> dict[str, Any]:
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    from urllib.parse import urljoin
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for a in soup.select("a[href]"):
        try:
            href = str(a.get("href") or "").strip()
            if not href or href.lower().startswith("javascript:"):
                continue
            text = _norm_ws(a.get_text(" ", strip=True))
            if not text:
                text = _norm_ws(str(a.get("aria-label") or ""))

            resolved = urljoin(base_url, href) if base_url else href
            sig = resolved + "|" + text
            if sig in seen:
                continue
            seen.add(sig)

            out.append({
                "href": _safe_truncate(href, 260),
                "url": _safe_truncate(resolved, 320),
                "text": _safe_truncate(text, 160),
            })
            if len(out) >= max_links:
                break
        except Exception:
            continue
    return {"ok": True, "count": len(out), "links": out}


def tool_list_cards(*, candidates: list, max_cards: int = 25, max_text: int = 900) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}

    for i, c in enumerate(candidates or []):
        try:
            if c.tag not in {"a", "button"}:
                if not (c.selector and c.selector.attribute == "href"):
                    continue

            key = (c.context or "").strip()
            if not key:
                key = "(no_context)"

            g = groups.get(key)
            if g is None:
                g = {"card_text": _safe_truncate(key, max_text), "candidate_ids": [], "actions": []}
                groups[key] = g

            g["candidate_ids"].append(i)
            if len(g["actions"]) < 6:
                g["actions"].append({
                    "candidate_id": i,
                    "tag": c.tag,
                    "text": _safe_truncate(c.text or "", 140),
                })
        except Exception:
            continue

    ranked = []
    for _k, g in groups.items():
        txt = str(g.get("card_text") or "")
        n_actions = len(g.get("actions") or [])
        L = len(txt)
        penalty = 0
        if L < 40:
            penalty += 400
        if L > 900:
            penalty += min(1200, L - 900)
        score = (1000 - penalty + min(L, 700), n_actions)
        ranked.append((score, g))

    ranked.sort(key=lambda x: x[0], reverse=True)
    cards = [g for _, g in ranked[:max_cards]]
    return {"ok": True, "count": len(cards), "cards": cards}


TOOL_REGISTRY = {
    "search_text": tool_search_text,
    "extract_forms": tool_extract_forms,
    "list_links": tool_list_links,
    "list_cards": tool_list_cards,
}


def run_tool(tool: str, args: dict[str, Any], *, html: str, url: str, candidates: list) -> dict[str, Any]:
    t = str(tool or "").strip()
    fn = TOOL_REGISTRY.get(t)
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {t}", "known": sorted(TOOL_REGISTRY.keys())}

    a = args if isinstance(args, dict) else {}
    if t == "list_cards":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"max_cards", "max_text"}})
    if t == "list_links":
        return fn(html=html, base_url=str(url or ""), **{k: v for k, v in a.items() if k in {"max_links"}})
    if t in {"search_text", "extract_forms"}:
        return fn(html=html, **a)

    return {"ok": False, "error": f"tool not wired: {t}"}
