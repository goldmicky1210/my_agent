"""HTML pruning, candidate extraction, and page IR generation.

Combines:
- Two-stage pruning (autoppia_top_miner) for clean extraction
- Context extraction per candidate (OJO Agent) for disambiguation
- Compact page IR format (Onyxdrift) for token efficiency
- Select element option extraction for dropdown tasks
"""
from __future__ import annotations
from bs4 import BeautifulSoup, Comment, Tag

from config import SELECTOR_PRIORITY, PAGE_IR_CHAR_LIMIT
from models import Candidate, Selector, PageContext, PageIR

STRIP_TAGS = {"script", "style", "svg", "noscript", "iframe"}
INTERACTIVE_CSS = (
    "button, a[href], input:not([type='hidden']), textarea, select, "
    "[role='button'], [role='link'], [role='tab'], [role='menuitem']"
)
MAX_TEXT_LEN = 100
MAX_CONTEXT_LEN = 180
MAX_OPTIONS = 10

_CSS_HIDDEN_CLASSES = frozenset({
    "hidden", "d-none", "invisible", "collapse", "sr-only",
    "visually-hidden", "offscreen", "screen-reader-only", "display-none",
})


# ---------------------------------------------------------------------------
# Stage 1: Prune non-semantic bloat
# ---------------------------------------------------------------------------

def prune_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    return soup


# ---------------------------------------------------------------------------
# Hidden / disabled detection
# ---------------------------------------------------------------------------

def _is_hidden_or_disabled(el: Tag) -> bool:
    if el.has_attr("hidden") or el.has_attr("disabled"):
        return True
    if el.get("type", "").lower() == "hidden":
        return True
    style = (el.get("style") or "").lower().replace(" ", "")
    if any(tok in style for tok in ("display:none", "visibility:hidden", "opacity:0")):
        return True
    if el.get("aria-hidden", "").lower() == "true":
        return True
    classes = el.get("class", [])
    class_set = {c.lower() for c in (classes if isinstance(classes, list) else str(classes).split())}
    if class_set & _CSS_HIDDEN_CLASSES:
        return True
    # Check parent (one level)
    parent = el.parent
    if parent and isinstance(parent, Tag):
        p_classes = parent.get("class", [])
        p_set = {c.lower() for c in (p_classes if isinstance(p_classes, list) else str(p_classes).split())}
        if p_set & _CSS_HIDDEN_CLASSES:
            return True
        p_style = (parent.get("style") or "").lower().replace(" ", "")
        if "display:none" in p_style or "visibility:hidden" in p_style:
            return True
    return False


# ---------------------------------------------------------------------------
# Selector building
# ---------------------------------------------------------------------------

def _pick_selector(el: Tag) -> Selector | None:
    for attr in SELECTOR_PRIORITY:
        if attr == "text":
            text = el.get_text(strip=True)
            if text:
                return Selector(type="tagContainsSelector", value=text[:60])
        else:
            val = el.get(attr)
            if val and isinstance(val, str) and val.strip():
                clean = val.strip()
                # Skip javascript: hrefs
                if attr == "href" and clean.lower().startswith("javascript:"):
                    continue
                return Selector(type="attributeValueSelector", attribute=attr, value=clean)
    return None


# ---------------------------------------------------------------------------
# Label inference (OJO Agent's priority chain)
# ---------------------------------------------------------------------------

def _infer_label(el: Tag) -> str:
    # Direct text for buttons/links
    if el.name in ("button", "a"):
        text = el.get_text(strip=True)
        if text:
            return text[:MAX_TEXT_LEN]

    # aria-label
    aria = el.get("aria-label")
    if aria:
        return str(aria).strip()[:MAX_TEXT_LEN]

    # placeholder
    ph = el.get("placeholder")
    if ph:
        return str(ph).strip()[:MAX_TEXT_LEN]

    # title
    title = el.get("title")
    if title:
        return str(title).strip()[:MAX_TEXT_LEN]

    # aria-labelledby
    labelledby = el.get("aria-labelledby")
    if labelledby and el.find_parent():
        root = el.find_parent(["html", "[document]"]) or el
        ref = root.find(id=labelledby)
        if ref:
            return ref.get_text(strip=True)[:MAX_TEXT_LEN]

    # <label for="id">
    el_id = el.get("id")
    if el_id and el.find_parent():
        root = el.find_parent(["html", "[document]"]) or el
        label = root.find("label", attrs={"for": el_id})
        if label:
            return label.get_text(strip=True)[:MAX_TEXT_LEN]

    # Parent <label> wrapper
    parent_label = el.find_parent("label")
    if parent_label:
        return parent_label.get_text(strip=True)[:MAX_TEXT_LEN]

    # Generic text
    text = el.get_text(strip=True)
    return text[:MAX_TEXT_LEN] if text else ""


# ---------------------------------------------------------------------------
# Context extraction (OJO Agent's container approach)
# ---------------------------------------------------------------------------

def _extract_context(el: Tag) -> str:
    """Walk up DOM to find a meaningful container and return its text."""
    container_tags = {"li", "tr", "article", "section", "div", "td", "card"}
    current = el.parent
    depth = 0
    while current and depth < 8:
        if not isinstance(current, Tag):
            break
        tag_name = current.name or ""
        # Check tag name or role
        if tag_name in container_tags or current.get("role") in ("listitem", "row", "article"):
            text = current.get_text(separator=" ", strip=True)
            text_len = len(text)
            # Sweet spot: not too short (no context), not too long (entire page)
            if 20 <= text_len <= 500:
                return " ".join(text.split())[:MAX_CONTEXT_LEN]
        current = current.parent
        depth += 1
    return ""


# ---------------------------------------------------------------------------
# Select element options
# ---------------------------------------------------------------------------

def _extract_options(el: Tag) -> list[str]:
    if el.name != "select":
        return []
    options = []
    for opt in el.find_all("option"):
        text = opt.get_text(strip=True)
        if text:
            options.append(text)
        if len(options) >= MAX_OPTIONS:
            break
    return options


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def extract_candidates(soup: BeautifulSoup) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_selectors: set[tuple] = set()
    index = 0

    for el in soup.select(INTERACTIVE_CSS):
        if _is_hidden_or_disabled(el):
            continue
        selector = _pick_selector(el)
        if selector is None:
            continue

        # Dedup by selector signature
        sig = (selector.type, selector.attribute, selector.value)
        if sig in seen_selectors:
            continue
        seen_selectors.add(sig)

        text = _infer_label(el)
        tag = el.name
        context = _extract_context(el)
        options = _extract_options(el)
        current_value = el.get("value", "") if tag == "input" else ""

        # Parent form
        form = el.find_parent("form")
        parent_form = None
        if form and isinstance(form, Tag):
            parent_form = form.get("id") or form.get("name") or form.get("action")

        candidate = Candidate(
            index=index,
            tag=tag,
            text=text,
            selector=selector,
            input_type=el.get("type") if tag in ("input", "button") else None,
            name=el.get("name"),
            placeholder=el.get("placeholder"),
            href=el.get("href") if tag == "a" else None,
            role=el.get("role"),
            context=context,
            parent_form=str(parent_form) if parent_form else None,
            options=options,
            current_value=str(current_value) if current_value else "",
        )
        candidates.append(candidate)
        index += 1

    return candidates


# ---------------------------------------------------------------------------
# Page context
# ---------------------------------------------------------------------------

def extract_page_context(soup: BeautifulSoup, url: str) -> PageContext:
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    headings = []
    for h in soup.find_all(["h1", "h2", "h3"]):
        text = h.get_text(strip=True)
        if text:
            headings.append(text)
        if len(headings) >= 10:
            break
    return PageContext(url=url, title=title, headings=headings)


# ---------------------------------------------------------------------------
# Page IR formatting
# ---------------------------------------------------------------------------

def _format_selector_display(sel: Selector) -> str:
    if sel.attribute == "id":
        return f"#{sel.value}"
    elif sel.attribute == "href":
        return f'href="{sel.value}"'
    elif sel.attribute == "name":
        return f"[name='{sel.value}']"
    elif sel.attribute == "data-testid":
        return f"[data-testid='{sel.value}']"
    elif sel.attribute == "aria-label":
        return f'[aria-label="{sel.value}"]'
    elif sel.attribute == "placeholder":
        return f'[placeholder="{sel.value}"]'
    elif sel.attribute == "title":
        return f'[title="{sel.value}"]'
    elif sel.type == "tagContainsSelector":
        return f'"{sel.value}"'
    return f'{sel.attribute}="{sel.value}"'


def _format_candidate_line(c: Candidate) -> str:
    parts = [f"[{c.index}]", c.tag]
    if c.input_type:
        parts.append(f"type={c.input_type}")
    if c.text:
        parts.append(f'"{c.text}"')
    parts.append(_format_selector_display(c.selector))
    if c.parent_form:
        parts.append(f"form={c.parent_form}")
    if c.options:
        preview = ", ".join(c.options[:5])
        parts.append(f"options=[{preview}]")
    if c.context:
        parts.append(f'-> "{c.context[:80]}"')
    return " ".join(parts)


def build_page_ir(soup: BeautifulSoup, url: str, candidates: list[Candidate]) -> PageIR:
    context = extract_page_context(soup, url)

    lines: list[str] = []
    lines.append(f"URL: {context.url}")
    if context.title:
        lines.append(f"Title: {context.title}")
    if context.headings:
        lines.append(f"Headings: {', '.join(context.headings[:5])}")
    lines.append("")
    lines.append("Interactive elements:")

    header_text = "\n".join(lines)
    remaining_budget = PAGE_IR_CHAR_LIMIT - len(header_text) - 1

    kept_candidates: list[Candidate] = []
    candidate_lines: list[str] = []
    running_len = 0

    for c in candidates:
        line = _format_candidate_line(c)
        line_len = len(line) + 1
        if running_len + line_len > remaining_budget:
            candidate_lines.append(f"... ({len(candidates) - len(kept_candidates)} more elements truncated)")
            break
        candidate_lines.append(line)
        kept_candidates.append(c)
        running_len += line_len

    raw_text = header_text + "\n" + "\n".join(candidate_lines)
    return PageIR(context=context, candidates=kept_candidates, raw_text=raw_text)


# ---------------------------------------------------------------------------
# DOM digest (from OJO Agent) - compact page structure summary
# ---------------------------------------------------------------------------

def build_dom_digest(soup: BeautifulSoup, max_chars: int = 600) -> str:
    """Compact summary of page structure for early steps."""
    parts: list[str] = []

    # Title
    if soup.title and soup.title.string:
        parts.append(f"TITLE: {soup.title.string.strip()}")

    # Headings
    headings = []
    for h in soup.find_all(["h1", "h2", "h3"], limit=8):
        text = h.get_text(strip=True)
        if text:
            headings.append(text[:60])
    if headings:
        parts.append(f"HEADINGS: {' | '.join(headings)}")

    # Forms summary
    forms = soup.find_all("form", limit=4)
    for form in forms:
        form_id = form.get("id") or form.get("name") or form.get("action") or "unnamed"
        inputs = form.find_all(["input", "textarea", "select"], limit=8)
        field_names = []
        for inp in inputs:
            if inp.get("type") == "hidden":
                continue
            name = inp.get("name") or inp.get("placeholder") or inp.get("aria-label") or inp.name
            field_names.append(name)
        if field_names:
            parts.append(f"FORM({form_id}): {', '.join(field_names)}")

    # CTAs (buttons not in navigation)
    buttons = soup.find_all("button", limit=6)
    ctas = []
    for btn in buttons:
        text = btn.get_text(strip=True)
        if text and text.lower() not in ("home", "logo", "menu"):
            ctas.append(text[:40])
    if ctas:
        parts.append(f"BUTTONS: {' | '.join(ctas)}")

    result = "\n".join(parts)
    return result[:max_chars]
