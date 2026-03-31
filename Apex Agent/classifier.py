"""Task type classification from natural-language prompts.

Combines keyword matching with priority ordering to handle compound tasks
(e.g. "login then search") correctly.
"""
from __future__ import annotations


def classify_task_type(prompt: str) -> str:
    """Return a task type string for the given prompt."""
    lower = prompt.lower()

    # Compound: login-then-X (must check before simple login)
    login_kw = ("log in", "login", "sign in")
    continuation_kw = (
        "then", "after", "once logged", "and then", "and add",
        "and check", "and go", "and navigate", "and search",
        "and click", "and open", "and view", "and delete",
        "and edit", "and update", "and submit",
    )
    if any(k in lower for k in login_kw) and any(k in lower for k in continuation_kw):
        return "login_then_action"

    # Registration (before login to avoid false-positive on "register")
    reg_kw = ("sign up", "registration", "create an account", "create account")
    if any(k in lower for k in reg_kw):
        return "registration"
    if "register" in lower and not any(
        exc in lower for exc in ("register a movie", "register a film", "register the ", "register for ")
    ):
        return "registration"

    # Logout (before login to avoid "log" substring match)
    if any(k in lower for k in ("log out", "logout", "sign out")):
        return "logout"

    # Login
    if any(k in lower for k in login_kw):
        return "login"

    # Contact/support form
    if "contact" in lower and any(k in lower for k in ("form", "message", "fill", "support", "submit", "expert")):
        return "contact"

    # Purchase / cart
    if any(k in lower for k in ("buy ", "purchase", "add to cart", "checkout", "place order")):
        return "purchase"

    # Delete / remove
    if any(k in lower for k in ("delete", "remove", "cancel")):
        return "delete"

    # Edit / update
    if any(k in lower for k in ("edit ", "update ", "change ", "modify")):
        return "edit"

    # Search
    if any(k in lower for k in ("search for", "search ", "find ", "look up", "look for")):
        return "search"

    # Filter
    if any(k in lower for k in ("filter", "apply filter", "narrow", "sort by")):
        return "filter"

    # Navigate / view detail
    if any(k in lower for k in (
        "show details", "view details", "navigate to", "go to", "open ",
        "click on the", "clicks on the", "visit",
    )):
        return "navigate_detail"

    # Form fill
    if any(k in lower for k in ("fill ", "submit", "complete the form", "create ")):
        return "form_fill"

    # Dropdown select
    if any(k in lower for k in ("select ", "choose ", "pick ")):
        return "dropdown_select"

    # Data retrieval
    if any(k in lower for k in ("retrieve", "show me", "display", "get ", "list ")):
        return "data_retrieval"

    return "general"


def classify_shortcut_type(prompt: str) -> str | None:
    """Return shortcut-eligible type or None."""
    lower = prompt.lower()
    if any(k in lower for k in ("sign up", "registration", "create an account", "create account")):
        return "registration"
    if "register" in lower and not any(
        exc in lower for exc in ("register a movie", "register a film", "register the ", "register for ")
    ):
        return "registration"
    if any(k in lower for k in ("log out", "logout", "sign out")):
        return "logout"
    if any(k in lower for k in ("log in", "login", "sign in")):
        return "login"
    if "contact" in lower and any(k in lower for k in ("form", "message", "fill", "support", "submit", "expert")):
        return "contact"
    return None
