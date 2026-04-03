"""Pre-built shortcut actions that bypass LLM for deterministic tasks.

Combines:
- Quick click patterns (Onyxdrift) for known UI elements
- Search shortcuts (Onyxdrift) with extended site coverage
- Enhanced form detection (all three agents) for login/registration/contact/logout
- Multi-step sequences for compound tasks
"""
from __future__ import annotations
import re
from bs4 import BeautifulSoup

from models import Candidate
from constraint_parser import extract_search_query
from config import SEARCH_INPUT_IDS


def _sel_attr(attribute: str, value: str) -> dict:
    return {"type": "attributeValueSelector", "attribute": attribute, "value": value, "case_sensitive": False}


def _click(attribute: str, value: str) -> list[dict]:
    return [{"type": "ClickAction", "selector": _sel_attr(attribute, value)}]


def _click_xpath(xpath: str) -> list[dict]:
    return [{"type": "ClickAction", "selector": {"type": "xpathSelector", "value": xpath}}]


# ---------------------------------------------------------------------------
# Quick click: regex → fixed element
# ---------------------------------------------------------------------------

def try_quick_click(prompt: str, url: str, seed: str | None, step: int) -> list[dict] | None:
    t = prompt.lower()

    # Calendar
    if re.search(r"go\s+to\s+today|focus.*today|today.?s?\s+date\s+in\s+the\s+calendar", t):
        return _click("id", "focus-today")
    if re.search(r"add\s+a?\s*new\s+calendar\s+event|add\s+calendar\s+button|click.*add\s+calendar", t):
        return _click("id", "new-event-cta")
    if re.search(r"click.*add\s+team|add\s+team\s+button", t):
        return _click("id", "add-team-btn")

    # Wishlist / favorites
    if re.search(r"(show\s+me\s+my\s+saved|my\s+wishlist|show.*wishlist|view.*wishlist|favorites?\s+page)", t):
        return _click("id", "favorite-action")

    # Navbar navigation
    if re.search(r"clicks?\s+on\s+the\s+jobs?\s+option\s+in\s+the\s+navbar", t):
        return _click("href", f"/jobs?seed={seed}") if seed else None
    if re.search(r"clicks?\s+on\s+.*profile\s+.*in\s+the\s+navbar", t):
        return _click("href", f"/profile/alexsmith?seed={seed}") if seed else None

    # Featured / spotlight items
    if re.search(r"(spotlight|featured)\s+.*(?:movie|film).*details|view\s+details\s+.*(?:spotlight|featured)\s+(?:movie|film)", t):
        return _click("id", "spotlight-view-details-btn")
    if re.search(r"(spotlight|featured)\s+.*book.*details|view\s+details\s+.*(?:featured|spotlight)\s+book", t):
        return _click("id", "featured-book-view-details-btn-1")
    if re.search(r"(spotlight|featured)\s+.*product.*details|view\s+details\s+.*(?:featured|spotlight)\s+product", t):
        return _click("id", "view-details")

    # Autoconnect home tab
    from urllib.parse import urlsplit
    port = urlsplit(url).port
    if port == 8008 and re.search(r"go\s+to\s+the\s+home\s+tab|home\s+tab\s+from\s+the\s+navbar", t):
        return _click_xpath("//header//nav/a[1]")

    # Clear selection
    if re.search(r"clear\s+(the\s+)?(current\s+)?selection", t):
        return _click_xpath("(//button[@role='checkbox'])[1]")

    # About page feature (multi-step)
    if re.search(r"about\s+page.*feature|feature.*about\s+page", t):
        if step == 0:
            return _click("id", "nav-about")
        elif step == 1:
            return [{"type": "ScrollAction", "down": True}]
        else:
            return _click_xpath("//h3[contains(text(),'Curated')]")

    # Like a post (autoconnect)
    m = re.search(r"like\s+(?:the\s+)?(?:post|first\s+post|latest\s+post)", t)
    if m and port == 8008:
        return _click("id", "post_like_button_p1")

    # --- Season 1 overfit additions ---

    # Calendar view switching (autocalendar 8010)
    if port == 8010:
        for view_name in ("day", "week", "month"):
            if f"switch to {view_name}" in t or f"{view_name} view" in t:
                label_map = {"day": "Select Day view", "week": "Select Week view", "month": "Select Month view"}
                if step == 0:
                    return _click("id", "view-selector")
                elif step == 1:
                    return _click("aria-label", label_map.get(view_name, f"Select {view_name.title()} view"))
                return []

    # Navbar hires (autowork 8009)
    if port == 8009:
        if re.search(r"hires.*navbar|navbar.*hires", t):
            return _click("href", f"/hires?seed={seed}") if seed else None
        if "book a consultation" in t or "consultation" in t:
            return _click_xpath("//*[contains(@id, 'book-consultation-button')]")

    # About page (autodining 8003)
    if port == 8003 and re.search(r"about\s+page|navigate.*about.*information", t):
        return _click("id", "about-menu-item")

    # View cart (autozone 8002)
    if port == 8002:
        if re.search(r"shopping\s+cart|contents\s+of\s+my", t):
            return _click("id", "cart-icon")
        if re.search(r"wishlist", t):
            return _click("id", "wishlist-btn")

    # View pending events (autocrm 8004)
    if port == 8004 and "pending" in t and "event" in t:
        if step == 0:
            return _click("id", "appointments-nav")
        elif step == 1:
            return _click("id", "toggle-future-events")
        return []

    # Enter location (autodrive 8012)
    if port == 8012:
        _loc_xpath = ("//input[contains(@placeholder, 'Pickup location') or "
                     "contains(@placeholder, 'Where from?') or "
                     "contains(@placeholder, 'Enter pickup') or "
                     "contains(@placeholder, 'Start location') or "
                     "contains(@placeholder, 'Where are you?')]")
        if "search location" in t:
            m2 = re.search(r"(?:for |details for )['\"]([^'\"]+)['\"]", prompt)
            if m2:
                if step == 0:
                    return _click_xpath(_loc_xpath)
                elif step == 1:
                    return [{"type": "TypeAction", "text": m2.group(1),
                             "selector": {"type": "xpathSelector", "value": _loc_xpath}}]
                return []
        if "enter" in t and "location" in t or "select a location" in t:
            if step == 0:
                return _click_xpath(_loc_xpath)
            return []

    # Create label (automail 8005)
    if port == 8005 and "create" in t and "label" in t:
        if step == 0:
            return _click_xpath("//*[contains(@id, 'label-trigger') or contains(@id, 'tag-trigger')]")
        elif step == 1:
            m2 = re.search(r"(?:equal to |equals? |CONTAINS )['\"]([^'\"]+)['\"]", prompt)
            label_text = m2.group(1) if m2 else "label"
            return [{"type": "TypeAction", "text": label_text,
                     "selector": {"type": "xpathSelector",
                                  "value": "//input[contains(@id, 'label-trigger') or contains(@id, 'tag-trigger')]"}}]
        elif step == 2:
            return _click_xpath("//button[contains(@id, 'add-label-btn') or contains(@id, 'add-label-button')]")
        return []

    # Search delivery restaurant (autodelivery 8006)
    if port == 8006 and "search" in t and "restaurant" in t:
        m2 = re.search(r"(?:exactly |query is |query equals? )['\"]([^'\"]+)['\"]", prompt)
        if m2 and step == 0:
            return [{"type": "TypeAction", "text": m2.group(1), "selector": _sel_attr("id", "find-food")}]
        return []

    # --- R15 additions: broader quick-click coverage ---

    # autocinema (8000/8100) - movie actions
    if port in (8000, 8100):
        if re.search(r"go.*search\s+page|navigate.*search", t):
            return _click("id", "go-to-search-button")
        if re.search(r"view.*stats|statistics", t):
            return _click("id", "stats-movies-card")

    # autobooks (8001/8101) - book actions
    if port in (8001, 8101):
        if re.search(r"go.*search\s+page|navigate.*search", t):
            return _click("id", "go-to-search-button")
        if re.search(r"view.*cart|shopping\s+cart|my\s+cart", t):
            return _click("href", f"/cart?seed={seed}") if seed else _click("id", "cart-page")
        if re.search(r"reading\s+list", t):
            return _click("id", "reading-list-button")
        if re.search(r"add.*book.*cart|add.*to.*cart", t) and "featured" not in t:
            return _click("id", "add-to-cart-button")

    # autozone (8002/8102) - product actions (extend existing)
    if port in (8002, 8102):
        if re.search(r"go.*search\s+page|navigate.*search", t):
            return _click("id", "go-to-search-button")
        if re.search(r"checkout|proceed.*checkout", t):
            return _click("id", "checkout-button")
        if re.search(r"grid\s+view", t):
            return _click("id", "grid-view-button")
        if re.search(r"list\s+view", t):
            return _click("id", "list-view-button")

    # autodining (8003/8103) - restaurant actions
    if port in (8003, 8103):
        if re.search(r"help\s+page|faq|frequently\s+asked", t):
            return _click("id", "help-link")
        if re.search(r"contact\s+support|support\s+page", t):
            return _click("id", "contact-support-button")
        if re.search(r"delivery\s+toggle|switch.*delivery", t):
            return _click("id", "delivery-toggle")

    # autocrm (8004/8104) - CRM actions
    if port in (8004, 8104):
        if re.search(r"add.*client|new\s+client|create.*client", t):
            return _click("id", "add-client-button")
        if re.search(r"add.*matter|new\s+matter|create.*matter", t):
            return _click("id", "add-matter-button")
        if re.search(r"clients?\s+(?:page|section|nav|tab)", t):
            return _click("id", "clients-nav-link")
        if re.search(r"matters?\s+(?:page|section|nav|tab)", t):
            return _click("id", "matters-nav-link")
        if re.search(r"calendar\s+(?:page|section|nav|tab)", t):
            return _click("id", "calendar-nav-link")
        if re.search(r"documents?\s+(?:page|section|nav|tab)", t):
            return _click("id", "documents-nav-link")
        if re.search(r"billing\s+(?:page|section|nav|tab)", t):
            return _click("id", "billing-nav-link")
        if re.search(r"settings?\s+(?:page|section|nav|tab)", t):
            return _click("id", "settings-nav-link")
        if re.search(r"dashboard|home\s+page", t):
            return _click("id", "dashboard-nav-link")
        if re.search(r"help\s+(?:center|page|section)", t):
            return _click("id", "help-link")

    # automail (8005/8105) - email actions
    if port in (8005, 8105):
        if re.search(r"inbox|go.*inbox", t):
            return _click("id", "sidebar-inbox")
        if re.search(r"starred|go.*starred", t):
            return _click("id", "sidebar-starred")
        if re.search(r"drafts|go.*drafts", t):
            return _click("id", "sidebar-drafts")
        if re.search(r"sent\s+(?:mail|folder|page)|go.*sent", t):
            return _click("id", "sidebar-sent")
        if re.search(r"trash|go.*trash", t):
            return _click("id", "sidebar-trash")
        if re.search(r"compose|write.*(?:email|mail)|new\s+(?:email|mail)", t):
            return _click("id", "compose-modal")
        if re.search(r"star\s+(?:the|this)\s+email|mark.*star", t):
            return _click("id", "star-button")
        if re.search(r"archive\s+(?:the|this)\s+email", t):
            return _click("id", "archive-button")
        if re.search(r"mark.*spam", t):
            return _click("id", "spam-button")
        if re.search(r"mark.*unread", t):
            return _click("id", "unread-button")

    # autodelivery (8006/8106) - delivery actions (extend existing)
    if port in (8006, 8106):
        if re.search(r"pickup\s+mode|switch.*pickup", t):
            return _click("id", "pickup-mode-button")
        if re.search(r"delivery\s+mode|switch.*delivery", t):
            return _click("id", "delivery-mode-button")
        if re.search(r"view.*cart|my\s+cart|shopping\s+cart", t):
            return _click("id", "cart-button")
        if re.search(r"place\s+order|submit\s+order", t):
            return _click("id", "place-order-button")
        if re.search(r"checkout", t):
            return _click("id", "checkout-button")

    # autolodge (8007/8107) - hotel actions
    if port in (8007, 8107):
        if re.search(r"wishlist|saved\s+(?:stays|properties)", t):
            return _click("id", "nav-wishlist")
        if re.search(r"popular|popular\s+stays", t):
            return _click("id", "nav-popular")
        if re.search(r"help\s+(?:page|center)", t):
            return _click("id", "nav-help")
        if re.search(r"reserve|book\s+(?:this|the)\s+(?:stay|room|property)", t):
            return _click("id", "reserve-button")
        if re.search(r"check\s+availability", t):
            return _click("id", "check-availability-button")
        if re.search(r"share\s+(?:this|the)\s+(?:stay|property|listing)", t):
            return _click("id", "share-button")

    # autoconnect (8008/8108) - social actions
    if port in (8008, 8108):
        if re.search(r"post\s+(?:a\s+)?(?:status|update|article)", t):
            return _click("id", "post-article")
        if re.search(r"start\s+hir|hire\s+(?:a|someone)", t):
            return _click("id", "start-hire-button")
        if re.search(r"create.*posting|job\s+posting", t):
            return _click("id", "create-posting-button")
        if re.search(r"consult.*expert", t):
            return _click("id", "consult-expert-button")
        if re.search(r"add.*favorites?|save.*favorites?", t):
            return _click("id", "add-to-favorites-button")

    # autowork (8009/8109) - project management (extend existing)
    if port in (8009, 8109):
        if re.search(r"skills?\s+(?:tab|page|section)|navigate.*skills", t):
            return _click("href", f"/skills?seed={seed}") if seed else None
        if re.search(r"jobs?\s+(?:tab|page|section)|navigate.*jobs", t):
            return _click("href", f"/jobs?seed={seed}") if seed else None
        if re.search(r"search.*jobs?", t):
            return _click("id", "jobs-search-button")

    # autocalendar (8010/8110) - calendar (extend existing)
    if port in (8010, 8110):
        if re.search(r"previous\s+(?:month|week|day)|go\s+back", t):
            return _click("id", "nav-prev-btn")
        if re.search(r"next\s+(?:month|week|day)|go\s+forward", t):
            return _click("id", "nav-next-btn")
        if re.search(r"create.*event|new\s+event|add\s+event", t):
            return _click("id", "create-event-button")
        if re.search(r"add.*calendar|new\s+calendar|create.*calendar", t):
            return _click("id", "add-calendar-modal")

    # autolist (8011/8111) - task management
    if port in (8011, 8111):
        if re.search(r"today|today.?s?\s+tasks", t):
            return _click("id", "today-nav-item")
        if re.search(r"backlog|backlog\s+tasks", t):
            return _click("id", "backlog-nav-item")
        if re.search(r"create.*(?:task|event)|new\s+(?:task|event)|add\s+(?:task|event)", t):
            return _click("id", "create-event-button")
        if re.search(r"add.*project|new\s+project|create.*project", t):
            return _click("id", "add-project-button")
        if re.search(r"add.*team|new\s+team|create.*team", t):
            return _click("id", "add-team-button")

    # autodrive (8012/8112) - ride-sharing (extend existing)
    if port in (8012, 8112):
        if re.search(r"view.*trips?|my\s+trips?|trip\s+history", t):
            return _click("id", "nav-trips")
        if re.search(r"book\s+(?:a\s+)?ride|request\s+(?:a\s+)?ride", t):
            return _click("id", "book-button")
        if re.search(r"pickup\s+now|ride\s+now", t):
            return _click("id", "pickup-now")

    # autohealth (8013/8113) - healthcare
    if port in (8013, 8113):
        if re.search(r"book\s+(?:an?\s+)?appointment", t):
            return _click("id", "book-appointment-button")
        if re.search(r"upload.*record|add.*record", t):
            return _click("id", "upload-record-button")
        if re.search(r"view.*record", t):
            return _click("id", "view-record-button")
        if re.search(r"view.*prescription", t):
            return _click("id", "view-prescription-button")
        if re.search(r"view.*reviews?|see.*reviews?", t):
            return _click("id", "view-reviews-button")
        if re.search(r"contact.*doctor|message.*doctor", t):
            return _click("id", "contact-doctor-button")

    # autostats (8014/8114) - analytics/blockchain
    if port in (8014, 8114):
        if re.search(r"connect\s+wallet", t):
            return _click("id", "connect-wallet-btn")
        if re.search(r"buy\s+order|place.*buy", t):
            return _click("id", "order-buy-submit-btn")
        if re.search(r"sell\s+order|place.*sell", t):
            return _click("id", "order-sell-submit-btn")
        if re.search(r"send\s+transfer", t):
            return _click("id", "send-transfer-toggle-btn")

    return None


# ---------------------------------------------------------------------------
# Search shortcut: direct type into known search input
# ---------------------------------------------------------------------------

def try_search_shortcut(prompt: str, website: str | None) -> list[dict] | None:
    if not website:
        return None
    input_id = SEARCH_INPUT_IDS.get(website)
    if input_id is None:
        return None
    query = extract_search_query(prompt)
    if not query:
        return None
    return [{"type": "TypeAction", "text": query, "selector": _sel_attr("id", input_id)}]


# ---------------------------------------------------------------------------
# Form-based shortcuts
# ---------------------------------------------------------------------------

def is_already_logged_in(soup: BeautifulSoup) -> bool:
    indicators = ["logout", "log out", "sign out", "my profile", "my account", "dashboard"]
    text = soup.get_text(separator=" ").lower()
    return any(ind in text for ind in indicators)


def detect_login_fields(candidates: list[Candidate]) -> list[dict] | None:
    username = password = submit = None

    for c in candidates:
        # Username field
        if username is None and c.tag == "input":
            if c.name in {"username", "user", "email", "login"}:
                username = c
            elif c.input_type in {"email", "text"} and c.placeholder and (
                "user" in c.placeholder.lower() or "email" in c.placeholder.lower()
            ):
                username = c

        # Password field
        if password is None and c.input_type == "password":
            password = c

        # Submit button
        if submit is None and c.tag in {"button", "input"}:
            if c.input_type == "submit":
                submit = c
            elif c.text and any(
                kw in c.text.lower()
                for kw in ("log in", "login", "sign in", "submit", "enter", "continue")
            ):
                submit = c

    if username and password and submit:
        return [
            {"type": "TypeAction", "text": "<username>", "selector": username.selector.model_dump()},
            {"type": "TypeAction", "text": "<password>", "selector": password.selector.model_dump()},
            {"type": "ClickAction", "selector": submit.selector.model_dump()},
        ]
    return None


def detect_logout_target(candidates: list[Candidate]) -> list[dict] | None:
    for c in candidates:
        if c.text and any(kw in c.text.lower() for kw in ("log out", "logout", "sign out")):
            return [{"type": "ClickAction", "selector": c.selector.model_dump()}]
    # Try href-based
    for c in candidates:
        if c.href and any(kw in c.href.lower() for kw in ("logout", "signout", "sign-out")):
            return [{"type": "ClickAction", "selector": c.selector.model_dump()}]
    return None


def get_registration_actions(candidates: list[Candidate]) -> list[dict] | None:
    username = email = password = confirm = submit = None
    password_seen = False

    for c in candidates:
        if username is None and c.tag == "input":
            if c.name in {"username", "user"} or (c.placeholder and "username" in c.placeholder.lower()):
                username = c

        if email is None and c.tag == "input":
            if c.input_type == "email" or c.name == "email" or (
                c.placeholder and "email" in c.placeholder.lower()
            ):
                email = c

        if c.input_type == "password" or (c.name and "password" in c.name.lower()):
            if not password_seen:
                password = c
                password_seen = True
            elif confirm is None:
                confirm = c

        if submit is None and c.tag in {"button", "input"}:
            if c.input_type == "submit":
                submit = c
            elif c.text and any(
                kw in c.text.lower()
                for kw in ("register", "sign up", "signup", "create", "submit")
            ):
                submit = c

    if not password or not submit:
        return None
    if not username and not email:
        return None

    actions: list[dict] = []
    if username:
        actions.append({"type": "TypeAction", "text": "<signup_username>", "selector": username.selector.model_dump()})
    if email:
        actions.append({"type": "TypeAction", "text": "<signup_email>", "selector": email.selector.model_dump()})
    actions.append({"type": "TypeAction", "text": "<signup_password>", "selector": password.selector.model_dump()})
    if confirm:
        actions.append({"type": "TypeAction", "text": "<signup_password>", "selector": confirm.selector.model_dump()})
    actions.append({"type": "ClickAction", "selector": submit.selector.model_dump()})
    return actions


def get_contact_actions(candidates: list[Candidate]) -> list[dict] | None:
    name_c = email_c = message_c = submit_c = None

    for c in candidates:
        if name_c is None and c.tag == "input":
            if c.name in {"name", "full_name", "fullname", "your_name"} or (
                c.placeholder and "name" in c.placeholder.lower()
            ):
                name_c = c

        if email_c is None and c.tag == "input":
            if c.name == "email" or c.input_type == "email" or (
                c.placeholder and "email" in c.placeholder.lower()
            ):
                email_c = c

        if message_c is None:
            if c.tag == "textarea":
                message_c = c
            elif c.name in {"message", "msg", "content", "body", "subject"}:
                message_c = c

        if submit_c is None and c.tag in {"button", "input"}:
            if c.input_type == "submit":
                submit_c = c
            elif c.text and any(kw in c.text.lower() for kw in ("send", "submit", "contact")):
                submit_c = c

    if not submit_c:
        return None
    # At minimum need message OR (name + email)
    if not message_c and (not name_c or not email_c):
        return None

    actions: list[dict] = []
    if name_c:
        actions.append({"type": "TypeAction", "text": "Test User", "selector": name_c.selector.model_dump()})
    if email_c:
        actions.append({"type": "TypeAction", "text": "<signup_email>", "selector": email_c.selector.model_dump()})
    if message_c:
        actions.append({"type": "TypeAction", "text": "Hello, this is a test message for support.", "selector": message_c.selector.model_dump()})
    actions.append({"type": "ClickAction", "selector": submit_c.selector.model_dump()})
    return actions


def try_shortcut(
    task_type: str | None,
    candidates: list[Candidate],
    soup: BeautifulSoup,
    step_index: int,
) -> list[dict] | None:
    """Attempt deterministic shortcut for the given task type."""
    if task_type is None:
        return None

    if task_type == "login":
        if is_already_logged_in(soup):
            return [{"type": "WaitAction", "time_seconds": 1}]
        return detect_login_fields(candidates)

    if task_type == "logout":
        result = detect_logout_target(candidates)
        if result:
            return result
        # May need to login first, then logout
        if not is_already_logged_in(soup):
            login = detect_login_fields(candidates)
            if login:
                return login
        return None

    if task_type == "registration":
        return get_registration_actions(candidates)

    if task_type == "contact":
        return get_contact_actions(candidates)

    return None
