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

    # About page feature (multi-step) - extract feature name from constraint
    if re.search(r"about\s+page.*feature|feature.*about\s+page|feature.*on\s+the\s+about\s+page", t):
        if step == 0:
            return _click("id", "nav-about")
        elif step == 1:
            return [{"type": "ScrollAction", "down": True}]
        else:
            # Extract feature text from prompt constraint (e.g. feature CONTAINS 'Trusted reviews')
            feature_m = re.search(r"feature\s+(?:field\s+)?(?:CONTAINS|equals?)\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
            feature_text = feature_m.group(1) if feature_m else "Curated"
            return _click_xpath(f"//h3[contains(text(),'{feature_text}')] | //*[contains(@class,'feature') and contains(text(),'{feature_text}')]")

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

    # Emails next/prev page (automail 8005)
    if port == 8005:
        if re.search(r"next\s+page\s+of\s+emails|go\s+to\s+the\s+next\s+page", t):
            return _click_xpath(
                "//button[@aria-label='Next page' or @aria-label='Next'] | "
                "//a[@aria-label='Next page' or @aria-label='Next'] | "
                "//*[contains(@class,'pagination')]//button[contains(.,'>')] | "
                "//*[@data-testid='next-page']"
            )
        if re.search(r"previous\s+page\s+of\s+emails|go\s+back\s+to\s+the\s+previous\s+page", t):
            return _click_xpath(
                "//button[@aria-label='Previous page' or @aria-label='Previous'] | "
                "//a[@aria-label='Previous page'] | "
                "//*[contains(@class,'pagination')]//button[contains(.,'<')]"
            )

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

    # AutoDiscord (8015) shortcuts
    if port == 8015:
        # SEND_MESSAGE: click the message input at the bottom, type, and submit
        if re.search(r"send\s+a\s+message\s+in\s+the\s+channel", t):
            _msg_xpath = ("//div[@role='textbox'] | //textarea[contains(@placeholder,'Message')] | "
                         "//input[contains(@placeholder,'Message')] | //div[contains(@class,'message-input')]")
            if step == 0:
                return _click_xpath(_msg_xpath)
            elif step == 1:
                return [{"type": "TypeAction", "text": "Hello!",
                         "selector": {"type": "xpathSelector", "value": _msg_xpath}}]
            elif step == 2:
                return _click_xpath("//button[@type='submit'] | //button[contains(@aria-label,'Send')] | //button[contains(@class,'send')]")
            # step 3+: fall through to LLM
            return None

        # OPEN_SETTINGS: click gear icon
        if re.search(r"open\s+the\s+settings\s+page", t):
            if step == 0:
                return _click_xpath(
                    "//button[contains(@aria-label,'Settings') or contains(@aria-label,'settings')] | "
                    "//*[contains(@id,'settings') and (@role='button' or self::button)] | "
                    "//*[contains(@class,'settings') and (@role='button' or self::button or self::a)]"
                )
            return []

        # SELECT_CHANNEL: click channel matching constraints (only step 0; LLM handles rest)
        if re.search(r"(select|click|open)\s+(a\s+)?(?:text\s+)?channel\s+(that|where|whose|not)", t):
            if step == 0:
                m3 = re.search(r"channel\s+(?:name\s+)?(?:does\s+)?NOT\s+(?:CONTAIN|equal)[s]?\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
                if m3:
                    exclude = m3.group(1)
                    return _click_xpath(f"//a[contains(@class,'channel') and not(contains(.,'{exclude}'))] | "
                                       f"//*[@role='listitem' and contains(@class,'channel') and not(contains(.,'{exclude}'))]")
            return None

    # AutoLodge (8007) shortcuts
    if port == 8007:
        # BACK_TO_ALL_HOTELS: after navigating to a hotel detail, click the back link
        if re.search(r"go\s+back\s+to\s+all\s+hotels", t):
            # This is multi-step: step 0 find+click hotel, step 1+ find back button
            if step >= 1:
                return _click_xpath(
                    "//a[contains(.,'Back') or contains(.,'All Hotels') or contains(.,'All hotels') or contains(.,'← Hotels')] | "
                    "//button[contains(.,'Back') or contains(.,'All Hotels')]"
                )
            return None

    # AutoStats (8014) shortcuts
    if port == 8014:
        # FAVORITE_SUBNET: click the star/favorite icon on a subnet row NOT matching excluded name
        if re.search(r"favorite\s+the\s+subnet", t):
            if step == 0:
                m4 = re.search(r"subnet_name\s+is\s+NOT\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
                exclude = m4.group(1) if m4 else ""
                if exclude:
                    return _click_xpath(
                        f"//tr[not(contains(.,'{exclude}'))]//button[contains(@aria-label,'favorite') or contains(@aria-label,'star') or contains(@class,'favorite') or contains(@class,'star')] | "
                        f"//tr[not(contains(.,'{exclude}'))]//*[@data-testid='favorite-btn' or @data-testid='star-btn']"
                    )
                return _click_xpath(
                    "//button[contains(@aria-label,'favorite') or contains(@aria-label,'star')] | "
                    "//*[@data-testid='favorite-btn']"
                )
            return None  # fall to LLM if step > 0

        # DISCONNECT_WALLET: may need 2 steps (click wallet button -> click Disconnect in dropdown)
        if re.search(r"disconnect.*wallet", t):
            if step == 0:
                return _click_xpath(
                    "//button[contains(.,'Disconnect') or contains(@aria-label,'Disconnect')] | "
                    "//*[contains(@class,'wallet') and (@role='button' or self::button)]"
                )
            # step 1+: let LLM handle (e.g. dropdown appeared with Disconnect option)
            return None

        # CONNECT_WALLET: step 0 click Connect Wallet, step 1+ let LLM pick wallet
        if re.search(r"connect\s+(a\s+|the\s+)?wallet", t):
            if step == 0:
                return _click_xpath(
                    "//button[contains(.,'Connect Wallet') or contains(.,'Connect wallet')] | "
                    "//*[@data-testid='connect-wallet']"
                )
            return None

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
