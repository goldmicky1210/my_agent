"""Central configuration: site knowledge, playbooks, constants."""
from __future__ import annotations
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Port → project mapping (IWA sandbox)
# ---------------------------------------------------------------------------
PORT_TO_PROJECT: dict[int, str] = {
    8000: "autocinema",
    8001: "autobooks",
    8002: "autozone",
    8003: "autodining",
    8004: "autocrm",
    8005: "automail",
    8006: "autodelivery",
    8007: "autolodge",
    8008: "autoconnect",
    8009: "autowork",
    8010: "autocalendar",
    8011: "autolist",
    8012: "autoride",
    8013: "autohealth",
    8014: "autostats",
    8015: "autodiscord",
}


def detect_website(url: str) -> str | None:
    port = urlsplit(url).port
    return PORT_TO_PROJECT.get(port) if port else None


# ---------------------------------------------------------------------------
# Selector priority (stable → fragile)
# ---------------------------------------------------------------------------
SELECTOR_PRIORITY: list[str] = [
    "id", "data-testid", "href", "aria-label", "name",
    "placeholder", "title", "text",
]

# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.15
LLM_MAX_TOKENS = 450
PAGE_IR_MAX_TOKENS = 1500
PAGE_IR_CHAR_LIMIT = PAGE_IR_MAX_TOKENS * 4

# ---------------------------------------------------------------------------
# Agent limits
# ---------------------------------------------------------------------------
AGENT_MAX_STEPS = 12
MAX_TASK_STATES = 8

# ---------------------------------------------------------------------------
# Per-website hints (domain knowledge injected into prompts)
# ---------------------------------------------------------------------------
WEBSITE_HINTS: dict[str, str] = {
    "autocinema": (
        "Movie booking site. Genre filters, movie cards with title/year/genre/rating, "
        "showtime selection, seat picker. Login required for watchlist/comments. "
        "Search bar at top. Film detail pages at /film/<id>."
    ),
    "autobooks": (
        "Online bookstore. Search bar, genre filters, book cards with title/author/price. "
        "Login required for reading list/comments/admin. Detail pages at /book/<id>."
    ),
    "autozone": (
        "E-commerce shop. Product cards with prices, category filters, search bar, "
        "shopping cart, wishlist. Checkout flow: cart → address → payment → confirm."
    ),
    "autodining": (
        "Restaurant discovery. Search, filter by cuisine/rating/price, restaurant cards "
        "with name/cuisine/rating. Review submission, table reservation. Detail pages."
    ),
    "autocrm": (
        "Legal/law firm CRM. Matters management (add/edit/delete/update matters). "
        "Time logging (add/edit/delete log entries). Document management. "
        "Calendar events. Login required for all actions."
    ),
    "automail": (
        "Email client. Inbox list with sender/subject/date. Email actions: star, archive, "
        "delete, forward, reply. Compose new, templates, pagination. Drafts/Sent/Spam folders."
    ),
    "autodelivery": (
        "Food delivery. Restaurant list with pagination, menu items, cart management, "
        "address selection, checkout. Search by restaurant name or food type."
    ),
    "autolodge": (
        "Hotel booking site. Search by location/dates, hotel cards with name/price/rating, "
        "guest selector, room types, payment methods, reviews, wishlist."
    ),
    "autoconnect": (
        "Social/professional network. Posts feed, jobs board, profiles, company pages, "
        "connections. Like buttons have IDs 'post_like_button_p<N>'. Profile at /profile/alexsmith."
    ),
    "autowork": (
        "Jobs/expert hiring platform. Browse experts with name/role/country/rating. "
        "Book consultations, favorite experts, hire later. Job search by title. "
        "Navbar has: jobs, hires, profile. "
        "For 'contact an expert': scroll down to expert cards, click Consult. "
        "Do NOT navigate directly -- URL slugs are unpredictable."
    ),
    "autocalendar": (
        "Calendar app. Day/week/month/5-day views. Event creation wizard with title, "
        "date/time, attendees. Search events. 'focus-today' button for today's view."
    ),
    "autolist": (
        "Task/todo manager. Task lists, team management, priority setting, "
        "task creation/deletion. Boards and list views."
    ),
    "autoride": (
        "Ride-hailing/transport service. Browse rides with name/location/destination/schedule. "
        "Enter/search locations, select times, reserve rides, cancel reservations. "
        "Next pickup scheduling. Location search by address."
    ),
    "autohealth": (
        "Medical portal. Doctor profiles with specialty/rating, appointment booking, "
        "medical analysis, contact forms. Search doctors by specialty."
    ),
    "autostats": (
        "Blockchain/Bittensor explorer. Subnets list with volume/name. Validators with rank/weight. "
        "Blocks with epoch/extrinsics. Wallet connect/disconnect. Favorite subnets. "
        "Browse and filter by various metrics."
    ),
    "autodiscord": (
        "Chat application. Server list, channels, messages, user search, "
        "server management. Channel messages in main panel."
    ),
}

# ---------------------------------------------------------------------------
# Task playbooks (step-by-step guidance per task type)
# ---------------------------------------------------------------------------
TASK_PLAYBOOKS: dict[str, str] = {
    "login": (
        "1) Find username/email field, type credentials. "
        "2) Find password field, type password. 3) Click submit/login button."
    ),
    "logout": (
        "Find logout/sign-out link (often in nav, header, or profile dropdown). Click it. "
        "If not visible, check for a user menu or hamburger icon first."
    ),
    "registration": (
        "1) Find registration form. 2) Fill username, email, password (and confirm). "
        "3) Click register/sign-up/create-account button. "
        "Use EXACT values from constraints for each field."
    ),
    "contact": (
        "1) Find contact/support form. 2) Fill name, email, subject, message fields. "
        "3) Click send/submit. Use constraint values or valid alternatives for NOT constraints."
    ),
    "login_then_action": (
        "1) Login first (username → password → submit). 2) After redirect, "
        "navigate to target section. 3) Complete the main action from the task. "
        "4) For constraint-based items, scroll through to find the matching one."
    ),
    "search": (
        "1) Find search input/bar. 2) Type the EXACT query from constraints. "
        "3) Submit search. For 'not equal' queries, type any valid alternative."
    ),
    "navigate_detail": (
        "1) Look at visible items/cards. 2) Check each item's text against ALL constraints. "
        "3) Click 'View Details' or the item link for the FIRST match. "
        "4) If no match visible, scroll down to load more items."
    ),
    "filter": (
        "1) Find filter controls (dropdowns, checkboxes). "
        "2) Select criteria matching constraints. 3) Apply filter."
    ),
    "purchase": (
        "1) Find the target item matching constraints. 2) Click Add to Cart. "
        "3) Go to cart/checkout. 4) Fill any required fields. 5) Confirm order."
    ),
    "form_fill": (
        "1) Identify each form field. 2) For EQUALS: type exact value. "
        "For NOT EQUALS: type any other valid value. For CONTAINS: include the substring. "
        "3) Submit the form."
    ),
    "dropdown_select": (
        "1) Find the dropdown/select element. 2) Click to open. "
        "3) Choose option matching constraints (exact match or exclude as needed)."
    ),
    "data_retrieval": (
        "1) Browse visible items checking constraints. 2) Click matching item for details. "
        "3) If not found, scroll down or navigate to find more items."
    ),
    "edit": (
        "1) Find the item matching constraints. 2) Click Edit. "
        "3) Update specified fields with exact constraint values. 4) Save."
    ),
    "delete": (
        "1) Find the item matching constraints. 2) Click Delete/Remove. "
        "3) Confirm if prompted."
    ),
    "general": (
        "1) Read page elements and task carefully. 2) Match constraints to visible items. "
        "3) Choose the most direct action. 4) If stuck, scroll to find more elements."
    ),
}

# ---------------------------------------------------------------------------
# Search input IDs per website (for search shortcuts)
# ---------------------------------------------------------------------------
SEARCH_INPUT_IDS: dict[str, str] = {
    "automail": "mail-search",
    "autocinema": "input",
    "autodining": "search-field",
    "autodelivery": "food-search",
    "autobooks": "input",
    "autozone": "input",
    "autoconnect": "input",
    "autohealth": "input",
    "autoride": "input",
}

# ---------------------------------------------------------------------------
# Known element IDs for quick-click shortcuts
# ---------------------------------------------------------------------------
QUICK_CLICK_IDS: dict[str, str] = {
    "focus_today": "focus-today",
    "new_event": "new-event-cta",
    "add_team": "add-team-btn",
    "wishlist": "favorite-action",
    "spotlight_movie": "spotlight-view-details-btn",
    "featured_book": "featured-book-view-details-btn-1",
    "featured_product": "view-details",
    "nav_about": "nav-about",
}
