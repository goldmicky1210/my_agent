"""Task type classification from natural-language prompts.

Expanded to 150+ specific task types across all 14 IWA websites,
matching the competitor's granularity for precise playbook selection.
"""
from __future__ import annotations
import re


def classify_task_type(prompt: str) -> str:
    """Return a specific task type label for the given prompt."""
    t = prompt.lower()

    # ---- AutoRide (8012) ----
    if re.search(r"(enter|type)\s+destination", t, re.IGNORECASE):
        return "ENTER_DESTINATION"
    if re.search(r"destination\s+(value\s+)?that\s+is\s+NOT", t, re.IGNORECASE):
        return "ENTER_DESTINATION"
    if re.search(r"enter\s+(and\s+select\s+)?a\s+location", t, re.IGNORECASE):
        return "ENTER_LOCATION"
    if re.search(r"location\s+equals\s+['\"]", t, re.IGNORECASE):
        return "ENTER_LOCATION"
    if re.search(r"search\s+ride\s+(details\s+)?where\s+the\s+location", t, re.IGNORECASE):
        return "SEARCH_RIDE"
    if re.search(r"(search|search\s+for)\s+.*location\s+.*destination", t, re.IGNORECASE):
        return "SEARCH_LOCATION"
    if re.search(r"search\s+location\s+(details|to\s+find|for)", t, re.IGNORECASE):
        return "SEARCH_LOCATION"
    if re.search(r"destination\s+equals\s+", t, re.IGNORECASE):
        return "SEARCH_LOCATION"
    if re.search(r"(reserve|book)\s+.*ride", t, re.IGNORECASE):
        return "RESERVE_RIDE"
    if re.search(r"cancel\s+reservation", t, re.IGNORECASE):
        return "CANCEL_RESERVATION"
    if re.search(r"select\s+(a\s+)?date\s+for\s+(the\s+|your\s+)?trip", t, re.IGNORECASE):
        return "SELECT_DATE"
    if re.search(r"select\s+(a\s+)?time\s+for\s+(my\s+|your\s+)?trip", t, re.IGNORECASE):
        return "SELECT_TIME"
    if re.search(r"select\s+time\s+for\s+my\s+trip", t, re.IGNORECASE):
        return "SELECT_TIME"
    if re.search(r"next\s+pickup", t, re.IGNORECASE):
        return "NEXT_PICKUP"

    # ---- AutoMail (8005) ----
    if re.search(r"mark\s+as\s+spam", t, re.IGNORECASE):
        return "MARK_AS_SPAM"
    if re.search(r"(mark|move)\s+.*(spam|junk)", t, re.IGNORECASE):
        return "MARK_AS_SPAM"
    if re.search(r"star\s+the\s+email", t, re.IGNORECASE):
        return "STAR_AN_EMAIL"
    if re.search(r"archive\s+the\s+email", t, re.IGNORECASE):
        return "ARCHIVE_EMAIL"
    if re.search(r"delete\s+the\s+email", t, re.IGNORECASE):
        return "DELETE_EMAIL"
    if re.search(r"forward\s+the\s+email", t, re.IGNORECASE):
        return "FORWARD_EMAIL"
    if re.search(r"mark.*email.*important|mark.*important.*email", t, re.IGNORECASE):
        return "MARK_EMAIL_AS_IMPORTANT"
    if re.search(r"mark\s+(the\s+)?email\s+as\s+unread", t, re.IGNORECASE):
        return "MARK_AS_UNREAD"
    if re.search(r"view\s+the\s+email\s+where", t, re.IGNORECASE):
        return "VIEW_EMAIL"
    if re.search(r"change\s+the\s+application\s+theme", t, re.IGNORECASE):
        return "THEME_CHANGED"
    if re.search(r"edit.*draft.*email", t, re.IGNORECASE):
        return "EDIT_DRAFT_EMAIL"
    if re.search(r"(next|go\s+to\s+the\s+next)\s+page\s+of\s+emails", t, re.IGNORECASE):
        return "EMAILS_NEXT_PAGE"
    if re.search(r"(previous|go\s+back\s+to\s+the\s+previous)\s+page\s+of\s+emails", t, re.IGNORECASE):
        return "EMAILS_PREV_PAGE"
    if re.search(r"(clear|deselect)\s+all\s+selected\s+emails", t, re.IGNORECASE):
        return "CLEAR_SELECTION"
    if re.search(r"send\s+.*using\s+the\s+template", t, re.IGNORECASE):
        return "TEMPLATE_SENT"
    if re.search(r"send\s+an\s+email\s+using\s+the\s+template", t, re.IGNORECASE):
        return "TEMPLATE_SENT"
    if re.search(r"save.*template.*draft", t, re.IGNORECASE):
        return "TEMPLATE_SAVED_DRAFT"
    if re.search(r"select\s+the\s+template", t, re.IGNORECASE):
        return "TEMPLATE_SELECTED"

    # ---- AutoCalendar (8010) ----
    if re.search(r"switch\s+to\s+week\s+view", t, re.IGNORECASE):
        return "SELECT_WEEK"
    if re.search(r"switch\s+to\s+month\s+view", t, re.IGNORECASE):
        return "SELECT_MONTH"
    if re.search(r"switch\s+to\s+day\s+view", t, re.IGNORECASE):
        return "SELECT_DAY"
    if re.search(r"switch\s+to\s+5.?day\s+view", t, re.IGNORECASE):
        return "SELECT_FIVE_DAYS"
    if re.search(r"(add\s+|click.*)\s*add\s+calendar\s+button", t, re.IGNORECASE):
        return "ADD_NEW_CALENDAR"
    if re.search(r"create\s+a\s+new\s+calendar", t, re.IGNORECASE):
        return "CREATE_CALENDAR"
    if re.search(r"add\s+an?\s+attendee\s+to\s+the\s+event", t, re.IGNORECASE):
        return "EVENT_ADD_ATTENDEE"
    if re.search(r"remove\s+an?\s+attendee\s+from\s+the\s+event", t, re.IGNORECASE):
        return "EVENT_REMOVE_ATTENDEE"
    if re.search(r"delete\s+an?\s+added\s+event", t, re.IGNORECASE):
        return "DELETE_ADDED_EVENT"
    if re.search(r"cancel\s+an?\s+event", t, re.IGNORECASE):
        return "CANCEL_ADD_EVENT"
    if re.search(r"open\s+the\s+event\s+creation\s+wizard", t, re.IGNORECASE):
        return "EVENT_WIZARD_OPEN"
    if re.search(r"click\s+on\s+cell\s+for\s+a\s+date", t, re.IGNORECASE):
        return "CELL_CLICKED"
    if re.search(r"click.*cell.*in\s+the\s+5\s+days\s+view", t, re.IGNORECASE):
        return "CELL_CLICKED"
    if re.search(r"add\s+a\s+new\s+calendar\s+event", t, re.IGNORECASE):
        return "NEW_CALENDAR_EVENT_ADDED"
    if re.search(r"add\s+an?\s+event\b", t, re.IGNORECASE):
        return "ADD_EVENT"
    if re.search(r"(show|view)\s+.*pending\s+events", t, re.IGNORECASE):
        return "VIEW_PENDING_EVENTS"
    if re.search(r"show\s+me\s+results\s+for\s+a\s+search\s+query", t, re.IGNORECASE):
        return "SEARCH_SUBMIT"

    # ---- AutoList (8011) ----
    if re.search(r"add\s+members?\s+to\s+the\s+team", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_MEMBERS_ADDED"
    if re.search(r"assign\s+a\s+role\s+.*team\s+member", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_ROLE_ASSIGNED"
    if re.search(r"edit\s+task\s+modal\s+open", t, re.IGNORECASE):
        return "AUTOLIST_EDIT_TASK_MODAL_OPENED"
    if re.search(r"button\s+to\s+add\s+a\s+task\s+is\s+clicked", t, re.IGNORECASE):
        return "AUTOLIST_ADD_TASK_CLICKED"
    if re.search(r"change\s+the\s+priority\s+to", t, re.IGNORECASE):
        return "AUTOLIST_SELECT_TASK_PRIORITY"
    if re.search(r"cancel\s+creating\s+the\s+task", t, re.IGNORECASE):
        return "AUTOLIST_CANCEL_TASK_CREATION"
    if re.search(r"create\s+a\s+team\s+whose", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_CREATED"
    if re.search(r"delete\s+task\s+whose", t, re.IGNORECASE):
        return "AUTOLIST_DELETE_TASK"
    if re.search(r"add\s+a\s+task\s+whose", t, re.IGNORECASE):
        return "AUTOLIST_TASK_ADDED"
    if re.search(r"add\s+a\s+task\s+where", t, re.IGNORECASE):
        return "AUTOLIST_TASK_ADDED"

    # ---- AutoMedic (8013) ----
    if re.search(r"(show|retrieve)\s+details\s+(for\s+a\s+doctor|of\s+the\s+doctor\s+education|of\s+a\s+doctor)", t, re.IGNORECASE):
        if re.search(r"education|certif", t, re.IGNORECASE):
            return "VIEW_DOCTOR_EDUCATION"
        if re.search(r"availab", t, re.IGNORECASE):
            return "VIEW_DOCTOR_AVAILABILITY"
        return "VIEW_DOCTOR_PROFILE"
    if re.search(r"show\s+details\s+for\s+a\s+doctor", t, re.IGNORECASE):
        return "VIEW_DOCTOR_PROFILE"
    if re.search(r"retrieve\s+details\s+of\s+the\s+doctor\s+education", t, re.IGNORECASE):
        return "VIEW_DOCTOR_EDUCATION"
    if re.search(r"show\s+me\s+the\s+availability\s+details\s+for\s+a\s+doctor", t, re.IGNORECASE):
        return "VIEW_DOCTOR_AVAILABILITY"
    if re.search(r"show\s+me\s+(details\s+(about|for)\s+(a\s+)?)?doctors?", t, re.IGNORECASE):
        return "SEARCH_DOCTORS"
    if re.search(r"show\s+details\s+for\s+a\s+doctor\s+where", t, re.IGNORECASE):
        return "VIEW_DOCTOR_PROFILE"
    if re.search(r"(search|retrieve)\s+(medical|details\s+of\s+medical)", t, re.IGNORECASE):
        return "SEARCH_MEDICAL_ANALYSIS"
    if re.search(r"view\s+medical\s+analysis", t, re.IGNORECASE):
        return "VIEW_MEDICAL_ANALYSIS"
    if re.search(r"open\s+appointment\s+form", t, re.IGNORECASE):
        return "OPEN_APPOINTMENT_FORM"
    if re.search(r"open\s+contact\s+doctor\s+form", t, re.IGNORECASE):
        return "OPEN_CONTACT_DOCTOR_FORM"
    if re.search(r"contact\s+a\s+doctor\s+where", t, re.IGNORECASE):
        return "DOCTOR_CONTACTED_SUCCESSFULLY"
    if re.search(r"contact\s+(a\s+)?doctor", t, re.IGNORECASE):
        return "CONTACT_DOCTOR"
    if re.search(r"retrieve\s+details\s+of\s+appointments|show\s+me\s+appointments\s+where", t, re.IGNORECASE):
        return "SEARCH_APPOINTMENT"
    if re.search(r"request\s+a\s+quick\s+appointment", t, re.IGNORECASE):
        return "REQUEST_QUICK_APPOINTMENT"
    if re.search(r"(book|schedule)\s+(an?\s+)?appointment\s+(with|where|for)", t, re.IGNORECASE):
        return "APPOINTMENT_BOOKED_SUCCESSFULLY"
    if re.search(r"doctor.*education|education.*doctor", t, re.IGNORECASE):
        return "VIEW_DOCTOR_EDUCATION"

    # ---- AutoDiscord (8015) ----
    if re.search(r"send\s+a\s+message\s+in\s+the\s+channel", t, re.IGNORECASE):
        return "SEND_MESSAGE"
    if re.search(r"open\s+the\s+settings\s+page", t, re.IGNORECASE):
        return "OPEN_SETTINGS"
    if re.search(r"join\s+(a\s+|the\s+)?voice\s+channel", t, re.IGNORECASE):
        return "JOIN_VOICE_CHANNEL"
    if re.search(r"(select|click|open)\s+(a\s+)?(?:text\s+)?channel\s+(that|where|whose|not)", t, re.IGNORECASE):
        return "SELECT_CHANNEL"
    if re.search(r"select\s+the\s+dm\s+(that|where|from|with)", t, re.IGNORECASE):
        return "SELECT_DM"
    if re.search(r"(view|open|go\s+to)\s+(your\s+|the\s+)?dms?\b", t, re.IGNORECASE):
        return "VIEW_DMS"
    if re.search(r"select\s+the\s+server\s+(that|where|whose|named|called)", t, re.IGNORECASE):
        return "SELECT_SERVER"
    if re.search(r"(mute|unmute|toggle\s+mute)", t, re.IGNORECASE):
        return "VOICE_MUTE_TOGGLE"
    if re.search(r"settings.*account\s+tab|account\s+tab.*settings|navigate.*account.*settings", t, re.IGNORECASE):
        return "SETTINGS_ACCOUNT"

    # ---- AutoStats (8014) ----
    if re.search(r"favorite\s+the\s+subnet", t, re.IGNORECASE):
        return "FAVORITE_SUBNET"
    if re.search(r"disconnect\s+the\s+wallet|disconnect.*wallet", t, re.IGNORECASE):
        return "DISCONNECT_WALLET"
    if re.search(r"connect\s+(a\s+|the\s+)?wallet", t, re.IGNORECASE):
        return "CONNECT_WALLET"
    if re.search(r"view\s+(the\s+)?block\s+(with|where|whose|number)", t, re.IGNORECASE):
        return "VIEW_BLOCK"
    if re.search(r"view\s+(the\s+)?subnet\s+(with|where|whose|named|called)", t, re.IGNORECASE):
        return "VIEW_SUBNET"
    if re.search(r"view\s+(the\s+)?validator\s+(with|where|whose|named)", t, re.IGNORECASE):
        return "VIEW_VALIDATOR"

    # ---- AutoConnect (8008) ----
    if re.search(r"comment\s+on\s+the\s+post", t, re.IGNORECASE):
        return "COMMENT_ON_POST"
    if re.search(r"save\s+the\s+post\s+where", t, re.IGNORECASE):
        return "SAVE_POST"
    if re.search(r"follow\s+the\s+company\s+page", t, re.IGNORECASE):
        return "FOLLOW_PAGE"
    if re.search(r"unfollow\s+the\s+company\s+page", t, re.IGNORECASE):
        return "UNFOLLOW_PAGE"
    if re.search(r"(withdraw|cancel)\s+application", t, re.IGNORECASE):
        return "CANCEL_APPLICATION"
    if re.search(r"(search\s+for|show\s+me)\s+users", t, re.IGNORECASE):
        return "SEARCH_USERS"
    if re.search(r"go\s+back\s+to\s+all\s+jobs", t, re.IGNORECASE):
        return "BACK_TO_ALL_JOBS"
    if re.search(r"navigate\s+to\s+the\s+'?home'?\s+tab", t, re.IGNORECASE):
        return "HOME_NAVBAR"
    if re.search(r"show\s+me\s+my\s+hidden\s+posts", t, re.IGNORECASE):
        return "VIEW_HIDDEN_POSTS"
    if re.search(r"search\s+for\s+jobs\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_JOBS"
    if re.search(r"apply\s+for\s+(a\s+)?job", t, re.IGNORECASE):
        return "APPLY_FOR_JOB"
    if re.search(r"edit\s+profile\s+to\s+set\s+the\s+bio", t, re.IGNORECASE):
        return "EDIT_PROFILE_BIO"
    if re.search(r"add\s+(a\s+)?new\s+experience\s+(with|where|at)", t, re.IGNORECASE):
        return "ADD_EXPERIENCE"
    if re.search(r"add\s+(an?\s+)?experience\s+(at|with|where)", t, re.IGNORECASE):
        return "ADD_EXPERIENCE"
    if re.search(r"(edit|update|modify)\s+(an?\s+)?experience\s+(where|with|at)", t, re.IGNORECASE):
        return "EDIT_EXPERIENCE"

    # ---- AutoHire (8009) ----
    if re.search(r"decide\s+to\s+remove\s+expert\s+(from|whose)", t, re.IGNORECASE):
        return "HIRE_LATER_REMOVED"
    if re.search(r"remove\s+expert\s+(whose|from\s+hire\s+later)", t, re.IGNORECASE):
        return "HIRE_LATER_REMOVED"
    if re.search(r"decide\s+to\s+hire\s+later", t, re.IGNORECASE):
        return "HIRE_LATER"
    if re.search(r"hire\s+(a\s+)?(consultant|expert|later)", t, re.IGNORECASE):
        if "later" in t:
            return "HIRE_LATER"
        return "HIRE_BTN_CLICKED"
    if re.search(r"show\s+me\s+details\s+about\s+a\s+hiring\s+team", t, re.IGNORECASE):
        return "SELECT_HIRING_TEAM"
    if re.search(r"select\s+a\s+project\s+size", t, re.IGNORECASE):
        return "CHOOSE_PROJECT_SIZE"
    if re.search(r"closing\s+the\s+job\s+posting\s+window", t, re.IGNORECASE):
        return "CLOSE_POST_A_JOB_WINDOW"
    if re.search(r"clicks?\s+on\s+the\s+jobs?\s+option\s+in\s+the\s+navbar", t, re.IGNORECASE):
        return "NAVBAR_JOBS_CLICK"
    if re.search(r"clicks?\s+on\s+.?hires?.?\s+from\s+the\s+navbar", t, re.IGNORECASE):
        return "NAVBAR_HIRES_CLICK"
    if re.search(r"searches?\s+for\s+a\s+skill", t, re.IGNORECASE):
        return "SEARCH_SKILL"
    if re.search(r"(job\s+posting|writing\s+(a\s+)?(strong\s+)?title\s+of\s+(the\s+)?job)", t, re.IGNORECASE):
        return "WRITE_JOB_TITLE"
    if re.search(r"edit\s+profile\s+about", t, re.IGNORECASE):
        return "EDIT_ABOUT"
    if re.search(r"update\s+my\s+profile\s+about\s+section", t, re.IGNORECASE):
        return "EDIT_ABOUT"
    if re.search(r"edit\s+profile\s+(location|email)", t, re.IGNORECASE):
        if "location" in t:
            return "EDIT_PROFILE_LOCATION"
        return "EDIT_PROFILE_EMAIL"
    if re.search(r"edit\s+profile\s+email", t, re.IGNORECASE):
        return "EDIT_PROFILE_EMAIL"

    # ---- AutoLodge (8007) ----
    if re.search(r"confirm\s+the\s+booking", t, re.IGNORECASE):
        return "BOOKING_CONFIRM"
    if re.search(r"(adjust|set|change)\s+the\s+number\s+of\s+guests", t, re.IGNORECASE):
        return "EDIT_NUMBER_OF_GUESTS"
    if re.search(r"(open\s+)?guest\s+selector\s+dropdown", t, re.IGNORECASE):
        return "PEOPLE_DROPDOWN_OPENED"
    if re.search(r"select\s+(a\s+)?payment\s+method", t, re.IGNORECASE):
        return "PAYMENT_METHOD_SELECTED"
    if re.search(r"(reserve|book)\s+the\s+hotel", t, re.IGNORECASE):
        return "RESERVE_HOTEL"
    if re.search(r"share\s+the\s+hotel\s+listing", t, re.IGNORECASE):
        return "SHARE_HOTEL"
    if re.search(r"show\s+(me\s+)?details\s+for\s+popular\s+hotels", t, re.IGNORECASE):
        return "POPULAR_HOTELS_VIEWED"
    if re.search(r"search\s+for\s+hotels?", t, re.IGNORECASE):
        return "SEARCH_HOTEL"
    if re.search(r"submit\s+a\s+review\b(?!.*restaurant)", t, re.IGNORECASE):
        return "SUBMIT_REVIEW"
    if re.search(r"add\s+to\s+wishlist.*hotel", t, re.IGNORECASE):
        return "ADD_TO_WISHLIST_HOTEL"
    if re.search(r"apply.*filter.*hotel|show\s+details\s+for\s+hotels", t, re.IGNORECASE):
        return "APPLY_FILTERS"
    if re.search(r"go\s+back\s+to\s+all\s+hotels", t, re.IGNORECASE):
        return "BACK_TO_ALL_HOTELS"
    if re.search(r"show\s+details\s+for\s+a\s+hotel\s+where", t, re.IGNORECASE):
        return "VIEW_HOTEL"

    # ---- AutoDelivery (8006) ----
    if re.search(r"(next|show\s+me\s+the\s+next)\s+set\s+of\s+restaurants", t, re.IGNORECASE):
        return "RESTAURANT_NEXT_PAGE"
    if re.search(r"go\s+back\s+to\s+the\s+previous\s+page\s+of\s+restaurants", t, re.IGNORECASE):
        return "RESTAURANT_PREV_PAGE"
    if re.search(r"return\s+to\s+all\s+restaurants", t, re.IGNORECASE):
        return "BACK_TO_ALL_RESTAURANTS"
    if re.search(r"increase\s+the\s+quantity\s+of\s+the\s+item\s+in\s+the\s+cart", t, re.IGNORECASE):
        return "ITEM_INCREMENTED"
    if re.search(r"search\s+for\s+restaurants?\s+(where|that)", t, re.IGNORECASE):
        return "SEARCH_DELIVERY_RESTAURANT"
    if re.search(r"submit\s+(a\s+)?review\s+for\s+(a\s+)?restaurant", t, re.IGNORECASE):
        return "REVIEW_SUBMITTED"
    if re.search(r"add\s+an?\s+address\s+that\s+is", t, re.IGNORECASE):
        return "ADDRESS_ADDED"
    if re.search(r"set\s+dropoff\s+preference", t, re.IGNORECASE):
        return "DROPOFF_PREFERENCE"
    if re.search(r"select\s+(a\s+)?delivery\s+priority", t, re.IGNORECASE):
        return "DELIVERY_PRIORITY_SELECTED"
    if re.search(r"view\s+the\s+details\s+of\s+a\s+restaurant\s+where", t, re.IGNORECASE):
        return "VIEW_DELIVERY_RESTAURANT"
    if re.search(r"show\s+all\s+restaurants", t, re.IGNORECASE):
        return "VIEW_ALL_RESTAURANTS"
    if re.search(r"(go\s+to\s+)?checkout\s+and\s+show\s+the\s+order", t, re.IGNORECASE):
        return "OPEN_CHECKOUT_PAGE"

    # ---- AutoRestaurant (8003) ----
    if re.search(r"search\s+for\s+restaurants?\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_RESTAURANT"
    if re.search(r"(please\s+)?collapse\s+the\s+(expanded\s+)?menu(\s+view)?", t, re.IGNORECASE):
        return "COLLAPSE_MENU"
    if re.search(r"click\s+the\s+contact\s+card\s+where", t, re.IGNORECASE):
        return "CONTACT_CARD_CLICK"
    if re.search(r"scroll\s+in\s+the\s+direction", t, re.IGNORECASE):
        return "SCROLL_VIEW"
    if re.search(r"show\s+details\s+for\s+the\s+help\s+category", t, re.IGNORECASE):
        return "HELP_CATEGORY_SELECTED"
    if re.search(r"(navigate\s+to|find)\s+the\s+help\s+page", t, re.IGNORECASE):
        return "HELP_PAGE_VIEW"
    if re.search(r"(open|show).*guest.*selector.*dropdown.*number\s+of\s+people", t, re.IGNORECASE):
        return "PEOPLE_DROPDOWN_OPENED"
    if re.search(r"select.*country.*dropdown|please\s+select\s+the\s+country", t, re.IGNORECASE):
        return "COUNTRY_SELECTED"
    if re.search(r"expand\s+the\s+faq\s+item", t, re.IGNORECASE):
        return "HELP_FAQ_TOGGLED"
    if re.search(r"open\s+the\s+help", t, re.IGNORECASE):
        return "HELP_VIEWED"
    if re.search(r"click\s+on\s+the\s+feature.*on\s+the\s+about\s+page", t, re.IGNORECASE):
        return "ABOUT_FEATURE_CLICK"
    if re.search(r"(show\s+details|details)\s+for\s+the\s+feature.*on\s+the\s+about\s+page", t, re.IGNORECASE):
        return "ABOUT_FEATURE_CLICK"
    if re.search(r"feature\s+(field\s+)?(?:CONTAINS|equals?)\s+.*about\s+page|about\s+page.*feature\s+(field\s+)?(?:CONTAINS|equals?)", t, re.IGNORECASE):
        return "ABOUT_FEATURE_CLICK"
    if re.search(r"contact\s+support\s+regarding", t, re.IGNORECASE):
        return "CONTACT_FORM_SUBMIT"
    if re.search(r"view\s+the\s+details\s+of\s+a\s+restaurant", t, re.IGNORECASE):
        return "VIEW_RESTAURANT"
    if re.search(r"show\s+details\s+for\s+(a|the)\s+restaurant", t, re.IGNORECASE):
        return "VIEW_RESTAURANT"

    # ---- AutoShop (8002) ----
    if re.search(r"update\s+quantity\s+of\s+item\s+with\s+title", t, re.IGNORECASE):
        return "QUANTITY_CHANGED"
    if re.search(r"update\s+the\s+quantity\s+of\s+the\s+item\s+in\s+my\s+cart", t, re.IGNORECASE):
        return "QUANTITY_CHANGED"
    if re.search(r"update\s+quantity\s+of\s+item", t, re.IGNORECASE):
        return "QUANTITY_CHANGED"
    if re.search(r"increase\s+the\s+quantity", t, re.IGNORECASE):
        return "ITEM_INCREMENTED"
    if re.search(r"show\s+details\s+for\s+a\s+product", t, re.IGNORECASE):
        return "VIEW_DETAIL"
    if re.search(r"filter\s+to\s+show\s+only\s+products\s+in\s+the\s+category", t, re.IGNORECASE):
        return "CATEGORY_FILTER"
    if re.search(r"(show\s+me\s+my\s+saved\s+items|my\s+wishlist|show.*wishlist)", t, re.IGNORECASE):
        return "VIEW_WISHLIST"
    if re.search(r"proceed\s+to\s+checkout", t, re.IGNORECASE):
        return "PROCEED_TO_CHECKOUT"
    if re.search(r"(complete\s+my\s+purchase|complete\s+my\s+order)", t, re.IGNORECASE):
        return "ORDER_COMPLETED"
    if re.search(r"scroll\s+(left|right)\s+in\s+the\s+carousel", t, re.IGNORECASE):
        return "CAROUSEL_SCROLL"
    if re.search(r"share\s+the\s+link\s+to\s+a\s+product", t, re.IGNORECASE):
        return "SHARE_PRODUCT"
    if re.search(r"add.*this.*item.*to.*cart", t, re.IGNORECASE):
        return "ADD_TO_CART"
    if re.search(r"(add|put).*wishlist\s+(a\s+)?(?:hotel|item|product|book)", t, re.IGNORECASE):
        return "ADD_TO_WISHLIST"
    if re.search(r"(show|view)\s+my\s+shopping\s+cart", t, re.IGNORECASE):
        return "VIEW_CART"

    # ---- AutoDoc (8004) ----
    if re.search(r"add\s+a\s+new\s+client", t, re.IGNORECASE):
        return "ADD_CLIENT"
    if re.search(r"add\s+a\s+new\s+matter", t, re.IGNORECASE):
        return "ADD_NEW_MATTER"
    if re.search(r"search\s+for\s+matters?\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_MATTER"
    if re.search(r"show\s+me\s+details\s+for\s+clients?\s+whose", t, re.IGNORECASE):
        return "FILTER_CLIENTS"
    if re.search(r"show\s+me\s+matters?\s+where\s+the\s+status", t, re.IGNORECASE):
        return "FILTER_MATTER_STATUS"
    if re.search(r"show\s+me\s+details\s+about\s+a\s+document", t, re.IGNORECASE):
        return "DOCUMENT_DELETED"
    if re.search(r"sort\s+matters?\s+so\s+that", t, re.IGNORECASE):
        return "SORT_MATTER_BY_CREATED_AT"
    if re.search(r"change\s+(user\s+)?name\s+to", t, re.IGNORECASE):
        return "CHANGE_USER_NAME"
    if re.search(r"show.*pending\s+events\s+on\s+the\s+calendar", t, re.IGNORECASE):
        return "VIEW_PENDING_EVENTS"
    if re.search(r"add\s+a\s+new\s+calendar\s+event\s+where", t, re.IGNORECASE):
        return "NEW_CALENDAR_EVENT_ADDED"

    # ---- AutoBooks (8001) ----
    if re.search(r"(delete|remove)\s+(your\s+)?(user[- _]?registered\s+)?book", t, re.IGNORECASE):
        if re.search(r"\b(login|sign.?in)\b", t, re.IGNORECASE):
            return "DELETE_BOOK"
    if re.search(r"modify\s+your\s+book|edit\s+(your\s+)?book\s+where", t, re.IGNORECASE):
        return "EDIT_BOOK"
    if re.search(r"remove\s+from\s+the\s+reading\s+list", t, re.IGNORECASE):
        return "REMOVE_FROM_READING_LIST"
    if re.search(r"go\s+to\s+the\s+contact\s+page\s+and\s+send", t, re.IGNORECASE):
        return "CONTACT_BOOK"
    if re.search(r"register\s+with\s+the\s+following\s+username", t, re.IGNORECASE):
        return "REGISTRATION_BOOK"
    if re.search(r"show\s+details\s+for\s+a\s+book\s+where", t, re.IGNORECASE):
        return "BOOK_DETAIL"
    if re.search(r"filter\s+books?\s+where", t, re.IGNORECASE):
        return "FILTER_BOOK"
    if re.search(r"search\s+for\s+(the\s+)?book\s+with\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_BOOK"
    if re.search(r"view\s+the\s+shopping\s+cart.*all\s+items|see\s+all\s+items.*cart", t, re.IGNORECASE):
        return "VIEW_CART_BOOK"
    if re.search(r"login\s+for\s+the\s+following\s+username", t, re.IGNORECASE):
        return "LOGIN_BOOK"
    if re.search(r"authenticate\s+with\s+username.*view\s+the\s+shopping\s+cart", t, re.IGNORECASE):
        return "VIEW_CART_BOOK"
    if re.search(r"(add|create)\s+a\s+book\s+(with|where)\s+genres?", t, re.IGNORECASE):
        return "ADD_BOOK"
    if re.search(r"leave\s+a\s+comment\s+on\s+a\s+book", t, re.IGNORECASE):
        return "ADD_COMMENT_BOOK"
    if re.search(r"open\s+preview\s+of\s+a\s+book", t, re.IGNORECASE):
        return "OPEN_PREVIEW"

    # ---- AutoCinema (8000) ----
    if re.search(r"add\s+(to\s+)?watchlist", t, re.IGNORECASE):
        return "ADD_TO_WATCHLIST"
    if re.search(r"remove\s+from\s+watchlist", t, re.IGNORECASE):
        return "REMOVE_FROM_WATCHLIST"
    if re.search(r"share\s+movie\s+details", t, re.IGNORECASE):
        return "SHARE_MOVIE"
    if re.search(r"watch\s+the\s+trailer\s+for\s+a\s+movie", t, re.IGNORECASE):
        return "WATCH_TRAILER"
    if re.search(r"(navigate\s+to\s+(a\s+)?movie\s+page|show\s+details?\s+for\s+(a\s+)?movie)\s+where", t, re.IGNORECASE):
        return "FILM_DETAIL"
    if re.search(r"search\s+for\s+(a\s+)?movie\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_FILM"

    # ---- AutoShop (8002) additional ----
    if re.search(r"click\s+on\s+buy\s+now\s+to\s+initiate\s+checkout", t, re.IGNORECASE):
        return "CHECKOUT_STARTED"

    # ---- AutoRestaurant (8003) additional ----
    if re.search(r"navigate\s+to\s+the\s+about\s+page", t, re.IGNORECASE):
        return "ABOUT_PAGE_VIEW"
    if re.search(r"open\s+the\s+date\s+selector", t, re.IGNORECASE):
        return "DATE_DROPDOWN_OPENED"
    if re.search(r"(open|show\s+details\s+for\s+opening)\s+the\s+time\s+(selection\s+)?dropdown", t, re.IGNORECASE):
        return "TIME_DROPDOWN_OPENED"
    if re.search(r"(retrieve\s+details\s+of\s+a\s+contact\s+form|submit.*contact.*form.*email.*contains)", t, re.IGNORECASE):
        return "CONTACT_FORM_SUBMIT"

    # ---- AutoDoc (8004) additional ----
    if re.search(r"(retrieve|show)\s+details\s+of\s+billing\s+(entries|records)|billing\s+entries\s+where", t, re.IGNORECASE):
        return "BILLING_SEARCH"
    if re.search(r"edit\s+log\s+entry\s+where", t, re.IGNORECASE):
        return "LOG_EDITED"
    if re.search(r"archive\s+the\s+matter\s+where", t, re.IGNORECASE):
        return "ARCHIVE_MATTER"
    if re.search(r"(retrieve|show)\s+details\s+(of|for)\s+a?\s*client\s+whose", t, re.IGNORECASE):
        return "VIEW_CLIENT_DETAILS"
    if re.search(r"(retrieve|show)\s+details\s+(of|for)\s+(the\s+)?matter\s+(whose|where)", t, re.IGNORECASE):
        return "VIEW_MATTER_DETAILS"

    # ---- AutoMail (8005) additional ----
    if re.search(r"reply\s+to\s+the\s+email|reply\s+the\s+email", t, re.IGNORECASE):
        return "REPLY_EMAIL"
    if re.search(r"add\s+a\s+label\s+to\s+the\s+email\s+where", t, re.IGNORECASE):
        return "ADD_LABEL"
    if re.search(r"send\s+an\s+email\s+to\s+['\"]", t, re.IGNORECASE):
        return "SEND_EMAIL"
    if re.search(r"search\s+for\s+emails?\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_EMAIL"

    # ---- AutoDelivery (8006) additional ----
    if re.search(r"delete\s+the\s+review\s+for\s+the\s+restaurant", t, re.IGNORECASE):
        return "DELETE_REVIEW"
    if re.search(r"show\s+me\s+restaurants?\s+(that\s+do\s+NOT|where\s+the\s+cuisine)", t, re.IGNORECASE):
        return "RESTAURANT_FILTER"
    if re.search(r"add\s+a?\s*menu\s+item\s+to\s+(my\s+)?cart", t, re.IGNORECASE):
        return "ADD_TO_CART_MENU_ITEM"
    if re.search(r"open\s+the\s+add.?to.?cart\s+modal", t, re.IGNORECASE):
        return "ADD_TO_CART_MODAL_OPEN"
    if re.search(r"start\s+a\s+quick\s+order", t, re.IGNORECASE):
        return "QUICK_ORDER_STARTED"

    # ---- AutoLodge (8007) additional ----
    if re.search(r"open\s+the\s+FAQ\s+item\s+where", t, re.IGNORECASE):
        return "FAQ_OPENED"
    if re.search(r"message\s+the\s+host\s+where", t, re.IGNORECASE):
        return "MESSAGE_HOST"
    if re.search(r"edit\s+check.?in.*check.?out\s+dates", t, re.IGNORECASE):
        return "EDIT_CHECK_IN_OUT_DATES"
    if re.search(r"open\s+my\s+wishlist\s+to\s+view\s+saved\s+hotels", t, re.IGNORECASE):
        return "WISHLIST_OPENED"
    if re.search(r"show\s+me\s+the\s+wishlist\s+so\s+i\s+can\s+view", t, re.IGNORECASE):
        return "WISHLIST_OPENED"
    if re.search(r"remove\s+from\s+my\s+wishlist", t, re.IGNORECASE):
        return "REMOVE_FROM_WISHLIST"

    # ---- AutoConnect (8008) additional ----
    if re.search(r"open\s+the\s+jobs?\s+tab\s+from\s+the\s+navbar", t, re.IGNORECASE):
        return "JOBS_NAVBAR"
    if re.search(r"edit\s+profile\s+information", t, re.IGNORECASE):
        return "EDIT_PROFILE"
    if re.search(r"edit\s+profile\s+for\s+the\s+user\s+whose", t, re.IGNORECASE):
        return "EDIT_PROFILE"
    if re.search(r"post\s+a\s+status\s+update", t, re.IGNORECASE):
        return "POST_STATUS"
    if re.search(r"remove\s+post\s+where", t, re.IGNORECASE):
        return "REMOVE_POST"

    # ---- AutoHire (8009) additional ----
    if re.search(r"edit\s+profile\s+title\s+where", t, re.IGNORECASE):
        return "EDIT_PROFILE_TITLE"
    if re.search(r"(user\s+clicks?|click)\s+'?post\s+a\s+job'?|initiate.*posting.*job|clicks?\s+'?post\s+a\s+job'?\s+button", t, re.IGNORECASE):
        return "POST_A_JOB"
    if re.search(r"clicks?\s+the\s+'?experts?'?\s+option\s+in\s+the\s+navbar|list\s+of\s+all\s+experts.*clicks?\s+the\s+'?experts?", t, re.IGNORECASE):
        return "NAVBAR_EXPERTS_CLICK"
    if re.search(r"show\s+the\s+list\s+of\s+all\s+experts", t, re.IGNORECASE):
        return "NAVBAR_EXPERTS_CLICK"
    if re.search(r"add\s+a\s+skill\s+where\s+skill", t, re.IGNORECASE):
        return "ADD_SKILL"
    if re.search(r"submit\s+a\s+job\s+with\s+title", t, re.IGNORECASE):
        return "SUBMIT_JOB"
    if re.search(r"decide\s+to\s+start\s+hiring", t, re.IGNORECASE):
        return "HIRE_LATER_START"

    # ---- AutoCalendar (8010) additional ----
    if re.search(r"select\s+the\s+calendar\s+that\s+contains", t, re.IGNORECASE):
        return "SELECT_CALENDAR"
    if re.search(r"unselect\s+the\s+calendar", t, re.IGNORECASE):
        return "UNSELECT_CALENDAR"
    if re.search(r"go\s+to\s+today'?s?\s+date\s+in\s+the\s+calendar", t, re.IGNORECASE):
        return "SELECT_TODAY"

    # ---- AutoList (8011) additional ----
    if re.search(r"complete\s+task\s+(where|whose)", t, re.IGNORECASE):
        return "AUTOLIST_COMPLETE_TASK"
    if re.search(r"(please\s+)?set\s+the\s+date\s+for\s+the\s+task\s+to", t, re.IGNORECASE):
        return "AUTOLIST_SELECT_DATE_FOR_TASK"

    # ---- AutoRide (8012) additional ----
    if re.search(r"view\s+trip\s+details\s+for\s+(a\s+)?(trip|ride)\s+where", t, re.IGNORECASE):
        return "TRIP_DETAILS"
    if re.search(r"select\s+car\s+options\s+where", t, re.IGNORECASE):
        return "SELECT_CAR"
    if re.search(r"search\s+destination\s+where\s+the\s+destination", t, re.IGNORECASE):
        return "SEARCH_DESTINATION"
    if re.search(r"select\s+date\s+for\s+(your|my)\s+trip\s+as", t, re.IGNORECASE):
        return "SELECT_DATE"

    # ---- AutoMedic (8013) additional ----
    if re.search(r"refill\s+prescription\s+where", t, re.IGNORECASE):
        return "REFILL_PRESCRIPTION"
    if re.search(r"(show\s+me\s+details\s+to\s+refill|show\s+details\s+for\s+a\s+prescription)", t, re.IGNORECASE):
        return "VIEW_PRESCRIPTION"
    if re.search(r"show\s+details\s+for\s+doctor\s+reviews\s+where", t, re.IGNORECASE):
        return "FILTER_DOCTOR_REVIEWS"

    # ---- AutoBooks login/logout ----
    if re.search(r"(login\s+for\s+the\s+following|login\s+with\s+(a\s+)?specific).*username.*then\s+logout", t, re.IGNORECASE):
        return "LOGOUT_BOOK"
    if re.search(r"first.*authenticate.*username.*then.*logout", t, re.IGNORECASE):
        return "LOGOUT_BOOK"

    # ---- AutoCinema specific login+action ----
    if re.search(r"delete\s+(your\s+)?movie|then.*delete.*movie|login.*then.*delete", t, re.IGNORECASE):
        if re.search(r"\b(login|sign.?in|authenticate)\b", t, re.IGNORECASE):
            return "LOGIN_THEN_DELETE_ITEM"
    if re.search(r"add\s+(a\s+)?film\s+(where|with|whose)|add\s+(a\s+)?new\s+film", t, re.IGNORECASE):
        if re.search(r"\b(login|sign.?in|authenticate)\b", t, re.IGNORECASE):
            return "LOGIN_THEN_ADD_ITEM"
    if re.search(r"edit\s+(your\s+)?film|edit\s+(a\s+)?movie\s+(where|with)", t, re.IGNORECASE):
        if re.search(r"\b(login|sign.?in|authenticate)\b", t, re.IGNORECASE):
            return "LOGIN_THEN_EDIT_ITEM"

    # ---- Multi-step compound tasks ----
    if re.search(r"\b(logout|sign.?out|log.?out)\b", t) and re.search(r"\b(login|sign.?in|log.?in)\b", t):
        return "LOGIN_THEN_LOGOUT"
    if re.search(r"\b(add|remove|delete).*(watchlist|reading.?list|wishlist|cart)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_LIST_ACTION"
    if re.search(r"\b(add|post|submit).*(comment|review|rating)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_COMMENT"
    if re.search(r"\b(add|insert|create|register).*(film|movie|book)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_ADD_ITEM"
    if re.search(r"\b(edit|update|modify).*(film|movie|book)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_EDIT_ITEM"
    if re.search(r"\b(delete|remove).*(film|movie|book)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_DELETE_ITEM"
    if re.search(r"\b(edit|update|modify).*(profile|account|user)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_EDIT_PROFILE"
    if re.search(r"\b(purchase|buy|checkout|order)\b", t) and re.search(r"\b(login|sign.?in|authenticate)\b", t):
        return "LOGIN_THEN_PURCHASE"

    # ---- AutoDelivery (8006) missing ----
    if re.search(r"reorder\s+the\s+(recent\s+)?item", t, re.IGNORECASE):
        return "QUICK_REORDER"
    if re.search(r"show\s+details\s+for\s+editing\s+a\s+cart\s+item", t, re.IGNORECASE):
        return "EDIT_CART_ITEM"

    # ---- AutoDoc (8004) missing ----
    if re.search(r"delete\s+the\s+matter\s+where", t, re.IGNORECASE):
        return "DELETE_MATTER"

    # ---- AutoMail (8005) missing ----
    if re.search(r"create\s+a\s+new\s+label", t, re.IGNORECASE):
        return "CREATE_LABEL"

    # ---- Task management ----
    if re.search(r"delete\s+task\b", t, re.IGNORECASE):
        return "DELETE_TASK"
    if re.search(r"(create|add|new)\s+task\b", t, re.IGNORECASE):
        return "CREATE_TASK"
    if re.search(r"(edit|update|modify)\s+task\b", t, re.IGNORECASE):
        return "EDIT_TASK"

    # ---- AutoRestaurant/generic logout ----
    if re.search(r"log\s+out\s+of\s+the\s+account|please\s+log\s+out|click.*log\s+out|logout\s+of\s+the", t, re.IGNORECASE):
        return "LOGOUT"

    # ---- Generic fallbacks ----
    if re.search(r"\b(register|sign.?up|create.*account|fill.*registration)\b", t):
        return "REGISTRATION"
    if re.search(r"\b(login|sign.?in|log.?in|fill.*login|authenticate)\b", t):
        return "LOGIN"
    if re.search(r"\b(search|look.?for|find|look.?up)\b", t) and re.search(r"\b(film|movie|book)\b", t):
        return "SEARCH_ITEM"
    if re.search(r"\b(filter|sort)\b", t) and re.search(r"\b(film|movie|book)\b", t):
        return "FILTER_ITEM"
    if re.search(r"\b(navigate|go.?to|view.?detail|detail.?page|film.?page|book.?page|movie.?page)\b", t):
        return "NAVIGATE_DETAIL"
    if re.search(r"\b(share)\b", t) and re.search(r"\b(film|movie|book)\b", t):
        return "SHARE_ITEM"
    if re.search(r"\b(watch.*trailer|play.*trailer|trailer)\b", t):
        return "WATCH_TRAILER"
    if re.search(r"\b(preview|open.*preview)\b", t):
        return "OPEN_PREVIEW"
    if re.search(r"\b(add|put).*(cart|basket)\b", t):
        return "ADD_TO_CART"
    if re.search(r"\b(remove|delete).*(cart|basket)\b", t):
        return "REMOVE_FROM_CART"
    if re.search(r"\b(view|show).*(cart|basket)\b", t):
        return "VIEW_CART"
    if re.search(r"\b(purchase|buy|checkout|order)\b", t):
        return "PURCHASE"
    if re.search(r"\b(contact|send.*message|fill.*contact)\b", t):
        return "CONTACT"
    if re.search(r"\b(add|post|submit).*(comment|review)\b", t):
        return "ADD_COMMENT"
    if re.search(r"\b(watchlist|reading.?list|wishlist)\b", t):
        return "LIST_ACTION"

    return "GENERAL"


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
