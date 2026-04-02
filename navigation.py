"""URL utilities: seed preservation, localhost validation, normalization."""
from __future__ import annotations
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode, parse_qsl

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1"}
_ALLOWED_SCHEMES = {"http", "https"}


def extract_seed(url: str) -> str | None:
    if not url:
        return None
    params = parse_qs(urlsplit(url).query)
    seed_vals = params.get("seed")
    return seed_vals[0] if seed_vals else None


def preserve_seed(target_url: str, current_url: str) -> str:
    seed = extract_seed(current_url)
    if seed is None:
        return target_url
    if extract_seed(target_url) == seed:
        return target_url
    parts = urlsplit(target_url)
    params = [(k, v) for k, v in parse_qsl(parts.query) if k != "seed"]
    params.append(("seed", seed))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname
    if hostname and hostname not in _LOCALHOST_HOSTS:
        port = parts.port
        new_netloc = f"localhost:{port}" if port else "localhost"
        return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))
    return url


def is_localhost_url(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        return False
    hostname = parts.hostname
    return hostname is not None and hostname in _LOCALHOST_HOSTS


def same_page(url_a: str, url_b: str) -> bool:
    a, b = urlsplit(url_a), urlsplit(url_b)
    return a.path == b.path and a.query == b.query
