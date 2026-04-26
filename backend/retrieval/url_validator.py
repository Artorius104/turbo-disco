"""Allowlist-based URL validator with concurrent HEAD/GET probing."""
import logging
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, Literal

import httpx

logger = logging.getLogger(__name__)

ALLOWED_HOSTS = {
    "protocols.io",
    "bio-protocol.org",
    "nature.com",
    "openwetware.org",
    "sigmaaldrich.com",
    "merckmillipore.com",
    "promega.com",
    "idtdna.com",
    "thermofisher.com",
    "fishersci.com",
    "atcc.org",
    "addgene.org",
    "europepmc.org",
    "ncbi.nlm.nih.gov",
    "jove.com",
    "doi.org",
    "biorxiv.org",
    "arxiv.org",
}

UrlStatus = Literal["ok", "unavailable", "disallowed"]

_TIMEOUT = 6.0
_MAX_WORKERS = 8


def _host(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""
    parts = host.lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host.lower()


def is_allowed(url: str) -> bool:
    return _host(url) in ALLOWED_HOSTS


def _ok_status(code: int) -> bool:
    # Accept any 2xx (live content) and 3xx (redirect chain), plus 403/405
    # (publisher blocks HEAD/range but the page exists).
    if 200 <= code < 400:
        return True
    if code in (403, 405):
        return True
    return False


def validate(url: str) -> UrlStatus:
    if not url or not url.startswith(("http://", "https://")):
        return "disallowed"
    if not is_allowed(url):
        return "disallowed"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, headers=headers) as c:
            try:
                r = c.head(url)
                if _ok_status(r.status_code):
                    return "ok"
            except Exception:
                pass
            # Fall back to range-GET (some publishers block HEAD).
            try:
                r = c.get(url, headers={**headers, "Range": "bytes=0-0"})
                if _ok_status(r.status_code):
                    return "ok"
            except Exception:
                pass
            # Last resort: full GET (some publishers ignore Range).
            r = c.get(url)
            return "ok" if _ok_status(r.status_code) else "unavailable"
    except Exception as e:
        logger.warning("URL validation failed for %r: %s", url, e)
        return "unavailable"


def validate_many(urls: Iterable[str]) -> Dict[str, UrlStatus]:
    """Validate URLs concurrently. Returns {url: status}. Deduplicated."""
    unique = sorted({u for u in urls if u})
    if not unique:
        return {}
    out: Dict[str, UrlStatus] = {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(unique))) as pool:
        futures = {pool.submit(validate, u): u for u in unique}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                out[url] = fut.result()
            except Exception:
                out[url] = "unavailable"
    return out
