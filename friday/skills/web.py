"""Web skills — real answers from the internet, not just browser links.

Two skills:

``web_search_results``
    Queries the DuckDuckGo Instant Answer API and returns actual text
    results — so FRIDAY can *answer* questions from the web instead of
    just opening a browser tab.

``fetch_webpage``
    Fetches any URL and returns cleaned, HTML-stripped text (first
    3 000 chars).  FRIDAY can read articles, docs, paste-bins, anything
    publicly accessible via HTTP.

Neither skill requires an API key.
"""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

from friday.skills.registry import skill

log = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds
_MAX_PAGE_CHARS = 3_000
_UA = "FRIDAY-AI/1.0 (voice assistant)"


# --------------------------------------------------------------------------- #
# DuckDuckGo Instant Answer                                                   #
# --------------------------------------------------------------------------- #


@skill(description="Search the web and return actual text answers via DuckDuckGo.")
def web_search_results(query: str) -> str:
    """Query DuckDuckGo Instant Answer API and return a text summary.

    Returns the abstract, direct answer, or top related snippets so
    FRIDAY can answer factual questions from web knowledge without
    opening a browser.

    Parameters
    ----------
    query:
        Natural-language question or search terms,
        e.g. ``"capital of Japan"``, ``"latest Python version"``.
    """
    q = query.strip()
    if not q:
        raise ValueError("web_search_results requires a non-empty query")

    encoded = urllib.parse.quote_plus(q)
    url = (
        f"https://api.duckduckgo.com/?q={encoded}"
        "&format=json&no_html=1&skip_disambig=1&no_redirect=1"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"web search failed: {exc}") from exc

    parts: list[str] = []

    # Direct answer (e.g. unit conversions, quick facts)
    answer = (data.get("Answer") or "").strip()
    if answer:
        parts.append(answer)

    # Abstract text (Wikipedia-style summary)
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        parts.append(abstract)

    # Related topic snippets (when there's no abstract)
    if not parts:
        for topic in data.get("RelatedTopics", [])[:5]:
            text = (topic.get("Text") or "").strip()
            if text:
                parts.append(text)

    if not parts:
        return (
            f"No instant answer found for '{q}'. "
            "Try fetch_webpage with a specific URL, or ask me to open a browser search."
        )

    combined = " | ".join(parts)
    # Cap at a reasonable length so we don't swamp the LLM context.
    return combined[:2_000]


# --------------------------------------------------------------------------- #
# Web page fetch + strip                                                       #
# --------------------------------------------------------------------------- #


@skill(description="Fetch any public URL and return its plain-text content (first 3 000 chars).")
def fetch_webpage(url: str) -> str:
    """Fetch a web page and return its cleaned plain text.

    Strips HTML tags, decodes entities, collapses whitespace.  Returns
    the first 3 000 characters so the LLM context stays manageable.

    Parameters
    ----------
    url:
        Full URL including scheme, e.g. ``"https://example.com/article"``.
        Must be a publicly accessible HTTP/HTTPS resource.
    """
    url = url.strip()
    if not url:
        raise ValueError("fetch_webpage requires a non-empty URL")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; FRIDAY-AI/1.0; voice assistant)"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(512_000)  # max 512 KB download
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"fetch failed: HTTP {exc.code} for {url}") from exc
    except Exception as exc:
        raise RuntimeError(f"fetch failed: {exc}") from exc

    # Decode
    encoding = _guess_encoding(content_type) or "utf-8"
    try:
        text = raw.decode(encoding, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")

    cleaned = _strip_html(text)
    if not cleaned.strip():
        return f"Fetched {url} but couldn't extract readable text."

    result = cleaned[:_MAX_PAGE_CHARS]
    if len(cleaned) > _MAX_PAGE_CHARS:
        result += f"\n[... truncated — {len(cleaned)} chars total]"
    return result


# --------------------------------------------------------------------------- #
# HTML stripping helpers                                                       #
# --------------------------------------------------------------------------- #

# Remove <script>, <style>, <head> blocks entirely (content not useful).
_BLOCK_RE = re.compile(
    r"<(script|style|head|nav|footer|header|noscript)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# Strip remaining tags.
_TAG_RE = re.compile(r"<[^>]+>")
# Collapse whitespace runs.
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _strip_html(raw: str) -> str:
    text = _BLOCK_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def _guess_encoding(content_type: str) -> str | None:
    """Extract charset from a Content-Type header, e.g. 'text/html; charset=utf-8'."""
    m = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
    return m.group(1).strip('"') if m else None


__all__ = ["fetch_webpage", "web_search_results"]
