"""Sites and URL sections this channel never publishes from (2026-10-31).

The user: a story from a section that never makes it should not be sent to the
Scorer at all ("根本就没必要送过去"). config/url_blocklist.txt is this channel's
own list, taken from its own logs: a site or section scored at least 10 times
since 10-01, never 6 or more, never into the candidate pool, never published.
Plus the sections the user named (/fashion, /auto, /leben-wissen, /sports)
where this channel's data shows the same.

A line is a host ("news.sky.com"), a host and section ("smh.com.au/politics",
matching that section and everything under it), or "*/name" for a section
segment of that name on any site ("*/fashion" also matches /fashion-news/).
Checked on the article's real URL, not a Google News link.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

_FILE = Path(__file__).resolve().parent.parent / "config" / "url_blocklist.txt"


@lru_cache(maxsize=1)
def _entries() -> tuple[tuple[str, ...], tuple[str, ...]]:
    prefixes, segments = [], []
    if _FILE.exists():
        for line in _FILE.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip().lower()
            if line.startswith("*/"):
                segments.append(line[2:].strip("/"))
            elif line:
                prefixes.append(line.strip("/"))
    return tuple(prefixes), tuple(segments)


def blocklisted(url: str) -> str | None:
    """The list entry this URL falls under, or None."""
    try:
        p = urlparse(url or "")
    except Exception:
        return None
    host = (p.hostname or "").lower().removeprefix("www.")
    parts = [s.lower() for s in (p.path or "").split("/") if s]
    key = "/".join([host] + parts)
    prefixes, segments = _entries()
    for e in prefixes:
        if key == e or key.startswith(e + "/"):
            return e
    for s in parts[:-1]:    # sections, not the article's own slug
        for name in segments:
            if s == name or s.startswith(name + "-"):
                return "*/" + name
    return None
