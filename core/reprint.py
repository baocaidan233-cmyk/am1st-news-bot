"""Reprints: the same article under another URL. Dropped at ingestion, before
any embedding, by exact keys instead of similarity (Linear AI-60).

Caidan, 2026-10-12: ingestion blocks only "the same article", the way Perigon
separates reprints from stories; the same event told by different outlets is
kept for the event store. Perigon hashes the whole article; ingestion here has
only the headline and the RSS description, so each gets its own key:

- the headline, normalised (case, punctuation, a trailing " - Outlet" or
  " | Outlet"), at least 4 words, seen on ANOTHER site. The same site reuses a
  column title every day for different content (Power Line's "Quote of the
  Day", Hot Air's "Saturday's Final Word", the NYT's "Here's the latest"), so
  a headline seen again on the same site is not a reprint by itself.
- the description, normalised, at least 80 characters, on any site: the same
  article moved to a second URL keeps its lead.

Measured on this channel's ingestion log since 2026-09-16: 3,464 headlines
appeared on more than one URL, mostly syndication (Daily Caller News
Foundation pieces on BizPac Review, Heartlander, Tennessee Star; AP on ABC
News and US News; one op-ed on Hot Air, Daily Signal and Patriot Post). In the
last 3 days 20 such reprints had reached the pool because one copy had no
description and the embedding similarity fell under the duplicate line.

The first copy seen is kept; which outlet published it first is for the event
store and the clustering step to sort out.

2026-10-14, user: the original over the relay. When the first copy came from
a site that only relays others' articles (_RELAYS), the original arriving
later is kept instead of dropped -- the relay's page often cannot be
extracted, so the story was lost (Hollywood Reporter's Getty bankruptcy and
Tennessee Star's Omar piece behind revolver.news; two Jonathan Turley posts
behind The Liberty Daily). A relay never displaces anything: arriving second,
it is still a reprint."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

_OUTLET_SUFFIX = re.compile(r"\s+[|\-–—]\s+[^|\-–—]{2,40}$")
_TAG = re.compile(r"<[^>]+>")
_MIN_TITLE_WORDS = 4
_MIN_DESCRIPTION_CHARS = 80
_RELAYS = frozenset({"revolver.news", "thelibertydaily.com", "realclearpolitics.com", "instapundit.com"})
_RELAY_MARK = "relay:"


def _norm(text: str) -> str:
    t = _TAG.sub(" ", (text or "").lower())
    t = re.sub(r"[^\w ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def site(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower().removeprefix("www.")


# Flags outlets add to the same headline (user 10-20, AI-69): "BREAKING:",
# "JUST IN:", "WATCH:" in front, "[WATCH]", "(VIDEO)" or an ellipsis at the end.
# Of the 10-20 pairs at cosine 0.93+, the same article under two titles
# differed mostly by these.
_FLAG_PREFIX = re.compile(r"^\s*(breaking|just in|watch)\s*[:\-–—]\s*", re.IGNORECASE)
_FLAG_SUFFIX = re.compile(r"\s*(\[(watch|video)\]|\((watch|video)\)|…|\.\.\.)\s*$", re.IGNORECASE)


def _strip_flags(title: str) -> str:
    t = (title or "").strip()
    for _ in range(3):
        t = _FLAG_SUFFIX.sub("", _FLAG_PREFIX.sub("", t)).strip()
    return t


def title_key(title: str) -> str | None:
    t = _norm(_OUTLET_SUFFIX.sub("", _strip_flags(title)))
    if len(t.split()) < _MIN_TITLE_WORDS:
        return None
    return "t:" + hashlib.sha1(t.encode("utf-8")).hexdigest()


def description_key(description: str) -> str | None:
    d = _norm(description)
    if len(d) < _MIN_DESCRIPTION_CHARS:
        return None
    return "d:" + hashlib.sha1(d.encode("utf-8")).hexdigest()


async def reprint_of(store, title: str, description: str, url: str) -> str | None:
    """Records this item's keys and returns why it is a reprint of an earlier
    item ("title" / "description"), or None. `store` is core.redis_store.RedisStore."""
    here = site(url)
    mine = (_RELAY_MARK + here) if here in _RELAYS else here
    reason = None
    tk = title_key(title)
    if tk:
        earlier = await _first(store, "reprint:" + tk, mine)
        if earlier is not None and earlier.removeprefix(_RELAY_MARK) != here:
            reason = f"title, first seen on {earlier.removeprefix(_RELAY_MARK)}"
    dk = description_key(description)
    if dk:
        earlier = await _first(store, "reprint:" + dk, mine)
        if earlier is not None and reason is None:
            reason = f"description, first seen on {earlier.removeprefix(_RELAY_MARK)}"
    return reason


async def _first(store, key: str, mine: str) -> str | None:
    """store.first_seen(), except that an original finding a relay's claim
    takes the key over and counts as first."""
    earlier = await store.first_seen(key, mine)
    relay_first = earlier is not None and (earlier.startswith(_RELAY_MARK) or earlier in _RELAYS)  # unmarked: claimed before 10-14
    if relay_first and not mine.startswith(_RELAY_MARK):
        await store.claim(key, mine)
        return None
    return earlier
