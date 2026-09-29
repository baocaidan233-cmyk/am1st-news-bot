"""Digests, rolling live pages and link lists — the channel does not run them.

User decision, carried over from Leading News and re-measured here: a post
has to be about one thing. A page that is several unrelated stories cannot be
written into one caption without the caption picking one of them and dropping
the rest, or describing the page instead of the news.

It cost a post on 2026-09-29: "Trump posts map showing 'Trump Strait' off
Iran, and other Mideast developments" is an ms.now omnibus whose body opens
"Here is a look at the latest news from the Middle East on Saturday" and then
runs Houthi missiles, Iraqi sanctions exemptions and West Bank springs under
separate headings. Fox's /live-news/ page went out the same way, with
"Coverage for this event has ended" sitting in the middle of it.

Leading News's version of this rule is multilingual and much wider. Ported
whole it would have been wrong here, which is the point of measuring on this
channel's own material instead:

- Its `^LIVE:` pattern is written for rolling pages like "Australia news
  live:". On this channel `LIVE:` means a single event being streamed --
  "LIVE: President Trump Holds a Rally in Mobile, AL" is the highest-scoring
  candidate in the whole 1730-row sample at 8.0, with three more rally and
  press-conference streams at 6.0. Blocking those would have removed the best
  thing the channel gets.
- Matching "briefing" in a URL caught /briefing-room/, which is a place in
  the White House.
- "Breitbart Business Digest" is a column name, not a digest: measured at
  5937 characters on one subject, the Fed's split over AI investment, with
  its subheadings arguing one case. The Dow Jones "Auto & Transport Roundup:
  Market Talk" and AP "Editorial Roundup: United States" really are lists.

Measured on this channel's own material (2026-09-29): 12 of 1730 scored
candidates (0.7%), of which 3 would otherwise have entered the pool, and 2 of
197 already-published articles -- both verified genuine omnibus pages by
reading their bodies.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# One named event being covered live or in full is a single subject, whatever
# the title calls itself.
_SINGLE_EVENT = re.compile(
    r"(?i)\b(rally|speech|remarks|address|announcement|press conference|news conference|"
    r"briefing room|hearing|testimony|interview|debate|summit|signing|roundtable|"
    r"ceremony|launch|vote|votes|verdict|ruling)\b")

# Columns whose brand name contains a digest word but which run one subject.
# A keep-list, so a name missing from it costs one story rather than letting a
# real digest through; add to it when one shows up.
_SINGLE_SUBJECT_COLUMNS = ("breitbart business digest",)

_TITLE_RE = re.compile(
    r"(\b(morning|evening|midday|weekly|weekend|daily|nightly)\s+"
    r"(briefing|brief|digest|round-?up|wrap|rundown|summary)\b"
    r"|\b(news|war|executive)\s+(briefing|digest|round-?up|wrap|rundown|summary)\b"
    r"|\bnews\s+(in brief|summary)\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
    r"(briefing|summary|round-?up|digest)\b"
    r"|\bnews live\s*[:|–-]|\bliveblog\b|\blive blog\b|\bas it happened\b"
    r"|\bthings (you need )?to know\b|\bweek in review\b|\bthe week in\b"
    r"|\band other [\w'’\s]{0,24}(news|stories|headlines|developments|updates|items)\b"
    r"|\btop (stories|headlines)\b|\bnews quiz\b|\bopen thread\b|\bmorning minute\b"
    r"|\broundup\s*:|\bdigest\s*:|\bwrap\s*:)", re.IGNORECASE)

# "briefing" is deliberately absent: the White House briefing room is a place.
_URL_RE = re.compile(
    r"/[^/]*(live-?blog|live-?updates?|live-news|newsblog|live-?ticker|digest|round-?up)[^/]*(?=/|$)",
    re.IGNORECASE)
# One entry of a live blog on its own page is a single story; the rolling page
# that collects them is not.
_LIVE_ENTRY_RE = re.compile(r"/(live-blog-update|liveblog_entry|liveblog-entry)/", re.IGNORECASE)


def roundup_rule(title: str, url: str) -> str | None:
    """The name of the rule this candidate breaks, or None."""
    text = title or ""
    lowered = text.lower()
    if any(name in lowered for name in _SINGLE_SUBJECT_COLUMNS):
        return None
    if _SINGLE_EVENT.search(text):
        return None
    if _TITLE_RE.search(text):
        return "title_roundup"
    path = urlparse(url or "").path or ""
    if _URL_RE.search(path) and not _LIVE_ENTRY_RE.search(path):
        return "url_roundup"
    return None
