"""Strip the publisher's own name off a link-preview headline.

The preview card's headline is `ttl` in the Gettr post payload — a field WE
send (agents/gettr_publisher.py), not something Gettr scrapes — and it is filled
from the source page's og:title. Publishers routinely append their own name to
that tag, so the card reads:

    Feeds nab illegal alien who allegedly dismembered victims | Blaze Media
    Jobless Claims Plunge to 196,000 as Layoffs Remain Historically Low › American Greatness
    Instapundit » Blog Archive » DEAR BOB. THERE SHOULD BE NO BALLOT FRAUD
    President Trump Just Put H-1B Employers On Notice – 100PercentFedUp.com – by Isaac

Measured 2026-10-02 over 927 published posts: 132 of them, 14.2%, about 6 a day.

NO LIST OF PUBLISHER NAMES. The publisher is derived from the article's own
domain, per article, so this works on a source nobody has seen before — the
standing rule here is that a list always misses the ninth entry. The one test
is whether the trailing segment's words appear in the hostname.

The error costs are lopsided and every rule below is set to the safe side:
missing a tail leaves a slightly ugly card, while a false positive cuts real
words out of a headline. Two false positives were found by running this over all
927 and both are fixed here; each has a comment and a guard, because both were
invisible until the real data was in front of it.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# " - " is in here and it is the risky one: an em-dash or hyphen also joins real
# headline clauses. It survives only because every cut additionally has to pass
# the position, length, digit and hostname tests below.
_SEPS = ("|", "›", "»", "–", "—", "·", "-")
_WORD = re.compile(r"[A-Za-z0-9]+")
_X_HANDLE = re.compile(r"\(@[A-Za-z0-9_]{2,}\)\s+on\s+X\s*$")
_BLOG_ARCHIVE = re.compile(r"^\s*[\w.]+\s*»\s*Blog Archive\s*»\s*(.+)$", re.S)
# "* The Gateway Pundit *" — a publisher fenced in asterisks rather than
# introduced by a separator.
_STARRED = re.compile(r"\s*[*]\s*[^*]{3,40}\s*[*]\s*$")
_DOMAINISH = re.compile(r"\.(com|org|net|news)\b")
# Prefixes worth splitting off a hostname stem so the remainder can still match:
# amgreatness.com has to be reachable from the words "American Greatness".
_HOST_PREFIXES = ("the", "am", "my", "real", "daily", "we")


def _hostname_tokens(url: str) -> set[str]:
    host = (urlparse(url or "").hostname or "").lower()
    host = re.sub(r"^(www|rss|feeds?|amp)\.", "", host)
    stem = host.rsplit(".", 2)[0] if host.count(".") >= 2 else host.split(".")[0]
    tokens = {stem}
    for prefix in _HOST_PREFIXES:
        if stem.startswith(prefix) and len(stem) > len(prefix) + 3:
            tokens.add(stem[len(prefix):])
    return {t for t in tokens if len(t) >= 4}


def strip_publisher(title: str, url: str) -> tuple[str, str | None]:
    """Returns (headline, rule) — rule is None when nothing was touched.

    rule == "x_handle" means the og:title is not a headline at all, it is the
    poster's display name ("Sean Davis (@seanmdav) on X"). There is nothing to
    strip there and the caller should fall back to the feed's own title."""
    text = (title or "").strip()
    if not text:
        return text, None
    if _X_HANDLE.search(text):
        return text, "x_handle"
    archive = _BLOG_ARCHIVE.match(text)
    if archive:
        return archive.group(1).strip(), "blog_archive_prefix"

    rule = None
    starred = _STARRED.search(text)
    # Has to run BEFORE the separator loop. Measured: with only the separator
    # rules, "… Unanimous 8–0 Verdict for Christian Groundskeeper Fired Over a
    # Religious Exemption — Awarded $671,000 * The Gateway Pundit *" lost
    # "Awarded $671,000" too, because the asterisk fence matched nothing and the
    # loop fell back to the em-dash further left.
    if starred and len(_WORD.findall(starred.group(0))) <= 5:
        text = text[:starred.start()].rstrip(" -–—|›»·*")
        rule = "starred_publisher"

    hosts = _hostname_tokens(url)
    for _ in range(3):   # "– 100PercentFedUp.com – by Isaac" needs two passes
        # Every occurrence of every separator, right to left. Taking only the
        # rightmost one per separator is not enough: in the line above the
        # rightmost " – " is followed by "by Isaac", which matches no rule, and
        # the scan has to keep going left to "– 100PercentFedUp.com" — that cut
        # takes the byline with it.
        positions = []
        for sep in _SEPS:
            at = text.find(f" {sep} ")
            while at > 0:
                positions.append(at)
                at = text.find(f" {sep} ", at + 1)
        cut = None
        for idx in sorted(set(positions), reverse=True):
            segment = text[idx:].lstrip(" " + "".join(_SEPS)).strip()
            words = _WORD.findall(segment.lower())
            if not words or len(words) > 7:
                continue
            # Only cut in the last 45%, and never leave a stub for a headline.
            if idx / len(text) < 0.55 or len(_WORD.findall(text[:idx])) < 5:
                continue
            if _DOMAINISH.search(segment.lower()):
                cut, rule = idx, "domain_tail"
            elif re.search(r"\d", segment):
                # A digit in the segment means headline content — a sum of
                # money, a vote count, a year. This is the second guard on the
                # $671,000 case above.
                continue
            elif hosts and any(w in h or h in w
                               for w in words if len(w) >= 4 for h in hosts):
                cut, rule = idx, "publisher_tail"
            if cut is not None:
                break
        if cut is None:
            break
        text = text[:cut].rstrip(" -–—|›»·*")

    # There is deliberately no byline rule. One was tried ("strip a trailing
    # 'by X'") and over the same 927 posts it fired exactly once, on
    # "America Spends More On Defense R&D Than Any Other OECD Country - By A
    # Mile" — where "By A Mile" is the headline. Every real byline in the data
    # trails a publisher name and is already removed by the two rules above, so
    # the rule was net negative and is gone.
    return text.strip(), rule
