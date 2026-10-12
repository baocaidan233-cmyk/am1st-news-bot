from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Donald Trump is the sitting president. A caption that calls him a former one
# is wrong on its face, and it is the single error this channel cannot ship.
#
# Deliberately a dedicated check on the caption alone, separate from
# agents/candidate_selector.py's filter_former_trump(). That one exists for a
# different job -- it screens the source article and its metadata, so it also
# catches a stale re-synced piece -- and it matches fixed substrings, which
# misses every variant that is not in its list. This one is the last gate on
# the text we are about to publish under our own name, so it is a pattern
# rather than a list: 6 fixed strings cannot cover a phrasing this open-ended,
# and a miss here is a published post.
#
# Measured 2026-09-27 over 672 published posts: zero occurrences. The check is
# insurance against the writer, which is told "always President Trump" but can
# still fall back on training-data habit, not a fix for an observed rate.
#
# What it must NOT block, because these are correct:
#   "President Trump"                    -- the plain current-tense title
#   "then-President Trump"               -- his first term, correctly marked
#   "Trump's former adviser"             -- former attaches to someone else
#   "the former president, Joe Biden"    -- former attaches to someone else
_FORMER = r"(?:former|ex[-\s]?)"
_TITLE = r"(?:u\.?\s?s\.?\s+)?(?:president|potus|commander[-\s]in[-\s]chief)"
_NAME = r"(?:donald\s+)?(?:j\.?\s+)?trump"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # former / ex- [U.S.] president [Donald] [J.] Trump
    ("former-president-trump", re.compile(rf"\b{_FORMER}\s*{_TITLE}\s+{_NAME}\b", re.I)),
    # Trump, the former president  /  Donald J. Trump, a former U.S. president
    # -- not "Trump, former President Obama and ...", a list going on to
    # somebody else (a name or "of" after the title)
    ("trump-the-former-president", re.compile(
        rf"\b{_NAME}\s*,\s*(?:the|a|an)?\s*{_FORMER}\s*{_TITLE}\b(?!\s*(?-i:[A-Z])|\s+of\b)", re.I)),
    # the former president, Donald Trump[,.] -- an apposition, so the name is
    # closed off; "after meeting the former president, Trump said" is not one
    ("the-former-president-trump", re.compile(
        rf"\b{_FORMER}\s*{_TITLE}\s*,\s*(?:the\s+)?{_NAME}\b(?=\s*(?:[,.;:!?)]|$))", re.I)),
)
# 2026-10-12, Caidan: 只拦截 former president trump 这种表述，别的前总统没问题.
# Gone: the name-less forms ("the former president said" in a post that names
# Trump), which blocked "the former President of Venezuela", and the "45th
# president" forms -- neither says Trump is a former president.

# Chinese-language captions. Several sibling bots publish translated copy, and
# an English-only pattern would pass their version of the same error straight
# through. 特朗普 and 川普 are both in use for the surname.
_ZH_NAME = r"(?:特朗普|川普)"
_ZH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 前总统特朗普 / 前美国总统川普 / 美国前总统特朗普 / 卸任总统特朗普
    ("zh-former-president-trump",
     re.compile(rf"(?:前|卸任)(?:美国)?总统{_ZH_NAME}|美国前总统{_ZH_NAME}")),
    # 特朗普前总统 — the inverted order
    ("zh-trump-former-president", re.compile(rf"{_ZH_NAME}前总统")),
)
# Must stay allowed: 特朗普总统 (correct), 时任总统特朗普 (his first term,
# correctly marked), 前总统拜登 (a real former president), 特朗普的前顾问
# (former attaches to someone else).
_ZH_ALLOW = re.compile(rf"时任(?:美国)?总统{_ZH_NAME}")

# Same kind of error for two of Trump's own people (2026-10-08, owner: "AI会幻觉他们还是议员",
# and keep it with this check because code is stable where the writer is not).
# JD Vance is the Vice President and Marco Rubio the Secretary of State; both
# left the Senate, and a model's training data still knows them as senators.
# Measured the same day over the channel's last 400 posts: zero occurrences --
# insurance, like the Trump rule. Correct historical forms stay allowed:
# "former Sen. Rubio", "then-Senator Vance", 前参议员卢比奥, 时任参议员万斯.
_SEN = r"(?:sen\.?|senator)"
_EX_SENATORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vance-called-senator", re.compile(
        rf"\b(?:ohio\s+)?{_SEN}\s+(?:j\.?\s?d\.?\s+)?vance\b"
        rf"|\bvance\s*,\s*(?:the|a|an)\s+(?:ohio\s+|republican\s+)?senator\b", re.I)),
    ("rubio-called-senator", re.compile(
        rf"\b(?:florida\s+)?{_SEN}\s+(?:marco\s+)?rubio\b"
        rf"|\brubio\s*,\s*(?:the|a|an)\s+(?:florida\s+|republican\s+)?senator\b", re.I)),
    ("zh-vance-called-senator", re.compile(r"(?:俄亥俄州?)?参议员万斯|万斯参议员")),
    ("zh-rubio-called-senator", re.compile(r"(?:佛罗里达州?|佛州)?参议员(?:卢比奥|鲁比奥)|(?:卢比奥|鲁比奥)参议员")),
)
_HISTORICAL = re.compile(r"(?:former|then|ex)[-\s]*$|(?:前|时任|曾任|昔日)$", re.I)



def former_president_violation(caption: str) -> str | None:
    """Returns the name of the rule a caption breaks, or None if it is clean.

    Checks only the caption -- the text this channel publishes under its own
    name. What the source article says about Trump's first term is the source's
    business and is screened elsewhere; what we write is ours."""
    if not caption:
        return None
    def _report(name: str, m: re.Match[str]) -> str:
        logger.warning(
            "caption_guard: blocked — rule %s matched %r",
            name, caption[max(0, m.start() - 40):m.end() + 40].replace("\n", " "),
        )
        return name

    for name, pattern in _PATTERNS:
        m = pattern.search(caption)
        if m:
            return _report(name, m)

    for name, pattern in _ZH_PATTERNS:
        m = pattern.search(caption)
        if m and not _ZH_ALLOW.search(caption[max(0, m.start() - 6):m.end() + 6]):
            return _report(name, m)

    for name, pattern in _EX_SENATORS:
        for m in pattern.finditer(caption):
            if not _HISTORICAL.search(caption[max(0, m.start() - 12):m.start()]):
                return _report(name, m)
    return None


# Joe Biden, in the scripts this channel's sources are written in. A caption
# that names him when its source never does got him from the model's memory,
# not from the article: on 2026-10-11 Market Watcher's writer (gpt-4o-mini,
# trained on 2023 data) turned TASS's "the US administration" into "the Biden
# administration". Any mention counts, not a list of phrasings -- "President
# Biden", "the Biden White House" and "Biden-era" are all invented when the
# article doesn't name him (Caidan, 2026-10-11: 这个规则好). On AM1ST the same day:
# "a 94% drop compared to the Biden era" under a ProPublica story that never
# names him.
_BIDEN_IN_CAPTION = re.compile(r"\bbiden|拜登", re.I)
_BIDEN_IN_SOURCE = re.compile(r"biden|拜登|байден|바이든|バイデン|بايدن|بایدن|ביידן|μπάιντεν", re.I)


def invented_biden(caption: str, source: str) -> str | None:
    """"invented-biden" when the caption names Biden and the source never does.

    Fails open (None) when the source is missing or under 400 characters, the
    same floor as core/title_guard.py: too little text to say what the article
    leaves out."""
    if not caption or not source or len(source) < 400:
        return None
    if _BIDEN_IN_CAPTION.search(caption) and not _BIDEN_IN_SOURCE.search(source):
        logger.warning("caption_guard: blocked — caption names Biden, source never does")
        return "invented-biden"
    return None
