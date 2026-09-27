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
    ("trump-the-former-president", re.compile(rf"\b{_NAME}\s*,\s*(?:the|a|an)?\s*{_FORMER}\s*{_TITLE}", re.I)),
    # the former president, Donald Trump
    ("the-former-president-trump", re.compile(rf"\b{_FORMER}\s*{_TITLE}\s*,\s*(?:the\s+)?{_NAME}\b", re.I)),
    # the 45th president Trump -- an ordinal used as if it were his current one
    ("45th-president-trump", re.compile(rf"\b45th\s+{_TITLE}\s*,?\s*(?:the\s+)?{_NAME}\b", re.I)),
)

# The same error with the name left out: "the former POTUS said", "the 45th
# president announced". These cannot be judged on the phrase alone, because
# they are correct about anyone who really is a former president, so they only
# fire when the caption is about Trump and mentions no other ex-president.
#
# They also have to exclude the attributive use, where the title modifies
# somebody else: "a former POTUS aide" is an aide to some past president, and
# "the former president's adviser" is that president's adviser. Neither says
# anything about who holds the office now. The named patterns above need no
# such guard, because they require Trump's name in the title's own slot.
_NOT_ATTRIBUTIVE = (
    r"(?!\s*(?:'s|\u2019s)\b)"
    r"(?!\s+(?:aide|aides|adviser|advisers|advisor|advisors|official|officials|staffer|staffers"
    r"|spokes\w+|appointee|appointees|nominee|nominees|attorney|attorneys|lawyer|lawyers"
    r"|candidate|candidates|hopeful|hopefuls|ally|allies|team|teams|administration|campaign"
    r"|era|library|museum|memoir|memoirs|son|daughter|family)\b)"
)
_UNNAMED: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("unnamed-former-president",
     re.compile(rf"\b(?:the\s+)?{_FORMER}\s*{_TITLE}\b{_NOT_ATTRIBUTIVE}", re.I)),
    ("unnamed-45th-president",
     re.compile(rf"\b(?:the\s+)?45th\s+{_TITLE}\b{_NOT_ATTRIBUTIVE}", re.I)),
)
_TRUMP = re.compile(r"\btrump\b", re.I)
_OTHER_EX = re.compile(r"\b(biden|obama|bush|clinton|carter|reagan|nixon)\b", re.I)


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

    # Name-less forms, only where they can only mean Trump.
    if _TRUMP.search(caption) and not _OTHER_EX.search(caption):
        for name, pattern in _UNNAMED:
            m = pattern.search(caption)
            if m:
                return _report(name, m)
    return None
