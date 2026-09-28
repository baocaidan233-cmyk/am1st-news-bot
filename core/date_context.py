from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# The writer is asked to summarise an article without being told when the
# article was written. Every relative date in the source ("next January",
# "明年1月") is then unresolvable, and a bare month with no year ("January to
# August") has no year to belong to -- so the model supplies one from
# parametric memory, which is how a 2023 lands in a 2026 post.
#
# Two separate holes, measured 2026-09-28: AM1ST passes today's date but not
# the article's, so it cannot tell a 3-day-old article's "next week" from its
# own; China Breaks and Market Watcher pass neither, so their writers know no
# date at all.
#
# This resolves what can be resolved deterministically and hands the writer
# the answer, rather than the inputs plus an instruction to compute. A model
# asked to do arithmetic AND recall AND write in one call fails at the
# arithmetic first -- the same reason Market Watcher's money handling is
# code and not prompt (see the AI-cannot-do-money-arithmetic note).
#
# Deliberately conservative. A mention is only resolved when no four-digit
# year sits near it, because an explicit year is the source telling us the
# answer and nothing here should override it. Anything it cannot resolve is
# simply left out, and an article with nothing resolvable adds nothing to the
# message at all -- byte-identical to the previous behaviour.

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_NAME = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")

_NEAR_YEAR = re.compile(r"\b(19[89]\d|20[0-5]\d)\b")
_NEAR = 25          # characters either side that count as "the source said the year"
_MAX_ITEMS = 6      # keep the block short; the long tail is rarely what breaks

_ZH_REL_MONTH = re.compile(r"(明年|去年|今年|本年)\s*(\d{1,2})\s*月")
_ZH_REL_YEAR = re.compile(r"(明年|去年|前年|今年)(?!\s*\d{1,2}\s*月)")
_ZH_BARE_MONTH = re.compile(r"(?<![\d年])(\d{1,2})\s*月(?!\s*\d{1,2}\s*日?\s*至)")
_EN_REL_MONTH = re.compile(
    r"\b(next|last|this)\s+(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b", re.I)
_EN_REL_YEAR = re.compile(r"\b(next|last|this)\s+year\b", re.I)
_EN_BARE_MONTH = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?", re.I)

_ZH_OFFSET = {"明年": 1, "去年": -1, "前年": -2, "今年": 0, "本年": 0}
_EN_OFFSET = {"next": 1, "last": -1, "this": 0}


def _has_year_near(text: str, start: int, end: int) -> bool:
    return bool(_NEAR_YEAR.search(text[max(0, start - _NEAR):end + _NEAR]))


def resolve_dates(article: str, article_year: int) -> list[tuple[str, str]]:
    """(what the source said, what it resolves to) for each mention this can
    settle from the article's own publication year."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    taken: list[tuple[int, int]] = []     # spans a relative rule already claimed

    def add(span: str, resolved: str, at: tuple[int, int] | None = None) -> None:
        key = span.strip().rstrip(".")
        if not key or key in seen or len(out) >= _MAX_ITEMS:
            return
        seen.add(key)
        out.append((key, resolved))
        if at:
            taken.append(at)

    def overlaps(start: int, end: int) -> bool:
        return any(start < b and a < end for a, b in taken)

    # Relative forms first: they are unambiguous and they also cover the bare
    # month sitting inside them ("next January" must not also emit "January").
    for m in _ZH_REL_MONTH.finditer(article):
        mo = int(m.group(2))
        if 1 <= mo <= 12:
            add(m.group(0), "%s %d" % (_MONTH_NAME[mo - 1], article_year + _ZH_OFFSET[m.group(1)]),
                (m.start(), m.end()))
    for m in _EN_REL_MONTH.finditer(article):
        mo = _MONTHS[m.group(2).lower()]
        add(m.group(0), "%s %d" % (_MONTH_NAME[mo - 1], article_year + _EN_OFFSET[m.group(1).lower()]),
            (m.start(), m.end()))
    for m in _ZH_REL_YEAR.finditer(article):
        add(m.group(0), str(article_year + _ZH_OFFSET[m.group(1)]), (m.start(), m.end()))
    for m in _EN_REL_YEAR.finditer(article):
        add(m.group(0), str(article_year + _EN_OFFSET[m.group(1).lower()]), (m.start(), m.end()))

    # Bare months: the year is missing rather than relative. Only where the
    # source gives no year nearby, and only once per month.
    months_done: set[int] = set()
    for pattern, is_zh in ((_ZH_BARE_MONTH, True), (_EN_BARE_MONTH, False)):
        for m in pattern.finditer(article):
            if overlaps(m.start(), m.end()) or _has_year_near(article, m.start(), m.end()):
                continue
            raw = m.group(1)
            mo = int(raw) if is_zh else _MONTHS[raw.lower()]
            if not 1 <= mo <= 12 or mo in months_done:
                continue
            months_done.add(mo)
            add(m.group(0), "%s %d" % (_MONTH_NAME[mo - 1], article_year))
    return out


def date_block(article: str, published_at: datetime | None, now: datetime | None = None) -> str:
    """The date metadata to put in the writer's user message, or "" when
    there is nothing to say."""
    now = now or datetime.now(timezone.utc)
    lines = ["Today's date: %s" % now.strftime("%Y-%m-%d")]
    if published_at is not None:
        lines.append("This article was published: %s" % published_at.strftime("%Y-%m-%d"))
        try:
            resolved = resolve_dates(article or "", published_at.year)
        except Exception:
            logger.exception("date_context: resolution failed — continuing with dates only")
            resolved = []
        if resolved:
            lines.append(
                "Dates already resolved from this article's publication date — use these "
                "values as written, do not work them out again:")
            lines += ['  %s = %s' % (span, val) for span, val in resolved]
    return "\n".join(lines)
