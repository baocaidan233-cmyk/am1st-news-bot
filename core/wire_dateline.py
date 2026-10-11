"""Recognising a wire dateline -- ported read-only from DailyNews core/dateline.py
(2026-10-11), only the part core/time_words.py needs.

A dateline is where and when an agency filed, not when anything happened:
"WASHINGTON, October 6. /TASS/." carries Moscow's date for an evening in New
York. A day a source gives only in its dateline does not license that day in
our post (core/time_words.py takes the dateline out before reading the source).
"""
from __future__ import annotations

import re

# "WASHINGTON, September 30." / "MOSCOW, Sept. 30." / "BAKU, Azerbaijan, September 29."
# / "TURKMENBASHI /Turkmenistan/, October 9. /TASS/."
_WIRE = re.compile(
    r"(?P<place>^[A-Z][A-Za-z .'-]{1,40}(?:\s*/[A-Z][A-Za-z .'-]{1,30}/)?"
    r"(?:,\s*[A-Z][A-Za-z .'-]{1,30})?,\s*)"
    r"(?P<mon>January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?\s+"
    r"(?P<day>\d{1,2})\b"
)
# （中央社台北28日電） -- the day alone; the month is the filing month.
_CNA = re.compile(r"(（中央社[^）]{0,12}?)(?P<day>\d{1,2})(日電）)")
# "On Friday, October 2, 2026, the Justice Department..." fits _WIRE with "On
# Friday" as a place; it is the event's own date in the body. Nor is
# "Correction, October 5, 2026: ..." a place.
_NOT_A_PLACE = re.compile(
    r"(?i)^\s*(?:on|in|at|by|since|from|until|last|this|next|early|late"
    r"|correction|clarification|update|editor's note)\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")


def wire_dateline(head: str):
    """The western-style dateline at the top of `head`, or None."""
    m = _WIRE.search(head or "")
    if m and _NOT_A_PLACE.search(m.group("place")):
        return None
    return m
