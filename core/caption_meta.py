"""Clauses in which the writer talks about its source instead of the news
(2026-10-27). In the gpt-6-luna tests the writer sometimes left a note for the
editor in the post: "the article does not specify the new charges", "that is
his prediction, not a charge announced in the source" (AM1ST 1 in 20,
International Scandal 2 in 20). Neither the fact check nor is_no_comment
catches them. Code takes the clause out; the rest of the post stays.

Scanned on 21,175 posts published by every Gettr channel up to 10-27: the 16
clauses it matches are all such notes ("The source doesn't say who she is or
where she was taken."), none a news sentence. Attribution to the source ("The
post says Senator Lucas was indicted") is not matched: it is news.
"""
from __future__ import annotations

import re

_SRC = r"(?:article|source|story|piece|text|report(?:ing)?)"
_META = re.compile(
    r"\b(?:the|this) " + _SRC + r"\b[^.;]*?\b(?:does(?:n['’]t| not)|did(?:n['’]t| not)|is(?:n['’]t| not)"
    r"|gives no|offers no|provides no|makes no)\b[^.;]*?\b(?:specify|specified|say|said|state|stated|mention"
    r"|mentioned|name|named|identify|identified|give|given|include|included|provide|provided|confirm|confirmed"
    r"|explain|explained|detail|clear)\b"
    r"|\bnot (?:a |an |the )?[^.;]{0,60}\b(?:announced|stated|given|named|reported|confirmed|mentioned) in the "
    + _SRC + r"\b"
    r"|\b(?:in|from|by) the (?:article|source)\b",
    re.I)
# A source tag hung on a news sentence: only the tag goes ("...in Europe, as
# noted by the article." keeps the sentence).
_TAG = re.compile(r",?\s*(?:as (?:noted|reported|stated|said|described|cited) (?:by|in)|according to) "
                  r"the (?:article|source)(?= *[.!?;]|$)", re.I)
# A full stop that does not end a clause: "U.S.", "D.C.", "J. Smith", "Sen. Cruz".
_NOT_AN_END = re.compile(r"\b([A-Z]|Mr|Mrs|Ms|Dr|Sen|Rep|Gov|Gen|Lt|Col|Sgt|Capt|Adm|Jr|Sr|St|No|Inc|Corp|Ltd|vs)\.")
_HOLD = "\x00"


def drop_meta_clauses(text: str) -> str:
    """The post without any clause that comments on the source; "" when
    nothing else is left. A clause ends at . ! ? or ; -- "X; the article does
    not say Y." keeps "X." """
    if not text or not _META.search(text):
        return text
    text = _TAG.sub("", text)
    out = []
    for para in _NOT_AN_END.sub(lambda m: m.group(1) + _HOLD, text).split("\n"):
        parts = re.split(r"(?<=[.!?;])\s+", para)
        kept = []
        for p in parts:
            if _META.search(p.replace(_HOLD, ".")):
                if kept and kept[-1].endswith(";"):
                    kept[-1] = kept[-1][:-1] + "."
            else:
                kept.append(p)
        out.append(" ".join(kept).strip() if len(kept) < len(parts) else para)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).replace(_HOLD, ".").strip()
