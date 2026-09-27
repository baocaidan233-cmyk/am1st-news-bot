from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Replaces the candidate database's dead `titel_check` Notion formula, which
# flagged a hardcoded list of eight names (Shigeru Ishiba, "Peter Hegseth",
# Sanae Takaichi, Hsiao Bi-khim, Tsai Ing-wen, Lai Ching-te, Alexander
# Tah-ray Yui). That formula read prop("post_content"), a column nothing has
# written since the 2026-08-05 restructure moved content generation into the
# publish cycle, so it has not fired once since. Measured 2026-09-27: 0 of
# 500 published rows carry a post_content value, and none of its eight names
# appears in the last 200 live captions.
#
# Who was on that list says what it was really for. "Peter Hegseth" is a
# misspelling -- he is Pete Hegseth -- and Tsai Ing-wen and Shigeru Ishiba
# both left office while this channel was running. So the list is a record of
# two failure modes, not one subject area: a name written wrong, and a title
# that expired. Both are checkable against the source article, which is in
# memory at the moment the caption exists, so this checks the general case
# instead of eight names. A list misses the ninth person; this does not.
#
# Same contract as core/caption_guard.py: returns a rule name or None, and
# blocks rather than annotates.

_TITLE_WORDS = (
    r"president|potus|vice[-\s]president|vp|prime\s+minister|premier|chancellor"
    r"|secretary(?:\s+of\s+(?:state|defense|defence|the\s+treasury|homeland\s+security))?"
    r"|attorney\s+general|solicitor\s+general|surgeon\s+general"
    r"|senator|sen\.|representative|rep\.|congressman|congresswoman|delegate"
    r"|governor|gov\.|lieutenant\s+governor|mayor|ambassador|envoy"
    r"|speaker|majority\s+leader|minority\s+leader|whip"
    r"|chairman|chairwoman|chairperson|chair|director|deputy\s+director|administrator"
    r"|commissioner|inspector\s+general|judge|justice|chief\s+justice|prosecutor"
    r"|general|admiral|colonel|captain|sheriff|chief"
    r"|minister|foreign\s+minister|defense\s+minister|finance\s+minister"
    r"|spokesman|spokeswoman|spokesperson|press\s+secretary"
    r"|ceo|chief\s+executive|chairman\s+of\s+the\s+board"
    r"|counsel|general\s+counsel|special\s+counsel|chief\s+of\s+staff|adviser|advisor"
)

# A qualifier changes who the title refers to in time or in rank. Dropping one
# is the error: "former Prime Minister Ishiba" -> "Prime Minister Ishiba" says
# he holds the office now, which is a claim the source did not make.
_QUALIFIER_WORDS = (
    r"former|ex|outgoing|incoming|acting|interim|past|previous|onetime|one-time"
    r"|then|deputy|vice|assistant|associate|former\s+acting|president-elect|elect"
    r"|ousted|disgraced|retired|late|ex-"
)
# Name parsing is done token by token rather than with one capture group.
# A single regex measured 6.9% false positives on 116 real published captions
# (2026-09-27, every one of them a parsing artefact, none a real error): it
# ran the capture past a possessive ("Michelle Wu's Office" -> surname
# "Office"), across a sentence boundary ("Jim Jordan. Brian" -> surname
# "Brian"), and compared only the token touching the surname, so a middle
# initial read as a contradiction ("Harmeet K. Dhillon" vs source "K.").
# Those are three different stop conditions, which is exactly what a
# tokeniser expresses and a capture group cannot.
_TITLE_RE = re.compile(rf"(?i:\b((?:{_QUALIFIER_WORDS})[-\s]+)?(?:{_TITLE_WORDS}))(?=\s+[A-Z])")
_STOPWORD = re.compile(rf"(?i:^(?:{_TITLE_WORDS}|{_QUALIFIER_WORDS}|mr|mrs|ms|dr|sir|dame|lord)$)")
# The trailing period is part of the token so a middle initial ("Roger T.
# Benitez") does not terminate the name -- it is a two-character token, and
# only a word-final period on a real word means the sentence ended.
_TOKEN = re.compile(r"\s+([A-Z][\w\u00C0-\u024F'\u2019-]*\.?)")
_POSSESSIVE = re.compile(r"['\u2019]s$")


def _read_name(text: str, pos: int, limit: int = 3) -> list[str]:
    """The personal name starting at `pos`, as clean tokens.

    Stops at a possessive (the owner is the name, what follows is a thing),
    at a word-final period (a sentence ended), and at a title or honorific."""
    tokens: list[str] = []
    while len(tokens) < limit:
        m = _TOKEN.match(text, pos)
        if not m:
            break
        tok = m.group(1)
        pos = m.end()
        if _STOPWORD.match(tok):
            if tokens:
                break
            continue          # a stacked title ("General Counsel James") — skip it
        if _POSSESSIVE.search(tok):
            tokens.append(_POSSESSIVE.sub("", tok))
            break
        if tok.endswith(".") and len(tok) > 2:   # "Jordan." — sentence end
            tokens.append(tok[:-1])
            break
        tokens.append(tok.rstrip("."))
    return [t for t in tokens if t]


def _titled_mentions(text: str) -> list[tuple[str | None, list[str]]]:
    """Every "<qualifier?> <title> <Name>" in a text, as (qualifier, tokens)."""
    out: list[tuple[str | None, list[str]]] = []
    for m in _TITLE_RE.finditer(text):
        tokens = _read_name(text, m.end())
        if tokens:
            out.append(((m.group(1) or "").strip() or None, tokens))
    return out


def _source_given_names(source: str, surname: str) -> set[str]:
    """Every capitalised token the SOURCE puts in the given-name slot before
    this surname -- the whole run, not just the adjacent one, so a middle
    name or initial is not read as a contradiction."""
    found: set[str] = set()
    for m in re.finditer(rf"\b{re.escape(surname)}\b", source):
        back = source[max(0, m.start() - 60):m.start()]
        toks = re.findall(r"[A-Z][\w\u00C0-\u024F'\u2019.-]*", back)
        tail: list[str] = []
        for tok in reversed(toks):
            if not back.rstrip().endswith(tok) and tok not in " ".join(back.split()[-3:]):
                break
            tail.append(tok)
            if len(tail) >= 3:
                break
        for tok in tail:
            if not _STOPWORD.match(tok):
                found.add(tok.lower().rstrip("."))
    return found


def title_violation(caption: str, source: str) -> str | None:
    """Returns the name of the rule the caption breaks against its source
    article, or None.

    Both arguments are required and the check is skipped entirely when the
    source is too short to judge against -- an empty or truncated source must
    never manufacture a violation. Fails open in every ambiguous case, which
    is the same convention every other enrichment here follows: this sits in
    front of publishing, so a false positive is a good story silently
    dropped."""
    if not caption or not source or len(source) < 400:
        return None

    for qual, tokens in _titled_mentions(caption):
        surname = tokens[-1]
        if len(surname) < 3 or surname.lower().rstrip(".") in ("the", "his", "her"):
            continue
        # The surname has to be in the source at all; a caption naming
        # someone the article never mentions is core/name-level, not a title
        # question, and is deliberately left to a separate check.
        if not re.search(rf"\b{re.escape(surname)}\b", source):
            continue

        # Rule A -- the given name contradicts the source's. Only fires on a
        # real conflict (the source consistently says something else), never
        # on a gap (the source only ever uses the surname).
        if len(tokens) >= 2:
            given = tokens[0].lower().rstrip(".")
            src_given = _source_given_names(source, surname)
            if src_given and given not in src_given:
                # An initial matching the source's given name is not a conflict.
                if not (len(given) <= 2 and any(g.startswith(given[:1]) for g in src_given)):
                    logger.warning(
                        "title_guard: blocked — caption says %r, source says %s for surname %s",
                        " ".join(tokens), sorted(src_given), surname,
                    )
                    return "given-name-conflict"

        # Rule B -- the caption drops a qualifier the source always carries.
        if qual is None:
            src_titled = [q for q, tk in _titled_mentions(source) if tk[-1] == surname]
            if src_titled and all(q is not None for q in src_titled):
                logger.warning(
                    "title_guard: blocked — caption titles %s with no qualifier, "
                    "source only ever says %s", surname, sorted({q.lower() for q in src_titled}),
                )
                return "title-qualifier-dropped"
    return None
