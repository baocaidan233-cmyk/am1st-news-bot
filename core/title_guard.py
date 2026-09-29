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
# The `s` is optional because a name that already ends in one takes a bare
# apostrophe: "Susan Collins' Biddeford office" is Collins possessing an
# office, exactly as "Michelle Wu's Office" is. Requiring the `s` read
# Biddeford -- a city in Maine -- as her surname and blocked the story
# (2026-09-28, live).
_POSSESSIVE = re.compile(r"['\u2019]s?$")


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


# One capitalised token immediately before whatever follows it, nothing but
# whitespace between them. Walking back with this instead of scanning a
# 60-character window is what keeps a neighbouring sentence out of the name:
# the window version collected every capitalised word near the surname, so
# an article reading "Jason Hughes, general manager for Asia, said the Hong
# Kong event in October..." offered {asia, hong, jason, kong, october} as his
# given names, and "Secretary Rubio met the delegation on Monday. Rubio said"
# offered {monday} (2026-09-28, live).
_BACK_TOKEN = re.compile(r"([A-Z][\w\u00C0-\u024F'\u2019-]*\.?)\s+\Z")


def _source_given_names(source: str, surname: str) -> set[str]:
    """Every capitalised token the SOURCE puts in the given-name slot before
    this surname -- the whole run, not just the adjacent one, so a middle
    name or initial is not read as a contradiction.

    The run has to be contiguous, and it ends at a title ("Secretary Rubio"
    gives nothing, which is right: the article never said his given name) or
    at a sentence boundary. Ending empty is the safe outcome -- Rule A does
    not fire at all without a name to contradict."""
    found: set[str] = set()
    for m in re.finditer(rf"\b{re.escape(surname)}\b", source):
        pos = m.start()
        for _ in range(3):
            bm = _BACK_TOKEN.search(source, 0, pos)
            if not bm or bm.end() != pos:
                break
            tok = bm.group(1)
            pos = bm.start()
            # A word-final period ends the sentence; a two-character token
            # ending in one is a middle initial and keeps the run going.
            if tok.endswith(".") and len(tok) > 2:
                break
            tok = tok.rstrip(".")
            if _STOPWORD.match(tok):
                break
            found.add(tok.lower())
    return found


# Short forms that are not a prefix of the formal name, so _is_short_form's
# prefix test cannot derive them. Only the direction nickname -> formal is
# listed, and only that direction is allowed: a caption saying "Russ" where
# the source says "Russell" is the same man written shorter, which is what
# every one of these people is actually called. The reverse -- a caption
# reaching for a formal name the source never uses -- is the error this rule
# was built for ("Peter Hegseth" against a source that says Pete), so it
# stays blocked.
_NICKNAMES = {
    "bill": {"william"}, "billy": {"william"}, "will": {"william"},
    "bob": {"robert"}, "bobby": {"robert"}, "rob": {"robert"},
    "dick": {"richard"}, "rick": {"richard"}, "ricky": {"richard"},
    "mike": {"michael"}, "mickey": {"michael"},
    "tom": {"thomas"}, "tommy": {"thomas"},
    "jim": {"james"}, "jimmy": {"james"}, "jamie": {"james"},
    "joe": {"joseph"}, "joey": {"joseph"},
    "jack": {"john"}, "johnny": {"john"},
    "ted": {"edward", "theodore"}, "teddy": {"edward", "theodore"},
    "ned": {"edward"}, "eddie": {"edward"},
    "dave": {"david"}, "steve": {"stephen", "steven"},
    "tony": {"anthony"}, "andy": {"andrew"}, "drew": {"andrew"},
    "jerry": {"gerald", "jerome"}, "larry": {"lawrence"},
    "chuck": {"charles"}, "charlie": {"charles"},
    "hank": {"henry"}, "harry": {"henry", "harold"},
    "ron": {"ronald"}, "don": {"donald"}, "donnie": {"donald"},
    "ken": {"kenneth"}, "gus": {"august", "gustav"},
    "frank": {"francis"}, "fran": {"francis", "frances"},
    "walt": {"walter"}, "art": {"arthur"}, "bernie": {"bernard"},
    "marty": {"martin"}, "mitch": {"mitchell"}, "sal": {"salvatore"},
    "vinny": {"vincent"}, "gabe": {"gabriel"}, "nate": {"nathan", "nathaniel"},
    "sue": {"susan"}, "susie": {"susan"},
    "liz": {"elizabeth"}, "beth": {"elizabeth"}, "betsy": {"elizabeth"},
    "libby": {"elizabeth"}, "peggy": {"margaret"}, "maggie": {"margaret"},
    "meg": {"margaret"}, "kathy": {"katherine"}, "kate": {"katherine"},
    "katie": {"katherine"}, "cindy": {"cynthia"}, "debbie": {"deborah"},
    "deb": {"deborah"}, "barb": {"barbara"}, "sandy": {"sandra"},
    "patty": {"patricia"}, "trish": {"patricia"}, "pat": {"patrick", "patricia"},
    "jenny": {"jennifer"}, "becky": {"rebecca"}, "mandy": {"amanda"},
    "nikki": {"nicole"}, "nick": {"nicholas"},
}


def _is_short_form(given: str, src_given: set[str]) -> bool:
    """Whether the caption's given name is the source's, written shorter.

    Three ways it can be: the same word, a truncation of it (Russ/Russell,
    Ben/Benjamin), or a nickname the table knows (Bill/William). A bare
    initial matching the source's first letter counts too -- that case was
    already handled here before the table existed."""
    for src in src_given:
        if given == src:
            return True
        if len(given) <= 2 and src.startswith(given[:1]):
            return True
        if len(given) >= 3 and len(src) > len(given) and src.startswith(given):
            return True
        if src in _NICKNAMES.get(given, ()):
            return True
    return False


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
        #
        # Every token before the surname is a candidate, not just the first,
        # which is the same latitude _source_given_names already gives the
        # source side. A title word this list shares with a job title carries
        # the job title into the name -- "General Manager Jason Hughes" reads
        # as given "Manager", surname "Hughes" -- and reading only the first
        # token made that a conflict although the source plainly said Jason
        # (2026-09-28, live, Market Watcher). One match anywhere in the run is
        # enough, so a stacked role, a middle name and a middle initial all
        # stop mattering without a list of role nouns to keep current.
        if len(tokens) >= 2:
            given_run = [t.lower().rstrip(".") for t in tokens[:-1]]
            src_given = _source_given_names(source, surname)
            if src_given and not any(_is_short_form(g, src_given) for g in given_run):
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
