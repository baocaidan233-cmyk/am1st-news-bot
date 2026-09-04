"""Second-opinion check on top of EventStore.peek()'s cosine match — see
core/config.py's EntityVerifierConfig docstring for the full derivation
and why it's shaped this way (validated on real historical am1st_events
data before this was written, not a guess).

Flow: main.py calls verify_compatibility() first (rule tier, no LLM). If
that comes back AMBIGUOUS, main.py calls EventVerifier.same_event() (the
one LLM call that actually gates whether this candidate gets treated as
the same event). classify_subtype() is a separate, optional second call —
only worth making when same_event() said True, and only for enriching the
logged training record; nothing downstream reads its output yet.

Every decision (rule-tier and LLM-tier) should be passed to log_decision()
by the caller — see that function's docstring for why."""

from __future__ import annotations

import html
import json
import logging
import re
import time
from itertools import combinations
from pathlib import Path

import redis.asyncio as redis
import spacy

from collections import Counter

from core.config import AppConfig
from core.hashing import tokenize, weighted_overlap
from core.language import is_english
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

# Gazetteer-based EntityRuler (2026-08-11) — en_core_web_sm's statistical
# NER misses real, important people in compressed headline syntax (no
# title, no verb context — e.g. "Colby pushes back on..."; confirmed 2026-
# 08-11 on a real production headline where the dependency parse correctly
# found "Colby" as the sentence's subject, but it was never tagged PERSON
# at all, so entity_covering() below correctly refused to accept it as an
# actor). The name list itself (537 sitting Congress members from the
# unitedstates/congress-legislators project + 24 Cabinet members, hand-
# verified from Wikipedia + 7 manually-curated "notable but not currently
# in office" figures) was already built and decided on 2026-08-09 — see
# project_am1st_ner_language_gate_decision memory — but was never actually
# wired into the running pipeline until now; it only existed as data in a
# published Artifact. core/gazetteer_names.json is that same data,
# recovered and committed here as a real tracked file.
#
# Only full-name patterns are added for Cabinet/notable (24+7 — no
# ambiguity check was ever computed for these, so a bare-last-name pattern
# risks false-positive PERSON tags on common surnames like "Wright"/
# "Turner"/"Collins"). For Congress, `last_name_ambiguous` (already
# computed in 2026-08-09's work, per-member, based on surname collisions
# WITHIN Congress — e.g. 3 Scotts, 5 Moores) gates whether a bare-last-name
# pattern is added at all: ~30 of the 537 are full-name-only for this
# reason. Known residual risk, not chased further: a bare last name that
# ALSO happens to be an ordinary English word (e.g. "Green"/"Paul"/"Cole")
# can still false-positive-tag unrelated capitalized text — same "accept
# imperfection, don't over-engineer entity rules" philosophy as this
# module's core_entities_of()/HubIndex design elsewhere.
_GAZETTEER_PATH = Path(__file__).parent / "gazetteer_names.json"


def _gazetteer_patterns() -> list[dict]:
    with open(_GAZETTEER_PATH, encoding="utf-8") as f:
        data = json.load(f)
    patterns = []
    for full_name, last, ambiguous in data["congress"]:
        patterns.append({"label": "PERSON", "pattern": full_name})
        if not ambiguous:
            patterns.append({"label": "PERSON", "pattern": last})
    # Cabinet/notable also get a bare-last-name pattern (2026-08-11, user
    # confirmed) — headlines almost always refer to these by surname alone
    # ("Bessent warns...", "Hegseth pushes back..."), and unlike the
    # Congress list, the actor field just stores a guessed name string, not
    # a resolved identity — a same-surname collision (Doug Collins/Susan
    # Collins, Joe Biden/Hunter Biden) doesn't make the guessed string
    # wrong, it's still a real, correct actor name either way.
    for full_name, last in data["cabinet"] + data["notable"]:
        patterns.append({"label": "PERSON", "pattern": full_name})
        # `last: null` opts a specific entry out of the bare-last-name
        # pattern — added for "Mehmet Oz" (2026-08-11): "Oz" alone collides
        # constantly with the "oz." weight abbreviation in ordinary text,
        # a real false-positive risk none of the other ~560 names have.
        if last:
            patterns.append({"label": "PERSON", "pattern": last})
    # Standalone nicknames/initialisms real headlines actually use instead
    # of any stored full/last form (2026-08-11) — "RFK Jr." for the
    # already-listed Robert F. Kennedy Jr., "AOC" for the already-listed
    # Alexandria Ocasio-Cortez. Matched on real production RSS data: both
    # were misses even with the underlying person already in the list.
    for alias in data.get("aliases", []):
        patterns.append({"label": "PERSON", "pattern": alias})
    return patterns


_ruler = nlp.add_pipe("entity_ruler", before="ner")
_ruler.add_patterns(_gazetteer_patterns())

# Not real people — collective/house pseudonyms whose byline shows up on
# unrelated articles, which would otherwise poison entity-based matching
# (see reference_known_byline_noise_entities memory: the BoJ/gold event
# false-merge case was traced back to this exact byline).
KNOWN_BYLINE_NOISE = {"tyler durden", "zero hedge", "zerohedge"}

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT"}
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "as", "by", "with", "from", "that", "this",
    "it", "its", "their", "his", "her", "he", "she", "they", "you", "i",
}
_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_TRAILING_POSSESSIVE_RE = re.compile(r"’s$|'s$")

# Acronym/full-name normalization (2026-09-04) — real production
# fragmentation traced to entity_tokens() decomposing "Immigration and
# Customs Enforcement" into {immigration, customs, enforcement} while the
# same org's own acronym "ICE" tokenizes to {ice} — zero word-level
# overlap despite being the identical entity, so verify_compatibility()'s
# NO_OVERLAP short-circuit wrongly split real duplicate events (confirmed:
# an ICE-whistleblower story and a USPS/mail-voting story both fragmented
# this way). Keyed on the ORG span's full cleaned+lowercased text (not
# decomposed into individual words) so this only fires on the complete
# phrase, never on an incidental shared word like "customs" alone.
_ORG_ACRONYM_MAP = {
    "immigration and customs enforcement": "ice",
    "u.s. immigration and customs enforcement": "ice",
    "customs and border protection": "cbp",
    "u.s. customs and border protection": "cbp",
    "u.s. postal service": "usps",
    "united states postal service": "usps",
    "department of justice": "doj",
    "u.s. department of justice": "doj",
    "federal bureau of investigation": "fbi",
    "department of homeland security": "dhs",
    "u.s. department of homeland security": "dhs",
    "department of health and human services": "hhs",
    "internal revenue service": "irs",
    "environmental protection agency": "epa",
    "centers for disease control and prevention": "cdc",
    "centers for disease control": "cdc",
    "food and drug administration": "fda",
    "drug enforcement administration": "dea",
    "central intelligence agency": "cia",
    "national security agency": "nsa",
    "department of defense": "dod",
    "department of state": "dos",
    "department of education": "ed",
    "justice department": "doj",
    "homeland security department": "dhs",
    "homeland security": "dhs",
    "postal service": "usps",
}

# Direct acronym scan, independent of spaCy NER entirely (2026-09-04) —
# confirmed on real production headlines that en_core_web_sm frequently
# fails to tag well-known government-agency acronyms as entities at all in
# Title-Case headline text ("DOJ threatens...", "ICE Skipped..."), instead
# mis-tagging unrelated capitalized words nearby (e.g. "Whistle-Blower",
# "Recruits" as PERSON/ORG). _ORG_ACRONYM_MAP above only helps when spaCy
# DID find the full-name ORG span; this catches the bare acronym directly
# via regex when spaCy found nothing there to normalize. False-positive
# risk is low — these are short, curated, low-ambiguity tokens (not
# generic words), and a spurious hit only adds a token that doesn't help
# a match, same as any other noise token already tolerated elsewhere in
# this module.
_KNOWN_GOV_ACRONYMS = {
    "ice", "doj", "fbi", "cbp", "usps", "dhs", "irs", "epa", "cdc", "fda",
    "dea", "cia", "nsa", "dod", "dos", "hhs", "atf", "swat", "nypd", "lapd",
    "opm", "gsa", "cbo", "gao", "ftc", "sec", "faa", "nsc", "nih", "who",
}
_ACRONYM_SCAN_RE = re.compile(r"\b[A-Z]{2,5}\b")


def _scan_known_acronyms(text: str) -> set[str]:
    return {tok.lower() for tok in _ACRONYM_SCAN_RE.findall(text) if tok.lower() in _KNOWN_GOV_ACRONYMS}


# Role-title -> current officeholder surname (2026-09-04) — the harder
# fragmentation case neither the acronym map nor a better NER model can
# close: one article names the officeholder ("Markwayne Mullin"), another
# refers to the same person only by role ("Trump's DHS Boss") with zero
# shared words between them. An LLM's own general knowledge was tested and
# found unreliable here (same_event() failed this exact DHS case with
# both the old and new prompt) — officeholders change and any model's
# knowledge can be stale, so this needs AM1ST's own ground truth, not the
# model's. Every mapping below was verified by grepping AM1ST's own
# ingested article text for the officeholder's name co-occurring with the
# title (logs/*.jsonl), NOT assumed from general knowledge — the DHS
# Secretary and Attorney General in this administration are not who
# real-world 2025 news would suggest, so an unverified guess would have
# been wrong. Deliberately excludes "Secretary of State" and "Attorney
# General" despite having verified names for both (Rubio, Blanche) — both
# titles collide with common state-level equivalents ("Texas Attorney
# General," a state's own "Secretary of State" election official), and a
# wrong match here risks a false MERGE (harder to recover from than a
# missed one, per this module's own design bias elsewhere). Needs manual
# upkeep on a cabinet reshuffle — same maintenance tradeoff as the rest of
# this gazetteer, accepted for this small, bounded list. "Vice President"
# is excluded too, for a different reason — too generic (corporate titles,
# foreign officials) to safely assume it means Vance.
_ROLE_TITLE_MAP = {
    "treasury secretary": "bessent",
    "secretary of the treasury": "bessent",
    "defense secretary": "hegseth",
    "secretary of defense": "hegseth",
    "interior secretary": "burgum",
    "secretary of the interior": "burgum",
    "agriculture secretary": "rollins",
    "secretary of agriculture": "rollins",
    "commerce secretary": "lutnick",
    "secretary of commerce": "lutnick",
    "hud secretary": "turner",
    "secretary of housing and urban development": "turner",
    "transportation secretary": "duffy",
    "secretary of transportation": "duffy",
    "energy secretary": "wright",
    "secretary of energy": "wright",
    "dhs secretary": "mullin",
    "dhs boss": "mullin",
    "dhs chief": "mullin",
    "secretary of homeland security": "mullin",
    "homeland security secretary": "mullin",
    "u.s. trade representative": "greer",
    "trade representative": "greer",
}
_ROLE_TITLE_RE = re.compile("|".join(re.escape(k) for k in _ROLE_TITLE_MAP), re.IGNORECASE)


def _scan_role_titles(text: str) -> set[str]:
    return {_ROLE_TITLE_MAP[m.group(0).lower()] for m in _ROLE_TITLE_RE.finditer(text)}


def _clean_entity_span(span_text: str) -> str:
    t = span_text.strip().strip("\"'“”‘’")
    t = _TRAILING_POSSESSIVE_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Some RSS feeds' description field is raw HTML (img tags, class
    attributes, byline spans), not plain text — confirmed 2026-09-02 by
    tracing a real false cross-event-link candidate back to the literal
    word "alt" (from an `alt="..."` attribute) getting NER-tagged as an
    entity, alongside other markup fragments ("item", "field", "em") from
    class names and tag scaffolding elsewhere in the same batch. Every
    caller that runs spaCy NER on article title+description should strip
    tags first — a whole tag (brackets and everything inside, including
    attribute values) is removed rather than just the bracket characters,
    since the attribute values themselves are the actual source of the
    noise words, not just the tag names."""
    return html.unescape(_HTML_TAG_RE.sub(" ", text))


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“])')


def _first_sentence(text: str) -> str:
    """A cheap, regex-based first-sentence cut — no second spaCy parse just
    to find a sentence boundary (entity_tokens()/extract_event_frame()
    already run the real parse on whatever this returns). Same fail-open
    spirit as the rest of this module: an imperfect split (missed
    abbreviation, no match at all) just returns more or less text than a
    perfect parser would, never raises, never blocks anything."""
    cleaned = _strip_html(text).strip()
    if not cleaned:
        return ""
    return _SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)[0].strip()


def event_identity_text(title: str, description: str) -> str:
    """Title + only the description's FIRST SENTENCE — the text every
    event-identity judgment (entity_tokens(), extract_event_frame(),
    same_event(), classify_subtype(), related_event(), and
    verify_compatibility()'s IDF lexical fallback corpus) is built from,
    2026-09-02. Modeled on the EU Joint Research Centre's EMM/NEXUS
    production system, which deliberately narrows per-cluster main-event
    detection to each article's title and first sentence rather than its
    full body — the same design independently explains a real anomaly
    found earlier (2026-08-14): a long, multi-topic Zero Hedge market-
    digest article's full description fanned out into an unusually high
    entity-token count, producing 31 spurious cross-event-link candidates
    from tangential mentions buried deep in the piece, none of which were
    its actual main event. RSS descriptions vary wildly in length and
    shape — some feeds give a one-line summary, others dump the entire
    article body into this field (see agents/rss_fetcher.py) — but the
    title and lead sentence are the one place a news article reliably
    states its own main event, regardless of how long or multi-topic the
    rest of the body is.

    Deliberately scoped to ONLY the event-identity chain — agents/
    scorer.py (relevance) and agents/writer.py (caption generation) still
    see the full article; this is a different, narrower question ("which
    event is this") than "is this worth covering" or "what should the post
    say," and narrowing their input would lose real substance those steps
    need."""
    return f"{title}\n{_first_sentence(description)}"


_LOCATION_LABELS = {"GPE", "LOC", "FAC"}
_NUMERIC_LABELS = {"CARDINAL", "QUANTITY", "PERCENT", "MONEY"}


def no_conflicting_specifics(text_a: str, text_b: str) -> bool:
    """True only if A and B name the same set of places/facilities and the
    same set of numbers. Added 2026-09-02 as a pre-check before
    classify_subtype(): high overall cosine similarity alone doesn't mean
    "no new information" — two sentences that are otherwise identical
    except for one swapped location ("Kuwait" -> "Qatar") or one changed
    figure ("10 dead" -> "20 dead") still score high on cosine while
    describing a materially different fact. Deliberately conservative:
    ANY difference in either set (including an added, not just a swapped,
    place/number) returns False, sending the pair to the LLM as before —
    this only short-circuits the genuinely unambiguous case."""
    if not text_a or not text_b:
        return False

    def _locations(text: str) -> set[str]:
        doc = nlp(_strip_html(text))
        return {
            tok
            for ent in doc.ents
            if ent.label_ in _LOCATION_LABELS
            for tok in _TOKEN_RE.findall(_clean_entity_span(ent.text).lower())
            if tok not in _STOPWORDS and len(tok) > 1
        }

    def _numbers(text: str) -> set[str]:
        doc = nlp(_strip_html(text))
        return {
            cleaned
            for ent in doc.ents
            if ent.label_ in _NUMERIC_LABELS
            for cleaned in [re.sub(r"[^0-9.]", "", ent.text)]
            if cleaned
        }

    return _locations(text_a) == _locations(text_b) and _numbers(text_a) == _numbers(text_b)


def entity_tokens(text: str) -> set[str]:
    """PERSON/ORG/GPE/LOC/NORP/FAC/EVENT spans, decomposed into lowercase
    word tokens (not kept as whole spans) so "Andy Ogles" and "Ogles"
    overlap. DATE/TIME excluded on purpose — date extraction needs its own
    relevance judgment, not NER on/off (see project_am1st_migration memory).

    2026-09-02: PERSON spans keep only their LAST word (the surname in
    Western name order) instead of every constituent word — a real false
    cross-event-link candidate was traced to "Tony" alone (from "Judge Tony
    Graf") coincidentally matching an unrelated article about "Gov. Tony
    Evers," two different people who just share a first name. A bare given
    name is common enough across unrelated people to be weak, risky
    identity evidence on its own; the surname is what actually
    distinguishes someone, and this keeps the original "Andy Ogles"/
    "Ogles" partial-match goal intact for the more identifying half of the
    name. Non-PERSON labels are unaffected — "Supreme Court" both words
    matter, "Kuwait" is one word regardless."""
    if not text:
        return set()
    doc = nlp(_strip_html(text))
    tokens: set[str] = set()
    for ent in doc.ents:
        if ent.label_ not in _ENTITY_LABELS:
            continue
        cleaned = _clean_entity_span(ent.text)
        if len(cleaned) < 2 or cleaned.lower() in KNOWN_BYLINE_NOISE:
            continue
        if ent.label_ == "ORG":
            org_key = cleaned.lower()
            for article in ("the ", "a ", "an "):
                if org_key.startswith(article):
                    org_key = org_key[len(article):]
                    break
            acronym = _ORG_ACRONYM_MAP.get(org_key)
            if acronym:
                tokens.add(acronym)
        words = _TOKEN_RE.findall(cleaned.lower())
        if ent.label_ == "PERSON" and len(words) > 1:
            words = words[-1:]
        for tok in words:
            if tok not in _STOPWORDS and len(tok) > 1:
                tokens.add(tok)
    tokens |= _scan_known_acronyms(text)
    tokens |= _scan_role_titles(text)
    return tokens


# Deliberately small and non-exhaustive (2026-08-10 design note: event_type
# is a coarse retrieval aid, not an identity-determining field — see
# 新闻事件库研究 2026.md §3.1's own "don't chase an exhaustive ontology"
# guidance) — an unmatched verb lemma just leaves event_type empty rather
# than guessing, same fail-open philosophy as the rest of this module.
# Keyed on the ROOT verb's lemma from extract_event_frame() below.
_ACTION_TYPE_MAP = {
    "sanction": "sanction", "impose": "sanction", "ban": "sanction", "tariff": "sanction",
    "arrest": "arrest", "charge": "arrest", "indict": "arrest", "detain": "arrest",
    "raid": "arrest", "deport": "arrest", "seize": "arrest",
    "rule": "court_ruling", "convict": "court_ruling", "acquit": "court_ruling",
    "sentence": "court_ruling", "overturn": "court_ruling", "dismiss": "court_ruling",
    "uphold": "court_ruling",
    "sue": "lawsuit", "appeal": "lawsuit", "settle": "lawsuit",
    "launch": "attack", "attack": "attack", "bomb": "attack", "invade": "attack",
    "kill": "attack", "strike": "attack",
    "elect": "election", "vote": "election", "win": "election", "lead": "election", "trail": "election",
    "meet": "meeting", "sign": "policy_action", "approve": "policy_action",
    "pass": "policy_action", "veto": "policy_action", "block": "policy_action", "revoke": "policy_action",
    "announce": "statement", "condemn": "statement", "praise": "statement", "back": "statement",
    "criticize": "statement", "warn": "statement", "deny": "statement", "confirm": "statement",
    "blast": "statement", "slam": "statement", "rip": "statement", "torch": "statement",
    "knock": "statement", "hail": "statement", "defend": "statement", "tout": "statement",
    "dump": "statement", "expose": "statement",
    "protest": "protest", "rally": "protest", "march": "protest", "boo": "protest",
    "resign": "resignation", "fire": "personnel_action", "appoint": "personnel_action",
    "nominate": "personnel_action", "endorse": "personnel_action",
    "surge": "economic_data", "plunge": "economic_data", "spike": "economic_data",
    "tank": "economic_data", "jump": "economic_data", "soar": "economic_data",
}


def extract_event_frame(text: str) -> dict:
    """Free, no-LLM ACTION/ACTOR/TARGET/event_type guess from the same kind
    of spaCy parse entity_tokens() already runs — 2026-08-10, replaces an
    earlier LLM-extraction design the user rejected as unnecessary cost
    once event_time was dropped from scope entirely (no consumer needs it;
    if a future feature does, the article URL is already on file to go
    re-read it). Reads the ROOT verb of the first sentence (usually the
    headline) and its nsubj/dobj/pobj children — this is meaningfully
    weaker than an LLM on complex sentences (subordinate clauses, passive
    voice, coordination) and is expected to leave actor/target empty often;
    that's intentional fail-open, not a bug — a wrong guess is worse than
    an empty field here, same philosophy as verify_compatibility(). actor/
    target are only accepted if they land inside a real recognized entity
    span (cross-checked against the same NER this module already does),
    not just any noun.

    Returns {"action": str|None, "actor": str|None, "target": str|None,
    "event_type": str|None} — all None if no clear verb root was found.

    2026-08-10, added after a real-data smoke test: en_core_web_sm run on
    non-English text (this collection has occasional Chinese/Portuguese
    items — see core/language.py's docstring for prior, unrelated
    incidents of the same root cause) produces fluent-looking nonsense —
    e.g. a Portuguese sentence's root verb lemma comes back as a real-
    looking but wrong token. is_english() already exists for exactly this
    class of problem; reused here rather than writing a second check."""
    empty = {"action": None, "actor": None, "target": None, "event_type": None}
    if not text or not is_english(text):
        return empty
    doc = nlp(_strip_html(text))
    entity_spans = [
        (ent.start, ent.end, _clean_entity_span(ent.text))
        for ent in doc.ents
        if ent.label_ in _ENTITY_LABELS and _clean_entity_span(ent.text).lower() not in KNOWN_BYLINE_NOISE
    ]

    def entity_covering(token) -> str | None:
        for start, end, cleaned in entity_spans:
            if start <= token.i < end:
                return cleaned
        return None

    for sent in doc.sents:
        root = sent.root
        if root.pos_ not in ("VERB", "AUX"):
            continue
        action = root.lemma_.lower()
        actor = target = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass") and actor is None:
                actor = entity_covering(child)
            elif child.dep_ == "dobj" and target is None:
                target = entity_covering(child)
            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj" and target is None:
                        target = entity_covering(grandchild)
        return {"action": action, "actor": actor, "target": target, "event_type": _ACTION_TYPE_MAP.get(action)}
    return empty


class HubIndex:
    """Persistent, cross-event 'how many distinct past events has this
    token/pair been the CORE of' counter — replaces a hand-maintained
    blocklist (cabinet officials, country names, ...) with a number that
    updates itself from real history and correctly tells apart, e.g.,
    Fauci (locally core to many of his OWN articles, but only ever the
    core of one storyline) from Rubio (globally rarer, but historically
    core to several unrelated events just because he's quoted on many
    different topics as Secretary of State).

    Each token/pair gets its own Redis SET of event_ids — SADD is
    idempotent, so re-bumping an event_id a token has already been
    credited for is a no-op — SCARD is then the count of distinct events.
    Same REDIS_URL as RedisStore, a separate key namespace
    (config.entity_verifier.hub_key_prefix) so the two never collide."""

    def __init__(self, config: AppConfig) -> None:
        self._prefix = config.entity_verifier.hub_key_prefix
        self._client = (
            redis.from_url(config.redis.url, decode_responses=True, socket_timeout=10, socket_connect_timeout=10)
            if config.redis.url
            else None
        )

    async def token_score(self, token: str) -> int:
        if self._client is None:
            return 0
        try:
            return await self._client.scard(f"{self._prefix}tok:{token}")
        except Exception:
            logger.exception("HubIndex: token_score failed for %s — treating as 0 (fail open)", token)
            return 0

    async def pair_score(self, token_a: str, token_b: str) -> int:
        if self._client is None:
            return 0
        t1, t2 = sorted((token_a, token_b))
        try:
            return await self._client.scard(f"{self._prefix}pair:{t1}|{t2}")
        except Exception:
            logger.exception("HubIndex: pair_score failed for %s|%s — treating as 0 (fail open)", token_a, token_b)
            return 0

    async def token_events(self, token: str) -> set[str]:
        """Reverse lookup (2026-08-31) — every distinct event_id this token
        has been the CORE of, not just the count token_score() already
        gave. Same underlying Redis SET bump() has always written
        (`tok:{token}`); this just reads the members instead of SCARD.
        Used by main.py's cross-event-linking pass to find storyline
        neighbors for a freshly-committed event: a token specific enough to
        pass token_score() < hub_event_count_threshold is unlikely to
        return more than a couple of event_ids here."""
        if self._client is None:
            return set()
        try:
            return await self._client.smembers(f"{self._prefix}tok:{token}")
        except Exception:
            logger.exception("HubIndex: token_events failed for %s — treating as empty (fail open)", token)
            return set()

    async def bump(self, event_id: str, core_tokens: set[str]) -> None:
        """Called once per EventStore.commit() with the event's current
        full core set — safe to call every time across an event's whole
        lifetime since SADD on an already-present event_id no-ops."""
        if self._client is None or not core_tokens:
            return
        try:
            pipe = self._client.pipeline()
            for tok in core_tokens:
                pipe.sadd(f"{self._prefix}tok:{tok}", event_id)
            for t1, t2 in combinations(sorted(core_tokens), 2):
                pipe.sadd(f"{self._prefix}pair:{t1}|{t2}", event_id)
            await pipe.execute()
        except Exception:
            logger.exception("HubIndex: bump failed for event %s — continuing without it (fail open)", event_id)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def core_entities_of(matched: dict) -> set[str]:
    """An event's persisted identity fingerprint: its seed article's own
    entities (permanent — this is what stops a chain of gradually-drifting
    articles from wandering arbitrarily far from where the event actually
    started) UNION whichever tokens have shown up in at least 2 of its own
    accumulated articles since. Deliberately a raw count, not a ratio — a
    ratio lets a single one-off token qualify as "core" while an event
    still only has 1-2 articles on record (validated real failure mode,
    2026-08-09)."""
    seed = set(matched.get("seed_entities", []))
    doc_freq = matched.get("entity_doc_freq", {})
    recurring = {t for t, c in doc_freq.items() if c >= 2}
    return seed | recurring


async def verify_compatibility(
    config: AppConfig,
    matched: dict,
    new_tokens: set[str],
    hub_index: HubIndex,
    new_text: str = "",
    doc_freq: Counter | None = None,
    doc_count: int = 0,
    cosine_score: float = 0.0,
) -> str:
    """Rule tier only — no LLM. Returns:
      NO_OVERLAP  — confident DIFFERENT_EVENT, no LLM needed
      COMPATIBLE  — confident SAME_EVENT, no LLM needed
      AMBIGUOUS   — needs the LLM (every shared entity token is a known
                    multi-event hub; new_tokens was empty and the lexical
                    fallback below couldn't confirm a match either; or
                    overlap was empty but cosine_score cleared
                    no_overlap_llm_review_floor)
      FAIL_OPEN   — new_tokens is empty AND no lexical fallback was available
                    to this call (doc_freq/new_text not supplied) — trust
                    cosine's own match, the pre-2026-08-20 behavior, kept only
                    as a defensive default for callers that don't pass a corpus

    2026-08-20: when new_tokens is empty (NER extracted nothing — very short
    text, or a genuine extraction miss), this used to blindly trust
    whatever cosine match it was handed, with zero independent check. Now,
    if the caller supplies a doc_freq/doc_count corpus (see main.py — built
    once per cycle from that cycle's own batch, no new database), an
    IDF-weighted lexical overlap check (core/hashing.py's weighted_overlap,
    ported from North_Korea_News's real-incident-driven design) gets a say:
    confirms COMPATIBLE if it agrees, otherwise downgrades to AMBIGUOUS
    (real LLM check) instead of blind trust. This can only make the
    zero-entity case MORE scrutinized than before, never less.
    Not designed to be perfect — see EntityVerifierConfig's docstring on
    the accepted residual miss rate."""
    if not new_tokens:
        rep_text = matched.get("representative_text", "")
        if doc_freq is not None and new_text and rep_text:
            overlap = weighted_overlap(tokenize(new_text), tokenize(rep_text), doc_freq, doc_count)
            verdict = "COMPATIBLE" if overlap >= config.entity_verifier.weighted_overlap_threshold else "AMBIGUOUS"
            # 2026-09-01: this branch's own hit rate/score distribution was
            # previously invisible in event_identity_decisions.jsonl — a rule-
            # tier COMPATIBLE/AMBIGUOUS from here looked identical to one from
            # the normal entity-overlap path below. Logged separately
            # (check_type) so weighted_overlap_threshold can eventually be
            # recalibrated on AM1ST's own data instead of North_Korea_News's.
            log_decision(config, {
                "check_type": "lexical_fallback",
                "candidate_event_id": matched.get("event_id"),
                "cosine_score": matched.get("_score"),
                "weighted_overlap_score": overlap,
                "threshold": config.entity_verifier.weighted_overlap_threshold,
                "rule_verdict": verdict,
            })
            return verdict
        return "FAIL_OPEN"
    core = core_entities_of(matched)
    if not core:
        return "COMPATIBLE"  # shouldn't normally happen once seed_entities is always set at event creation
    overlap = core & new_tokens
    if not overlap:
        # 2026-09-04: zero literal entity overlap at high cosine is not
        # always a different event — see EntityVerifierConfig's
        # no_overlap_llm_review_floor docstring. Below the floor, still a
        # confident NO_OVERLAP (no LLM call); at/above it, downgraded to
        # AMBIGUOUS so the LLM's own world knowledge gets a chance to
        # catch the "same actor/institution, different surface form" case
        # that no dictionary or NER fix can close.
        if cosine_score >= config.entity_verifier.no_overlap_llm_review_floor:
            return "AMBIGUOUS"
        return "NO_OVERLAP"
    threshold = config.entity_verifier.hub_event_count_threshold
    non_hub = set()
    for tok in overlap:
        if await hub_index.token_score(tok) < threshold:
            non_hub.add(tok)
    if non_hub:
        return "COMPATIBLE"
    pair_max = config.entity_verifier.pair_cooccur_max
    for t1, t2 in combinations(sorted(overlap), 2):
        if await hub_index.pair_score(t1, t2) <= pair_max:
            return "COMPATIBLE"
    return "AMBIGUOUS"


class EventVerifier:
    """The LLM tier for whatever verify_compatibility() couldn't resolve.
    same_event() gates the actual merge decision. classify_subtype() is a
    deliberately SEPARATE second call, made only when same_event() said
    True — see EntityVerifierConfig's docstring for the ablation test
    (2026-08-09) that found asking both in one prompt biases the model
    toward SAME_EVENT on ~30% of real ambiguous pairs.

    On config.openai.chat_model (gpt-4o-mini). Moved onto gpt-5-nano
    2026-08-26, then back on 2026-09-01 — a live multi-cycle test found
    real judgment failures on both same_event() and related_event(), in
    OPPOSITE directions: same_event() wrongly split a genuine same-event
    paraphrase pair ("charged" vs "indicted" on the same fraud case) into
    two different events, while related_event() (see prompts/
    related_event_prompt.txt) rationalized a string of topically-adjacent
    but unrelated financial-news articles as the same storyline. The
    2026-08-26 switch had only verified well-formed, non-empty output at
    the time, never actual judgment quality — see agents/scorer.py's
    docstring, which found the same gap independently for Scorer's own
    role. Reverted every nano_model consumer at once rather than
    re-litigating per call site, per the user's explicit call."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._same_event_prompt = Path(config.entity_verifier.same_event_prompt_file).read_text(encoding="utf-8")
        self._subtype_prompt = Path(config.entity_verifier.update_subtype_prompt_file).read_text(encoding="utf-8")
        self._related_event_prompt = Path(config.entity_verifier.related_event_prompt_file).read_text(encoding="utf-8")

    async def _ask(self, prompt: str, max_tokens: int) -> str:
        kwargs = dict(model=self._model, messages=[{"role": "user", "content": prompt}])
        if self._model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = max_tokens
        resp = await self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _extract_field(text: str, field: str) -> str:
        for line in text.splitlines():
            if line.upper().startswith(field):
                return line.split(":", 1)[1].strip()
        return ""

    async def same_event(self, text_a: str, text_b: str) -> tuple[bool, str]:
        raw = await self._ask(self._same_event_prompt.format(a=text_a, b=text_b), max_tokens=80)
        verdict = self._extract_field(raw, "VERDICT").upper()
        return verdict.startswith("SAME"), raw

    async def classify_subtype(self, text_a: str, text_b: str) -> tuple[str, str]:
        raw = await self._ask(self._subtype_prompt.format(a=text_a, b=text_b), max_tokens=60)
        subtype = self._extract_field(raw, "SUBTYPE").upper()
        return subtype, raw

    async def related_event(self, text_a: str, text_b: str) -> tuple[bool, str]:
        """Cross-event storyline check (2026-08-31) — called only on two
        events ALREADY confirmed distinct (see main.py's cross-event-
        linking pass); asks whether B is a genuine follow-up/consequence of
        A's storyline, not whether they're the same occurrence. See
        prompts/related_event_prompt.txt."""
        raw = await self._ask(self._related_event_prompt.format(a=text_a, b=text_b), max_tokens=80)
        verdict = self._extract_field(raw, "RELATED").upper()
        return verdict.startswith("YES"), raw


def log_decision(config: AppConfig, record: dict) -> None:
    """Appends one JSON line per rule-tier/LLM decision — the training-
    data asset the 2026-08-09 design discussion committed to keeping, so
    that residual rule-tier misses (an accepted, non-blocking cost — see
    EntityVerifierConfig's docstring) become future hard-negative examples
    for distilling a cheap pairwise model, instead of silently recurring
    forever with no record. Best-effort: a logging failure must never take
    down the ingestion cycle."""
    try:
        path = Path(config.entity_verifier.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full_record = {"logged_at": int(time.time()), **record}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("log_decision: failed to write training-data log entry — continuing")
