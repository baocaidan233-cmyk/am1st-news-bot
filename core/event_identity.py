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

import json
import logging
import re
import time
from itertools import combinations
from pathlib import Path

import redis.asyncio as redis
import spacy

from core.config import AppConfig
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

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


def _clean_entity_span(span_text: str) -> str:
    t = span_text.strip().strip("\"'“”‘’")
    t = _TRAILING_POSSESSIVE_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def entity_tokens(text: str) -> set[str]:
    """PERSON/ORG/GPE/LOC/NORP/FAC/EVENT spans, decomposed into lowercase
    word tokens (not kept as whole spans) so "Andy Ogles" and "Ogles"
    overlap. DATE/TIME excluded on purpose — date extraction needs its own
    relevance judgment, not NER on/off (see project_am1st_migration memory)."""
    if not text:
        return set()
    doc = nlp(text)
    tokens: set[str] = set()
    for ent in doc.ents:
        if ent.label_ not in _ENTITY_LABELS:
            continue
        cleaned = _clean_entity_span(ent.text)
        if len(cleaned) < 2 or cleaned.lower() in KNOWN_BYLINE_NOISE:
            continue
        for tok in _TOKEN_RE.findall(cleaned.lower()):
            if tok not in _STOPWORDS and len(tok) > 1:
                tokens.add(tok)
    return tokens


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


async def verify_compatibility(config: AppConfig, matched: dict, new_tokens: set[str], hub_index: HubIndex) -> str:
    """Rule tier only — no LLM, no article text beyond new_tokens. Returns:
      NO_OVERLAP  — confident DIFFERENT_EVENT, no LLM needed
      COMPATIBLE  — confident SAME_EVENT, no LLM needed
      AMBIGUOUS   — every shared token is itself a known multi-event hub; needs the LLM
      FAIL_OPEN   — new_tokens is empty (nothing extracted, e.g. very short text) — trust
                    cosine's own match rather than treat "no evidence" as evidence of difference
    Not designed to be perfect — see EntityVerifierConfig's docstring on
    the accepted residual miss rate."""
    if not new_tokens:
        return "FAIL_OPEN"
    core = core_entities_of(matched)
    if not core:
        return "COMPATIBLE"  # shouldn't normally happen once seed_entities is always set at event creation
    overlap = core & new_tokens
    if not overlap:
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
    toward SAME_EVENT on ~30% of real ambiguous pairs."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._same_event_prompt = Path(config.entity_verifier.same_event_prompt_file).read_text(encoding="utf-8")
        self._subtype_prompt = Path(config.entity_verifier.update_subtype_prompt_file).read_text(encoding="utf-8")

    async def _ask(self, prompt: str, max_tokens: int) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
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
