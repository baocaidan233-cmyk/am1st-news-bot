from __future__ import annotations

import asyncio
import logging
import time
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from core.config import AppConfig

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small


async def ensure_collection_with_retry(store, name: str, attempts: int = 3, delay_seconds: float = 3.0) -> None:
    """Wraps any of this module's `ensure_collection()` methods with retry +
    tolerant continue — called once at process startup in main.py/
    main_publish.py, OUTSIDE run_cycle's own per-cycle try/except. A
    transient connectivity blip here previously crashed the whole
    long-running daemon outright (confirmed live 2026-08-06: a Qdrant
    ConnectTimeout at startup killed main.py entirely, needing a manual
    restart) — unlike every per-item operation elsewhere in this codebase,
    which already fails open. Retries a few times with a short delay, then
    gives up and continues anyway: the collection almost certainly already
    exists from a prior run, and every Qdrant call inside run_cycle already
    fails open on its own if the underlying issue persists."""
    for attempt in range(1, attempts + 1):
        try:
            await store.ensure_collection()
            return
        except Exception:
            logger.exception("ensure_collection_with_retry: failed for %s (attempt %d/%d)", name, attempt, attempts)
            if attempt < attempts:
                await asyncio.sleep(delay_seconds)
    logger.error("ensure_collection_with_retry: never succeeded for %s after %d attempts — continuing anyway", name, attempts)


class QdrantStore:
    """Cross-cycle semantic dedup cache for the INGESTION cycle only — NOT
    the permanent archive (Notion's candidate pool is), and NOT the same
    collection/purpose as PostedHistoryStore below (that one is the
    separate publish cycle's "already posted" cache, keyed on post_content).

    Populated from a candidate's title+description, using the same
    embedding already computed for intra-batch dedup — written once, right
    when that candidate is accepted into the Notion candidate pool. This
    corrects a 2026-08-03 mistake (this class's write method was originally
    called at Gettr-publish time, on post_content, before the "ingestion
    and publish are two separate cycles" architecture was understood — see
    project_am1st_migration memory's 2026-08-04 note).

    Payload schema (content/url/urlHash/publishedAt) matches the real,
    pre-existing n8n system's schema exactly — confirmed 2026-08-05 by
    reading the actual upsert node in v2.15_gdelt_america_first_major_feed_to_notion.json,
    after discovering this collection already held ~2900 real historical
    points under this schema that our own "logged_at"/"title" field names
    made invisible to every query (see project_am1st_migration memory's
    2026-08-05 "field-name mismatch" note — a real bug, not a cold cache).
    publishedAt is the article's own original publish time in Unix seconds
    (matches the n8n source exactly), not when this row was written.

    Periodic delete-by-filter cleanup (retention_days) is intentionally NOT
    implemented here as something the main cycle calls — see standing dedup
    architecture note: it must be a separate, low-frequency scheduled job,
    out of scope for this first build.
    """

    def __init__(self, config: AppConfig) -> None:
        self._collection = config.qdrant.collection
        self._window_seconds = config.qdrant.cross_cycle_window_hours * 3600
        self._client = (
            AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None, timeout=config.qdrant.timeout_seconds)
            if config.qdrant.url
            else None
        )

    async def ensure_collection(self) -> None:
        if self._client is None:
            return
        existing = await self._client.get_collections()
        if self._collection not in {c.name for c in existing.collections}:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("QdrantStore: created collection %s", self._collection)

    async def most_similar_recent(self, embedding: list[float]) -> float:
        """Highest cosine similarity against title+description embeddings
        whose source article was published in the last cross_cycle_window_hours.
        Returns 0.0 if Qdrant isn't configured or nothing matches (fail
        open — never blocks a candidate just because this cache is cold).

        Pure duplicate detection only — as of 2026-08-06, the corroboration/
        heat signal moved to its own dedicated EventStore below (see that
        class's docstring for why: heat needed to persist and accumulate
        across cycles, which a single frozen-at-write-time neighbor lookup
        against THIS collection couldn't do, especially for a near-
        duplicate that gets dropped — see project_am1st_migration memory's
        2026-08-06 "event aggregation" note)."""
        if self._client is None:
            return 0.0
        cutoff = time.time() - self._window_seconds
        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=1,
                query_filter=Filter(must=[FieldCondition(key="publishedAt", range=Range(gte=cutoff))]),
                with_payload=False,
            )
        except Exception:
            logger.exception("QdrantStore: query failed, treating as no match")
            return 0.0
        points = result.points
        return points[0].score if points else 0.0

    async def write_embedding(self, url: str, url_hash: str, content: str, published_at_unix: int, embedding: list[float]) -> None:
        """Called once, right when a candidate is accepted into the Notion
        candidate pool — pass the title+description embedding already
        computed for intra-batch dedup (see class docstring). `content`
        should be the same title+description text the embedding was
        computed from."""
        if self._client is None:
            return
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"content": content, "url": url, "urlHash": url_hash, "publishedAt": published_at_unix},
                )
            ],
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


class EventStore:
    """Event aggregation collection (qdrant.events_collection, default
    am1st_events) — added 2026-08-06 after the first heat-scoring attempt
    (bucketing QdrantStore's own dedup-query neighbors) turned out to lose
    corroboration credit whenever a near-duplicate article got dropped, and
    to be vulnerable to the two classic Topic-Detection-and-Tracking
    failure modes: cluster fragmentation (a differently-worded report of
    the same event fails to match a single fixed anchor) and semantic
    drift (a multi-day evolving story's phrasing moves away from its own
    first report). See project_am1st_migration memory's 2026-08-06 "event
    aggregation" note for the full design discussion.

    Follows the standard TDT "topic tracking" pattern: an event is a GROUP
    of points sharing one `event_id` payload field — not one fixed vector —
    so new representative points can be added as an event's coverage picks
    up new phrasing over time (mitigates fragmentation/drift), while
    heat_score/first_seen_at/last_updated_at/sources stay in sync across
    every point of that event via one filtered payload update (Qdrant's
    set_payload accepts a Filter as the points selector, so this is a
    single API call regardless of how many representative points an event
    has accumulated).

    Payload per point:
      event_id        — uuid shared by every representative point of this event
      source           — the RSS source name that contributed THIS point
      url              — this point's own article's URL — 2026-08-11, was missing entirely (only am1st_embeddings/am1st_posting_news_embedding had it); the user's own earlier "drop event_time, re-derive from the article URL later if ever needed" reasoning depended on this actually being stored, which it wasn't until now
      published_at     — this point's own article's publish time (unix seconds)
      heat_score        — 1.0 + weighted sum of distinct corroborating sources (kept in sync across the event's points)
      first_seen_at     — earliest published_at seen across the event's sources, unix seconds (can only move earlier)
      last_updated_at   — unix seconds of the most recent corroboration — used as the recency filter for matching, NOT first_seen_at, so a still-developing event stays matchable even if it started outside heat.window_hours
      sources           — list of distinct source names already credited, to avoid double-counting the same outlet re-syndicating its own story
      seed_entities     — entity tokens (core/event_identity.entity_tokens) of this event's FIRST article, set once at creation and never changed — see core/event_identity.py's core_entities_of()
      entity_doc_freq   — {token: how many of this event's own accumulated articles contain it}, updated every commit()
      representative_text — title+description of whichever article was this commit()'s representative — a rolling anchor (updates to the latest), used as the "TEXT A" side of EventVerifier's LLM comparisons
      related_event_ids  — [{event_id, cosine_score, linked_at}], set once at this event's creation when it exists BECAUSE a cosine-matched, entity-related candidate was ruled DIFFERENT_EVENT (see core/event_identity.py and main.py's 2026-08-10 note) — a breadcrumb for a future storyline-linking pass, not itself storyline grouping
      canonical_title, canonical_summary — the seed article's own title/description, set once at creation and never changed (same permanence as seed_entities) — deliberately NOT an LLM-rewritten title/summary; see 2026-08-10 event-identity note: the user explicitly ruled out a new LLM call for this once event_time was dropped from scope
      event_type, canonical_action, actor, target — core/event_identity.extract_event_frame()'s free, no-LLM guess (spaCy dependency parse of the seed article, not an LLM) — set once at creation from `seed_frame`, fail-open to empty string when unclear rather than guessing; NOT re-derived on later commits to the same event

    Vector = whichever article's embedding this particular point represents
    (the first point of an event uses that event's originating article;
    later representative points use whichever new-angle article added
    them) — never a recomputed centroid, per the "fixed representative,
    not rolling average" TDT guidance.

    2026-08-06, same day — split from a single record() into peek() +
    preview_heat() + commit(), after the user flagged that record()'s
    "query, then always write" design wrote an event (or bumped one) for
    EVERY local news cluster regardless of whether anything in that
    cluster would end up politically/thematically relevant — sports
    scores, celebrity gossip, etc. all created event-store noise, since
    llm_score-based relevance is only known AFTER scoring, which itself
    happens after the old record() call. Rather than add a second,
    separate (and imperfect) cheap classifier just to gate the write, this
    reuses the score gate that's ALREADY being paid for: main.py now calls
    peek() (read-only) BEFORE scoring to get a preview heat_score to feed
    the scoring prompt, then calls commit() (the actual write) AFTER
    scoring, and ONLY for a cluster that had at least one member clear the
    score threshold — a cluster where every member scored below threshold
    (irrelevant content) never touches this collection at all. See
    project_am1st_migration memory's 2026-08-06 "event aggregation, take
    2" note."""

    def __init__(self, config: AppConfig) -> None:
        self._collection = config.qdrant.events_collection
        self._related_threshold = config.heat.related_threshold
        self._window_seconds = config.heat.window_hours * 3600
        self._major_outlets = set(config.heat.major_outlets)
        self._major_outlet_weight = config.heat.major_outlet_weight
        self._client = (
            AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None, timeout=config.qdrant.timeout_seconds)
            if config.qdrant.url
            else None
        )

    async def ensure_collection(self) -> None:
        """Also creates payload indexes on last_updated_at (range-filtered
        in peek()'s query) and event_id (exact-match-filtered in commit()'s
        set_payload calls) — Qdrant rejects a filter on an unindexed field
        ("Index required but not found"), which the collection didn't have
        from 2026-08-06's first real run: every query failed open that
        cycle (logged, not crashed — the fail-open design worked as
        intended), so event aggregation silently did nothing all cycle.
        create_payload_index is a no-op if the index already exists, so
        this always runs, not just on first creation — it needs to
        backfill the index on the collection created before this fix
        existed."""
        if self._client is None:
            return
        existing = await self._client.get_collections()
        if self._collection not in {c.name for c in existing.collections}:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("EventStore: created collection %s", self._collection)
        await self._client.create_payload_index(
            collection_name=self._collection, field_name="last_updated_at", field_schema=PayloadSchemaType.INTEGER,
        )
        await self._client.create_payload_index(
            collection_name=self._collection, field_name="event_id", field_schema=PayloadSchemaType.KEYWORD,
        )

    async def peek(self, embedding: list[float]) -> dict | None:
        """Read-only — does NOT write anything. Called once per LOCAL
        CLUSTER (see main.py — every member of a cluster is already known,
        via cheap in-memory cosine comparison, to be mutually similar, so
        only one representative embedding needs to be checked against this
        collection, not one query per candidate). Returns the matched
        event's current payload dict (with the match's cosine similarity
        added under "_score") if the top match is within
        heat.related_threshold and heat.window_hours, else None. Fails
        open (returns None, i.e. "no match found") if Qdrant isn't
        configured or the query fails.

        "_score" (2026-08-07) lets main.py distinguish "same event, genuinely
        different angle" (0.6-0.8, still worth its own post even if the
        event was already published) from "same event, near-verbatim rehash"
        (>=0.8) when deciding whether an already-published event's cluster
        should be dropped outright — see mark_published()'s docstring."""
        if self._client is None:
            return None
        cutoff = time.time() - self._window_seconds
        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=1,
                query_filter=Filter(must=[FieldCondition(key="last_updated_at", range=Range(gte=cutoff))]),
                with_payload=True,
            )
            if result.points and result.points[0].score >= self._related_threshold:
                payload = dict(result.points[0].payload or {})
                payload["_score"] = result.points[0].score
                return payload
        except Exception:
            logger.exception("EventStore: peek query failed, treating as no match")
        return None

    async def peek_top_k(self, embedding: list[float], k: int) -> list[dict]:
        """Read-only, like peek() — but returns up to `k` distinct candidate
        events (by cosine descending, each ≥ heat.related_threshold, within
        heat.window_hours) instead of trusting the single most-similar one.
        Added 2026-08-14 (P0 item from the event-identity research memo,
        "难点五"): the true match isn't guaranteed to be Top-1 — a
        coincidentally-closer but actually-unrelated event can outrank the
        real match, and peek()'s old limit=1 design gave main.py no way to
        even see, let alone check, anything past that first (possibly
        wrong) candidate. Caller (main.py's Layer 3) should walk this list
        in order and verify each with the same rule/LLM tiers as before,
        stopping at the first SAME_OCCURRENCE — not assume rank 0 is right.

        Only used by main.py's multi-candidate verification flow — the
        simpler existing peek() (Top-1) stays as-is for main_publish.py's
        mark_published() check, which just needs "does this map to a
        tracked event at all," not a full verification walk.

        Over-fetches raw points (k*4) and dedupes to distinct event_ids,
        keeping each event's highest-scoring point — necessary because an
        event can have multiple representative points (see class
        docstring), so a naive top-k point query could return several
        points that all belong to the same one or two events."""
        if self._client is None:
            return []
        cutoff = time.time() - self._window_seconds
        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=k * 4,
                query_filter=Filter(must=[FieldCondition(key="last_updated_at", range=Range(gte=cutoff))]),
                with_payload=True,
            )
        except Exception:
            logger.exception("EventStore: peek_top_k query failed, treating as no candidates")
            return []
        seen_event_ids: set[str] = set()
        candidates: list[dict] = []
        for point in result.points:
            if point.score < self._related_threshold:
                break  # Qdrant returns points sorted by score descending — nothing after this clears the floor either
            payload = dict(point.payload or {})
            event_id = payload.get("event_id")
            if event_id in seen_event_ids:
                continue  # keep only the first (highest-scoring) point seen per event
            seen_event_ids.add(event_id)
            payload["_score"] = point.score
            candidates.append(payload)
            if len(candidates) >= k:
                break
        return candidates

    def preview_heat(
        self,
        matched: dict | None,
        sources: set[str],
        earliest_source: str,
        earliest_published_at_unix: int,
        heat_multiplier: float = 1.0,
    ) -> tuple[float, int]:
        """Pure computation, no I/O — shared by main.py's pre-scoring
        preview (fed into the scoring prompt) and commit()'s final write,
        so the two never diverge as long as `sources`/`earliest_*` haven't
        changed between the two calls (they don't, in main.py's flow — a
        local cluster's membership is fully resolved before either call
        happens).

        `sources` is the FULL set of distinct source names this cluster
        has accumulated so far (including ones dropped as near-duplicates
        before ever reaching scoring — see main.py). `earliest_source` is
        whichever of those sources has the earliest published_at — it gets
        the flat "1.0 baseline" for a brand-new event (mirrors the original
        record() convention: the very first report doesn't get outlet-
        weighted just for being a major wire service; only SUBSEQUENT
        corroboration does). Picking a fixed, deterministic source for that
        baseline matters — iterating a plain set in arbitrary order would
        make the total heat_score depend on hash order, not on anything
        meaningful.

        `heat_multiplier` (2026-08-14, P0 subtype-weighted heat) — scales
        only the INCREMENTAL contribution from this cluster's new sources
        (never the brand-new-event 1.0 baseline, which has no "existing
        event" to be a subtype of). Caller (main.py) sets this from
        EventVerifier.classify_subtype() — RESTATEMENT barely moves heat,
        CORE_UPDATE moves it more than plain CORROBORATION. Defaults to
        1.0 (today's unweighted behavior) when the caller has no subtype
        opinion, e.g. matched is None."""
        if matched is not None:
            existing_sources = set(matched.get("sources", []))
            heat = matched.get("heat_score", 1.0)
            delta = 0.0
            for s in sources - existing_sources:
                delta += self._major_outlet_weight if s in self._major_outlets else 1.0
            heat += delta * heat_multiplier
            first_seen = min(matched.get("first_seen_at", earliest_published_at_unix), earliest_published_at_unix)
            return heat, first_seen

        heat = 1.0
        for s in sources - {earliest_source}:
            heat += self._major_outlet_weight if s in self._major_outlets else 1.0
        return heat, earliest_published_at_unix

    async def commit(
        self,
        matched: dict | None,
        sources: set[str],
        earliest_source: str,
        earliest_published_at_unix: int,
        representative: tuple[list[float], str, int, str, set[str], str],
        extra_points: list[tuple[list[float], str, int, str, set[str], str]],
        related_links: list[dict] | None = None,
        seed_frame: dict | None = None,
        heat_multiplier: float = 1.0,
        hot_until: int = 0,
        timeline_entry: dict | None = None,
    ) -> tuple[float, int, str, set[str], int]:
        """The actual write — called AFTER scoring, and only for a local
        cluster where at least one member cleared the score gate (see
        main.py). `representative` and each of `extra_points` are now
        (embedding, source, published_at_unix, text, entity_tokens, url) —
        `text` and `entity_tokens` added 2026-08-09 (core/event_identity.py)
        so this event's identity fingerprint (seed_entities/entity_doc_freq)
        and its rolling LLM-comparison anchor (representative_text) can be
        maintained here, instead of re-deriving them from scratch on every
        check by re-reading every past article's text (which is how the
        design was validated in scratchpad, but is not how it should run
        in production — see project_am1st_migration memory's 2026-08-09
        "event identity" note). `url` added 2026-08-11 — was missing
        entirely until the user pointed out the event library has no way
        to go re-read a full article, which the earlier "drop event_time,
        look it up from the URL later" decision had assumed was already
        possible. Each embedding still becomes its own representative
        point sharing one event_id, so future cross-cycle matches can hit
        any of their phrasings, not just the first one (mitigates
        fragmentation/drift, see class docstring).

        `related_links` (2026-08-10, plural since 2026-08-14's Top-K
        redesign) — only passed when `matched` is None BECAUSE the entity
        verifier rejected one or more OTHER candidate events as
        RELATED_DIFFERENT_EVENT while walking main.py's Top-K list (shared/
        hub entities, cosine matched, but a distinct action — e.g.
        "condemns the launch" vs the launch itself): a list of
        `{"event_id": ..., "cosine_score": ...}`, one per rejected
        candidate. Persisted once, at this new event's creation, as a seed
        for storyline linking later — deliberately NOT recorded for a
        NO_OVERLAP rejection, since zero entity relation is more likely an
        unrelated cosine false-positive than a real storyline neighbor.
        This does not itself group events into a storyline — that needs
        its own, separately-thresholded pass (main.py's 2026-08-10 comment)
        — it only keeps the links from being silently discarded before
        that pass exists.

        `seed_frame` (2026-08-10) — core/event_identity.extract_event_frame()
        run on the representative's own title+description, always passed
        by the caller (cheap — no LLM, no I/O) but only consulted when
        `matched` is None: canonical_title/canonical_summary/event_type/
        canonical_action/actor/target get set once from it at this event's
        creation and are never re-derived on later commits, same as
        seed_entities. Deliberately not LLM-generated — see the class
        docstring's payload note on why.

        `heat_multiplier` (2026-08-14) — see preview_heat()'s docstring;
        passed straight through so commit()'s final heat_score exactly
        matches whatever preview_heat() returned earlier for this cluster.

        `hot_until` (2026-08-31) — unix timestamp until which this event
        counts as manually-flagged-hot (core/hot_topics.py, core/config.py's
        HotTopicsConfig), or 0 if this cluster didn't match any currently-
        live flag. The final stored value is max(existing hot_until already
        on a matched event, this call's own hot_until) — so once ANY
        commit to a given event_id has matched a hot-topic flag, every
        later commit to that SAME event_id (future corroborating articles,
        via the usual `matched is not None` carry-forward) inherits hot
        status automatically, without needing to re-match the flag text
        again each time. Never decreases on its own — it only moves
        forward when a fresh match extends it, exactly like heat_score
        only ever accumulating.

        Returns (heat_score, first_seen_at_unix, event_id, core_entities,
        hot_until) — heat_score/first_seen_at_unix should exactly equal
        whatever preview_heat() returned earlier for this same cluster;
        event_id and core_entities (core/event_identity.core_entities_of()
        on the just-written payload) are for the caller to pass to
        HubIndex.bump(); hot_until is the final stored value described
        above, for the caller to derive is_hot = hot_until > now."""
        heat, first_seen = self.preview_heat(matched, sources, earliest_source, earliest_published_at_unix, heat_multiplier)
        now = int(time.time())
        event_id = matched.get("event_id") if matched is not None else str(uuid.uuid4())

        all_points = [representative] + list(extra_points)
        seed_entities = set(matched.get("seed_entities", [])) if matched is not None else all_points[0][4]
        entity_doc_freq = dict(matched.get("entity_doc_freq", {})) if matched is not None else {}
        for _, _, _, _, entities, _ in all_points:
            for tok in entities:
                entity_doc_freq[tok] = entity_doc_freq.get(tok, 0) + 1
        representative_text = representative[3]
        related_event_ids = (
            list(matched.get("related_event_ids", []))
            if matched is not None
            else [{**link, "linked_at": now} for link in (related_links or [])]
        )
        frame = seed_frame or {}
        canonical_title = matched.get("canonical_title", "") if matched is not None else frame.get("canonical_title", "")
        canonical_summary = matched.get("canonical_summary", "") if matched is not None else frame.get("canonical_summary", "")
        event_type = matched.get("event_type", "") if matched is not None else (frame.get("event_type") or "")
        canonical_action = matched.get("canonical_action", "") if matched is not None else (frame.get("action") or "")
        actor = matched.get("actor", "") if matched is not None else (frame.get("actor") or "")
        target = matched.get("target", "") if matched is not None else (frame.get("target") or "")
        final_hot_until = max(matched.get("hot_until", 0) if matched is not None else 0, hot_until)

        # Timeline (2026-08-31) — only a genuine new development (caller
        # passes timeline_entry when EventVerifier.classify_subtype() said
        # CORE_UPDATE, see main.py) gets appended; a brand-new event starts
        # with an empty list (the origin article itself isn't "a new
        # development" of anything). Capped at the most recent 20 entries —
        # a self-imposed defensive limit (this list is denormalized onto
        # every point of the event, so an unbounded list isn't free long-
        # term), not something the design called for; drop the cap if it
        # turns out unnecessary.
        timeline = list(matched.get("timeline", [])) if matched is not None else []
        if timeline_entry is not None:
            timeline.append(timeline_entry)
        timeline = timeline[-20:]

        shared_payload = {
            "heat_score": heat,
            "first_seen_at": first_seen,
            "last_updated_at": now,
            "sources": list(set(matched.get("sources", [])) | sources) if matched is not None else list(sources),
            "seed_entities": list(seed_entities),
            "entity_doc_freq": entity_doc_freq,
            "representative_text": representative_text,
            "related_event_ids": related_event_ids,
            "canonical_title": canonical_title,
            "canonical_summary": canonical_summary,
            "event_type": event_type,
            "canonical_action": canonical_action,
            "actor": actor,
            "target": target,
            "hot_until": final_hot_until,
            "timeline": timeline,
        }

        if matched is not None:
            try:
                await self._client.set_payload(
                    collection_name=self._collection,
                    payload=shared_payload,
                    points=Filter(must=[FieldCondition(key="event_id", match=MatchValue(value=event_id))]),
                )
            except Exception:
                logger.exception("EventStore: failed to update event %s", event_id)

        try:
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={"event_id": event_id, "source": src, "published_at": pub, "url": url, **shared_payload},
                    )
                    for emb, src, pub, _, _, url in all_points
                ],
            )
        except Exception:
            logger.exception("EventStore: failed to add representative point(s) for event %s", event_id)

        core_entities = seed_entities | {t for t, c in entity_doc_freq.items() if c >= 2}
        return heat, first_seen, event_id, core_entities, final_hot_until

    async def mark_published(self, event_id: str) -> None:
        """Called by main_publish.py right after a candidate is actually
        posted to Gettr — flags every point of this event as published, so a
        LATER ingestion cycle's peek() can tell "we already told our
        audience about this" apart from "this event exists in our tracking
        but we never actually published anything about it" (most events:
        most rows in am1st_events never had any member selected for
        publish — see project_am1st_migration memory's 2026-08-07 note).

        2026-08-07: added because publish.posted_dedup_window_hours (the
        publish cycle's own "don't repost something too similar to what we
        posted in the last N hours" check) is deliberately short (24h,
        tightened from the ingestion side's dedup for a reason — see
        main_publish.py's docstring) and compares post_content, which
        doesn't exist yet at ingestion time. This lets main.py catch a
        near-verbatim rehash of an event we ALREADY published, even days
        later, using the same title+description embedding space it already
        has on hand — no cross-text-type comparison, no new collection.

        Writes `posted_to_gettr_at`, NOT `published_at` — found 2026-08-09:
        the original version wrote `published_at`, which every point ALSO
        uses for its own article's original publish time (see class
        docstring). Since this is a blanket payload update across every
        point of the event, it was silently clobbering each point's real
        article-publish timestamp with "whenever we happened to post to
        Gettr" the first time any event got marked published — a real,
        already-shipped bug, fixed here rather than carried forward into
        the new seed_entities/entity_doc_freq fields this same file adds."""
        if self._client is None:
            return
        try:
            await self._client.set_payload(
                collection_name=self._collection,
                payload={"published": True, "posted_to_gettr_at": int(time.time())},
                points=Filter(must=[FieldCondition(key="event_id", match=MatchValue(value=event_id))]),
            )
        except Exception:
            logger.exception("EventStore: failed to mark event %s as published", event_id)

    async def get_by_id(self, event_id: str) -> dict | None:
        """Read-only, filter-only fetch (scroll(), not a vector query) —
        translates an event_id (e.g. one HubIndex.token_events() returned)
        back into that event's current payload, for the cross-event-linking
        pass (main.py) to hand to EventVerifier.related_event() and to pull
        a display title from. Returns None if not found or Qdrant isn't
        configured — fail open, same convention as peek()/peek_top_k()."""
        if self._client is None:
            return None
        try:
            points, _ = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(must=[FieldCondition(key="event_id", match=MatchValue(value=event_id))]),
                limit=1,
                with_payload=True,
            )
        except Exception:
            logger.exception("EventStore: get_by_id query failed for %s, treating as not found", event_id)
            return None
        return dict(points[0].payload or {}) if points else None

    async def link_related_events(self, event_id_a: str, title_a: str, event_id_b: str, title_b: str) -> None:
        """Cross-event storyline linking (2026-08-31) — a separate,
        callable-anytime mutation path from commit()'s own related_event_ids
        (which is only ever set ONCE, at a NEW event's creation, from a
        cosine-matched-but-rejected candidate during THAT SAME cycle — see
        commit()'s docstring). This is the other half: it can reach back
        and update an OLD event that wasn't touched by this cycle's commit
        at all, which is exactly what "storyline linking, found later, from
        either side" requires.

        Writes to BOTH sides (two separate set_payload calls) so either
        event's related_event_ids can be walked to find the other,
        regardless of which one was the "new" event when the link was
        found. Skips (no-op) if this exact pair is already linked, so
        calling this repeatedly across cycles is safe — main.py's caller
        doesn't need to track what it's already linked itself."""
        if self._client is None:
            return
        a = await self.get_by_id(event_id_a)
        b = await self.get_by_id(event_id_b)
        if a is None or b is None:
            return
        a_links = list(a.get("related_event_ids", []))
        b_links = list(b.get("related_event_ids", []))
        if any(link.get("event_id") == event_id_b for link in a_links):
            return  # already linked — same check from either side, since both are written together below
        now = int(time.time())
        a_links.append({"event_id": event_id_b, "title": title_b, "linked_at": now})
        b_links.append({"event_id": event_id_a, "title": title_a, "linked_at": now})
        try:
            await self._client.set_payload(
                collection_name=self._collection,
                payload={"related_event_ids": a_links},
                points=Filter(must=[FieldCondition(key="event_id", match=MatchValue(value=event_id_a))]),
            )
            await self._client.set_payload(
                collection_name=self._collection,
                payload={"related_event_ids": b_links},
                points=Filter(must=[FieldCondition(key="event_id", match=MatchValue(value=event_id_b))]),
            )
        except Exception:
            logger.exception("EventStore: failed to link events %s <-> %s", event_id_a, event_id_b)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


class PostedHistoryStore:
    """The publish cycle's "already posted" cache — a separate collection
    from QdrantStore above, and a separate purpose: QdrantStore dedups
    fresh candidates against each other (title+description) before scoring;
    this dedups the publish cycle's chosen winner against what this channel
    has *actually posted* in the last publish.posted_dedup_window_hours,
    keyed on post_content (the generated caption), not title+description.
    See agents/posted_dedup_checker.py.

    Same real n8n payload schema as QdrantStore above (content/url/urlHash/
    publishedAt) — confirmed against v1.4_am1st_notion_to_gettr_auto
    posting.json's upsert node and the "similarity check flow (posting_
    channels_vectorDB)" query node, 2026-08-05."""

    def __init__(self, config: AppConfig) -> None:
        self._collection = config.qdrant.posted_collection
        self._window_seconds = config.publish.posted_dedup_window_hours * 3600
        self._client = (
            AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None, timeout=config.qdrant.timeout_seconds)
            if config.qdrant.url
            else None
        )

    async def ensure_collection(self) -> None:
        if self._client is None:
            return
        existing = await self._client.get_collections()
        if self._collection not in {c.name for c in existing.collections}:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("PostedHistoryStore: created collection %s", self._collection)

    async def most_similar_recent(self, embedding: list[float]) -> tuple[float, str, str]:
        """Highest cosine similarity against post_content embeddings whose
        source article was published in the last window, plus that match's
        url and its own stored content (2026-09-02, for agents/
        posted_dedup_checker.py's observational entity-overlap logging — see
        that module's docstring) for logging. Returns (0.0, "", "") if
        Qdrant isn't configured, the collection is empty, or the query
        fails — fail open, same as QdrantStore.most_similar_recent."""
        if self._client is None:
            return 0.0, "", ""
        cutoff = time.time() - self._window_seconds
        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=5,
                query_filter=Filter(must=[FieldCondition(key="publishedAt", range=Range(gte=cutoff))]),
                with_payload=True,
            )
        except Exception:
            logger.exception("PostedHistoryStore: query failed, treating as no match")
            return 0.0, "", ""
        points = result.points
        if not points:
            return 0.0, "", ""
        best = max(points, key=lambda p: p.score)
        payload = best.payload or {}
        return best.score, payload.get("url", ""), payload.get("content", "")

    async def write(self, url: str, url_hash: str, content: str, published_at_unix: int, embedding: list[float]) -> None:
        """Called once, right after the publish cycle's winner is chosen —
        never for a rejected/duplicate candidate. `content` should be the
        post_content the embedding was computed from."""
        if self._client is None:
            return
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"content": content, "url": url, "urlHash": url_hash, "publishedAt": published_at_unix},
                )
            ],
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
