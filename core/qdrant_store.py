from __future__ import annotations

import logging
import time
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from core.config import AppConfig

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small


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
            AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None)
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
      published_at     — this point's own article's publish time (unix seconds)
      heat_score        — 1.0 + weighted sum of distinct corroborating sources (kept in sync across the event's points)
      first_seen_at     — earliest published_at seen across the event's sources, unix seconds (can only move earlier)
      last_updated_at   — unix seconds of the most recent corroboration — used as the recency filter for matching, NOT first_seen_at, so a still-developing event stays matchable even if it started outside heat.window_hours
      sources           — list of distinct source names already credited, to avoid double-counting the same outlet re-syndicating its own story

    Vector = whichever article's embedding this particular point represents
    (the first point of an event uses that event's originating article;
    later representative points use whichever new-angle article added
    them) — never a recomputed centroid, per the "fixed representative,
    not rolling average" TDT guidance."""

    def __init__(self, config: AppConfig) -> None:
        self._collection = config.qdrant.events_collection
        self._related_threshold = config.heat.related_threshold
        self._window_seconds = config.heat.window_hours * 3600
        self._major_outlets = set(config.heat.major_outlets)
        self._major_outlet_weight = config.heat.major_outlet_weight
        self._client = (
            AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None)
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
            logger.info("EventStore: created collection %s", self._collection)

    async def record(
        self, embedding: list[float], source: str, published_at_unix: int, add_representative: bool,
    ) -> tuple[float, int]:
        """Called for EVERY candidate that survives intra-batch dedup —
        including one about to be dropped as a near-duplicate against
        QdrantStore's am1st_embeddings collection. That's the whole point:
        even a dropped near-duplicate still counts as one more source
        covering this event, so its corroboration credit must be recorded
        BEFORE the caller's duplicate-drop decision, not after.

        `add_representative` should be True only for a candidate that will
        actually get its own candidate-pool row (i.e. NOT a near-duplicate
        drop) — a near-duplicate doesn't add matching robustness (it's
        already textually redundant with an existing representative), so
        it bumps the matched event's tally but doesn't become a new point.

        Returns (heat_score, first_seen_at_unix) reflecting this event's
        state right after this call — the caller attaches these to the
        candidate. Fails open (returns (1.0, published_at_unix), i.e. "no
        corroboration found yet") if Qdrant isn't configured or the query
        fails — never blocks the ingestion cycle."""
        if self._client is None:
            return 1.0, published_at_unix

        cutoff = time.time() - self._window_seconds
        match = None
        try:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=1,
                query_filter=Filter(must=[FieldCondition(key="last_updated_at", range=Range(gte=cutoff))]),
                with_payload=True,
            )
            if result.points:
                match = result.points[0]
        except Exception:
            logger.exception("EventStore: query failed, treating as no match")

        weight = self._major_outlet_weight if source in self._major_outlets else 1.0
        now = int(time.time())

        if match is not None and match.score >= self._related_threshold:
            payload = match.payload or {}
            event_id = payload.get("event_id")
            sources = list(payload.get("sources", []))
            new_heat = payload.get("heat_score", 1.0)
            if source not in sources:
                new_heat += weight
                sources.append(source)
            new_first_seen = min(payload.get("first_seen_at", published_at_unix), published_at_unix)
            shared_payload = {
                "heat_score": new_heat,
                "first_seen_at": new_first_seen,
                "last_updated_at": now,
                "sources": sources,
            }
            try:
                await self._client.set_payload(
                    collection_name=self._collection,
                    payload=shared_payload,
                    points=Filter(must=[FieldCondition(key="event_id", match=MatchValue(value=event_id))]),
                )
            except Exception:
                logger.exception("EventStore: failed to update event %s", event_id)
            if add_representative:
                try:
                    await self._client.upsert(
                        collection_name=self._collection,
                        points=[
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding,
                                payload={
                                    "event_id": event_id,
                                    "source": source,
                                    "published_at": published_at_unix,
                                    **shared_payload,
                                },
                            )
                        ],
                    )
                except Exception:
                    logger.exception("EventStore: failed to add representative point for event %s", event_id)
            return new_heat, new_first_seen

        if not add_representative:
            # Dropped as a near-duplicate AND no matching event found within
            # the window — nothing to attach a heat_score to (this candidate
            # never becomes a pool row), and no value in creating an orphan
            # event for something we're discarding anyway.
            return 1.0, published_at_unix

        event_id = str(uuid.uuid4())
        try:
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "event_id": event_id,
                            "source": source,
                            "published_at": published_at_unix,
                            "heat_score": 1.0,
                            "first_seen_at": published_at_unix,
                            "last_updated_at": now,
                            "sources": [source],
                        },
                    )
                ],
            )
        except Exception:
            logger.exception("EventStore: failed to create new event")
        return 1.0, published_at_unix

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
            AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None)
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

    async def most_similar_recent(self, embedding: list[float]) -> tuple[float, str]:
        """Highest cosine similarity against post_content embeddings whose
        source article was published in the last window, plus that match's
        url for logging. Returns (0.0, "") if Qdrant isn't configured, the
        collection is empty, or the query fails — fail open, same as
        QdrantStore.most_similar_recent."""
        if self._client is None:
            return 0.0, ""
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
            return 0.0, ""
        points = result.points
        if not points:
            return 0.0, ""
        best = max(points, key=lambda p: p.score)
        payload = best.payload or {}
        return best.score, payload.get("url", "")

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
