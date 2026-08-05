from __future__ import annotations

import logging
import time
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
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
        open — never blocks a candidate just because this cache is cold)."""
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
