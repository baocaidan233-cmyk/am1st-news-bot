from __future__ import annotations

import logging

from agents.embedder import Embedder
from core.config import AppConfig
from core.models import PublishCandidate
from core.qdrant_store import PostedHistoryStore

logger = logging.getLogger(__name__)


def content_for_embedding(post_content: str, url: str) -> str:
    """post_content always has "\\n\\n{url}" appended after generation (see
    main_publish.py's run_cycle) — needed for the actual Gettr post, but
    embedding the literal URL string dilutes the semantic dedup signal.
    Real case caught 2026-08-06: two different sources' takes on the exact
    same event (a 2020 Maricopa County voter-data hack) scored 0.731 on
    caption text alone — comfortably over the 0.70 duplicate threshold —
    but only 0.698 with the URL included, missing the duplicate entirely.
    Strips the exact suffix that was appended, so dedup compares
    like-for-like; returns the input unchanged if that suffix isn't
    present (defensive, shouldn't happen given how post_content is built)."""
    suffix = f"\n\n{url}"
    return post_content[: -len(suffix)] if post_content.endswith(suffix) else post_content


async def find_publishable(
    ranked_batch: list[PublishCandidate],
    embedder: Embedder,
    posted_store: PostedHistoryStore,
    config: AppConfig,
) -> PublishCandidate | None:
    """Walks `ranked_batch` in priority order (highest first) and returns the
    first candidate that is NOT a near-duplicate of something this channel
    already posted in the last publish.posted_dedup_window_hours. Duplicates
    are skipped (logged only, not written anywhere) — never causes the whole
    cycle to abort. Returns None only if every candidate in the batch is a
    duplicate (or the batch is empty) — the correct outcome is "publish
    nothing this cycle", not a fallback."""
    threshold = config.publish.posted_dedup_threshold

    for candidate in ranked_batch:
        embedding = await embedder.embed(content_for_embedding(candidate.post_content, candidate.url))
        similarity, matched_url = await posted_store.most_similar_recent(embedding)

        if similarity > threshold:
            logger.info(
                "find_publishable: %s dropped — duplicate of already-posted content (%.3f > %.2f, matched %s)",
                candidate.url,
                similarity,
                threshold,
                matched_url,
            )
            continue

        logger.info("find_publishable: %s selected (priority_score=%.1f)", candidate.url, candidate.priority_score)
        return candidate

    logger.info("find_publishable: all %d candidate(s) were duplicates — nothing to publish this cycle", len(ranked_batch))
    return None
