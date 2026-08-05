from __future__ import annotations

import logging

from agents.embedder import Embedder
from core.config import AppConfig
from core.models import PublishCandidate
from core.qdrant_store import PostedHistoryStore

logger = logging.getLogger(__name__)


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
        embedding = await embedder.embed(candidate.post_content)
        similarity, matched_url, matched_title = await posted_store.most_similar_recent(embedding)

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
