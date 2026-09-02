from __future__ import annotations

import logging

from agents.embedder import Embedder
from core.config import AppConfig
from core.event_identity import entity_tokens, log_decision
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
    nothing this cycle", not a fallback.

    2026-09-02: every comparison (not just ones that end up over threshold)
    now also logs an entity-token overlap between the candidate and its
    closest posted match — observational only, does NOT affect the
    similarity > threshold decision below. posted_dedup_threshold (0.70)
    has never actually been calibrated against a real score distribution
    the way dedup.semantic_threshold/hot_topics.match_threshold were — the
    only real precedent behind it is the single 2026-08-06 incident in this
    module's content_for_embedding() docstring. The user separately asked
    whether entity overlap (already used at ingestion time, see
    core/event_identity.py) should factor into this decision too — logging
    it here first, unused, so both questions (is 0.70 the right cosine
    line, would requiring entity overlap too change any real verdict) can
    be answered from real accumulated data before either one changes actual
    behavior."""
    threshold = config.publish.posted_dedup_threshold

    for candidate in ranked_batch:
        candidate_content = content_for_embedding(candidate.post_content, candidate.url)
        embedding = await embedder.embed(candidate_content)
        similarity, matched_url, matched_content = await posted_store.most_similar_recent(embedding)

        is_duplicate = similarity > threshold
        if matched_url:
            candidate_entities = entity_tokens(candidate_content)
            matched_entities = entity_tokens(matched_content)
            log_decision(config, {
                "check_type": "posted_dedup",
                "candidate_url": candidate.url,
                "matched_url": matched_url,
                "cosine_score": similarity,
                "threshold": threshold,
                "rule_verdict": "duplicate" if is_duplicate else "kept",
                "candidate_entities": sorted(candidate_entities),
                "matched_entities": sorted(matched_entities),
                "entity_overlap": sorted(candidate_entities & matched_entities),
            })

        if is_duplicate:
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
