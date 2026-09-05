from __future__ import annotations

import logging

from agents.embedder import Embedder
from core.config import AppConfig
from core.event_identity import EventVerifier, entity_tokens, log_decision
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
    event_verifier: EventVerifier,
    config: AppConfig,
) -> PublishCandidate | None:
    """Walks `ranked_batch` in priority order (highest first) and returns the
    first candidate that is NOT a near-duplicate of something this channel
    already posted in the last publish.posted_dedup_window_hours. Duplicates
    are skipped (logged only, not written anywhere) — never causes the whole
    cycle to abort. Returns None only if every candidate in the batch is a
    duplicate (or the batch is empty) — the correct outcome is "publish
    nothing this cycle", not a fallback.

    2026-09-05: cosine similarity alone no longer decides a duplicate — it
    only decides whether to ASK. A real production audit (same day) found
    cosine > threshold flags genuine next-stage developments in an ongoing
    story as duplicates of the earlier stage just as often as it flags
    actual reprints: "Missouri Supreme Court Tosses New Congressional Map"
    (0.784 cosine) was NOT the same event as the later "Missouri asks US
    Supreme Court to allow use of new congressional districts" — a state
    court rejection vs. a federal appeal of that rejection — and "Excavation
    Work On Trump's Arch To Begin" (0.741 cosine) was NOT the same event as
    "Opponents Seek Emergency Order to Stop Trump's Arch" — a construction
    announcement vs. a legal challenge trying to halt it. Both got silently
    dropped as duplicates in production, cascading the actual winner down to
    a much weaker, unrelated story. Meanwhile two other same-cycle pairs
    (a DOJ-Mount Sinai settlement and a Pentagon-leak polygraph story, each
    reported by a second outlet a bit later) really were the same event —
    cosine alone can't tell these apart, because "opposing/next-stage
    action on the same underlying facts" and "reprint of the same facts"
    look equally similar in embedding space.

    core/event_identity.py's EventVerifier.same_event() already exists for
    exactly this question (used at ingestion time to decide whether two
    articles describe the same occurrence) and already carries this
    session's stage-distinction guidance (prompts/same_event_prompt.txt:
    considering->announcing->passing->signing->implementing are different
    events unless clearly the same stage) — reused here as-is, no new
    prompt, no keyword heuristics. Only called when cosine > threshold
    (typically 0-3 times per publish cycle per the 2026-09-05 audit), so
    the added cost is negligible."""
    threshold = config.publish.posted_dedup_threshold

    for candidate in ranked_batch:
        candidate_content = content_for_embedding(candidate.post_content, candidate.url)
        embedding = await embedder.embed(candidate_content)
        similarity, matched_url, matched_content_raw = await posted_store.most_similar_recent(embedding)

        looks_similar = similarity > threshold
        is_duplicate = False
        same_event_raw = ""
        if looks_similar:
            matched_content = content_for_embedding(matched_content_raw, matched_url)
            is_duplicate, same_event_raw = await event_verifier.same_event(candidate_content, matched_content)

        if matched_url:
            log_record = {
                "check_type": "posted_dedup",
                "candidate_url": candidate.url,
                "matched_url": matched_url,
                "cosine_score": similarity,
                "threshold": threshold,
                "cosine_flagged": looks_similar,
                "final_verdict": "duplicate" if is_duplicate else "kept",
            }
            if looks_similar:
                candidate_entities = entity_tokens(candidate_content)
                matched_entities = entity_tokens(matched_content_raw)
                log_record.update({
                    "same_event_raw": same_event_raw,
                    "candidate_entities": sorted(candidate_entities),
                    "matched_entities": sorted(matched_entities),
                    "entity_overlap": sorted(candidate_entities & matched_entities),
                })
            log_decision(config, log_record)

        if is_duplicate:
            logger.info(
                "find_publishable: %s dropped — same_event() confirmed duplicate of already-posted content (cosine=%.3f > %.2f, matched %s)",
                candidate.url,
                similarity,
                threshold,
                matched_url,
            )
            continue
        if looks_similar:
            logger.info(
                "find_publishable: %s cosine-flagged (%.3f > %.2f) but same_event() said DIFFERENT — not treating as duplicate, matched %s",
                candidate.url,
                similarity,
                threshold,
                matched_url,
            )

        logger.info("find_publishable: %s selected (priority_score=%.1f)", candidate.url, candidate.priority_score)
        return candidate

    logger.info("find_publishable: all %d candidate(s) were duplicates — nothing to publish this cycle", len(ranked_batch))
    return None
