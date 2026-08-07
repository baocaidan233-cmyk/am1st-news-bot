"""
AM1ST (America First) — ported from the original n8n workflow
(v2.15_gdelt_america_first_major_feed_to_notion.json) to a plain Python
process, per the standing dedup/model architecture. RSS-only rebuild: the
GDELT source branch and the score>8 auto-publish sub-workflow are explicitly
out of scope (see am1st_pipeline.html's "新机器人方案" sections).

This is the INGESTION cycle only. As of 2026-08-04 it no longer publishes
to Gettr directly — that turned out to be a mistaken assumption from the
initial 2026-08-03 build, made before discovering the real n8n design is
two separate workflows (see project_am1st_migration memory). This cycle's
job ends at writing a candidate into the shared Notion candidate pool;
main_publish.py is the separate, independently-scheduled process that
later reads that pool and decides what (if anything) actually gets posted.

Pipeline order per cycle (cheapest filter first):
  load sources (Notion) -> fetch RSS (UTC-normalized, 3h publish-age filter)
  -> URL-hash Redis dedup -> intra-batch semantic CLUSTERING (title+
  description; groups this batch's own candidates into local event
  clusters, not just a drop/keep decision — see Layer 2 below) -> per
  cluster: read-only peek at core/qdrant_store.py's EventStore (a separate
  Qdrant collection tracking heat_score/event_first_seen_at per underlying
  event, not per article) for a heat_score PREVIEW to feed the AI score
  gate, plus per-candidate cross-cycle Qdrant dedup (vs title+description
  written in the last 72h) -> AI score gate (gpt-4o-mini, >=5) -> for each
  cluster with at least one scoring survivor: COMMIT that cluster's
  accumulated heat/sources to the EventStore (a cluster where nothing
  cleared the score gate never touches it at all — see 2026-08-06 "event
  aggregation, take 2" design note in project_am1st_migration memory) ->
  write survivors to Notion candidate pool -> Qdrant embedding write (the
  same title+description embeddings used for clustering, now persisted
  for future cross-cycle checks).

Full-text extraction and content generation are deliberately NOT part of
this cycle (moved to main_publish.py, 2026-08-05) — every candidate that
merely cleared the cheap title+description score gate used to pay for a
full extraction + LLM-written post, even though only a handful per 30-min
publish cycle would ever actually get posted before aging out of the 12h
candidate-pool window. See agents/extractor.py's docstring and
project_am1st_migration memory.

Usage:
  python3 main.py              # normal run
  python3 main.py --dry-run    # logs what would be added, never writes to Notion/Qdrant
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents.embedder import Embedder
from agents.og_metadata import fetch_link_preview
from agents.rss_fetcher import fetch_all
from agents.scorer import Scorer
from core.config import load_config
from core.hashing import cosine_similarity
from core.language import is_english
from core.notion_candidates import write_candidate
from core.notion_sources import load_rss_sources
from core.qdrant_store import EventStore, QdrantStore, ensure_collection_with_retry
from core.redis_store import RedisStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


async def run_cycle(
    config, redis_store, qdrant_store, event_store, embedder, scorer, dry_run,
) -> None:
    sources = await load_rss_sources(config)
    if not sources:
        logger.info("run_cycle: no in_use sources — nothing to do")
        return

    candidates = await fetch_all(config, sources)
    if not candidates:
        return

    # --- Layer 1: exact-duplicate URL-hash dedup (Redis) ---
    survivors = [c for c in candidates if await redis_store.claim_new(c.url_hash)]
    logger.info("run_cycle: %d/%d new after URL-hash dedup", len(survivors), len(candidates))
    if not survivors:
        return

    # --- Layer 1.5: English-only filter (2026-08-06) — AM1ST is an
    # English-only channel; a non-English candidate wastes embedding/
    # scoring cost and, if it slips all the way through, can only be
    # caught downstream by the writer's own judgment (see core/language.py
    # docstring for the real incident that showed that alone isn't
    # reliable). Cheap, on title+description only — the full extracted
    # article gets its own check at publish time in main_publish.py. ---
    before_lang_filter = len(survivors)
    english_survivors = []
    for c in survivors:
        if is_english(f"{c.title} {c.description}"):
            english_survivors.append(c)
        else:
            logger.info("run_cycle: %s dropped — non-English title/description", c.url)
    survivors = english_survivors
    logger.info("run_cycle: %d/%d survive English-language filter", len(survivors), before_lang_filter)
    if not survivors:
        return

    # --- Layer 1.6: description backfill via the article page's own
    # og:description meta tag (2026-08-07) — some RSS feeds give a title
    # with no description at all, which under-informs both clustering
    # (embeds on title alone) and the AI score gate (see
    # project_am1st_migration memory's "GOP donor memo scored only 5"
    # investigation). This is deliberately just the page's existing
    # meta-description tag (agents/og_metadata.py, already used for Gettr
    # preview cards) — NOT full-article extraction, which the user
    # explicitly rejected as a fix for this same problem and which stays
    # deferred to the publish cycle for cost reasons (agents/extractor.py's
    # docstring). A failed/empty fetch just leaves the candidate as-is. ---
    backfilled = 0
    empty_before = sum(1 for c in survivors if not c.description)
    for c in survivors:
        if c.description:
            continue
        try:
            og = await fetch_link_preview(c.url)
        except Exception:
            continue
        desc = og.get("prev_desc")
        if desc:
            c.description = desc
            backfilled += 1
    if empty_before:
        logger.info("run_cycle: backfilled og:description for %d/%d description-less candidate(s)", backfilled, empty_before)

    # --- Layer 2: intra-batch semantic CLUSTERING (title+description) ---
    # Groups this batch's own candidates into local clusters instead of a
    # plain drop/keep decision — same pairwise cosine comparisons as
    # before, just also tracking WHICH existing cluster a match belongs to.
    # score >= dedup.semantic_threshold: near-duplicate WITHIN this batch —
    #   not its own candidate, but its source still credits the cluster
    #   (fixes a bug symmetric to the one caught in cross-cycle dedup: this
    #   used to be silently dropped with zero corroboration credit).
    # heat.related_threshold <= score < dedup.semantic_threshold: distinct
    #   angle on the same local event — joins the cluster, becomes its own
    #   candidate.
    # score < heat.related_threshold: starts a new local cluster.
    # See project_am1st_migration memory's 2026-08-06 "event aggregation,
    # take 2" note for the full design (peek/preview/commit split below).
    threshold = config.dedup.semantic_threshold
    related_threshold = config.heat.related_threshold
    clusters: list[dict] = []  # {"sources": set(), "earliest_source": str, "earliest_unix": int}
    cluster_members: list[list[tuple]] = []  # parallel to clusters: [(candidate, embedding), ...]

    for c in survivors:
        try:
            # Some RSS feeds dump full article text into "description" instead
            # of a short summary — truncate defensively so a single oversized
            # item can't blow past the embedding model's 8192-token limit.
            embedding = await embedder.embed(f"{c.title}\n{c.description}"[:6000])
        except Exception:
            logger.exception("run_cycle: %s dropped — embedding failed, skipping this item", c.url)
            continue

        best_ci, best_score = None, 0.0
        for ci, members in enumerate(cluster_members):
            for _, member_embedding in members:
                score = cosine_similarity(embedding, member_embedding)
                if score > best_score:
                    best_score, best_ci = score, ci

        published_unix = int(c.published_at.timestamp())

        if best_score >= threshold:
            cluster = clusters[best_ci]
            cluster["sources"].add(c.source_name)
            if published_unix < cluster["earliest_unix"]:
                cluster["earliest_unix"] = published_unix
                cluster["earliest_source"] = c.source_name
            logger.info("run_cycle: %s dropped — intra-batch semantic duplicate (credited to cluster %d)", c.url, best_ci)
            continue

        if best_score >= related_threshold:
            cluster_idx = best_ci
        else:
            cluster_idx = len(clusters)
            clusters.append({"sources": set(), "earliest_source": c.source_name, "earliest_unix": published_unix})
            cluster_members.append([])

        cluster = clusters[cluster_idx]
        cluster["sources"].add(c.source_name)
        if published_unix < cluster["earliest_unix"]:
            cluster["earliest_unix"] = published_unix
            cluster["earliest_source"] = c.source_name
        cluster_members[cluster_idx].append((c, embedding))

    total_members = sum(len(m) for m in cluster_members)
    logger.info(
        "run_cycle: %d/%d survive intra-batch semantic dedup, forming %d local cluster(s)",
        total_members, len(survivors), len(clusters),
    )

    # --- Layer 3: per-cluster event-store PREVIEW (read-only) + per-candidate
    # cross-cycle Qdrant dedup (vs title+description written in the last 72h) ---
    cluster_peeks: list[dict | None] = []
    scoring_candidates = []  # list of (Candidate, embedding, cluster_idx)
    for cluster_idx, members in enumerate(cluster_members):
        if not members:
            cluster_peeks.append(None)
            continue
        matched = await event_store.peek(members[0][1])
        cluster_peeks.append(matched)

        # Already-published guard (2026-08-07) — if this cluster is a
        # near-verbatim rehash (>= dedup.semantic_threshold, the same bar
        # used for "not worth its own candidate" everywhere else) of an
        # event we already actually posted to Gettr (matched["published"],
        # set by main_publish.py's EventStore.mark_published()), skip the
        # whole cluster: scoring/candidate-pool cost on something that
        # would just tell our audience the same news twice again isn't
        # worth paying. A genuinely different angle on that same event
        # (0.6-0.8) is NOT blocked here — see EventStore.mark_published()'s
        # docstring; that's still a legitimate fresh update.
        if matched is not None and matched.get("published") and matched.get("_score", 0.0) >= threshold:
            logger.info(
                "run_cycle: cluster %d dropped — near-duplicate of an already-published event (%.3f)",
                cluster_idx, matched.get("_score", 0.0),
            )
            continue

        cluster = clusters[cluster_idx]
        preview_heat, preview_first_seen = event_store.preview_heat(
            matched, cluster["sources"], cluster["earliest_source"], cluster["earliest_unix"],
        )
        preview_first_seen_dt = datetime.fromtimestamp(preview_first_seen, tz=timezone.utc)

        for c, embedding in members:
            best_score = await qdrant_store.most_similar_recent(embedding)
            if best_score >= threshold:
                logger.info("run_cycle: %s dropped — cross-cycle semantic duplicate (%.3f)", c.url, best_score)
                continue
            c.heat_score = preview_heat
            c.event_first_seen_at = preview_first_seen_dt
            scoring_candidates.append((c, embedding, cluster_idx))
    logger.info("run_cycle: %d/%d survive cross-cycle semantic dedup", len(scoring_candidates), total_members)

    # --- Score every survivor first, THEN decide per cluster whether to
    # commit anything to the EventStore — a cluster where nothing clears
    # the score gate never touches am1st_events at all, without needing a
    # separate (and inevitably imperfect) cheap relevance classifier: the
    # score gate we're already paying for IS that classifier. ---
    scored: list[tuple] = []  # (candidate, embedding, cluster_idx, passed)
    for c, embedding, cluster_idx in scoring_candidates:
        try:
            score_output = await scorer.score(c)
            if score_output is None:
                continue
            c.llm_score = score_output.llm_score
            c.llm_comment = score_output.llm_comment
            passed = c.llm_score >= config.openai.score_threshold
            if not passed:
                logger.info("run_cycle: %s scored %.1f, below threshold", c.url, c.llm_score)
            scored.append((c, embedding, cluster_idx, passed))
        except Exception:
            logger.exception("run_cycle: unhandled error scoring %s, skipping this item", c.url)

    added_count = 0
    for cluster_idx, cluster in enumerate(clusters):
        survivors_in_cluster = [(c, emb) for c, emb, ci, passed in scored if ci == cluster_idx and passed]
        if not survivors_in_cluster:
            continue

        if dry_run:
            for c, _ in survivors_in_cluster:
                logger.info("run_cycle: [dry-run] would add to candidate pool: %s (score=%.1f)", c.url, c.llm_score)
                added_count += 1
            continue

        rep_c, rep_embedding = survivors_in_cluster[0]
        extra_points = [
            (emb, c.source_name, int(c.published_at.timestamp())) for c, emb in survivors_in_cluster[1:]
        ]
        heat_score, first_seen_unix = await event_store.commit(
            cluster_peeks[cluster_idx],
            cluster["sources"],
            cluster["earliest_source"],
            cluster["earliest_unix"],
            representative=(rep_embedding, rep_c.source_name, int(rep_c.published_at.timestamp())),
            extra_points=extra_points,
        )
        first_seen_dt = datetime.fromtimestamp(first_seen_unix, tz=timezone.utc)

        for c, embedding in survivors_in_cluster:
            try:
                c.heat_score = heat_score
                c.event_first_seen_at = first_seen_dt
                if not await write_candidate(config, c):
                    logger.warning("run_cycle: candidate-pool write failed for %s", c.url)
                    continue
                content_for_embedding = f"{c.title}\n{c.description}"[:6000]
                await qdrant_store.write_embedding(
                    c.url, c.url_hash, content_for_embedding, int(c.published_at.timestamp()), embedding,
                )
                added_count += 1
                logger.info("run_cycle: added to candidate pool: %s (score=%.1f)", c.url, c.llm_score)
            except Exception:
                logger.exception("run_cycle: unhandled error writing %s, skipping this item", c.url)

    logger.info("run_cycle: %d added to candidate pool this cycle", added_count)


async def main() -> None:
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    config = load_config("config/config.yaml")

    redis_store = RedisStore(config)
    qdrant_store = QdrantStore(config)
    event_store = EventStore(config)
    await ensure_collection_with_retry(qdrant_store, "am1st_embeddings")
    await ensure_collection_with_retry(event_store, "am1st_events")
    embedder = Embedder(config)
    scorer = Scorer(config)

    if dry_run:
        logger.info("Running in --dry-run mode: Notion/Qdrant writes will be logged, not sent")

    try:
        while True:
            try:
                await run_cycle(config, redis_store, qdrant_store, event_store, embedder, scorer, dry_run)
            except Exception:
                logger.exception("run_cycle failed")
            jitter = config.poll_interval_seconds * random.uniform(-0.1, 0.1)
            await asyncio.sleep(config.poll_interval_seconds + jitter)
    finally:
        await redis_store.close()
        await qdrant_store.close()
        await event_store.close()


if __name__ == "__main__":
    asyncio.run(main())
