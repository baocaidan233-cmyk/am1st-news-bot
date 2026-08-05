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
  -> URL-hash Redis dedup -> intra-batch embedding dedup (title+description)
  -> cross-cycle Qdrant dedup (title+description vs title+description
  written in the last 72h) -> AI score gate (gpt-4o-mini, >=5) -> full-text
  extraction (cookie-mapped, alerts on failure instead of silently dropping)
  -> content generation (gpt-4o-mini, "No comment" = quality-filtered out)
  -> write to Notion candidate pool -> Qdrant embedding write (the same
  title+description embedding used for intra-batch dedup, now persisted for
  future cross-cycle checks).

Usage:
  python3 main.py              # normal run
  python3 main.py --dry-run    # logs what would be added, never writes to Notion/Qdrant
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys

from dotenv import load_dotenv

from agents.embedder import Embedder
from agents.extractor import Extractor
from agents.rss_fetcher import fetch_all
from agents.scorer import Scorer
from agents.writer import Writer
from core.alerts import AlertNotifier
from core.config import load_config
from core.hashing import cosine_similarity
from core.notion_candidates import write_candidate
from core.notion_sources import load_rss_sources
from core.qdrant_store import QdrantStore
from core.redis_store import RedisStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


async def run_cycle(
    config, redis_store, qdrant_store, embedder, scorer, extractor, writer, dry_run,
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

    # --- Layer 2: intra-batch semantic dedup (title+description) ---
    threshold = config.dedup.semantic_threshold
    accepted: list[tuple] = []  # (candidate, embedding)
    for c in survivors:
        embedding = await embedder.embed(f"{c.title}\n{c.description}")
        if any(cosine_similarity(embedding, prev_emb) >= threshold for _, prev_emb in accepted):
            logger.info("run_cycle: %s dropped — intra-batch semantic duplicate", c.url)
            continue
        accepted.append((c, embedding))
    logger.info("run_cycle: %d/%d survive intra-batch semantic dedup", len(accepted), len(survivors))

    # --- Layer 3: cross-cycle semantic dedup (vs title+description written in the last 72h) ---
    scoring_candidates = []  # list of (Candidate, title+description embedding)
    for c, embedding in accepted:
        similarity = await qdrant_store.most_similar_recent(embedding)
        if similarity >= threshold:
            logger.info("run_cycle: %s dropped — cross-cycle semantic duplicate (%.3f)", c.url, similarity)
            continue
        scoring_candidates.append((c, embedding))
    logger.info("run_cycle: %d/%d survive cross-cycle semantic dedup", len(scoring_candidates), len(accepted))

    added_count = 0
    for c, title_desc_embedding in scoring_candidates:
        try:
            score_output = await scorer.score(c)
            if score_output is None:
                continue
            c.llm_score = score_output.llm_score
            c.llm_comment = score_output.llm_comment
            if c.llm_score < config.openai.score_threshold:
                logger.info("run_cycle: %s scored %.1f, below threshold", c.url, c.llm_score)
                continue

            await extractor.extract(c, sources)

            post_content = await writer.write(c)
            if Writer.is_no_comment(post_content):
                logger.info("run_cycle: %s — writer returned No comment, filtered out", c.url)
                continue
            c.post_content = post_content

            if not dry_run:
                if not await write_candidate(config, c):
                    logger.warning("run_cycle: candidate-pool write failed for %s", c.url)
                    continue
                await qdrant_store.write_embedding(c.url, c.title, title_desc_embedding)
            added_count += 1
            logger.info("run_cycle: added to candidate pool: %s (score=%.1f)", c.url, c.llm_score)
        except Exception:
            logger.exception("run_cycle: unhandled error processing %s, skipping this item", c.url)

    logger.info("run_cycle: %d added to candidate pool this cycle", added_count)


async def main() -> None:
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    config = load_config("config/config.yaml")

    redis_store = RedisStore(config)
    qdrant_store = QdrantStore(config)
    await qdrant_store.ensure_collection()
    embedder = Embedder(config)
    scorer = Scorer(config)
    alerts = AlertNotifier(config)
    extractor = Extractor(config, alerts)
    writer = Writer(config)

    if dry_run:
        logger.info("Running in --dry-run mode: Notion/Qdrant writes will be logged, not sent")

    try:
        while True:
            try:
                await run_cycle(
                    config, redis_store, qdrant_store, embedder, scorer, extractor, writer, dry_run,
                )
            except Exception:
                logger.exception("run_cycle failed")
            jitter = config.poll_interval_seconds * random.uniform(-0.1, 0.1)
            await asyncio.sleep(config.poll_interval_seconds + jitter)
    finally:
        await redis_store.close()
        await qdrant_store.close()


if __name__ == "__main__":
    asyncio.run(main())
