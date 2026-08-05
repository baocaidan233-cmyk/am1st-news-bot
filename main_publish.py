"""
AM1ST — the separate publish cycle, ported from v1.4_am1st_notion_to_gettr_auto
posting.json. Runs independently of main.py's ingestion cycle (see
project_am1st_migration memory's 2026-08-04 "two separate workflows" note):
main.py only ever writes candidates into the shared Notion candidate pool;
this process is the only thing that ever reads that pool to actually pick
something to post.

Cycle order (every publish.interval_seconds, 30 min by default):
  query eligible candidates (Notion: not sent, <=12h old, llm_score>=6)
  -> drop stale "former president Trump" phrasing
  -> tiered batch selection (fresh+high-score preferred, cascading fallback,
     3-5 candidates)
  -> LLM priority re-rank (gpt-4o-mini, on post_content, second opinion vs
     the ingestion-side llm_score)
  -> walk the ranked list, skipping anything that's a near-duplicate of
     content this channel already posted in the last 24h (threshold 0.70 —
     stricter than the ingestion side's 0.8, deliberately, since this is a
     fully-autonomous post: see feedback in project_am1st_migration memory)
  -> the first survivor is the winner; mark it sent + record its embedding
     in the posted-history collection.

The actual Gettr publish API call is deliberately NOT wired in yet — the
user asked to discuss that separately. Where it will go is marked with
_publish_stub below.

Usage:
  python3 main_publish.py              # normal run
  python3 main_publish.py --dry-run    # logs the winner, never touches Notion/Qdrant
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys

from dotenv import load_dotenv

from agents.candidate_selector import filter_former_trump, select_batch
from agents.embedder import Embedder
from agents.posted_dedup_checker import find_publishable
from agents.priority_ranker import PriorityRanker
from core.config import load_config
from core.models import PublishCandidate
from core.notion_candidates import mark_send_status, query_eligible_candidates
from core.qdrant_store import PostedHistoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main_publish")


def _publish_stub(winner: PublishCandidate, dry_run: bool) -> bool:
    """Placeholder for the actual Gettr publish call — deliberately not
    implemented yet, see this module's docstring. Always returns False (no
    real post happened): mark_send_status and the posted-history write
    below are gated on this return value, not on dry_run, precisely so that
    running this module for real (real Notion/Qdrant credentials, no
    --dry-run) can't falsely mark a candidate as sent or pollute the
    posted-history collection before Gettr publishing is actually wired in.
    """
    prefix = "[dry-run] " if dry_run else "[DEFERRED — not actually posted to Gettr] "
    logger.info("%swould publish: %s -- %s", prefix, winner.url, winner.post_content[:120])
    return False


async def run_cycle(config, embedder: Embedder, ranker: PriorityRanker, posted_store: PostedHistoryStore, dry_run: bool) -> None:
    candidates = await query_eligible_candidates(config)
    if not candidates:
        logger.info("run_cycle: no eligible candidates this cycle")
        return

    candidates = filter_former_trump(candidates)
    if not candidates:
        logger.info("run_cycle: all candidates dropped by former-Trump filter")
        return

    batch = select_batch(candidates, config)
    logger.info("run_cycle: selected batch of %d for ranking", len(batch))

    ranked = await ranker.rank(batch)

    winner = await find_publishable(ranked, embedder, posted_store, config)
    if winner is None:
        return

    published = _publish_stub(winner, dry_run)

    if published and not dry_run:
        await mark_send_status(config, winner.page_id)
        winner_embedding = await embedder.embed(winner.post_content)
        await posted_store.write(winner.url, winner.title, winner_embedding)


async def main() -> None:
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    config = load_config("config/config.yaml")

    embedder = Embedder(config)
    ranker = PriorityRanker(config)
    posted_store = PostedHistoryStore(config)
    await posted_store.ensure_collection()

    if dry_run:
        logger.info("Running in --dry-run mode: Notion/Qdrant writes will be logged, not sent")

    try:
        while True:
            try:
                await run_cycle(config, embedder, ranker, posted_store, dry_run)
            except Exception:
                logger.exception("run_cycle failed")
            jitter = config.publish.interval_seconds * random.uniform(-0.1, 0.1)
            await asyncio.sleep(config.publish.interval_seconds + jitter)
    finally:
        await posted_store.close()


if __name__ == "__main__":
    asyncio.run(main())
