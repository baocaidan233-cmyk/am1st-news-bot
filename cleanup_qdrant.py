"""Daily retention cleanup for the two short-window dedup-cache collections
(2026-09-06) — the job core/qdrant_store.py's QdrantStore/PostedHistoryStore
docstrings and config.qdrant.cleanup_retention_days have referenced since
2026-08-xx as "not yet built." Deliberately a separate, low-frequency
scheduled script, not something either main cycle calls itself.

Only touches the two collections whose own query methods already ignore
anything past a fixed age window (QdrantStore.most_similar_recent(),
PostedHistoryStore.most_similar_recent(), both filtering on `publishedAt`
>= now - their own window_seconds) — deleting a point older than that same
cutoff is invisible to the running system, it could never have been
returned by either query again anyway:
  - am1st_embeddings:            cutoff = qdrant.cross_cycle_window_hours (72h)
  - am1st_posting_news_embedding: cutoff = publish.posted_dedup_window_hours (240h/10d)

am1st_events (the event store) is deliberately EXCLUDED, per the user's
explicit instruction (2026-09-06) — it's the aggregated identity/heat/
timeline record, not a short-window cache; a blind age-based delete there
would destroy real history a future storyline/heat feature still needs,
not just an expired lookup cache.

Usage: DRY_RUN=1 (default) python cleanup_qdrant.py — logs counts only.
       DRY_RUN=0 python cleanup_qdrant.py — actually deletes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, Range

from core.config import load_config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cleanup_qdrant")

DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"


async def _cleanup_collection(client: AsyncQdrantClient, name: str, window_seconds: float) -> None:
    cutoff = time.time() - window_seconds
    stale_filter = Filter(must=[FieldCondition(key="publishedAt", range=Range(lt=cutoff))])

    count_result = await client.count(collection_name=name, count_filter=stale_filter, exact=True)
    stale_count = count_result.count
    total_result = await client.count(collection_name=name, exact=True)
    logger.info(
        "%s: %d/%d points older than cutoff (%.1fh window)",
        name, stale_count, total_result.count, window_seconds / 3600,
    )

    if stale_count == 0:
        return
    if DRY_RUN:
        logger.info("%s: DRY_RUN=1, not deleting — set DRY_RUN=0 to actually delete these %d points", name, stale_count)
        return

    await client.delete(collection_name=name, points_selector=stale_filter, wait=True)
    logger.info("%s: deleted %d stale points", name, stale_count)


async def main() -> None:
    config = load_config("config/config.yaml")
    if not config.qdrant.url:
        logger.info("cleanup_qdrant: no qdrant.url configured, nothing to do")
        return

    client = AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None, timeout=config.qdrant.timeout_seconds)
    try:
        await _cleanup_collection(client, config.qdrant.collection, config.qdrant.cross_cycle_window_hours * 3600)
        await _cleanup_collection(client, config.qdrant.posted_collection, config.publish.posted_dedup_window_hours * 3600)
        logger.info("cleanup_qdrant: %s excluded from cleanup by design — event store, not a lookup cache", config.qdrant.events_collection)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
