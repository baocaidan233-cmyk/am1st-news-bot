"""
One-shot backfill (2026-10-09, ported from China Breaks): give every posted-history point still inside
publish.posted_dedup_window_hours its source article's title + description,
so agents/posted_dedup_checker.py's gray-zone same_event() call can compare
source articles against the posts made before PostedHistoryStore.write()
started storing them. Points it can't resolve keep comparing captions, which
is what they did before.

Reads the candidate-pool Notion database by url (read only) and writes only
the two new payload fields on this channel's own posted-history collection.
Safe to re-run: points that already have a title are skipped.

Usage:
  python3 backfill_posted_source_text.py           # report what would change
  python3 backfill_posted_source_text.py --apply   # write it
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, Range

from core.config import load_config
from core.notion_candidates import NOTION_VERSION


async def _source_text(client: httpx.AsyncClient, config, url: str) -> tuple[str, str] | None:
    notion = config.notion
    props = notion.candidate_props
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    for _ in range(3):
        resp = await client.post(
            f"https://api.notion.com/v1/databases/{notion.candidate_db_id}/query",
            headers=headers,
            json={"filter": {"property": props.url, "url": {"equals": url}}, "page_size": 1},
        )
        if resp.status_code == 429:
            await asyncio.sleep(2)
            continue
        resp.raise_for_status()
        rows = resp.json().get("results", [])
        if not rows:
            return None
        p = rows[0]["properties"]
        title = "".join(x.get("plain_text", "") for x in p.get(props.title, {}).get("title", []))
        description = "".join(x.get("plain_text", "") for x in p.get(props.description, {}).get("rich_text", []))
        return (title, description) if title else None
    return None


async def main() -> None:
    load_dotenv()
    apply = "--apply" in sys.argv
    config = load_config("config/config.yaml")
    collection = config.qdrant.posted_collection
    cutoff = time.time() - config.publish.posted_dedup_window_hours * 3600
    qdrant = AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None, timeout=config.qdrant.timeout_seconds)

    points, offset = [], None
    while True:
        batch, offset = await qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[FieldCondition(key="publishedAt", range=Range(gte=cutoff))]),
            limit=256, offset=offset, with_payload=True, with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    todo = [p for p in points if not (p.payload or {}).get("title") and (p.payload or {}).get("url")]
    print(f"{len(points)} points in window, {len(todo)} without source text")

    resolved = missing = 0
    sem = asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=30) as client:
        async def one(p) -> None:
            nonlocal resolved, missing
            async with sem:
                found = await _source_text(client, config, p.payload["url"])
            if not found:
                missing += 1
                print(f"  no Notion row: {p.payload['url']}")
                return
            resolved += 1
            if apply:
                await qdrant.set_payload(
                    collection_name=collection,
                    payload={"title": found[0], "description": found[1]},
                    points=[p.id],
                )
        await asyncio.gather(*(one(p) for p in todo))
    await qdrant.close()
    print(f"resolved {resolved}, unresolved {missing}, {'written' if apply else 'dry run — nothing written'}")


if __name__ == "__main__":
    asyncio.run(main())
