"""
AM1ST — the separate publish cycle, ported from v1.4_am1st_notion_to_gettr_auto
posting.json. Runs independently of main.py's ingestion cycle (see
project_am1st_migration memory's 2026-08-04 "two separate workflows" note):
main.py only ever writes candidates into the shared Notion candidate pool;
this process is the only thing that ever reads that pool to actually pick
something to post.

Cycle order (every publish.interval_seconds, 30 min by default):
  query eligible candidates (Notion: not sent, <=12h old, llm_score>=6)
  -> drop stale "former president Trump" phrasing (cheap pass, title+
     description only)
  -> tiered batch selection (fresh+high-score preferred, cascading fallback,
     3-5 candidates)
  -> full-text extraction + content generation for just this small batch
     (moved here from main.py, 2026-08-05 — see agents/extractor.py's
     docstring for why), dropping anything the writer judges "No comment"
  -> re-check the former-Trump filter now that full text/post_content
     exist, in case only the article body (not title/description) had it
  -> LLM priority re-rank (gpt-4o-mini, on post_content, second opinion vs
     the ingestion-side llm_score), given a read-only snapshot of Google
     News' current top US-politics headlines as trending context (see
     agents/trending.py — never ingested/scored/published from directly)
  -> walk the ranked list, skipping anything that's a near-duplicate of
     content this channel already posted in the last 24h (threshold 0.70 —
     stricter than the ingestion side's 0.8, deliberately, since this is a
     fully-autonomous post: see feedback in project_am1st_migration memory)
  -> the first survivor is the winner; mark it sent + record its embedding
     in the posted-history collection.

The Gettr publish call uses agents/gettr_publisher.py's GettrPublisher
(wired in 2026-08-05, at the user's explicit request, after they supplied a
real test Gettr account for this purpose — see project_am1st_migration
memory. Text-only post, matching the original n8n design).

2026-08-06: also fetches OG link-preview metadata (agents/og_metadata.py)
for the winner's own article URL right before publishing, so the post
shows a real preview card instead of a bare appended URL with no card —
see agents/gettr_publisher.py's docstring for the field names involved.

2026-08-06, same day: the posted-dedup embedding (both the check in
find_publishable and the final write below) now uses
agents/posted_dedup_checker.py's content_for_embedding() to strip the
appended "\n\n{url}" suffix before embedding — a real duplicate slipped
through (two different sources' takes on the same 2020 Maricopa County
voter-data-hack story) because the literal URL text diluted the
similarity score just under the 0.70 threshold (0.698 with the URL vs
0.731 on the caption alone). The URL itself is still appended to the post
that actually goes out — only what gets embedded for comparison changed.

Usage:
  python3 main_publish.py              # normal run
  python3 main_publish.py --dry-run    # logs the winner, never touches Notion/Qdrant/Gettr
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents.candidate_selector import filter_former_trump, select_batch
from agents.embedder import Embedder
from agents.extractor import Extractor
from agents.gettr_publisher import GettrPublisher
from agents.og_metadata import fetch_link_preview
from agents.posted_dedup_checker import content_for_embedding, find_publishable
from agents.priority_ranker import PriorityRanker
from agents.trending import fetch_trending_headlines
from agents.writer import Writer
from core.alerts import AlertNotifier
from core.config import load_config
from core.language import is_english
from core.notion_candidates import has_unpublished_hot_candidate, mark_send_status, query_eligible_candidates
from core.notion_sources import load_rss_sources
from core.qdrant_store import EventStore, PostedHistoryStore, ensure_collection_with_retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main_publish")


def _build_background(matched: dict | None) -> str:
    """Formats a matched event's timeline + related_event_ids (2026-08-31,
    core/qdrant_store.py's EventStore) into a short plain-text Background
    for agents/writer.py's Writer.write(context=...) — see that method and
    prompts/content_gen_prompt.txt's "OPTIONAL BACKGROUND" section for how
    it's used. Most recent 3 of each, oldest to newest for the timeline
    (reads as a chronology). Returns "" if there's nothing to say — the
    caller then omits `context` entirely, reproducing today's behavior."""
    if not matched:
        return ""
    parts = []
    timeline = matched.get("timeline", [])[-3:]
    entries = [
        f"{e.get('summary', '')} ({datetime.fromtimestamp(e['ts'], tz=timezone.utc).strftime('%b %d')})"
        for e in timeline if e.get("ts") and e.get("summary")
    ]
    if entries:
        parts.append("Prior developments: " + "; ".join(entries) + ".")
    titles = [r.get("title") for r in matched.get("related_event_ids", [])[-3:] if r.get("title")]
    if titles:
        parts.append("Related storylines: " + "; ".join(titles) + ".")
    return " ".join(parts)


async def run_cycle(
    config,
    embedder: Embedder,
    ranker: PriorityRanker,
    posted_store: PostedHistoryStore,
    event_store: EventStore,
    publisher: GettrPublisher,
    extractor: Extractor,
    writer: Writer,
    dry_run: bool,
) -> None:
    candidates = await query_eligible_candidates(config)
    if not candidates:
        logger.info("run_cycle: no eligible candidates this cycle")
        return

    candidates = filter_former_trump(candidates)
    if not candidates:
        logger.info("run_cycle: all candidates dropped by former-Trump filter")
        return

    batch = select_batch(candidates, config)
    logger.info("run_cycle: selected batch of %d for extraction/content-gen", len(batch))

    sources = await load_rss_sources(config)
    generated = []
    for c in batch:
        text = await extractor.extract(c.url, sources)
        if not text:
            logger.info("run_cycle: %s dropped — full-text extraction failed (paywall/blocked/empty), refusing to publish off title+description alone", c.url)
            continue
        c.content = text

        # English-only channel — check the actual extracted article text,
        # not just title/description (main.py's cheaper ingestion-time
        # filter already covers those). A real published post (2026-08-06)
        # was a Portuguese-language Reuters article that made it all the
        # way to publish because the writer's own "decline" signal was, in
        # that instance, missed by a separate formatting bug — see
        # core/language.py's docstring. This check doesn't depend on the
        # model noticing at all.
        if not is_english(c.content[:1000]):
            logger.info("run_cycle: %s dropped — non-English article content", c.url)
            continue

        # Background for the writer (2026-08-31) — peek() against the same
        # title+description embedding space main.py already uses, so this
        # is checked against every candidate in the batch (not just the
        # eventual winner, since the winner isn't known until after
        # ranking, but content-gen runs on the whole batch) — see
        # _build_background()'s docstring and agents/writer.py's `context`
        # param. Fails open to no background on any error, same as every
        # other best-effort Qdrant read in this codebase.
        background = ""
        try:
            title_desc_embedding = await embedder.embed(f"{c.title}\n{c.description}"[:6000])
            background = _build_background(await event_store.peek(title_desc_embedding))
        except Exception:
            logger.exception("run_cycle: failed to build writer background for %s — continuing without it", c.url)

        post_content = await writer.write(c.title, c.content, context=background)
        if Writer.is_no_comment(post_content):
            logger.info("run_cycle: %s — writer returned No comment, dropped from batch", c.url)
            continue
        # Link appended after generation, not counted against the writer's
        # word cap — the AI's own output stays pure caption text.
        c.post_content = f"{post_content}\n\n{c.url}"
        generated.append(c)

    if not generated:
        logger.info("run_cycle: nothing survived extraction/content-gen this cycle")
        return

    # Re-check now that full text/post_content exist — the first pass only
    # had title+description to work with, so this catches stale phrasing
    # that only shows up in the article body or the generated caption.
    generated = filter_former_trump(generated)
    if not generated:
        logger.info("run_cycle: all candidates dropped by post-extraction former-Trump filter")
        return

    trending_headlines = await fetch_trending_headlines()
    ranked = await ranker.rank(generated, trending_headlines)

    winner = await find_publishable(ranked, embedder, posted_store, config)
    if winner is None:
        return

    og = await fetch_link_preview(winner.url)
    post_id = await publisher.publish(
        winner.post_content,
        log_ref=winner.url,
        prev_desc=og.get("prev_desc") or winner.description or None,
        prev_img=og.get("prev_img"),
        prev_src_link=og.get("prev_src_link") or winner.url,
        prev_ttl=og.get("prev_ttl") or winner.title,
    )
    published = post_id is not None
    logger.info(
        "run_cycle: publish %s for %s (post_id=%s)",
        "succeeded" if published else "FAILED",
        winner.url,
        post_id,
    )

    if published and not dry_run:
        await mark_send_status(config, winner.page_id)
        winner_embedding = await embedder.embed(content_for_embedding(winner.post_content, winner.url))
        await posted_store.write(
            winner.url, winner.url_hash, winner.post_content, int(winner.published_at.timestamp()), winner_embedding,
        )

        # Flag the underlying event as published (2026-08-07) — so a later
        # ingestion cycle's EventStore.peek() can drop a near-verbatim
        # rehash of it outright instead of only catching a duplicate at the
        # publish cycle's own, much shorter posted_dedup_window_hours check.
        # Re-embeds title+description (not post_content — this needs to
        # land in the same embedding space main.py's peek() already uses)
        # to find which event this candidate belongs to; skips silently if
        # no match is found (fail open, never blocks on this).
        try:
            title_desc_embedding = await embedder.embed(f"{winner.title}\n{winner.description}"[:6000])
            matched = await event_store.peek(title_desc_embedding)
            if matched and matched.get("event_id"):
                await event_store.mark_published(matched["event_id"])
        except Exception:
            logger.exception("run_cycle: failed to mark event as published for %s", winner.url)


async def main() -> None:
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    config = load_config("config/config.yaml")

    embedder = Embedder(config)
    ranker = PriorityRanker(config)
    posted_store = PostedHistoryStore(config)
    event_store = EventStore(config)
    publisher = GettrPublisher(config, dry_run=dry_run)
    alerts = AlertNotifier(config)
    extractor = Extractor(config, alerts)
    writer = Writer(config)
    await ensure_collection_with_retry(posted_store, "am1st_posting_news_embedding")
    await ensure_collection_with_retry(event_store, "am1st_events")

    if dry_run:
        logger.info("Running in --dry-run mode: Notion/Qdrant writes will be logged, not sent")

    try:
        while True:
            started = time.monotonic()
            try:
                await asyncio.wait_for(
                    run_cycle(config, embedder, ranker, posted_store, event_store, publisher, extractor, writer, dry_run),
                    timeout=config.cycle_timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Same self-loop cutoff as main.py's ingestion cycle — this
                # process runs independently of it, so publishing must not
                # stall just because one cycle got stuck on e.g. a slow
                # extraction (2026-08-12 discussion).
                logger.error("run_cycle exceeded %ds — cutting it off, will retry next cycle", config.cycle_timeout_seconds)
            except Exception:
                logger.exception("run_cycle failed")
            logger.info("run_cycle: cycle took %.1fs", time.monotonic() - started)
            jitter = config.publish.interval_seconds * random.uniform(-0.1, 0.1)
            # Manual hot-topic fast lane (2026-08-31, core/hot_topics.py) —
            # instead of one flat sleep, wait in fast_poll_seconds chunks and
            # check in between whether a manually-flagged-hot candidate is
            # sitting unsent; if so, cut the wait short and run the next
            # cycle now instead of waiting out the full interval. The check
            # itself is a cheap, existence-only Notion query (no LLM cost),
            # so this is safe to run often. Bounded risk if something stays
            # hot-flagged but never wins (e.g. repeatedly filtered/declined):
            # worst case is one full cycle (real extraction+writer cost)
            # every fast_poll_seconds instead of every interval_seconds —
            # acceptable since it only happens while the user has
            # deliberately flagged something as breaking, not automatically.
            remaining = config.publish.interval_seconds + jitter
            while remaining > 0:
                chunk = min(config.hot_topics.fast_poll_seconds, remaining)
                await asyncio.sleep(chunk)
                remaining -= chunk
                if remaining > 0 and await has_unpublished_hot_candidate(config):
                    logger.info("run_cycle: unpublished hot-flagged candidate detected — triggering cycle early")
                    break
    finally:
        await posted_store.close()
        await event_store.close()


if __name__ == "__main__":
    asyncio.run(main())
