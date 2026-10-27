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

Pipeline order per cycle (2026-10-21, Linear AI-69, user 10-19/10-20):
  load sources (Notion) -> fetch RSS (3h publish-age filter) -> URL-hash
  Redis dedup -> full text (AI-29, user 10-21: every article; saved for the
  publish cycle; no text -> one more try an hour later, then dropped; a
  short/empty summary becomes the first paragraph) -> reprints (same headline
  on another site / same summary / same full text) -> English only -> no roundups or newsletters -> embedding
  pre-score -> same article inside the batch (cosine >= 0.96, real headline
  and summary only) -> score gate (gpt-4o-mini, >= 5; under 6 goes on only
  while the pool holds fewer than 10 usable 6+ candidates, AI-31) -> merged
  judge (core/merged_judge.py, Haiku, one call per article: which event;
  anything new for readers compared with OUR POSTS, NOTHING_NEW / NOT_WORTH
  -> event store only, not Notion; same update already waiting in Notion ->
  event store only; the answer is stored on the article's point) -> event store
  commit -> storyline links -> topic + appeal tags (Haiku) -> Notion
  candidate pool -> Qdrant embedding write.
  Replaced on 10-21: same-event clustering inside the batch, the event-store
  judge, classify_subtype, the published-event and stale-event guards, and
  cross-cycle dedup.

Content generation is deliberately NOT part of this cycle (main_publish.py,
2026-08-05). Full-text extraction moved back here on 2026-10-21 (AI-29): it
costs no model call, and an article whose text cannot be fetched should not
be scored or reach the pool at all.

Usage:
  python3 main.py              # normal run
  python3 main.py --dry-run    # logs what would be added, never writes to Notion/Qdrant
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents.embedder import Embedder
from agents.extractor import Extractor
from agents.rss_fetcher import fetch_all
from agents.scorer import Scorer
from agents.scorer_shadow import ShadowScorer
from agents.appeal_tagger import AppealTagger
from agents.candidate_selector import usable_count
from agents.topic_tagger import TopicTagger
from core import fulltext
from core.config import load_config
from core.event_identity import EventVerifier, HubIndex, entity_tokens, event_identity_text, event_judge_text, extract_event_frame, log_decision
from core.hashing import cosine_similarity, sha256_url_hash
from core.hot_topics import fetch_active_hot_topics
from core.language import is_english
from core.merged_judge import MergedJudge, log_verdict
from core.notion_candidates import query_eligible_candidates, retire_by_url, write_candidate
from core.prescore import PreScorer, log_prescore_decision
from core.reprint import RELAYS, reprint_of, site
from core.roundup import roundup_rule, newsletter_rule
from core.notion_sources import load_rss_sources
from core.qdrant_store import EventStore, PostedHistoryStore, QdrantStore, ensure_collection_with_retry
from core.redis_store import AppealLabels, RedisStore, StuckVersions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

# Summaries (2026-10-21, AI-69, user 10-20): what counts as no summary.
_BOILERPLATE = re.compile(r"the post .{0,300} appeared first on .{0,80}$", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _judge_summary(text: str, description: str, limit: int = 400) -> str:
    """What the merged judge reads under the headline (2026-10-27, user: 可以):
    the article's first paragraph(s), up to `limit` characters, instead of the
    RSS summary -- the user's rule is that a repeat check compares the headline
    and the first paragraph. Some RSS summaries are a photo caption or the
    headline again, and the judge then answered CANNOT_TELL and let the repeat
    through. Replayed on 84 AM1ST articles (judge1027): CANNOT_TELL 6 -> 3,
    NOTHING_NEW 10 -> 15, 2 of 5 leaked repeats caught (DOJ charging 48
    immigrants, U.S. ready to strike Iran in 96 hours), 1 real development of 30
    held back once and passed on another run. The RSS summary when there is no
    text."""
    out = ""
    for p in (x.strip() for x in (text or "").split("\n")):
        if len(p) < 60:
            continue
        out = f"{out} {p}".strip()
        if len(out) >= limit:
            break
    return out[:limit] or description


def _summary_text(description: str) -> str:
    """The summary without tags and without the RSS "The post ... appeared first on ..." line."""
    return _BOILERPLATE.sub("", " ".join(_TAG.sub(" ", description or "").split())).strip()


def _comparable(c) -> bool:
    """Layer 2 compares only articles with a real headline and summary."""
    return len((c.title or "").split()) >= 4 and len(_summary_text(c.description)) >= 40 and "{{" not in (c.description or "")


class _SilentAlerts:
    """agents/extractor.py raises a Notion alert when a cookie source fails; at
    ingestion that is just an article without text, not something to act on."""

    async def alert(self, page_id: str, message: str) -> None:
        return None


async def _fetch_full_text(config, candidates, sources, retried: set[str]):
    """Layer 1.05 (2026-10-21, AI-29, user 10-21): the full text of every new
    article (agents/extractor.py, the render service for the sites that need
    it), saved for the publish cycle (core/fulltext.py). An article with no
    fetchable text is not scored. A plain fetch that fails goes straight to the
    headless browser in the same cycle (user 10-22: the sites that refuse the
    plain fetch -- Washington Times, BizPac Review, Daily Signal, Tennessee
    Star, Rumble -- are blocking a client, not down: the browser got 57 of 72
    articles the plain retry had given up, while the plain retry an hour later
    got 1 of 245). What the browser cannot get either waits
    ingestion.fulltext_retry_minutes for one more browser try, then is
    dropped. Browser fetches run two at a time: the render service is shared.
    A summary shorter than ingestion.min_summary_chars becomes the article's
    first paragraph.

    Returns (articles with text, {url_hash: text})."""
    floor = config.ingestion.min_summary_chars
    extractor = Extractor(config, _SilentAlerts())
    plain_gate, browser_gate = asyncio.Semaphore(6), asyncio.Semaphore(2)
    texts: dict[str, str] = {}
    how: Counter = Counter()
    wait: list = []

    async def extract(c, browser: bool):
        # attempts >= browser_after_attempts makes the extractor render the page
        # whenever the plain fetch comes back empty or too thin.
        async with (browser_gate if browser else plain_gate):
            try:
                return await extractor.extract(c.url, sources,
                                               attempts=config.extraction.browser_after_attempts if browser else 0)
            except Exception:
                logger.exception("run_cycle: extraction failed for %s", c.url)
                return None

    def keep(c, text, how_key):
        texts[c.url_hash] = text
        fulltext.save(config, c.url_hash, text)
        how[how_key] += 1
        if len(_summary_text(c.description)) < floor:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 60]
            c.description = (paragraphs[0] if paragraphs else text.strip())[:400]
            how["summary_from_text"] += 1

    async def fetch(c):
        if c.url_hash in retried:
            text = await extract(c, browser=True)
            if text:
                keep(c, text, "second_try")
            else:
                how["given_up"] += 1
                logger.info("run_cycle: %s dropped — no article text on the second try (%s)", c.url, (c.title or "")[:120])
            return
        text = await extract(c, browser=False)
        if text:
            keep(c, text, "fetched")
            return
        text = await extract(c, browser=True)
        if text:
            keep(c, text, "browser")
            return
        how["retry_later"] += 1
        wait.append(c)

    await asyncio.gather(*(fetch(c) for c in candidates))
    fulltext.add_retry(config, wait)
    logger.info("run_cycle: full text for %d of %d: %s", len(texts), len(candidates), dict(how))
    return [c for c in candidates if c.url_hash in texts], texts


def _stronger_version(c, queued_url: str, queued: dict | None) -> bool:
    """Is this report a stronger version of the update already waiting in
    Notion (2026-10-22, AI-24)? The original over a relay site's copy (user
    10-14: 优先原稿不发转述).

    2026-10-26 (user approved): no longer "a higher score wins". Of 46
    replacements on 10-23..26, 42 were by score, and two versions of one update
    differ mostly by one point, the scorer's own noise (10-22: 20 of 22). The
    noise favoured rewrites with louder headlines: in 6 pairs the article
    swapped in links the one it replaced, none the other way round. The other
    version still counts toward the row's versions."""
    return site(queued_url) in RELAYS and site(c.url) not in RELAYS


async def _usable_supply(config) -> int:
    """How many candidates of 6+ the publish batch could draw from right now
    (2026-10-21, AI-31): 8+ inside the age ceiling, 6-7.x published within
    publish.fresh_hours -- agents/candidate_selector.select_batch()'s tiers 1
    and 2. Code only: one Notion query for 6+ rows. 0 when the query fails."""
    pool = await query_eligible_candidates(config, min_score=config.ingestion.low_score_below)
    return usable_count(pool, config, datetime.now(timezone.utc))


async def run_cycle(
    config, redis_store, qdrant_store, event_store, embedder, scorer, topic_tagger, hub_index, event_verifier, dry_run,
    prescorer=None, shadow_scorer=None, appeal_tagger=None, appeal_labels=None, stuck_versions=None,
    posted_store=None, merged_judge=None,
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
    # Articles whose text failed an hour ago get their second try now.
    retry = fulltext.take_due(config)
    if retry:
        logger.info("run_cycle: %d article(s) back for a second full-text try", len(retry))
    survivors += retry
    pruned = fulltext.prune(config)
    if pruned:
        logger.info("run_cycle: pruned %d saved full text(s)", pruned)
    if not survivors:
        return

    # --- Layer 1.05: full text (2026-10-21, AI-29, user 10-21) — no text, no
    # scoring; every later step reads title + summary. See _fetch_full_text(). ---
    survivors, full_texts = await _fetch_full_text(config, survivors, sources, {c.url_hash for c in retry})
    if not survivors:
        return

    # --- Layer 1.1: reprints (2026-10-12, core/reprint.py) — the same article
    # under another URL, by exact headline (on another site), description or
    # full text (10-21). ---
    before_reprint = len(survivors)
    kept = []
    for c in survivors:
        why = await reprint_of(redis_store, c.title or "", c.description or "", c.url, full_texts.get(c.url_hash, ""))
        if why:
            logger.info("run_cycle: %s dropped — reprint (%s): %s", c.url, why, (c.title or "")[:120])
        else:
            kept.append(c)
    survivors = kept
    if len(survivors) != before_reprint:
        logger.info("run_cycle: %d/%d survive the reprint check", len(survivors), before_reprint)
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

    # --- Layer 1.52: no digests, rolling live pages or link lists
    # (2026-09-29, core/roundup.py) — user rule: a post has to be about one
    # thing, and a page that is several unrelated stories cannot be written
    # into one caption without the caption picking one and dropping the rest.
    # Before the pre-score so a dropped page costs not even an embedding. ---
    before_roundup = len(survivors)
    kept = []
    for c in survivors:
        rule = roundup_rule(c.title or "", c.url) or newsletter_rule(c.description or "", c.url)
        if rule:
            logger.info("run_cycle: %s dropped — %s (%s)", c.url, rule, (c.title or "")[:120])
        else:
            kept.append(c)
    survivors = kept
    if len(survivors) != before_roundup:
        logger.info("run_cycle: %d/%d survive the roundup filter", len(survivors), before_roundup)
    if not survivors:
        return

    # --- Layer 1.55: embedding pre-score (2026-09-29, core/prescore.py) —
    # drop a candidate whose title the model is confident cannot reach the
    # pool gate, before anything expensive touches it. It sits here rather
    # than beside the Scorer on purpose: Layers 1.6 to 3 below spend an HTTP
    # fetch, a clustering embed and, when the rule layer is unsure, an LLM
    # dedup judge on every candidate that gets this far, and one dropped here
    # costs none of them. Title embeddings go out in one batched request,
    # fractions of a cent per cycle. Any failure scores everyone. ---
    if prescorer is not None and prescorer.enabled:
        try:
            vectors = await embedder.embed_many([(c.title or " ")[:500] for c in survivors])
            kept, decisions = [], Counter()
            for c, v in zip(survivors, vectors):
                decision, detail = prescorer.decide(c.url_hash, c.title or "", v)
                decisions[decision] += 1
                log_prescore_decision(config.prescore.log_path,
                                      {"url": c.url, "title": (c.title or "")[:160],
                                       "decision": decision, **detail})
                if decision != "skip":
                    kept.append(c)
            logger.info("run_cycle: pre-score %s (model %s) — %d/%d go on to be scored",
                        dict(decisions), prescorer.version, len(kept), len(survivors))
            survivors = kept
        except Exception:
            logger.exception("run_cycle: pre-score failed — scoring everything this cycle")
        if not survivors:
            return

    # Manual breaking-news override (2026-08-31, core/hot_topics.py) — every
    # currently-live flag for this bot's own channel, embedded once per
    # cycle. Empty in the common case (no flag active right now); a failed
    # Notion read also comes back empty (fail open), so this never blocks
    # the cycle. See core/config.py's HotTopicsConfig docstring.
    hot_topic_texts = await fetch_active_hot_topics(config)
    hot_topics: list[tuple[str, list[float]]] = []
    for text in hot_topic_texts:
        try:
            hot_topics.append((text, await embedder.embed(text[:2000])))
        except Exception:
            logger.exception("run_cycle: failed to embed hot-topic flag %r, skipping it this cycle", text)
    if hot_topics:
        logger.info("run_cycle: %d active hot-topic flag(s) this cycle", len(hot_topics))

    # --- Layer 2: the same article inside this batch, by vector (2026-10-21,
    # AI-69, user 10-20). Only near-verbatim copies: cosine >=
    # ingestion.near_verbatim_cosine, and only between two articles that both
    # have a real headline (4+ words) and summary -- two LifeSiteNews items
    # with no title and the same page-code "summary" scored 0.99 apart. At the
    # old 0.85, this step dropped other outlets' reports of the same event
    # (10-20 measurement); those now go on to the merged judge below.
    # Every survivor's title+summary embedding is kept for the judge and the
    # event store. ---
    near_verbatim = config.ingestion.near_verbatim_cosine
    texts = [f"{c.title}\n{c.description}"[:6000] for c in survivors]
    try:
        vectors = await embedder.embed_many(texts)
    except Exception:
        logger.exception("run_cycle: batched embedding failed, embedding one by one")
        vectors = []
        for t in texts:
            try:
                vectors.append(await embedder.embed(t))
            except Exception:
                vectors.append(None)
    embedded: list[tuple] = []  # (candidate, embedding)
    for c, embedding in zip(survivors, vectors):
        if embedding is None:
            logger.warning("run_cycle: %s dropped — embedding failed", c.url)
            continue
        twin = None
        if _comparable(c):
            for other, other_embedding in embedded:
                if _comparable(other) and cosine_similarity(embedding, other_embedding) >= near_verbatim:
                    twin = other
                    break
        if twin is not None:
            logger.info("run_cycle: %s dropped — same article as %s in this batch (cosine >= %.2f)", c.url, twin.url, near_verbatim)
            continue
        embedded.append((c, embedding))
    logger.info("run_cycle: %d/%d survive the same-article check", len(embedded), len(survivors))

    # Manual hot-topic flags, per article (was per cluster; same threshold and log).
    hot_untils: dict[str, int] = {}
    if hot_topics:
        for c, embedding in embedded:
            best_hot_score, best_hot_text = max(
                ((cosine_similarity(embedding, emb), text) for text, emb in hot_topics), key=lambda x: x[0],
            )
            is_hot_match = best_hot_score >= config.hot_topics.match_threshold
            log_decision(config, {
                "check_type": "hot_topic_match", "cluster_representative_url": c.url,
                "best_hot_score": best_hot_score, "matched_flag_text": best_hot_text,
                "threshold": config.hot_topics.match_threshold, "matched": is_hot_match,
            })
            if is_hot_match:
                hot_untils[c.url_hash] = int(time.time()) + config.hot_topics.ttl_hours * 3600
                logger.info("run_cycle: %s matched an active hot-topic flag (%.3f)", c.url, best_hot_score)

    # --- Score gate (unchanged: gpt-4o-mini, >= openai.score_threshold) ---
    # Trending headlines fetched once per cycle (agents/trending.py). Since
    # 2026-10-21 the scorer runs before the event store is consulted, so a
    # candidate carries the defaults heat_score=1.0 / event_first_seen_at=None
    # and the prompt's "hours since first seen" is its own publish time: that
    # field only served to spot old news rehashed, which the publish cycle's
    # 48h timeliness and first-report checks cover.
    # No trending headlines in the scoring request any more (2026-10-23, user:
    # 那15条Google新闻头条现在就能换掉). They never did their job; a hot-topic
    # input may take their place later (AI-78).
    trending_headlines = None
    passed: list[tuple] = []  # (candidate, embedding)
    for c, embedding in embedded:
        try:
            score_output = await scorer.score(c, trending_headlines)
            if score_output is None:
                continue
            c.llm_score = score_output.llm_score
            c.llm_comment = score_output.llm_comment
            if c.llm_score >= config.openai.score_threshold:
                passed.append((c, embedding))
            else:
                # The only record a rejected candidate ever existed (see
                # tools/reject_report.py): title and llm_comment included.
                logger.info("run_cycle: %s scored %.1f, below threshold — %s — %s",
                            c.url, c.llm_score, c.llm_comment or "-", c.title[:160])
        except Exception:
            logger.exception("run_cycle: unhandled error scoring %s, skipping this item", c.url)
            continue
        # Shadow scoring — observes only, in its own try/except (2026-10-01).
        if shadow_scorer is not None:
            try:
                await shadow_scorer.observe(c, c.llm_score, c.llm_comment, trending_headlines)
            except Exception:
                logger.debug("run_cycle: shadow scoring failed for %s", c.url)

    # --- The 5-score band only when news is short (2026-10-21, AI-31, user
    # 10-21: 新闻不够的时候5分档可以入库，新闻够的时候就不要入库). See
    # _usable_supply(); a failed count lets them through, as before. ---
    low = [c for c, _ in passed if c.llm_score < config.ingestion.low_score_below]
    if low:
        usable = await _usable_supply(config)
        if usable >= config.ingestion.low_score_supply_floor:
            passed = [(c, e) for c, e in passed if c.llm_score >= config.ingestion.low_score_below]
            for c in low:
                logger.info("run_cycle: %s held back — scored %.1f and the pool has %d usable %.0f+ candidates (%s)",
                            c.url, c.llm_score, usable, config.ingestion.low_score_below, c.title[:120])
        else:
            logger.info("run_cycle: %d article(s) under %.0f go on — the pool has only %d usable %.0f+ candidates",
                        len(low), config.ingestion.low_score_below, usable, config.ingestion.low_score_below)

    # --- Merged judge (2026-10-21, core/merged_judge.py, Linear AI-69) ---
    # Replaces the event-store judge, classify_subtype, both event guards and
    # cross-cycle dedup. One Haiku call per article that cleared the score
    # gate, oldest first and one at a time, so a later article sees the
    # earlier ones' verdicts: which event it belongs to (or a new one); whether
    # it tells our readers anything our posts have not (NOTHING_NEW / NOT_WORTH
    # with a named post -> dropped); whether the same update is already
    # waiting in Notion (-> recorded in the event store only, no Notion row).
    # Anything the judge cannot read or answer keeps the article (fails open).
    passed.sort(key=lambda pair: pair[0].published_at)
    judge_ready = bool(merged_judge is not None and config.merged_judge.enabled and passed
                       and await merged_judge.prepare_cycle())
    added_count = folded_count = dropped_count = replaced_count = 0
    for c, embedding in passed:
        verdict = None
        if judge_ready:
            try:
                verdict = await merged_judge.judge(c.url, c.title,
                                                   _judge_summary(full_texts.get(c.url_hash, ""), c.description), embedding)
            except Exception:
                logger.exception("run_cycle: merged judge failed on %s — kept", c.url)
            if verdict is not None:
                log_verdict(config, verdict.record)
        action = verdict.action if verdict is not None else "keep"
        # The same update already waiting in Notion (2026-10-22, AI-24, user:
        # 更强的版本换进 Notion): a stronger report takes that row's place --
        # the original over a relay, else a higher score -- instead of folding
        # into it; a weaker one folds and adds to the row's version count.
        replaced, queued = None, None
        if action == "fold":
            try:
                queued = (await event_store.points_for_urls([verdict.fold_url], with_vectors=False)).get(verdict.fold_url)
            except Exception:
                logger.exception("run_cycle: could not read the queued version %s", verdict.fold_url)
            if _stronger_version(c, verdict.fold_url, queued):
                action, replaced = "keep", verdict.fold_url
        if action == "drop":
            # Still filed under its event, outside Notion (user 10-21, AI-69):
            # one more source of a story we already told.
            dropped_count += 1
            logger.info("run_cycle: %s dropped — %s, our post(s) %s already told readers (%s)", c.url, verdict.relation,
                        ",".join(verdict.record.get("covered_by", [])), c.title[:120])
        if dry_run:
            if action != "drop":
                logger.info("run_cycle: [dry-run] would %s: %s (score=%.1f, event=%s)%s",
                            "record in the event store only" if action == "fold" else "add to candidate pool",
                            c.url, c.llm_score, (verdict.event_id if verdict else None) or "new",
                            f", replacing {replaced}" if replaced else "")
                if action == "fold":
                    folded_count += 1
                else:
                    added_count += 1
            continue
        try:
            matched = await event_store.get_by_id(verdict.event_id) if verdict is not None and verdict.event_id else None
            pub = int(c.published_at.timestamp())
            rep_text = event_identity_text(c.title, c.description)
            own_text = event_judge_text(c.title, c.description)
            # Free, no-LLM frame guess, only used when this starts a new event;
            # canonical title/summary are the seed article's own (EventStore.commit()).
            seed_frame = {**extract_event_frame(rep_text), "canonical_title": c.title,
                          "canonical_summary": c.description or c.title}
            # A timeline entry for a genuine new development of an existing
            # event (was classify_subtype()'s CORE_UPDATE): the judge's HAS_NEW.
            timeline_entry = ({"ts": pub, "source": c.source_name, "url": c.url, "summary": c.title}
                              if matched is not None and verdict.relation == "HAS_NEW" else None)
            heat_score, first_seen_unix, event_id, core_entities, hot_until = await event_store.commit(
                matched, {c.source_name}, c.source_name, pub,
                representative=(embedding, c.source_name, pub, rep_text, entity_tokens(rep_text), c.url, own_text),
                extra_points=[], related_links=None, seed_frame=seed_frame, heat_multiplier=1.0,
                hot_until=hot_untils.get(c.url_hash, 0), timeline_entry=timeline_entry,
                in_notion=action == "keep",
                # What the judge said about our posts, for main_publish.py (AI-24);
                # none when the judge did not actually answer.
                relation=verdict.relation if verdict is not None and verdict.relation != "ERROR" else None,
                # For the writer (10-22): what this report adds, when it adds something.
                new_info=(verdict.record.get("new_info", "") if verdict is not None and verdict.relation == "HAS_NEW" else ""),
                score=c.llm_score,
                versions=(int((queued or {}).get("versions") or 1) + 1) if replaced else 1,
            )
            await hub_index.bump(event_id, core_entities)
            if judge_ready:
                merged_judge.note_added(c.url, own_text, embedding, action == "keep", event_id)
            if action == "fold" and verdict.fold_url:
                await event_store.add_version(verdict.fold_url)
        except Exception:
            logger.exception("run_cycle: event-store commit failed for %s, skipping this item", c.url)
            continue
        if replaced:
            # The weaker row leaves Notion, the judge's queue, and the fold
            # targets (StuckVersions), so later reports fold into this one.
            replaced_count += 1
            try:
                await event_store.leave_queue(replaced)
                retired = await retire_by_url(config, replaced)
                if stuck_versions is not None:
                    await stuck_versions.mark(sha256_url_hash(replaced), "replaced")
                if judge_ready:
                    merged_judge.forget(replaced)
                logger.info("run_cycle: %s replaces %s in the candidate pool — the stronger report of the same update "
                            "(%d row(s) retired)", c.url, replaced, retired)
            except Exception:
                logger.exception("run_cycle: could not retire the replaced version %s", replaced)
        if action == "drop":
            continue
        if action == "fold":
            folded_count += 1
            logger.info("run_cycle: %s recorded in the event store only — same update as %s, already in Notion",
                        c.url, verdict.fold_url)
            continue

        # Cross-event storyline linking (2026-08-31, unchanged): this article's
        # specific entity tokens, reverse-looked-up against other events.
        this_event_title = (matched or {}).get("canonical_title", "") or c.title
        candidate_event_ids: set[str] = set()
        for tok in entity_tokens(rep_text):
            if await hub_index.token_score(tok) < config.entity_verifier.hub_event_count_threshold:
                candidate_event_ids |= await hub_index.token_events(tok)
        candidate_event_ids.discard(event_id)
        for other_event_id in candidate_event_ids:
            other = await event_store.get_by_id(other_event_id)
            if other is None:
                continue
            other_text = other.get("representative_text", "") or other.get("canonical_summary", "")
            related, llm_raw = await event_verifier.related_event(other_text, rep_text)
            log_decision(config, {
                "event_id": event_id, "linked_candidate_event_id": other_event_id,
                "llm_related_raw": llm_raw, "final_verdict": "RELATED" if related else "NOT_RELATED",
                "candidate_text": rep_text, "matched_representative_text": other_text,
            })
            if related:
                other_title = other.get("canonical_title", "") or other_text
                await event_store.link_related_events(event_id, this_event_title, other_event_id, other_title)
                logger.info("run_cycle: linked event %s <-> %s as related storyline", event_id, other_event_id)

        try:
            c.heat_score = heat_score
            c.event_first_seen_at = datetime.fromtimestamp(first_seen_unix, tz=timezone.utc)
            c.is_hot = hot_until > int(time.time())
            # Topic and appeal tags (Haiku, gpt-4o-mini when Haiku fails), only
            # for what reaches the pool. Both fail open to None.
            c.topic = await topic_tagger.tag(c)
            # Iranian regime propaganda stays out of the pool (owner, 2026-10-23,
            # agents/topic_tagger.py). Its event-store row leaves the judge's queue
            # view too, so later reports of the same update are not folded into a
            # row that will never be posted (the 10-22 Hubbard hole).
            if c.topic == "伊朗宣传":
                await event_store.leave_queue(c.url)
                logger.info("run_cycle: %s dropped — Iranian regime propaganda (%s)", c.url, c.title[:120])
                continue
            appeal = await appeal_tagger.tag(c) if appeal_tagger is not None else None
            if not await write_candidate(config, c):
                logger.warning("run_cycle: candidate-pool write failed for %s", c.url)
                continue
            await qdrant_store.write_embedding(
                c.url, c.url_hash, f"{c.title}\n{c.description}"[:6000], int(c.published_at.timestamp()), embedding,
            )
            if appeal is not None and appeal_labels is not None:
                await appeal_labels.set(c.url_hash, appeal)
            added_count += 1
            logger.info("run_cycle: added to candidate pool: %s (score=%.1f, topic=%s, appeal=%s, event=%s)",
                        c.url, c.llm_score, c.topic or "-", {True: "yes", False: "no"}.get(appeal, "-"),
                        "existing" if matched is not None else "new")
        except Exception:
            logger.exception("run_cycle: unhandled error writing %s, skipping this item", c.url)

    logger.info("run_cycle: %d added to candidate pool, %d recorded in the event store only, %d dropped by the merged judge this cycle"
                " (%d replaced a weaker queued version)", added_count, folded_count, dropped_count, replaced_count)


async def main() -> None:
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    config = load_config("config/config.yaml")

    redis_store = RedisStore(config)
    qdrant_store = QdrantStore(config)
    event_store = EventStore(config)
    posted_store = PostedHistoryStore(config)
    hub_index = HubIndex(config)
    event_verifier = EventVerifier(config)
    await ensure_collection_with_retry(qdrant_store, "am1st_embeddings")
    await ensure_collection_with_retry(event_store, "am1st_events")
    await ensure_collection_with_retry(posted_store, "am1st_posting_news_embedding")
    embedder = Embedder(config)
    scorer = Scorer(config)
    # Built once alongside Scorer, not per cycle: it holds an AsyncOpenAI client
    # and rebuilding one every ten minutes leaks httpx connections. Observes
    # only — see agents/scorer_shadow.py.
    shadow_scorer = None
    if config.openai.shadow_scoring_enabled:
        try:
            shadow_scorer = ShadowScorer(config)
            logger.info("shadow scoring is ON (observes only, writes %s)",
                        config.openai.shadow_scoring_log_path)
        except Exception:
            logger.exception("shadow scorer failed to start — continuing without it")
    topic_tagger = TopicTagger(config)
    appeal_tagger = AppealTagger(config)
    appeal_labels = AppealLabels(config)
    stuck_versions = StuckVersions(config)
    prescorer = PreScorer(config)
    merged_judge = MergedJudge(config, event_store, posted_store, qdrant_store, stuck_versions)

    if dry_run:
        logger.info("Running in --dry-run mode: Notion/Qdrant writes will be logged, not sent")

    try:
        while True:
            started = time.monotonic()
            try:
                await asyncio.wait_for(
                    run_cycle(config, redis_store, qdrant_store, event_store, embedder, scorer, topic_tagger, hub_index, event_verifier, dry_run, prescorer, shadow_scorer,
                              appeal_tagger, appeal_labels, stuck_versions, posted_store, merged_judge),
                    timeout=config.cycle_timeout_seconds,
                )
            except asyncio.TimeoutError:
                # This process self-loops (see module docstring) instead of
                # being re-triggered externally like the old n8n workflow —
                # without this cutoff, one truly stuck cycle would freeze
                # every future cycle forever, since nothing else would ever
                # call run_cycle() again (2026-08-12 discussion).
                logger.error("run_cycle exceeded %ds — cutting it off, will retry next cycle", config.cycle_timeout_seconds)
            except Exception:
                logger.exception("run_cycle failed")
            logger.info("run_cycle: cycle took %.1fs", time.monotonic() - started)
            jitter = config.poll_interval_seconds * random.uniform(-0.1, 0.1)
            await asyncio.sleep(config.poll_interval_seconds + jitter)
    finally:
        await redis_store.close()
        await qdrant_store.close()
        await event_store.close()
        await posted_store.close()
        await hub_index.close()
        await appeal_labels.close()
        await stuck_versions.close()


if __name__ == "__main__":
    asyncio.run(main())
