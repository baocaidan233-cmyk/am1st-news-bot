"""
AM1ST — the separate publish cycle, ported from v1.4_am1st_notion_to_gettr_auto
posting.json. Runs independently of main.py's ingestion cycle (see
project_am1st_migration memory's 2026-08-04 "two separate workflows" note):
main.py only ever writes candidates into the shared Notion candidate pool;
this process is the only thing that ever reads that pool to actually pick
something to post.

Cycle order (every publish.interval_seconds by default, 30 min — but see
compute_dynamic_interval() below, 2026-09-05: the actual wait each cycle
scales 0.6x-1.3x that base on real-time news volume, clamped to
[15min, 39min], on top of which core/hot_topics.py's manual fast lane can
still cut things short for a human-flagged breaking story; a cycle that
finds nothing worth publishing just publishes nothing — see run_cycle()'s
widen-on-empty give-up, not a longer wait until the next check):
  query eligible candidates (Notion: not sent, <=24h old at the query level,
     llm_score>=6 — select_batch() then applies the real day-aware freshness
     ceiling: 12h on weekdays, 24h on weekends, see agents/candidate_selector.py)
  -> drop stale "former president Trump" phrasing (cheap pass, title+
     description only)
  -> tiered batch selection (fresh+high-score preferred, cascading fallback,
     3-10 candidates) -> extraction/content-gen/rank/dedup on that batch;
     if nothing survives to publish, widen to the next batch_max-sized
     chunk of the still-untried eligible pool and repeat, up to
     publish.max_widen_attempts times (2026-09-05 — see run_cycle())
  -> full-text extraction + content generation for just this small batch
     (moved here from main.py, 2026-08-05 — see agents/extractor.py's
     docstring for why), dropping anything the writer judges "No comment"
  -> re-check the former-Trump filter now that full text/post_content
     exist, in case only the article body (not title/description) had it
  -> deterministic priority formula (agents/priority_ranker.py, 2026-09-04 —
     replaced the old LLM re-rank call after a live audit found it unstable
     in exactly the range that decides most cycles): llm_score + heat_bonus
     + trending_bonus - freshness_penalty, given the same trending-headlines
     snapshot as before (agents/trending.py — never ingested/scored/
     published from directly)
  -> walk the ranked list, skipping anything whose embedding is a near-
     match (threshold 0.70 — stricter than the ingestion side's 0.8,
     deliberately, since this is a fully-autonomous post: see feedback in
     project_am1st_migration memory) of content this channel already
     posted in the last 24h AND that core/event_identity.py's
     EventVerifier.same_event() also confirms is the same real-world
     occurrence, not just a lexically-similar next-stage development
     (2026-09-05 — a live audit found cosine alone wrongly blocked genuine
     escalations in an ongoing story, e.g. a state court ruling vs. the
     later federal Supreme Court appeal of that same ruling, as "duplicates"
     of the earlier stage; see agents/posted_dedup_checker.py's docstring)
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

from agents.candidate_selector import filter_former_trump, select_batch, shortlist
from agents.embedder import Embedder
from agents.extractor import Extractor
from agents.gettr_publisher import GettrPublisher
from agents.og_metadata import fetch_link_preview
from agents.poster import card_for_broken_preview
from agents.posted_dedup_checker import content_for_embedding, find_publishable
from agents.priority_ranker import PriorityRanker, log_publish_outcome
from agents.trending import fetch_trending_headlines
from agents.staleness_checker import StalenessChecker
from agents.writer import Writer
from agents.editor import EditorPicker
from core.selection_trace import SelectionTrace
from core.alerts import AlertNotifier
from core.config import load_config
from core.event_identity import EventVerifier, HubIndex
from core.caption_guard import former_president_violation
from core.language import is_english
from core.notion_candidates import has_unpublished_hot_candidate, mark_dedup_rejected, mark_extraction_failed, mark_send_status, mark_writer_rejected, query_eligible_candidates, recent_published_topic_counts, record_extraction_failure
from core.notion_candidates import topic_publication_debt
from core.topic_mix import compute_adjustments
from core.publish_cadence import compute_dynamic_interval
from core.notion_sources import load_rss_sources
from core.qdrant_store import EventStore, PostedHistoryStore, ensure_collection_with_retry
from core.redis_store import AppealLabels, BatchSeen, CaptionCache, CycleCounter, PostedDupStrikes
from core.title_guard import title_violation
from core.stance_guard import StanceVectorGuard, epithet_violation

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
    event_verifier: EventVerifier,
    publisher: GettrPublisher,
    extractor: Extractor,
    writer: Writer,
    staleness_checker: StalenessChecker,
    caption_cache: CaptionCache,
    dup_strikes: PostedDupStrikes,
    hub_index: HubIndex,
    editor: EditorPicker,
    batch_seen: BatchSeen,
    cycle_counter: CycleCounter,
    dry_run: bool,
    appeal_labels: AppealLabels | None = None,
    stance_vec: StanceVectorGuard | None = None,
) -> bool:
    """Returns True iff this cycle actually published something — main()'s
    loop uses this to track how recently the channel last posted, so the
    hot-topic fast lane (below) can be kept from firing more often than
    dynamic_publish.min_interval_seconds even when a permanently-stuck
    hot-flagged candidate keeps re-qualifying every fast_poll_seconds (see
    2026-09-05 incident: a paywalled FT article that could never actually
    extract kept is_hot=true and unsent, firing the fast lane every 3min
    for 2+ hours straight, 31 real publishes instead of the ~4 the dynamic
    interval alone would have produced)."""
    candidates = await query_eligible_candidates(config)
    if not candidates:
        logger.info("run_cycle: no eligible candidates this cycle")
        return False

    candidates = filter_former_trump(candidates)
    if not candidates:
        logger.info("run_cycle: all candidates dropped by former-Trump filter")
        return False

    # Reader-appeal labels written at pool entry (2026-10-06). Only read when
    # publish.appeal_order is on; an unlabelled candidate keeps appeal=None,
    # which agents/candidate_selector.py orders exactly like no appeal.
    if config.publish.appeal_order and appeal_labels is not None:
        labels = await appeal_labels.get_many([c.url_hash for c in candidates])
        for c in candidates:
            c.appeal = labels.get(c.url_hash)
        logger.info("run_cycle: appeal labels — %d yes, %d no, %d unlabelled of %d eligible",
                    sum(c.appeal is True for c in candidates), sum(c.appeal is False for c in candidates),
                    sum(c.appeal is None for c in candidates), len(candidates))

    sources = await load_rss_sources(config)
    trending_headlines = await fetch_trending_headlines()

    # Subject-mix adjustments for this cycle — computed once here and handed
    # to both select_batch() and the ranker, so the two stages optimise the
    # same thing rather than each re-deriving it. Fails open to {} (no
    # adjustment anywhere), which is the pre-2026-09-25 behaviour.
    topic_adjustments = compute_adjustments(config, await recent_published_topic_counts(config))
    # Subject debt is no longer read: as of 2026-10-05 neither selection nor
    # ranking orders by it. core/notion_candidates.py's topic_publication_debt
    # and agents/candidate_selector.py's debt_level are kept for the
    # measurement they document, not called.
    cycle_token = str(int(datetime.now(timezone.utc).timestamp()) // 60)
    # Observation only — see core/selection_trace.py. Fails open everywhere, so
    # a trace that cannot be written cannot change what gets published.
    trace = None
    try:
        if getattr(config.publish, "selection_trace", False):
            trace = SelectionTrace(config.publish.selection_trace_path)
    except Exception:
        trace = None

    # Widen-on-empty (2026-09-05, per the user's "扩大范围，如果找不到合适的"
    # request): select_batch() only ever looks at `remaining` — the pool
    # shrinks each attempt as tried page_ids are removed, so a widen never
    # re-extracts/re-writes a candidate it already paid for this cycle.
    # Bounded by max_widen_attempts (not unbounded — extraction+content-gen
    # is real per-candidate cost, not free): each attempt is up to
    # batch_max candidates, so the default of 3 tries up to 30 total before
    # accepting "nothing to publish this cycle" is the real outcome, not
    # just "the first 10 happened to all be duplicates/stale."
    remaining = candidates
    winner = None
    ranked_len = 0
    # Which arm this cycle runs. Decided once, before the widen loop, so a
    # widen never silently changes paths mid-cycle.
    cycle_no = await cycle_counter.next() if config.editor.ab_alternate else 0
    use_editor = config.editor.enabled and (not config.editor.ab_alternate or cycle_no % 2 == 1)
    arm = "editor" if use_editor else "legacy"
    if config.editor.enabled:
        logger.info("run_cycle: selection arm = %s (cycle %d, ab_alternate=%s)",
                    arm, cycle_no, config.editor.ab_alternate)

    editor_order: dict[str, int] = {}
    editor_meta: dict[str, tuple[str | None, str]] = {}
    for attempt in range(1, config.publish.max_widen_attempts + 1):
        # Reset per attempt: a widen that falls back to select_batch must not
        # inherit the previous attempt's ordering.
        editor_order, editor_meta = {}, {}
        batch = []
        if use_editor:
            # One editorial judgement over the whole shortlist, in place of a
            # score tier plus sha1(url). See agents/editor.py for why the
            # per-story judgements it replaces could not discriminate.
            seen_counts = await batch_seen.counts([c.url_hash for c in remaining])
            short = shortlist(remaining, config, seen_counts)
            if short:
                await batch_seen.mark([c.url_hash for c in short])
                recent = await posted_store.recent_captions(
                    config.editor.recent_titles_hours, config.editor.recent_titles_max)
                picks = await editor.pick(short, recent, topic_adjustments,
                                          datetime.now(timezone.utc), trending_headlines)
                if picks:
                    batch = [p.candidate for p in picks]
                    editor_order = {p.candidate.page_id: i for i, p in enumerate(picks)}
                    editor_meta = {p.candidate.page_id: (p.subject, p.why) for p in picks}
                    for p in picks:
                        if p.subject:
                            p.candidate.topic = p.subject
        if not batch:
            # Editor off, or it failed/returned nothing — the previous path,
            # unchanged, which is the point of it being a separate branch.
            batch = select_batch(remaining, config, topic_adjustments,
                                 cycle_token=cycle_token, trace=trace)
        if not batch:
            logger.info("run_cycle: widen attempt %d — no more candidates left to try", attempt)
            break
        if trace is not None:
            try:
                trace.flush(batch)
            except Exception:
                pass
        logger.info("run_cycle: widen attempt %d — selected batch of %d for extraction/content-gen", attempt, len(batch))
        tried_ids = {c.page_id for c in batch}
        remaining = [c for c in remaining if c.page_id not in tried_ids]

        # Cheap feasibility pass, before extraction and writing. See
        # PublishConfig.batch_event_dedup_cosine for why it is re-asked here
        # and why the bar is so much higher than the publish-side one.
        if config.publish.batch_event_dedup_cosine > 0:
            kept = []
            try:
                batch_vectors = await embedder.embed_many(
                    [f"{c.title}\n{c.description}"[:6000] for c in batch])
            except Exception:
                logger.exception("run_cycle: pre-batch embedding failed — skipping this check")
                batch_vectors = []
            for i, c in enumerate(batch):
                if not batch_vectors:
                    kept.append(c)
                    continue
                try:
                    matched = await event_store.peek(batch_vectors[i])
                except Exception:
                    logger.exception(
                        "run_cycle: pre-batch event dedup failed for %s — keeping it", c.url)
                    kept.append(c)
                    continue
                score = (matched or {}).get("_score", 0.0)
                if matched and matched.get("published") and score >= config.publish.batch_event_dedup_cosine:
                    logger.info(
                        "run_cycle: %s dropped before extraction — this event was already published "
                        "(cosine %.3f), saving a fetch and a caption", c.url, score)
                    continue
                kept.append(c)
            dropped = len(batch) - len(kept)
            batch = kept
            if dropped:
                logger.info("run_cycle: widen attempt %d — %d of the batch were already-published events",
                            attempt, dropped)
            if not batch:
                logger.info("run_cycle: widen attempt %d — every candidate was an already-published event", attempt)
                continue

        # Second stance net, on the headline alone and before anything is paid
        # for: a headline written from the hostile side is dropped here instead
        # of being extracted and captioned first. See StanceVectorGuard.
        if stance_vec is not None and stance_vec.enabled:
            kept = []
            for c in batch:
                rule = await stance_vec.violation(getattr(c, "title", "") or "")
                if rule:
                    logger.warning("run_cycle: %s — blocked by stance_guard rule %s, dropped from batch", c.url, rule)
                    continue
                kept.append(c)
            batch = kept
            if not batch:
                logger.info("run_cycle: widen attempt %d — every candidate failed the vector stance check", attempt)
                continue

        generated = []
        for c in batch:
            text = await extractor.extract(c.url, sources, attempts=c.extraction_attempts)
            if not text:
                logger.info("run_cycle: %s dropped — full-text extraction failed (paywall/blocked/empty), refusing to publish off title+description alone", c.url)
                # Counted, not flagged on the first failure (2026-09-26). The
                # 2026-09-05 rule gave up immediately, which was right for the
                # paywall that prompted it but wrong for everything else: all
                # 303 candidates then carrying the flag were extraction
                # failures rather than writer rejections, and 8 of 14 flagged
                # 7+ scorers extracted fine on a retry. See
                # core/notion_candidates.py's record_extraction_failure() and
                # PublishConfig.extraction_max_attempts. A real paywall still
                # drops out, just after a bounded number of tries instead of one.
                await record_extraction_failure(config, c.page_id, c.extraction_attempts)
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

            # Staleness classification (2026-09-05) — a separate, single-
            # purpose call BEFORE the writer runs; see agents/
            # staleness_checker.py's docstring for why this isn't folded
            # into content_gen_prompt.txt (three attempts to make Writer
            # self-police this in one call all failed on real test
            # articles). Three-way, not binary — per the user's explicit
            # "两种处理方法，要么...发观点，要么...不发": a genuine analysis
            # piece with real argument/expert input (OPINION) still gets
            # written, just framed as opinion rather than dropped outright;
            # only a pure rehash with no new angle (STALE) gets dropped.
            #
            # Gated behind a free pre-filter, not run unconditionally on
            # every candidate — per the user's explicit cost concern: this
            # would otherwise double the LLM calls for every article that
            # reaches extraction, when only a minority (ones about an
            # already-old underlying event) are actually at risk. Reuses
            # event_first_seen_at — already computed at ingestion time,
            # zero extra cost — the same field agents/scorer.py already
            # uses for this exact "how old is the underlying event"
            # question. Only when that gap clears staleness_check_hours_floor
            # is there real ambiguity worth spending the LLM call on; a
            # freshly-first-seen event skips the check entirely (treated
            # as FRESH for free).
            first_seen = c.event_first_seen_at or c.published_at
            hours_since_first_seen = (datetime.now(timezone.utc) - first_seen).total_seconds() / 3600
            is_opinion = False
            if hours_since_first_seen >= config.publish.staleness_check_hours_floor:
                try:
                    verdict, verdict_raw = await staleness_checker.classify(c.title, c.content)
                except Exception:
                    logger.exception("run_cycle: staleness check failed for %s — failing open, treating as fresh", c.url)
                    verdict = "FRESH"
                if verdict == "STALE":
                    logger.info("run_cycle: %s dropped — stale rehash of an old event (%s)", c.url, verdict_raw.replace("\n", " "))
                    continue
                is_opinion = verdict == "OPINION"
                if is_opinion:
                    logger.info("run_cycle: %s classified OPINION — will write framed as analysis, not breaking news (%s)", c.url, verdict_raw.replace("\n", " "))

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

            # Cache by url_hash (2026-09-06) — without this, the same still-
            # unpublished candidate gets a freshly-reworded caption every time
            # a later cycle reconsiders it, drifting its embedding and thus
            # its cosine score against posted history; see CaptionCache's
            # docstring for the real duplicate this caused in production.
            post_content = await caption_cache.get(c.url_hash)
            cached = post_content is not None
            if post_content is None:
                post_content = await writer.write(c.title, c.content, context=background, is_opinion=is_opinion, published_at=c.published_at)

            # Last gate on the text we are about to publish under our own name.
            # Donald Trump is the sitting president, so a caption calling him a
            # former one cannot go out, whatever else is right about it. Checked
            # here rather than trusted to the prompt, which already says "always
            # President Trump" and is still only an instruction to a model; and
            # checked on every caption including ones served from the cache,
            # since a caption written before this gate existed would otherwise
            # be replayed straight past it. See core/caption_guard.py.
            violation = former_president_violation(post_content)
            if violation:
                logger.warning(
                    "run_cycle: %s — caption blocked by caption_guard rule %s, dropped from batch",
                    c.url, violation,
                )
                continue


            # Second gate on the same caption, and a different question from
            # caption_guard's: not "is this one fixed error present" but "does
            # every title this caption hands out match the article it came
            # from". Replaces the candidate board's `titel_check` Notion
            # formula, which listed eight names and has been dead since the
            # 2026-08-05 restructure stopped anything writing post_content
            # (0 of 500 published rows carry one, measured 2026-09-27).
            # Measured on 116 real caption/source pairs: 0 false positives,
            # and on the same pairs with the two errors injected, 34/37 and
            # 56/58 caught. Fails open whenever the source text is missing or
            # too short, so a cache hit that skipped extraction is unaffected.
            title_rule = title_violation(post_content, getattr(c, "content", "") or "")
            if title_rule:
                logger.warning(
                    "run_cycle: %s — caption blocked by title_guard rule %s, dropped from batch",
                    c.url, title_rule,
                )
                continue

            # Third gate on the same caption, plus the headline it will carry.
            # A different question again: not "is this fact wrong" but "are we
            # sneering at our own side". On 2026-10-05 this channel published
            # a post whose card read "Trump Goons Melt Down at CNN Star Over
            # Propaganda Ads" and whose caption called the administration's
            # own ads "propaganda ads" in its own voice. It took 26 engagement
            # against a channel median near 60. Nothing could have stopped it:
            # scoring.py's anti_admin field went offline-only with the 10-02
            # revert, and the editor prompt's one sentence on framing failed
            # on 3 of its first 9 picks.
            #
            # Only the free, code half is wired. Measured over 22 real
            # published captions x 3 runs: the epithet check caught the one
            # known violation every time with no false positives, while the
            # model half caught zero true positives and blocked 4 different
            # good posts (85, 69, 82 and 49 engagement, all at or above the
            # median) across two prompt designs. See core/stance_guard.py.
            stance_rule = epithet_violation(post_content, getattr(c, "title", "") or "")
            if stance_rule:
                logger.warning(
                    "run_cycle: %s — blocked by stance_guard rule %s, dropped from batch",
                    c.url, stance_rule,
                )
                continue

            if not cached and not Writer.is_no_comment(post_content):
                await caption_cache.set(c.url_hash, post_content)
            if Writer.is_no_comment(post_content):
                logger.info("run_cycle: %s — writer returned No comment, dropped from batch", c.url)
                # 2026-09-08, per the user's explicit request: a static
                # prompt on the same extracted text gives the same verdict
                # every time, so give up on this candidate permanently
                # instead of re-writing (and re-billing) it every future
                # cycle — see mark_writer_rejected()'s docstring for the
                # real repeat-offender that prompted this.
                await mark_writer_rejected(config, c.page_id)
                continue
            # Link appended after generation, not counted against the writer's
            # word cap — the AI's own output stays pure caption text.
            c.post_content = f"{post_content}\n\n{c.url}"
            generated.append(c)

        if not generated:
            logger.info("run_cycle: widen attempt %d — nothing survived extraction/content-gen", attempt)
            continue

        # Re-check now that full text/post_content exist — the first pass
        # only had title+description to work with, so this catches stale
        # phrasing that only shows up in the article body or the generated
        # caption.
        generated = filter_former_trump(generated)
        if not generated:
            logger.info("run_cycle: widen attempt %d — all candidates dropped by post-extraction former-Trump filter", attempt)
            continue

        if editor_order:
            # The editor already ranked these, reading all of them together
            # with the brief. Re-scoring them one at a time with the arithmetic
            # the editor replaced would just undo that.
            ranked = sorted(generated, key=lambda c: editor_order.get(c.page_id, 10**6))
            for c in ranked:
                subject, why = editor_meta.get(c.page_id, (None, ""))
                logger.info("run_cycle: editor rank %d — %s [%s] %s",
                            editor_order.get(c.page_id, -1) + 1, c.url, subject, why)
        else:
            ranked = await ranker.rank(generated, trending_headlines, topic_adjustments)
        ranked_len = len(ranked)
        # 2026-09-23 — retire a candidate the dedup check keeps rejecting,
        # instead of re-extracting and re-writing it every cycle for the rest
        # of its 24h window (172 of 184 repeatedly-judged candidates in the
        # full posted_dedup log never changed verdict = 1539 wasted cycles).
        # Notion is only written on the strike that reaches the threshold, so
        # the common case (a candidate seen once and dropped) still writes
        # nothing at all. See PublishConfig.posted_dedup_strikes_before_retire.
        async def _retire_if_settled(c) -> None:
            strikes = await dup_strikes.strike(c.url_hash)
            if strikes < config.publish.posted_dedup_strikes_before_retire:
                return
            if dry_run:
                logger.info("run_cycle: dry-run — would retire %s after %d duplicate verdicts", c.url, strikes)
                return
            if await mark_dedup_rejected(config, c.page_id):
                logger.info("run_cycle: %s retired from the pool — %d consecutive duplicate verdicts", c.url, strikes)

        winner = await find_publishable(ranked, embedder, posted_store, event_verifier, hub_index, config, on_duplicate=_retire_if_settled)
        if winner is not None:
            logger.info("run_cycle: widen attempt %d — found a publishable candidate", attempt)
            break
        logger.info("run_cycle: widen attempt %d — all candidates were duplicates/errored, widening", attempt)

    log_publish_outcome(ranked_len, winner)
    if winner is None:
        logger.info("run_cycle: no publishable candidate found after widening — nothing to publish this cycle")
        return False

    og = await fetch_link_preview(winner.url)
    # The card title is the publisher's own og:title, which the loop above
    # could not see -- it is fetched once, for the winner only. It is also
    # the first thing a reader sees, so an epithet here cannot be published
    # even though the caption already passed. Nothing goes out this cycle.
    #
    # The candidate is retired from the pool (2026-10-08). It used to stay,
    # on the reasoning that being refused again was free -- but the og:title
    # does not change, so the verdict never does, and the same candidate could
    # win the ranking again next cycle and the next, publishing nothing each
    # time until it aged out of the pool, up to 24 hours later.
    card_rule = epithet_violation("", og.get("prev_ttl") or winner.title or "")
    if card_rule:
        logger.warning(
            "run_cycle: %s — card title blocked by stance_guard rule %s, "
            "nothing published this cycle", winner.url, card_rule,
        )
        if not dry_run and await mark_writer_rejected(config, winner.page_id):
            logger.info("run_cycle: %s retired from the pool — its card title will never pass", winner.url)
        return False
    # 2026-09-22 — when the preview image Gettr would render is missing or
    # broken, attach our own headline card instead (agents/poster.py). Returns
    # None whenever the preview is fine, when the feature is off, or when
    # anything in render/upload fails, and in all of those cases publish()
    # behaves exactly as it did before.
    card_media = await card_for_broken_preview(
        config,
        title=og.get("prev_ttl") or winner.title or "",
        source_url=winner.url,
        image_url=og.get("prev_img"),
        dry_run=dry_run,
    )
    post_id = await publisher.publish(
        winner.post_content,
        log_ref=winner.url,
        prev_desc=og.get("prev_desc") or winner.description or None,
        prev_img=og.get("prev_img"),
        prev_src_link=og.get("prev_src_link") or winner.url,
        prev_ttl=og.get("prev_ttl") or winner.title,
        media=card_media,
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
            winner.url, winner.url_hash, winner.post_content, int(winner.published_at.timestamp()),
            winner_embedding, arm=arm,
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

    return published


async def main() -> None:
    load_dotenv()
    dry_run = "--dry-run" in sys.argv

    config = load_config("config/config.yaml")

    embedder = Embedder(config)
    ranker = PriorityRanker(config)
    posted_store = PostedHistoryStore(config)
    event_store = EventStore(config)
    event_verifier = EventVerifier(config)
    hub_index = HubIndex(config)
    publisher = GettrPublisher(config, dry_run=dry_run)
    alerts = AlertNotifier(config)
    extractor = Extractor(config, alerts)
    writer = Writer(config)
    staleness_checker = StalenessChecker(config)
    caption_cache = CaptionCache(config)
    dup_strikes = PostedDupStrikes(config)
    editor = EditorPicker(config)
    batch_seen = BatchSeen(config)
    appeal_labels = AppealLabels(config)
    stance_vec = StanceVectorGuard(config)
    cycle_counter = CycleCounter(config)
    await ensure_collection_with_retry(posted_store, "am1st_posting_news_embedding")
    await ensure_collection_with_retry(event_store, "am1st_events")

    if dry_run:
        logger.info("Running in --dry-run mode: Notion/Qdrant writes will be logged, not sent")

    # 2026-09-07: seed from the real last-published-post timestamp (Qdrant
    # posted_history, survives restarts) instead of always starting at None
    # — a bare `None` start meant every restart (a deploy, a crash) forgot
    # how recently the channel had actually posted and ran its very first
    # cycle immediately with no gating at all, regardless of
    # dynamic_publish.min_interval_seconds. Confirmed via 3 real same-day
    # deploy restarts each producing a publish gap under the 15min floor
    # (down to 4.5min). If the real last publish was more recent than the
    # floor, sleep out the remainder before the loop's first run_cycle()
    # call; otherwise (never published, or already long enough ago) start
    # immediately as before.
    last_publish_monotonic: float | None = None
    most_recent_ts = await posted_store.most_recent_publish_ts()
    if most_recent_ts is not None:
        seconds_since = time.time() - most_recent_ts
        last_publish_monotonic = time.monotonic() - seconds_since
        floor = config.dynamic_publish.min_interval_seconds
        if seconds_since < floor:
            wait = floor - seconds_since
            logger.info(
                "main: last real publish was %.0fs ago (< %ds floor) — waiting %.0fs before the first cycle",
                seconds_since, floor, wait,
            )
            await asyncio.sleep(wait)
    try:
        while True:
            started = time.monotonic()
            published_this_cycle = False
            try:
                published_this_cycle = await asyncio.wait_for(
                    run_cycle(config, embedder, ranker, posted_store, event_store, event_verifier, publisher, extractor, writer, staleness_checker, caption_cache, dup_strikes, hub_index, editor, batch_seen, cycle_counter, dry_run,
                              appeal_labels, stance_vec),
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
            if published_this_cycle:
                last_publish_monotonic = time.monotonic()
            logger.info("run_cycle: cycle took %.1fs", time.monotonic() - started)
            base_interval = await compute_dynamic_interval(config)
            jitter = base_interval * random.uniform(-0.1, 0.1)
            # Manual hot-topic fast lane (2026-08-31, core/hot_topics.py) —
            # instead of one flat sleep, wait in fast_poll_seconds chunks and
            # check in between whether a manually-flagged-hot candidate is
            # sitting unsent; if so, cut the wait short and run the next
            # cycle now instead of waiting out the full interval. The check
            # itself is a cheap, existence-only Notion query (no LLM cost),
            # so this is safe to run often.
            #
            # 2026-09-05 incident: a paywalled FT article that could never
            # actually extract kept is_hot=true and unsent for hours, and
            # this fast lane fired every fast_poll_seconds (180s) the whole
            # time — 31 real publishes in 2 hours instead of the ~4 the
            # dynamic interval alone would have produced, because nothing
            # here checked how recently the channel had actually last
            # published. Fixed two ways: (1) mark_extraction_failed() now
            # permanently excludes a candidate like that one after its
            # first failed extraction (see run_cycle()), so it stops
            # re-qualifying at all; (2) as a hard backstop regardless of
            # cause, the fast lane may never fire more often than
            # dynamic_publish.min_interval_seconds (15min, the user's
            # explicit "频道上限15分钟一条" ceiling) since the last actual
            # publish — a second hot-flagged candidate that DOES extract
            # fine could otherwise reproduce the same problem.
            remaining = base_interval + jitter
            while remaining > 0:
                chunk = min(config.hot_topics.fast_poll_seconds, remaining)
                await asyncio.sleep(chunk)
                remaining -= chunk
                if remaining <= 0:
                    break
                since_last_publish = (
                    time.monotonic() - last_publish_monotonic
                    if last_publish_monotonic is not None
                    else config.dynamic_publish.min_interval_seconds
                )
                if since_last_publish < config.dynamic_publish.min_interval_seconds:
                    continue
                if await has_unpublished_hot_candidate(config):
                    logger.info("run_cycle: unpublished hot-flagged candidate detected — triggering cycle early")
                    break
    finally:
        await posted_store.close()
        await event_store.close()
        await caption_cache.close()
        await dup_strikes.close()
        await hub_index.close()
        await appeal_labels.close()


if __name__ == "__main__":
    asyncio.run(main())
