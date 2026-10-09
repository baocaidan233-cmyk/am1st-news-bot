from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from agents.embedder import Embedder
from core.config import AppConfig
from core.event_identity import EventVerifier, HubIndex, entity_tokens, event_identity_text, has_date_conflict, log_decision, posted_dedup_rule_verdict
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
    hub_index: HubIndex,
    config: AppConfig,
    on_duplicate: Callable[[PublishCandidate], Awaitable[None]] | None = None,
) -> PublishCandidate | None:
    """Walks `ranked_batch` in priority order (highest first) and returns the
    first candidate that is NOT a near-duplicate of something this channel
    already posted in the last publish.posted_dedup_window_hours. Duplicates
    (and candidates whose dedup check itself failed — see Fallback below)
    are skipped (never causes the whole
    cycle to abort. Returns None only if every candidate in the batch was a
    duplicate or errored (or the batch is empty) — the correct outcome for
    an actual all-duplicate batch is "publish nothing this cycle", not a
    fallback that fakes freshness.

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
    the added cost is negligible.

    Rule-tier pre-filter (2026-09-05, same day, per the user's "先用算法，
    然后哪个阈值交给llm做判断" request): before asking the LLM, check
    has_date_conflict() (core/event_identity.py, already validated on 1384
    real historical event-identity pairs at 0.6% flag rate, 0 harmful) — if
    both sides name a single explicit "Month Day" date and they disagree,
    that's strong independent evidence of a different specific occurrence
    regardless of how similar the text otherwise reads, so skip the LLM
    call and treat as NOT duplicate outright. Deliberately NOT adding the
    mirror-image shortcut (skip the LLM and assume duplicate above some
    high cosine/entity-overlap cutoff) — real same-day data argues against
    it: the SCMP Witkoff/Kushner-to-Moscow story scored 0.738-0.747 cosine
    against an older Putin-meeting story across several cycles (all
    wrongly auto-flagged "duplicate" under the pre-LLM pure-cosine rule)
    before same_event() correctly called it DIFFERENT_EVENT at a similar
    0.731 — i.e. the exact cosine/entity-overlap range that would tempt a
    "confident duplicate" shortcut is also where a real false positive
    just happened. Revisit only once enough same_event()-adjudicated
    posted_dedup pairs accumulate to bucket-calibrate a genuinely safe
    floor, the same way no_overlap_llm_review_floor was calibrated —
    not before.

    Fallback (2026-09-05, same day): same_event() is a real OpenAI call
    with no fail-open wrapper of its own (core/openai_client.py's
    FallbackOpenAI only retries same-cause RateLimitError across keys —
    any other error, e.g. a timeout or a malformed response, propagates).
    Before this fallback, that exception would escape find_publishable()
    entirely and abort the WHOLE cycle via main_publish.py's run_cycle()
    exception handler — turning one transient API hiccup on candidate #1
    into zero publishes this cycle, even if candidates #2-#10 never
    needed an LLM call at all. Each candidate's whole dedup check
    (embedding, posted-history lookup, same_event()) is now wrapped in its
    own try/except: on any failure, that ONE candidate is skipped (logged
    as check_type=posted_dedup_error, not conflated with a real "duplicate"
    verdict) and the walk continues to the next-ranked candidate, so the
    cycle still very likely finds something to publish.

    Rule-tier bypass (2026-09-06, per the user's "cosine查重加入entity和
    行动的比对" request, after a real duplicate published when the ONE
    same_event() call on it came back wrong — see core/event_identity.py's
    posted_dedup_rule_verdict() docstring for the full incident and the
    394-pair validation behind this): once has_date_conflict() has cleared
    a cosine-flagged pair, entity+actor overlap (posted_dedup_rule_verdict())
    now gets first say. COMPATIBLE decides "duplicate" outright, no LLM
    call at all — deliberately trusting the deterministic rule over a
    single non-deterministic LLM call for the cases it's confident about.
    Only the residual AMBIGUOUS/NO_OVERLAP cases still go to same_event()
    as before — same LLM, same prompt, just asked less often.

    on_duplicate (2026-09-23) is awaited once per CONFIRMED duplicate verdict,
    in priority order, and is how main_publish.py retires a candidate that
    keeps coming back with the same verdict — see
    core/notion_candidates.py's mark_dedup_rejected(). It is a callback rather
    than a return value so this module keeps knowing nothing about Notion;
    it is not called for the Fallback path above, because a dedup check that
    itself errored is explicitly not a verdict. A failing callback must never
    cost this cycle its publish, so it is wrapped.

    2026-10-09, two changes ported from China Breaks (c127fee, 0ce2df3), owner:
    "把CB的查重先移植给几个频道". Published duplicates scanned the same day:
    about four stories posted two or three times in 3.8 days (a shipyard, a
    diesel order three times, Vance and Microsoft, the FBI memos).

    1. Every one of the top-5 posted matches above `threshold` is checked,
       most similar first, not just the best one; the first confirmed
       duplicate ends the walk. This channel's own threshold applies to every
       rank -- China Breaks' 0.75 floor for ranks 2-5 was measured on its own
       corpus and is not copied.

    2. When the judge, reading our two captions, says DIFFERENT, it is asked
       once more on the two source articles' title + lead
       (event_identity_text(); publish.posted_dedup_source_second_opinion).
       Captions carry the writer's own slips and different figures, and the
       judge reads "$3.7B" vs "$6.6B" as two shipyards. Only a "different" is
       re-asked. On 93 caption-judged "kept" pairs from this channel's log
       (labelled by Claude): 14 of 28 real duplicates caught, 9 of 52
       different stories killed. A posted point without stored source text
       (written before 2026-10-09 and not backfilled) is judged on captions
       only, as before.

    China Breaks' 0.60-0.80 gray zone is NOT ported: this channel never
    judged pairs below 0.70, so there is no labelled data to set a floor
    with."""
    threshold = config.publish.posted_dedup_threshold
    second_opinion = config.publish.posted_dedup_source_second_opinion

    for candidate in ranked_batch:
        comparisons: list[dict] = []
        deciding: dict | None = None
        top: dict | None = None
        try:
            candidate_content = content_for_embedding(candidate.post_content, candidate.url)
            candidate_source = event_identity_text(candidate.title, candidate.description)
            embedding = await embedder.embed(candidate_content)
            matches = await posted_store.similar_recent(embedding)
            top = matches[0] if matches else None
            candidate_entities = entity_tokens(candidate_content) if matches else set()

            for m in matches:
                similarity = m["score"]
                if not m["url"] or similarity <= threshold:
                    continue
                matched_content = content_for_embedding(m["content"], m["url"])
                if has_date_conflict(candidate_content, matched_content):
                    is_duplicate = False
                    same_event_raw = "RULE: has_date_conflict() — explicit conflicting dates, skipped LLM call"
                    resolved_by = "date_conflict_rule"
                else:
                    rule_verdict = await posted_dedup_rule_verdict(config, hub_index, candidate_content, matched_content, similarity)
                    if rule_verdict == "COMPATIBLE":
                        is_duplicate = True
                        same_event_raw = "RULE: posted_dedup_rule_verdict() — non-hub entity overlap, no actor conflict, skipped LLM call"
                        resolved_by = "entity_rule"
                    else:
                        is_duplicate, same_event_raw = await event_verifier.same_event(candidate_content, matched_content)
                        resolved_by = "llm"
                        if not is_duplicate and second_opinion and m["title"] and candidate.title:
                            second, second_raw = await event_verifier.same_event(
                                candidate_source, event_identity_text(m["title"], m["description"]))
                            same_event_raw = f"{same_event_raw}\n[source second opinion] {second_raw}"
                            if second:
                                is_duplicate = True
                                resolved_by = "llm_source_second_opinion"
                matched_entities = entity_tokens(matched_content)
                comparison = {
                    "matched_url": m["url"],
                    "cosine_score": similarity,
                    "resolved_by": resolved_by,
                    "same_event_raw": same_event_raw,
                    "verdict": "duplicate" if is_duplicate else "kept",
                    "matched_entities": sorted(matched_entities),
                    "entity_overlap": sorted(candidate_entities & matched_entities),
                    # the texts exactly as the judge saw them, so a verdict can be replayed offline
                    "candidate_text": candidate_content,
                    "matched_text": matched_content,
                    "candidate_source": candidate_source,
                    "matched_source": event_identity_text(m["title"], m["description"]) if m["title"] else "",
                }
                comparisons.append(comparison)
                if is_duplicate:
                    deciding = comparison
                    break
        except Exception:
            logger.exception(
                "find_publishable: dedup check failed for %s — skipping this candidate (not a confirmed verdict), trying next",
                candidate.url,
            )
            log_decision(config, {"check_type": "posted_dedup_error", "candidate_url": candidate.url})
            continue

        is_duplicate = deciding is not None
        if top is not None:
            # Top-level fields describe the comparison that decided the verdict
            # (the duplicate hit, else the first one asked, else the top match),
            # so older replay scripts keep reading the same shape; `comparisons`
            # holds every match that was actually asked about.
            shown = deciding or (comparisons[0] if comparisons else None)
            log_record = {
                "check_type": "posted_dedup",
                "candidate_url": candidate.url,
                "matched_url": shown["matched_url"] if shown else top["url"],
                "cosine_score": shown["cosine_score"] if shown else top["score"],
                "threshold": threshold,
                "cosine_flagged": shown is not None,
                "final_verdict": "duplicate" if is_duplicate else "kept",
                "matches_seen": len(matches),
                "comparisons_asked": len(comparisons),
            }
            if shown:
                log_record.update({
                    "resolved_by": shown["resolved_by"],
                    "same_event_raw": shown["same_event_raw"],
                    "candidate_entities": sorted(candidate_entities),
                    "matched_entities": shown["matched_entities"],
                    "entity_overlap": shown["entity_overlap"],
                    "candidate_text": shown["candidate_text"],
                    "matched_text": shown["matched_text"],
                    "comparisons": comparisons,
                })
            log_decision(config, log_record)

        if is_duplicate:
            logger.info(
                "find_publishable: %s dropped — confirmed duplicate of already-posted content by %s (cosine=%.3f > %.2f, match %d of %d, matched %s)",
                candidate.url,
                deciding["resolved_by"],
                deciding["cosine_score"],
                threshold,
                len(comparisons),
                len(matches),
                deciding["matched_url"],
            )
            if on_duplicate is not None:
                try:
                    await on_duplicate(candidate)
                except Exception:
                    logger.exception("find_publishable: on_duplicate callback failed for %s — continuing", candidate.url)
            continue
        if comparisons:
            logger.info(
                "find_publishable: %s flagged against %d posted match(es) but every one was judged DIFFERENT — not treating as duplicate (closest %s, cosine=%.3f)",
                candidate.url,
                len(comparisons),
                comparisons[0]["matched_url"],
                comparisons[0]["cosine_score"],
            )

        logger.info("find_publishable: %s selected (priority_score=%.1f)", candidate.url, candidate.priority_score)
        return candidate

    logger.info("find_publishable: all %d candidate(s) were duplicates — nothing to publish this cycle", len(ranked_batch))
    return None
