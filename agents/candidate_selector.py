from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.config import AppConfig
from core.models import PublishCandidate

# Which timezone's calendar day decides "weekday vs weekend" — US/Eastern,
# since this is a US-audience channel and that's the standard reference for
# "the US news day," not UTC or wherever this process happens to run.
_DAY_TZ = ZoneInfo("America/New_York")

# Exact phrase list from the original n8n "check former" node (confirmed
# 2026-08-05 by reading its actual JS in v1.4_am1st_notion_to_gettr_auto
# posting.json — not a guess). The real system also had a second,
# independent check ("check former1") reading a Notion formula column's
# precomputed flag — that formula's own definition isn't in the export, so
# it isn't replicated here, but this phrase list is the confirmed one that
# runs in code either way. Checked against title+description+content+
# post_content combined, same as the original. Now serves double duty
# after 2026-08-05's extraction/content-gen move: still catches a stale
# re-synced article, and also the writer's own occasional hallucination
# (content_gen_prompt.txt says "always President Trump," but an LLM can
# still default to "former president" out of training-data habit) once
# post_content is checked here too.
_FORMER_TRUMP_PHRASES = (
    "former president trump",
    "former us president trump",
    "former u.s. president trump",
    "former president donald trump",
    "former us president donald trump",
    "former u.s. president donald trump",
)

# Independent of publish.candidate_min_score (the floor used by the Notion
# query) — the original n8n "batch of top 5" node hardcodes this tier
# boundary at 7 regardless of what the floor is set to.
_TIER1_MIN_SCORE = 7.0


def _is_weekday(now: datetime) -> bool:
    return now.astimezone(_DAY_TZ).weekday() < 5  # Mon=0 ... Sun=6


def filter_former_trump(candidates: list[PublishCandidate]) -> list[PublishCandidate]:
    """Drops anything calling Trump a former president — see module
    docstring for the exact phrase list and why this now runs twice
    (before and after content-gen)."""
    kept = []
    for c in candidates:
        combined = " ".join([c.title, c.description, c.content, c.post_content]).lower()
        if any(phrase in combined for phrase in _FORMER_TRUMP_PHRASES):
            continue
        kept.append(c)
    return kept


def select_batch(candidates: list[PublishCandidate], config: AppConfig) -> list[PublishCandidate]:
    """Tiered batch selection — same cascade as the original n8n "batch of
    top 5" node: prefer fresh+high-scoring, progressively relax until at
    least batch_min survive (or give up and just take the newest ones),
    capped at batch_max. `candidates` should already be former-Trump-filtered
    and come from the Notion eligibility query (send_status/age/score gate
    already applied there, at the lower of weekday/weekend_min_score so
    both are actually fetched).

    Weekday/weekend-aware floor (2026-08-05, user request): weekdays see
    much more real news volume, so prefer publish.weekday_min_score first
    and only relax to the lower publish.weekend_min_score if that doesn't
    fill batch_min. Weekends see much less volume, so go straight to
    weekend_min_score — being picky first would usually just waste a tier.

    2026-08-06: the score-based tiers (1-4) now keep cascading until
    batch_max is full, not just until batch_min is met — this used to stop
    as soon as 3 candidates were found, even though batch_max allows 10 and
    extraction/content-gen only runs on this small selected batch (cheap).
    A moderately-scored-but-currently-trending story could end up excluded
    from the batch entirely if 3+ higher-scoring fresh stories already
    existed that cycle — by the time the priority-ranker sees trending
    context (agents/priority_ranker.py), it's too late, that story was
    never in the running. Filling the full batch_max with every genuinely
    scored candidate (not just the top 3) gives the AI re-rank step, which
    DOES see trending_headlines, a real chance to surface it. Tier 5 (the
    pure "newest regardless of score" last resort) still gates on
    batch_min only — it exists for when there's nothing real to pick from
    at all, not to pad out a batch that already has legitimate candidates.

    2026-09-01 — hard, unconditional published_at ceiling added (user's
    explicit "iron rule": only same-day news, publish nothing rather than
    something stale). Confirmed live: a 3-day-old CNBC article that only
    entered the candidate pool THAT DAY (query_eligible_candidates()'s own
    eligibility window is keyed on Notion's created_time — when this bot
    first saw it — not the article's own published_at) sailed through
    tier 3 below, which only checks llm_score, not freshness at all, and
    got published as if it were breaking. Every tier below, including the
    hot-topic force-include and the batch_min last-resort fallback, now
    only ever draws from `candidates` after this filter — none of them can
    bypass it.

    2026-09-05 — this ceiling is now day-aware, same weekday/weekend split
    as the score floor above: weekday_max_age_hours (12h) keeps the
    original same-day rule; weekend_max_age_hours (24h, widened per the
    user's explicit request) reflects real lower weekend news volume, so a
    Friday-night story is still eligible through Saturday instead of the
    batch running under batch_min or empty. query_eligible_candidates()'s
    own Notion-query ceiling (config.publish.candidate_max_age_hours) is
    deliberately the WIDER of the two (24h) so weekend-eligible candidates
    are never excluded before reaching this actual day-aware filter — same
    "fetch the superset, then apply the real floor here" pattern as
    candidate_min_score/weekday_min_score/weekend_min_score above. On a
    genuinely slow news day this can still leave the batch under
    batch_min, even empty — that's the accepted trade-off, not a bug to
    work around."""
    pub = config.publish
    now = datetime.now(timezone.utc)
    is_weekday = _is_weekday(now)
    preferred_floor = pub.weekday_min_score if is_weekday else pub.weekend_min_score
    fallback_floor = pub.weekend_min_score
    max_age_hours = pub.weekday_max_age_hours if is_weekday else pub.weekend_max_age_hours

    def hours_old(c: PublishCandidate) -> float:
        return (now - c.published_at).total_seconds() / 3600

    candidates = [c for c in candidates if hours_old(c) <= max_age_hours]

    fresh = [c for c in candidates if hours_old(c) <= pub.fresh_hours]

    batch: list[PublishCandidate] = sorted(
        (c for c in fresh if c.llm_score >= _TIER1_MIN_SCORE),
        key=lambda c: c.llm_score,
        reverse=True,
    )

    if len(batch) < pub.batch_max:
        picked_ids = {c.page_id for c in batch}
        fill = sorted(
            (c for c in fresh if preferred_floor <= c.llm_score < _TIER1_MIN_SCORE and c.page_id not in picked_ids),
            key=lambda c: c.llm_score,
            reverse=True,
        )
        batch.extend(fill[: pub.batch_max - len(batch)])

    if len(batch) < pub.batch_max:
        picked_ids = {c.page_id for c in batch}
        fill = sorted(
            (c for c in candidates if c.llm_score >= preferred_floor and c.page_id not in picked_ids),
            key=lambda c: c.llm_score,
            reverse=True,
        )
        batch.extend(fill[: pub.batch_max - len(batch)])

    if is_weekday and len(batch) < pub.batch_max:
        # Weekday-only extra fallback: still room in the batch, so relax
        # down to the weekend's lower floor before giving up on score entirely.
        picked_ids = {c.page_id for c in batch}
        fill = sorted(
            (c for c in candidates if fallback_floor <= c.llm_score < preferred_floor and c.page_id not in picked_ids),
            key=lambda c: c.llm_score,
            reverse=True,
        )
        batch.extend(fill[: pub.batch_max - len(batch)])

    if len(batch) < pub.batch_min:
        # Last resort: newest overall, regardless of score.
        batch = sorted(candidates, key=hours_old)[: pub.batch_min]

    # Manual hot-topic force-include (2026-08-31, core/hot_topics.py) — a
    # candidate the user has explicitly flagged as breaking must never be
    # silently excluded from the batch just because its llm_score tier
    # didn't make the cut above; by the time agents/priority_ranker.py sees
    # it, it's too late (same reasoning as the 2026-08-06 tier-cascade
    # change above, applied to a stronger signal). Evicts the current
    # lowest-scored non-hot member if the batch is already full, rather
    # than growing past batch_max.
    picked_ids = {c.page_id for c in batch}
    missing_hot = [c for c in candidates if c.is_hot and c.page_id not in picked_ids]
    for c in missing_hot:
        if len(batch) < pub.batch_max:
            batch.append(c)
            continue
        evict_idx = min(
            (i for i, b in enumerate(batch) if not b.is_hot),
            key=lambda i: batch[i].llm_score,
            default=None,
        )
        if evict_idx is not None:
            batch[evict_idx] = c

    return batch[: pub.batch_max]
