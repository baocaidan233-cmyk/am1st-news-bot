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
    weekend_min_score — being picky first would usually just waste a tier."""
    pub = config.publish
    now = datetime.now(timezone.utc)
    is_weekday = _is_weekday(now)
    preferred_floor = pub.weekday_min_score if is_weekday else pub.weekend_min_score
    fallback_floor = pub.weekend_min_score

    def hours_old(c: PublishCandidate) -> float:
        return (now - c.published_at).total_seconds() / 3600

    fresh = [c for c in candidates if hours_old(c) <= pub.fresh_hours]

    batch: list[PublishCandidate] = sorted(
        (c for c in fresh if c.llm_score >= _TIER1_MIN_SCORE),
        key=lambda c: c.llm_score,
        reverse=True,
    )

    if len(batch) < pub.batch_min:
        picked_ids = {c.page_id for c in batch}
        fill = sorted(
            (c for c in fresh if preferred_floor <= c.llm_score < _TIER1_MIN_SCORE and c.page_id not in picked_ids),
            key=lambda c: c.llm_score,
            reverse=True,
        )
        batch.extend(fill[: pub.batch_max - len(batch)])

    if len(batch) < pub.batch_min:
        picked_ids = {c.page_id for c in batch}
        fill = sorted(
            (c for c in candidates if c.llm_score >= preferred_floor and c.page_id not in picked_ids),
            key=lambda c: c.llm_score,
            reverse=True,
        )
        batch.extend(fill[: pub.batch_max - len(batch)])

    if is_weekday and len(batch) < pub.batch_min:
        # Weekday-only extra fallback: not enough at the 6+ floor, so relax
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

    return batch[: pub.batch_max]
