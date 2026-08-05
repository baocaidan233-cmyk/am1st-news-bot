from __future__ import annotations

from datetime import datetime, timezone

from core.config import AppConfig
from core.models import PublishCandidate

# Same skip-list as the original n8n "check former"/"check former1" nodes,
# ported verbatim. Applied as a plain keyword filter here instead of a
# Notion formula column (the original also had a formula-column version of
# this same check gating the Notion query itself — this Python rebuild does
# it once, in code, right after querying eligible candidates).
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


def filter_former_trump(candidates: list[PublishCandidate]) -> list[PublishCandidate]:
    """Drops anything still referring to Trump as a former president —
    stale phrasing that slips through when an old article gets re-synced
    into a source feed."""
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
    already applied there)."""
    pub = config.publish
    now = datetime.now(timezone.utc)

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
        fill = [c for c in fresh if c.llm_score == pub.candidate_min_score and c.page_id not in picked_ids]
        batch.extend(fill[: pub.batch_max - len(batch)])

    if len(batch) < pub.batch_min:
        picked_ids = {c.page_id for c in batch}
        fill = [c for c in candidates if c.llm_score >= pub.candidate_min_score and c.page_id not in picked_ids]
        batch.extend(fill[: pub.batch_max - len(batch)])

    if len(batch) < pub.batch_min:
        # Last resort: newest overall, regardless of score.
        batch = sorted(candidates, key=hours_old)[: pub.batch_min]

    return batch[: pub.batch_max]
