from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.embedder import Embedder
from core.config import AppConfig
from core.hashing import cosine_similarity
from core.models import PublishCandidate

logger = logging.getLogger(__name__)

_LOG_PATH = Path("logs/priority_rank_decisions.jsonl")

# heat_score bands — deliberately the SAME cutoffs as prompts/scoring_prompt.txt's
# section 2 corroboration bullets (2.0/4.0/10.0), so a "hot" story is treated
# consistently at both the ingestion and publish stage.
_HEAT_BANDS = ((10.0, 3.0), (4.0, 2.0), (2.0, 1.0))

# Trending-overlap cosine thresholds — carried over from
# core/hot_topics.py's HotTopicsConfig.match_threshold (0.5), validated
# 2026-08-31 on real embedding calls (same embedding model): a genuine
# same-story follow-up scored 0.588-0.636, a same-broad-topic-different-
# event story scored 0.220, an unrelated story scored 0.134. 0.65 is a
# second, higher band for an unusually tight match, not independently
# validated — a reasonable starting point given the 0.588-0.636 range
# already observed for genuine matches.
_TRENDING_SIM_HIGH = 0.65
_TRENDING_SIM_LOW = 0.5

# Freshness decay coefficient — see class docstring for the reasoning
# (logarithmic, not linear: a candidate_max_age_hours=12h-old story loses
# ~3.85 points, an hour-old story loses ~1, an 18-minutes-old story loses
# ~0.14 — proportionate to the ~5-15 range llm_score+heat_bonus+
# trending_bonus produces, without dominating it).
_FRESHNESS_DECAY_K = 1.5


def _heat_bonus(heat_score: float) -> float:
    for floor, bonus in _HEAT_BANDS:
        if heat_score >= floor:
            return bonus
    return 0.0


def _log_decision(record: dict) -> None:
    """Same append-only JSONL convention as core/event_identity.py's
    log_decision(), kept separate (own file, own module) since this isn't
    an event-identity decision — it's the publish-time ranking breakdown,
    the thing the user asked to have fully recorded for a 2026-09-05
    review of this rewrite's real behavior."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        full = {"logged_at": int(time.time()), **record}
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(full, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("PriorityRanker: failed to log decision — continuing (fail open)")


def log_publish_outcome(batch_size: int, winner: PublishCandidate | None) -> None:
    """Called once per publish cycle, after find_publishable() resolves —
    links the per-candidate breakdowns above to what actually got
    published (or didn't), so a review can walk logs/priority_rank_decisions.jsonl
    and see both "how was this batch scored" and "which one won" without
    cross-referencing main_publish.log."""
    _log_decision({
        "check_type": "publish_outcome",
        "batch_size": batch_size,
        "winner_page_id": winner.page_id if winner else None,
        "winner_url": winner.url if winner else None,
        "winner_priority_score": winner.priority_score if winner else None,
    })


class PriorityRanker:
    """Formula-based priority ranking (2026-09-04 rewrite, replacing the
    previous LLM-based holistic 1-10 rank call).

    Why: a live audit found the LLM version was unstable in exactly the
    range that decides most publish cycles. Re-running the SAME batch
    through the SAME ranker 3x with nothing else changed produced
    different scores for the same candidate (e.g. 2.0/2.0/5.0,
    6.0/6.0/3.0 across 3 identical calls) — real, reproducible run-to-run
    noise, not just batch-composition sensitivity (which was ALSO
    independently confirmed: one specific candidate scored 0 in one batch
    and a stable 4 in a later batch with different competing candidates
    — meaning priority_score was only ever meaningful relative to
    whatever else happened to be in that specific LLM call, not a
    portable, independently-checkable number). One case (a heat_score=12.4,
    hours_since_update=1.3h story) consistently scored 0 across repeats,
    directly contradicting the prompt's own stated rules — not noise, a
    real miscalibration.

    Every input here is already a reliable, already-computed number:
    llm_score (ingestion-side editorial severity — this session's own
    redesigned scoring_prompt.txt), heat_score (corroboration), and
    hours_since_update (this candidate's own freshness, deliberately NOT
    hours_old/event age — a fresh update in a long-running story should
    be judged on its own freshness, not discounted for the event's total
    age; heat_score already captures "how big/enduring has this event
    been," a separate axis). The only thing that isn't already a number
    is "does this topic overlap a trending headline" — that reduces
    cleanly to an embedding cosine-similarity check, no LLM judgment
    needed. Removing the LLM call removes the only place noise could
    enter.

    is_hot still sorts ahead of priority_score unconditionally, same as
    before — a manually-flagged breaking candidate always wins the slot.

    Every scored candidate's full breakdown is appended to
    logs/priority_rank_decisions.jsonl (one line per candidate per
    cycle), plus one summary line per cycle (see log_publish_outcome())
    recording which candidate, if any, actually got published."""

    def __init__(self, config: AppConfig) -> None:
        self._embedder = Embedder(config)

    async def rank(self, batch: list[PublishCandidate], trending_headlines: list[str] | None = None) -> list[PublishCandidate]:
        if not batch:
            return []
        trending_headlines = trending_headlines or []
        trending_embeddings = [await self._embedder.embed(h) for h in trending_headlines if h]

        now = datetime.now(timezone.utc)
        scored: list[tuple[PublishCandidate, float]] = []
        for c in batch:
            hours_since_update = max(0.0, (now - c.published_at).total_seconds() / 3600)
            heat_bonus = _heat_bonus(c.heat_score)

            best_sim = 0.0
            trending_bonus = 0.0
            if trending_embeddings:
                cand_embedding = await self._embedder.embed((c.post_content or c.title)[:2000])
                best_sim = max(cosine_similarity(cand_embedding, e) for e in trending_embeddings)
                if best_sim >= _TRENDING_SIM_HIGH:
                    trending_bonus = 2.0
                elif best_sim >= _TRENDING_SIM_LOW:
                    trending_bonus = 1.0

            freshness_penalty = _FRESHNESS_DECAY_K * math.log(1 + hours_since_update)
            priority_score = c.llm_score + heat_bonus + trending_bonus - freshness_penalty

            _log_decision({
                "page_id": c.page_id,
                "url": c.url,
                "title": c.title,
                "llm_score": c.llm_score,
                "heat_score": c.heat_score,
                "heat_bonus": heat_bonus,
                "trending_max_similarity": round(best_sim, 4),
                "trending_bonus": trending_bonus,
                "hours_since_update": round(hours_since_update, 2),
                "freshness_penalty": round(freshness_penalty, 3),
                "priority_score": round(priority_score, 3),
                "is_hot": c.is_hot,
            })

            scored.append((c.model_copy(update={"priority_score": priority_score}), hours_since_update))

        scored.sort(key=lambda item: (item[0].is_hot, item[0].priority_score, -item[1]), reverse=True)
        return [c for c, _ in scored]
