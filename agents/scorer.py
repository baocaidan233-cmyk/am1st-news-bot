from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)


class ScoreOutput(BaseModel):
    llm_score: float
    llm_comment: str


class Scorer:
    """AI relevancy scoring — same prompt/role/theme list as the original
    AM1ST n8n workflow's Scoring node, ported verbatim (prompts/scoring_prompt.txt).

    Moved BACK to config.openai.chat_model (gpt-4o-mini) on 2026-09-01,
    reverting the 2026-08-14 move to gpt-5-nano — a ~19-hour live test run
    (real RSS data, real Notion candidate pool, real Gettr test-account
    publishes) surfaced a judgment-quality regression the original switch's
    verification never checked: 2026-08-14 only confirmed gpt-5-nano
    returned well-formed, non-empty JSON, never whether its actual scores
    stayed editorially sound. Confirmed live, 2026-09-01, by re-scoring
    several REAL candidates gpt-5-nano had just passed at the 5.0 floor: a
    Rheinmetall drone story with zero US angle (Germany's own aviation
    authority certifying a German company's drone), a Karim Benzema soccer
    transfer, a Cuba retail-policy story, a Texas crane rescue — all
    scored >=5, each with reasoning that rationalized a tenuous "could
    plausibly relate if reframed" angle rather than applying
    prompts/scoring_prompt.txt's own explicit "sports, weather, celebrity
    gossip with no political dimension" rejection list. The prompt's "lean
    toward passing when thin" guidance (written for genuinely ambiguous
    cases) was apparently read by gpt-5-nano as blanket permission to
    always find SOME angle rather than firmly reject clearly off-theme
    content — a failure mode gpt-4o-mini did not previously exhibit in
    this role. 2026-09-01, same day: PriorityRanker and EventVerifier also
    moved back to chat_model — a live test found each of their own
    nano_model-era judgment calls unreliable too (see
    agents/priority_ranker.py and core/event_identity.py), so this was not
    a Scorer-only problem after all; nothing in this codebase runs on
    gpt-5-nano anymore.

    No secondary Gemini autofix model; a single retry with the parse error
    appended does the same job the original's autoFix/second-model fallback
    did."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._system_prompt = Path(config.openai.scoring_prompt_file).read_text(encoding="utf-8")

    async def _call(self, user_message: str) -> str:
        kwargs = dict(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        if self._model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = 500
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["temperature"] = 0.3
            kwargs["max_tokens"] = 500
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def score(self, candidate: Candidate, trending_headlines: list[str] | None = None) -> ScoreOutput | None:
        # Corroboration signal (2026-08-06 — see core/config.py's HeatConfig
        # and project_am1st_migration memory's 2026-08-05 design note):
        # heat_score/event_first_seen_at are set by main.py's Layer 3, from
        # the cross-cycle Qdrant query, before this is called. heat_score=1.0
        # means only this one source so far; event_first_seen_at falls back
        # to this article's own published_at when nothing earlier was found.
        first_seen = candidate.event_first_seen_at or candidate.published_at
        hours_since_first_seen = round((datetime.now(timezone.utc) - first_seen).total_seconds() / 3600, 1)
        # Trending headlines (2026-09-04) — same free Google News feed
        # agents/trending.py already supplies to main_publish.py's
        # priority_ranker, reused here so the Scorer has an EXTERNAL signal
        # of what's actually getting mainstream attention right now,
        # distinct from heat_score (which only reflects how many of AM1ST's
        # own RSS sources have corroborated THIS specific candidate's
        # underlying event). Optional/best-effort — an empty list (fetch
        # failure, or a caller that doesn't pass one) just omits the
        # section below, same fail-open convention as everywhere else.
        trending_block = ""
        if trending_headlines:
            headlines = "\n".join(f"- {h}" for h in trending_headlines)
            trending_block = f"\n\nCurrently trending in US news (Google News, for context only):\n{headlines}"
        user_message = (
            f"Title: {candidate.title}\n\nDescription: {candidate.description}"
            f"\n\nCorroboration: heat_score={candidate.heat_score:.1f} (1.0 = only this one source"
            f" reporting it so far; higher means more outlets, weighted, are covering the same event),"
            f" hours_since_event_first_seen={hours_since_first_seen}"
            f"{trending_block}"
        )
        raw = await self._call(user_message)
        try:
            return ScoreOutput.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning("Scorer: malformed output for %s, retrying once: %s", candidate.url, e)
            retry_message = (
                f"{user_message}\n\nYour previous response could not be parsed as "
                f'{{"llm_score": float, "llm_comment": string}}. Error: {e}. Return valid JSON only.'
            )
            raw_retry = await self._call(retry_message)
            try:
                return ScoreOutput.model_validate(json.loads(raw_retry))
            except (json.JSONDecodeError, ValidationError):
                logger.error("Scorer: gave up on %s after retry", candidate.url)
                return None
