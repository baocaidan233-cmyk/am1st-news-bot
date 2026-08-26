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
    On gpt-5-nano (config.openai.nano_model), not the chat_model Writer
    uses — switched 2026-08-14 since this call is pure numeric triage
    (never user-facing prose), and the ~2.4x real cost saving is worth
    taking here specifically (2026-08-26: PriorityRanker and EventVerifier
    moved onto this same nano_model for the same reason, see core/config.py).
    No secondary Gemini autofix model; a single retry with the parse error
    appended does the same job the original's autoFix/second-model fallback
    did.

    gpt-5-nano is a reasoning model — two real, empirically-found quirks
    that don't apply to gpt-4o-mini:
    - It rejects `temperature`/`max_tokens`; needs `max_completion_tokens`.
    - Without `reasoning_effort="minimal"`, it can burn the entire
      max_completion_tokens budget on invisible reasoning tokens (still
      billed at the output rate) and return EMPTY content — verified
      2026-08-14: with the default reasoning effort and this budget, the
      scoring call came back with content=None. `minimal` fixed it and
      also made the real per-call cost cheaper than gpt-4o-mini, not more
      expensive (default effort's hidden reasoning tokens made a "cheaper"
      model cost MORE per call)."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.nano_model
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

    async def score(self, candidate: Candidate) -> ScoreOutput | None:
        # Corroboration signal (2026-08-06 — see core/config.py's HeatConfig
        # and project_am1st_migration memory's 2026-08-05 design note):
        # heat_score/event_first_seen_at are set by main.py's Layer 3, from
        # the cross-cycle Qdrant query, before this is called. heat_score=1.0
        # means only this one source so far; event_first_seen_at falls back
        # to this article's own published_at when nothing earlier was found.
        first_seen = candidate.event_first_seen_at or candidate.published_at
        hours_since_first_seen = round((datetime.now(timezone.utc) - first_seen).total_seconds() / 3600, 1)
        user_message = (
            f"Title: {candidate.title}\n\nDescription: {candidate.description}"
            f"\n\nCorroboration: heat_score={candidate.heat_score:.1f} (1.0 = only this one source"
            f" reporting it so far; higher means more outlets, weighted, are covering the same event),"
            f" hours_since_event_first_seen={hours_since_first_seen}"
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
