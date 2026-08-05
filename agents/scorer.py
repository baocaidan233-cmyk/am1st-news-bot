from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from core.config import AppConfig
from core.models import Candidate

logger = logging.getLogger(__name__)


class ScoreOutput(BaseModel):
    llm_score: float
    llm_comment: str


class Scorer:
    """AI relevancy scoring — same prompt/role/theme list as the original
    AM1ST n8n workflow's Scoring node, ported verbatim (prompts/scoring_prompt.txt).
    gpt-4o-mini only, per standing model-tier rule — no secondary Gemini
    autofix model; a single retry with the parse error appended does the
    same job the original's autoFix/second-model fallback did."""

    def __init__(self, config: AppConfig) -> None:
        self._client = AsyncOpenAI(api_key=config.openai.api_key)
        self._model = config.openai.chat_model
        self._system_prompt = Path(config.openai.scoring_prompt_file).read_text(encoding="utf-8")

    async def _call(self, user_message: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content or ""

    async def score(self, candidate: Candidate) -> ScoreOutput | None:
        user_message = f"Title: {candidate.title}\n\nDescription: {candidate.description}"
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
