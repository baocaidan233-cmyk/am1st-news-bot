from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from core.config import AppConfig
from core.models import PublishCandidate

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```json|```")
_SMART_QUOTES_RE = re.compile(r"[“”]")


class _RankEntry(BaseModel):
    id: str
    priority_score: float


class PriorityRanker:
    """Second, independent LLM pass on top of the batch agents/candidate_selector.py
    picked — same prompt/rubric as the original n8n "LLM Rank Stories" node
    (prompts/priority_rank_prompt.txt), ranking on post_content itself
    (not title+description, which is what the ingestion-side llm_score used).
    gpt-4o-mini, per standing model-tier rule."""

    def __init__(self, config: AppConfig) -> None:
        self._client = AsyncOpenAI(api_key=config.openai.api_key)
        self._model = config.openai.chat_model
        self._system_prompt = Path(config.publish.priority_rank_prompt_file).read_text(encoding="utf-8")

    async def _call(self, batch: list[PublishCandidate], trending_headlines: list[str]) -> str:
        now = datetime.now(timezone.utc)
        user_message = json.dumps(
            {
                "trending_headlines": trending_headlines,
                "stories": [
                    {
                        "id": c.page_id,
                        "post_content": c.post_content,
                        "hours_old": round((now - c.published_at).total_seconds() / 3600, 1),
                    }
                    for c in batch
                ],
            },
            ensure_ascii=False,
        )
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            max_tokens=500,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse(raw: str) -> list[_RankEntry]:
        cleaned = _SMART_QUOTES_RE.sub('"', _FENCE_RE.sub("", raw)).strip()
        data = json.loads(cleaned)
        return [_RankEntry.model_validate(item) for item in data]

    async def rank(self, batch: list[PublishCandidate], trending_headlines: list[str] | None = None) -> list[PublishCandidate]:
        """Sets priority_score on each item in `batch` (in place, on copies)
        and returns them sorted by priority_score descending, tie-broken by
        published_at descending. Falls back to the existing llm_score order
        (priority_score left at 0) if the LLM output can't be parsed even
        after one retry — never blocks the publish cycle on this step.

        `trending_headlines` is a read-only context snapshot (see
        agents/trending.py) — current top US-politics headlines from
        Google News, given to the model so it can judge whether a
        candidate overlaps with what's actively trending right now. An
        empty list (the default, or whatever agents/trending.py returns on
        a failed fetch) just means no trending context this cycle — never
        blocks ranking."""
        if not batch:
            return []
        trending_headlines = trending_headlines or []

        raw = await self._call(batch, trending_headlines)
        try:
            entries = self._parse(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.warning("PriorityRanker: malformed output, retrying once: %s", e)
            retry_raw = await self._call(batch, trending_headlines)
            try:
                entries = self._parse(retry_raw)
            except (json.JSONDecodeError, ValidationError, TypeError):
                logger.error("PriorityRanker: gave up after retry — falling back to llm_score order")
                return sorted(batch, key=lambda c: c.llm_score, reverse=True)

        scores = {e.id: e.priority_score for e in entries}
        ranked = [c.model_copy(update={"priority_score": scores.get(c.page_id, 0.0)}) for c in batch]
        ranked.sort(key=lambda c: (c.priority_score, c.published_at), reverse=True)
        return ranked
