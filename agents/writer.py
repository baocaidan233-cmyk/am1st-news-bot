from __future__ import annotations

import re
from pathlib import Path

from openai import AsyncOpenAI

from core.config import AppConfig

NO_COMMENT = "No comment"

# Strips markdown emphasis (*_`#) and trailing punctuation before comparing —
# a real published post (2026-08-06) revealed the model sometimes wraps its
# "decline to write this" signal in markdown ("**No comment**"), which an
# exact-string match doesn't recognize as the same thing. That post then
# proceeded through ranking/dedup/publish as if it were real content — see
# project_am1st_migration memory's 2026-08-06 note for the full incident.
_MARKDOWN_RE = re.compile(r"[*_`#]+")


class Writer:
    """Content generation — same prompt as the original n8n workflow's 'Gen
    Gettr Post' node, ported verbatim (prompts/content_gen_prompt.txt).
    Deliberately a separate LLM call from Scorer, not merged into one
    request — Scoring decides "is this worth reporting", this decides "how
    to write it"; keeping them separate matches the original's own division
    of labor (see am1st_pipeline.html #p5).

    Called from the publish cycle only (2026-08-05 — moved out of
    ingestion, same reasoning as agents/extractor.py's docstring): takes
    plain title/article text rather than a specific Candidate type so it
    works for whichever model the caller has on hand."""

    def __init__(self, config: AppConfig) -> None:
        self._client = AsyncOpenAI(api_key=config.openai.api_key)
        self._model = config.openai.chat_model
        self._system_prompt = Path(config.openai.content_gen_prompt_file).read_text(encoding="utf-8")

    async def write(self, title: str, article: str) -> str:
        user_message = f"Title:  {title}\n\nArticle: {article}"
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def is_no_comment(text: str) -> bool:
        cleaned = _MARKDOWN_RE.sub("", text).strip().rstrip(".!").strip().lower()
        return cleaned == NO_COMMENT.lower()
