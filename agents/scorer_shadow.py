"""Shadow scorer. Observes only.

Nothing in this module can change what AM1ST publishes. main.py calls it after
the production Scorer has already decided, inside its own try/except, and
discards everything it returns except one log line. Deleting this file changes
production output by exactly zero bytes.

2026-10-02 — THE TWO SCORERS SWAPPED PLACES. Until today this ran the
whitelist-free v2 prompt in shadow while prompts/scoring_prompt.txt decided
production. v2 won the comparison and now decides production
(agents/scorer.py), and the retired 21-theme prompt runs here.

Why keep paying for the retired prompt at all, instead of just deleting it:
the thing we cannot see from a cutover alone is what the NEW prompt throws
away. v2 rejects 27% of what we were publishing. That 27% measured 0.86x on
engagement, which is why the cutover was made, but a median is not a promise
about any single story, and the one failure mode that actually costs this
channel is missing a story that becomes a hot topic. With the old prompt
running here, `tools/shadow_report.py` keeps printing the disagreements in both
directions, so a story v2 kills and the old prompt wanted is visible the same
day instead of never.

This is a few days of running, not a permanent fixture. Turn it off with
shadow_scoring_enabled: false once the reverse direction has been reviewed;
that costs one gpt-4o-mini call per candidate and nothing else.

Note the asymmetry in what gets logged. The retired prompt is given the FULL
input it was written for — title, description, heat_score,
hours_since_event_first_seen and the trending headlines — because the point of
keeping it is to see its best judgment, not a crippled version of it. The new
production scorer deliberately gets the title alone (see agents/scorer.py). So
the two columns in the log are not an input-matched comparison, and the
cutover was not decided on them; they are a watch list.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)


def legacy_user_message(candidate: Candidate, trending_headlines: list[str] | None) -> str:
    """Byte-for-byte the message the retired Scorer built for this prompt, kept
    here so the shadow sees exactly what it used to see. If you change this,
    the shadow stops being a record of what the old scorer would have done."""
    first_seen = candidate.event_first_seen_at or candidate.published_at
    hours_since_first_seen = round(
        (datetime.now(timezone.utc) - first_seen).total_seconds() / 3600, 1)
    trending_block = ""
    if trending_headlines:
        headlines = "\n".join(f"- {h}" for h in trending_headlines)
        trending_block = (
            f"\n\nCurrently trending in US news (Google News, for context only):\n{headlines}")
    return (
        f"Title: {candidate.title}\n\nDescription: {candidate.description}"
        f"\n\nCorroboration: heat_score={candidate.heat_score:.1f} (1.0 = only this one source"
        f" reporting it so far; higher means more outlets, weighted, are covering the same event),"
        f" hours_since_event_first_seen={hours_since_first_seen}"
        f"{trending_block}"
    )


class ShadowScorer:
    """One extra gpt-4o-mini call per scored candidate. Fails open everywhere:
    any error, any malformed answer, and it logs a debug line and returns."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._prompt = Path(config.openai.shadow_scoring_prompt_file).read_text(encoding="utf-8")
        self._log = Path(config.openai.shadow_scoring_log_path)

    async def observe(
        self,
        candidate: Candidate,
        production_score: float,
        production_comment: str = "",
        trending_headlines: list[str] | None = None,
    ) -> None:
        try:
            message = legacy_user_message(candidate, trending_headlines)
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.3,     # what the retired Scorer used
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": message},
                ],
            )
            answer = json.loads(resp.choices[0].message.content or "{}")
            legacy_score = answer.get("llm_score")
            if not isinstance(legacy_score, (int, float)):
                logger.debug("ShadowScorer: no llm_score in answer for %s", candidate.url)
                return
            legacy_comment = str(answer.get("llm_comment") or "")[:300]
        except Exception as e:
            logger.debug("ShadowScorer: %s — %s", type(e).__name__, candidate.url)
            return
        # Column names stay as they were so tools/shadow_report.py and the rows
        # already on disk keep their meaning: `production_score` is whatever
        # decided this cycle, `shadow_score` is whatever only watched. What
        # swapped on 2026-10-02 is which prompt sits behind each name.
        row = {
            "ts": int(time.time()),
            "url": candidate.url,
            "title": (candidate.title or "")[:200],
            "production_score": production_score,
            "production_comment": production_comment[:300],
            "shadow_score": float(legacy_score),
            "shadow_comment": legacy_comment,
            "scorers": "prod=v2_fields shadow=legacy_themes",
        }
        try:
            self._log.parent.mkdir(parents=True, exist_ok=True)
            with self._log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("ShadowScorer: could not write the shadow log")
