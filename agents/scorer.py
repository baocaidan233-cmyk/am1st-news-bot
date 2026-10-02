from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client
from core.scoring import compute_score, describe, normalise

logger = logging.getLogger(__name__)


class ScoreOutput(BaseModel):
    llm_score: float
    llm_comment: str


class Scorer:
    """AI relevancy scoring. Asks four closed-set judgments and computes the
    score in code — see core/scoring.py for every weight and the measurements
    behind it.

    2026-10-02: swapped onto prompts/scoring_prompt_v2.txt after 13 hours and
    734 candidates of shadow running. prompts/scoring_prompt.txt, the 21-theme
    whitelist this replaces, is NOT deleted — it now runs in shadow
    (agents/scorer_shadow.py), so for the next few days we can still see what
    the new prompt throws away and whether any of it became a story. That
    reversal is the only reason the old prompt file is still in the repo.

    Three things the shadow run settled, all on real production candidates:

      * Hard rejections. 64 `trivia` rejections in 13 hours, and the retired
        prompt would have published ZERO of them — pure redundancy, free.
        17 `anti_admin`, of which the retired prompt would have published 6.
        On those 6 the new judgment is right 4 times and wrong twice ("Obama
        Judge Rules President Trump Cannot Fire..." is our side being attacked
        BY a judge, not an attack on us; "ICE changes course again in tweaking
        traffic stop policies" is neutral reporting). The prompt already states
        the rule it is getting wrong, in those words, so this is not fixed by
        rewording it — adding stance rules to a prompt has been falsified
        repeatedly on these bots. Accepted deliberately: two bad rejections out
        of ~450 passing candidates a day, against the leak it closes. Seven
        posts flagged anti_admin went live on the channel in the 14 days before
        this shipped, which is the rule this project cares most about.
      * It is NOT fed the description. The shadow ran on title + description;
        both engagement validations (716 of our own published posts, 900 peer
        posts) ran on the title alone, so the title alone is what ships. Do not
        "improve" this by adding the description without re-measuring the
        ladder, because the grievance rate moves a lot with it.
      * heat_score, hours_since_event_first_seen and the trending headlines are
        no longer sent. The retired prompt's score bands referenced heat_score
        directly; this prompt has no band that uses it, and nothing was fitted
        with it in the input. agents/priority_ranker.py still uses both signals
        at ranking time, which is where they belong. The `trending_headlines`
        argument is kept so callers need no change, and is ignored.

    temperature is 0, not the retired prompt's 0.3: the measurements were taken
    at 0, and a closed-set judgment has nothing to gain from sampling."""

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
            kwargs["max_completion_tokens"] = 120
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 120
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def score(
        self, candidate: Candidate, trending_headlines: list[str] | None = None
    ) -> ScoreOutput | None:
        # trending_headlines is accepted and ignored — see the class docstring.
        text = (candidate.title or "").strip()[:900]
        if not text:
            logger.warning("Scorer: no title to score for %s", candidate.url)
            return None
        fields = None
        for attempt in (1, 2):
            try:
                raw = await self._call(text)
                fields = normalise(json.loads(raw))
            except json.JSONDecodeError as e:
                logger.warning("Scorer: unparseable JSON for %s (attempt %d): %s",
                               candidate.url, attempt, e)
                fields = None
            if fields is not None:
                break
            if attempt == 1:
                logger.warning("Scorer: answer outside the closed sets for %s, retrying once",
                               candidate.url)
        if fields is None:
            # Same failure branch the retired Scorer had: main.py logs it and
            # skips this one candidate. Measured at 1 in 500 on 500 real items.
            logger.error("Scorer: gave up on %s after retry", candidate.url)
            return None
        return ScoreOutput(llm_score=compute_score(fields, text), llm_comment=describe(fields))
