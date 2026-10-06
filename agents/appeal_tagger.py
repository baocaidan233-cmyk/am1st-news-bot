from __future__ import annotations

import json
import logging

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client
from core.scoring import normalise

logger = logging.getLogger(__name__)

# Does this story give a MAGA reader something to care about -- the way his side
# is wronged, or the way it is taking something back -- or nothing? One yes/no,
# used only to order candidates that share a score (agents/candidate_selector.py).
#
# The labeler is prompts/scoring_prompt_v2.txt, unchanged, on the title alone, at
# temperature 0. That exact setup is what was measured, so changing any of it
# means measuring again:
#   - our own 716 published posts (2026-10-05): within the same v2.3 score band,
#     appeal vs none ran at median engagement 62 vs 50 at 5, 70 vs 60 at 6,
#     75 vs 61 at 7; permutation p=0.0005 over the 5/6/7 bands.
#   - 1,042 posts of the MAGA big accounts (2026-10-06), same labeler: inside
#     the 5/6/7 bands gatewaypundit 0.99/1.12/1.23x vs 0.84/0.91/1.00x (p=0.001),
#     newsmax and the commentary accounts the same direction (p=0.0005). Still
#     holds with the 17 posts about the prompt's own example stories removed.
#
# Only the field "g" is read, and only as none / not none. It is free text --
# the prompt says "There is no list to pick from" and "answer none when unsure"
# -- which is the difference from the reader-want tagger removed in 7788e7c:
# that one picked the best fit from seventeen fixed wants, so it found one for
# nearly everything and could not order anything.
_PROMPT_PATH = "prompts/scoring_prompt_v2.txt"


class AppealTagger:
    """One gpt-4o-mini call per candidate that reaches the pool.

    Fails open: any error or unparseable answer returns None, and the selector
    treats None exactly like "no appeal" -- the candidate is ordered after the
    labelled-appeal ones in its own score band, never dropped."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = create_openai_client(config)
        with open(_PROMPT_PATH, encoding="utf-8") as fh:
            self._prompt = fh.read()

    async def tag(self, item: Candidate) -> bool | None:
        title = (item.title or "").strip()[:900]
        if not title:
            return None
        try:
            resp = await self._client.chat.completions.create(
                model=self._config.openai.chat_model,
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": title},
                ],
            )
            d = normalise(json.loads(resp.choices[0].message.content or "{}"))
        except Exception:
            logger.exception("AppealTagger: tagging failed for %s — continuing unlabelled", item.url)
            return None
        if not d:
            logger.warning("AppealTagger: unparseable answer for %s — continuing unlabelled", item.url)
            return None
        return d.get("g") not in ("none", "", None)
