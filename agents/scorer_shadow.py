"""Shadow-mode v2 scorer. Observes only.

Nothing in this module can change what AM1ST publishes. main.py calls it after
the production Scorer has already decided, inside its own try/except, and
discards whatever it returns except for one log line. Deleting this file
changes production output by exactly zero bytes.

Why it exists: prompts/scoring_prompt.txt is a 21-theme whitelist plus event-
severity bands. Measured 2026-10-01, that costs real stories — 66 candidates on
the Lindsay Clancy retrial, the most-rejected single story in the whole log,
all scored 3.0-4.0 against a 5.0 floor and none were published, because an
ordinary murder case is not on the theme list. A whitelist that misses a
category loses it permanently; a blacklist that misses one costs a single extra
call. prompts/scoring_prompt_v2.txt replaces the list with four judgments and
lets the code compute the score.

What was measured before writing this (on 677 of our own published posts with
12h fixed-age, hour-normalised engagement total, and again on 900 posts from
four peer MAGA news channels, channel-normalised — a corpus nothing here was
fitted on):

    g non-empty      1.23x [1.15, 1.33]   (open-ended; the 11-item enumerated
                                           version of the same question was
                                           weaker, 1.12x — the list hurt)
    d = stopped      1.16x [1.07, 1.23]   ours   1.59x [1.44, 1.75]  peers
    who = ally       1.21x [1.13, 1.35]   ours   1.52x [1.25, 1.79]  peers
    who = opponent   1.06x [1.01, 1.14]   ours   1.28x [1.19, 1.38]  peers
    who = ordinary   1.05x [0.97, 1.17]   not significant on either corpus
    d = happening    1.01x [0.98, 1.10]   zero — scores nothing
    d = price        1.10x [0.81, 1.20]   NOT ESTABLISHED, n=24. The 0.5 it
                                          carries below is a placeholder; do not
                                          cite it as a measured weight.

Two further dimensions (how decisive the action was, and whether the story is
the first instalment or the Nth) each looked strong alone -- 1.43x and 1.30x on
the peer corpus -- and were REJECTED: inside a fixed (d, who) cell they add
nothing (1.10x on peers), the head-to-head with and without them is inside the
noise, and asking for them shifted the four existing fields on 7-11% of items.
They were measuring the same underlying thing. Do not re-add them without a
within-cell test.

Head-to-head on the peer corpus, top quartile vs bottom quartile:

                    as a gate            above its own gate     ties   levels
    v2 (this)       1.67x [1.49,1.90]    1.38x [1.22,1.62]      30%      6
    production      1.61x [1.42,1.80]    1.18x [1.02,1.37]      58%      5

So the gate is a wash and the whole gain is in ranking. Note also that the
production prompt almost never emits 8+: 7 of 3,049 peer posts. Its real range
is 2-6, and above its own floor it has two values.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

_D = {"stopped", "price", "happening", "none"}
_WHO = {"opponent", "ally", "ordinary", "none"}
_DROP = {"anti_admin", "trivia", "none"}

# Channel duty, not a fitted weight: the user's standing instruction
# (2026-10-01) is that China, Xi, Taiwan and CCP influence are part of what this
# channel exists to cover, and are not to be cut because their measured
# engagement is low — AM1ST's China topic measured 0.83x and stays regardless.
# Implemented as a FLOOR so it can only ever raise a score. A keyword this
# misses costs that story a lower score, never a rejection, which is the whole
# reason it is not a gate.
_CHINA = re.compile(
    r"\b(china|chinese|ccp|xi jinping|beijing|taiwan|taipei|hong kong|huawei)\b", re.I)
_CHINA_FLOOR = 5.5


def compute_score(fields: dict, text: str = "") -> float:
    """Weights fitted on our own published corpus, direction confirmed on the
    peer corpus. See the module docstring for every number."""
    if fields["drop"] != "none":
        return 0.0
    s = 4.0
    if fields["g"] != "none":
        s += 1.5
    if fields["d"] == "stopped":
        s += 1.0
    elif fields["d"] == "price":
        s += 0.5          # placeholder, n=24, CI crosses 1.0
    if fields["who"] == "ally":
        s += 1.0
    elif fields["who"] in ("opponent", "ordinary"):
        s += 0.5
    if _CHINA.search(text or ""):
        s = max(s, _CHINA_FLOOR)
    return min(s, 9.0)


def _normalise(d) -> dict | None:
    """The model answers 'none' in whichever field has nothing in it, so every
    field's empty value is 'none' and the two that spell it differently are
    mapped back. Anything outside the closed sets is a parse failure, not a
    guess."""
    if not isinstance(d, dict):
        return None
    direction = (d.get("d") or "none").strip().lower()
    if direction == "neither":
        direction = "none"
    who = (d.get("who") or "none").strip().lower()
    grievance = (d.get("g") or "none").strip()
    if grievance.lower() in ("none", "n/a", ""):
        grievance = "none"
    drop = (d.get("drop") or "none").strip().lower()
    if direction not in _D or who not in _WHO or drop not in _DROP:
        return None
    return {"g": grievance, "d": direction, "who": who, "drop": drop}


class ShadowScorer:
    """One extra gpt-4o-mini call per scored candidate, on title + description.

    1,416 prompt tokens against the production prompt's 4,118, and over the
    1,024-token minimum that makes a prompt prefix cacheable at all — a
    1,024-token floor the first draft of this prompt sat under, at 833, which
    would have cost MORE per call than the longer production prompt it replaces.

    Fails open everywhere. Any error, any malformed answer, any field outside
    its closed set returns None and logs nothing but a debug line."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._prompt = Path(config.openai.shadow_scoring_prompt_file).read_text(encoding="utf-8")
        self._log = Path(config.openai.shadow_scoring_log_path)

    async def observe(self, candidate: Candidate, production_score: float) -> None:
        text = f"{candidate.title}\n\n{candidate.description or ''}".strip()[:900]
        if not text:
            return
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=90,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": text},
                ],
            )
            fields = _normalise(json.loads(resp.choices[0].message.content or "{}"))
        except Exception as e:
            logger.debug("ShadowScorer: %s — %s", type(e).__name__, candidate.url)
            return
        if fields is None:
            logger.debug("ShadowScorer: unparseable answer for %s", candidate.url)
            return
        row = {
            "ts": int(time.time()),
            "url": candidate.url,
            "title": (candidate.title or "")[:200],
            "production_score": production_score,
            "shadow_score": compute_score(fields, text),
            **fields,
        }
        try:
            self._log.parent.mkdir(parents=True, exist_ok=True)
            with self._log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("ShadowScorer: could not write the shadow log")
