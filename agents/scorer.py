from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

# 2026-10-03 — restored. This file was replaced on 2026-10-02 by a four-judgment
# scorer (prompts/scoring_prompt_v2.txt, scored in core/scoring.py) and the
# channel owner reverted it the next day. Both the decision and the reason are
# recorded here so the same mistake is not repeated:
#
# v2 scored the channel's #1, #2 and #3 posts by engagement at 5.5, below the
# 6.0 weekday publishing floor — 344.5x, 229.4x and 214.3x hour-normalised
# against a channel median of 70. It read their grievances correctly ("they have
# divided loyalties and our country is at risk") and then gave all three
# who=none despite each having a named actor, because its `d` and `who` axes
# reward completed institutional action and are blind to someone NAMING a wrong.
# A procedural story scored higher: "Kansas, Roblox strike $10 million-plus
# deal to strengthen child safety" got 7.5 and realised 0.53x; "20 Alaska
# scholarship groups approved" got 6.0 with no grievance at all.
#
# The case for replacing this file was weaker than it was presented as. The
# whitelist argument was structural and still stands — 66 candidates on one
# ordinary murder retrial were refused, 44 at exactly 4.0, because the case was
# on none of the 21 Core Themes. But the evidence that v2 was BETTER was a
# quartile ratio, and a quartile ratio cannot see the tail where this channel's
# value is concentrated. Worse, this prompt puts 37% of its scores on the single
# value 6.0, so the bottom-quartile cut lands inside that tie block and the
# ratio moves with input order: re-measured on the same 716 posts it ranges
# 1.19x-1.24x across shuffles, against the single 1.16x figure that was reported.
# v2's 1.32x was a real but much smaller edge than claimed, bought by blocking
# the three best posts the channel has made.
#
# Do not replace this file again on an aggregate separation metric. The decision
# needs a fixed publish budget, total engagement as the primary number, and the
# high-engagement tail as a separate constraint that a change must not damage.


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
        # 2026-10-29 (AI-105, user: 上线): scorer_model first; a call that fails
        # outright is asked once of openai.fallback_model (gpt-6-luna, user
        # 10-29: 所有回退全部改成 luna). Blank scorer_model = chat_model, as before.
        self._scorer_model = config.openai.scorer_model or self._model
        self._scorer_effort = config.openai.scorer_reasoning_effort
        self._fallback_model = config.openai.fallback_model
        self._system_prompt = Path(config.openai.scoring_prompt_file).read_text(encoding="utf-8")
        self._focus_note = config.openai.scoring_focus_note.strip()
        self._focus_until = config.openai.scoring_focus_until.strip()
        self._standing_note = config.openai.scoring_standing_note.strip()
        self._calls_log = Path(config.openai.scorer_calls_log_path)
        self._prompt_sha = hashlib.sha256(self._system_prompt.encode("utf-8")).hexdigest()[:12]

    def _log_call(self, candidate: Candidate, user_message: str, result: ScoreOutput | None, retried: bool,
                  model: str = "") -> None:
        """One JSON line per scoring call (2026-10-27): the exact message and
        what came back. Best-effort: a logging failure never fails the score."""
        try:
            record = {"logged_at": int(time.time()), "url": candidate.url, "title": candidate.title,
                      "model": model or self._model, "prompt_sha": self._prompt_sha, "user_message": user_message,
                      "llm_score": result.llm_score if result else None,
                      "llm_comment": result.llm_comment if result else None, "retried": retried}
            self._calls_log.parent.mkdir(parents=True, exist_ok=True)
            with self._calls_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Scorer: failed to write %s -- continuing", self._calls_log)

    def _focus_block(self) -> str:
        """The editor's current focus (2026-10-22), in the user message rather
        than the 18k-character system prompt, until scoring_focus_until; then
        the standing note (2026-10-27), which has no end date."""
        notes = []
        if self._focus_note and not (self._focus_until and
                                     datetime.now(timezone.utc).date().isoformat() > self._focus_until):
            notes.append(self._focus_note)
        if self._standing_note:
            notes.append(self._standing_note)
        return "".join(f"\n\n{n}" for n in notes)

    async def _call(self, user_message: str, model: str) -> str:
        kwargs = dict(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        if model.startswith("gpt-6"):
            # Reasoning counts against this cap; at low effort the 10-29 test
            # (886 stories) used about 200 output tokens a story in all.
            kwargs["max_completion_tokens"] = 4000
            kwargs["reasoning_effort"] = self._scorer_effort
        elif model.startswith("gpt-5"):
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
            f"{self._focus_block()}"
        )
        model = self._scorer_model
        try:
            raw = await self._call(user_message, model)
        except Exception as e:
            logger.warning("Scorer: %s failed for %s (%s: %s) -- asking %s", model, candidate.url,
                           type(e).__name__, str(e)[:200], self._fallback_model)
            model = self._fallback_model
            raw = await self._call(user_message, model)
        result, retried = None, False
        try:
            result = ScoreOutput.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            retried = True
            logger.warning("Scorer: malformed output for %s, retrying once: %s", candidate.url, e)
            retry_message = (
                f"{user_message}\n\nYour previous response could not be parsed as "
                f'{{"llm_score": float, "llm_comment": string}}. Error: {e}. Return valid JSON only.'
            )
            raw_retry = await self._call(retry_message, model)
            try:
                result = ScoreOutput.model_validate(json.loads(raw_retry))
            except (json.JSONDecodeError, ValidationError):
                logger.error("Scorer: gave up on %s after retry", candidate.url)
        self._log_call(candidate, user_message, result, retried, model)
        return result
