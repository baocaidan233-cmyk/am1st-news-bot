"""Second opinion before an event-store guard drops an article (2026-10-13,
Linear AI-69).

main.py's two guards ("already-published event", "stale event with nothing
new") drop on the subtype judge's RESTATEMENT / CORROBORATION. 220 drops from
10-12/13, labelled blind: 49 (22%) were wrong -- 26 carried a major
development by the user's 10-12 standard (Trump's own response, the other
side answering, a motive revealed), 23 were not the same event at all.

Claude Haiku now reads the event's article and the new one
(prompts/guard_judge_prompt.txt) and answers DIFFERENT_EVENT /
MAJOR_DEVELOPMENT / NO_NEW; the guard drops only on NO_NEW. Offline on those
220: 135 of the 170 right drops still dropped, 42 of the 49 wrong ones let
through (gpt-4o-mini: 100 and 33). It is asked only when a guard is about to
drop, so the subtype judge still sets heat as before.

No key, an API error or an unreadable answer returns "ERROR", which the guard
treats like NO_NEW: the drop goes ahead as it did before this existed."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic

from core import haiku
from core.config import AppConfig

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdict", "reason", "quote"],
    "properties": {"verdict": {"type": "string", "enum": ["DIFFERENT_EVENT", "MAJOR_DEVELOPMENT", "NO_NEW"]},
                   "reason": {"type": "string"}, "quote": {"type": "string"}},
}


class GuardJudge:
    def __init__(self, config: AppConfig) -> None:
        ev = config.entity_verifier
        self.enabled = ev.guard_judge_enabled and bool(ev.guard_judge_api_key)
        self._model = ev.guard_judge_model
        self._system = Path(ev.guard_judge_prompt_file).read_text(encoding="utf-8") if self.enabled else ""
        self._client = (anthropic.AsyncAnthropic(api_key=ev.guard_judge_api_key, timeout=60, max_retries=1)
                        if self.enabled else None)
        if ev.guard_judge_enabled and not ev.guard_judge_api_key:
            logger.warning("GuardJudge: FIRST_REPORT_ANTHROPIC_API_KEY not set -- guards drop on the subtype alone")

    async def judge(self, event_text: str, new_text: str) -> tuple[str, dict]:
        """(verdict, raw). verdict is DIFFERENT_EVENT, MAJOR_DEVELOPMENT, NO_NEW, OFF or ERROR."""
        if not self.enabled:
            return "OFF", {}
        if not haiku.available():  # out of credit: drop as before, without calling
            return "ERROR", {"error": "haiku out of credit"}
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=16000, system=self._system,
                messages=[{"role": "user", "content": f"TEXT A:\n{event_text}\n\nTEXT B:\n{new_text}"}],
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA}})
            if resp.stop_reason == "refusal":
                return "ERROR", {"refusal": True}
            data = json.loads(next(b.text for b in resp.content if b.type == "text"))
            data["usage"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
            return data["verdict"], data
        except Exception as e:
            haiku.note_error(e)
            logger.warning("GuardJudge: %s -- the guard drops as before", type(e).__name__)
            return "ERROR", {"error": type(e).__name__}


_judge: GuardJudge | None = None


def get_guard_judge(config: AppConfig) -> GuardJudge:
    global _judge
    if _judge is None:
        _judge = GuardJudge(config)
    return _judge
