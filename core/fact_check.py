"""Pre-publish fact check (2026-10-13, Linear AI-74).

Claude Haiku reads the article and the caption and labels each factual claim
in the caption SUPPORTED / CONTRADICTED / NOT_FOUND / UNCLEAR against the
article only. The writer's own characterisation ("part of Beijing's push",
"another bloc ally") is house style and is not checked -- user, 10-13: "只管
文章内事实".

Measured offline on 250 posts already checked by hand (AI-51): blocking on
FAIL only, Haiku caught 25-27 of 39 posts with a real factual error and
flagged 12-15 of 197 the hand check had passed, about half of which turned out
to be real errors too. 4o-mini and 5.4-mini, on the same posts, caught 9-11.
A FAIL goes back to the writer once with these reasons (main_publish.py): 12
of 13 wrong posts came back right, the 13th failed again and was not sent.

Fails open: no key, a timeout, an API error or an unreadable answer returns
verdict "ERROR", which the caller treats like PASS. Every call is logged to
logs/fact_check.jsonl."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from core.config import AppConfig

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims", "verdict"],
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "label", "reason"],
            "properties": {"text": {"type": "string"},
                           "label": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "NOT_FOUND", "UNCLEAR"]},
                           "reason": {"type": "string"}}}},
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "UNSURE"]},
    },
}


class FactChecker:
    def __init__(self, config: AppConfig) -> None:
        fc = config.fact_check
        self.enabled = fc.enabled and bool(fc.api_key)
        self._model = fc.model
        self._effort = fc.effort
        self._system = Path(fc.prompt_file).read_text(encoding="utf-8") if self.enabled else ""
        self._client = (anthropic.AsyncAnthropic(api_key=fc.api_key, timeout=fc.timeout_seconds, max_retries=1)
                        if self.enabled else None)
        if fc.enabled and not fc.api_key:
            logger.warning("FactChecker: ANTHROPIC_API_KEY not set -- fact check off")

    async def check(self, title: str, article: str, post: str, url: str = "", stage: str = "draft") -> dict:
        """{"verdict": PASS|FAIL|UNSURE|ERROR|OFF, "problems": [ "claim: reason", ...]}"""
        if not self.enabled:
            return {"verdict": "OFF", "problems": []}
        user = (f"TODAY: {datetime.now(timezone.utc).date().isoformat()}\n\nARTICLE TITLE: {title}"
                f"\n\nARTICLE:\n{article}\n\nPOST:\n{post}")
        verdict, problems, claims, usage = "ERROR", [], [], None
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=16000, system=self._system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self._effort, "format": {"type": "json_schema", "schema": _SCHEMA}})
            usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
            if resp.stop_reason == "refusal":
                logger.warning("FactChecker: refusal for %s -- letting it through", url)
            else:
                data = json.loads(next(b.text for b in resp.content if b.type == "text"))
                verdict, claims = data["verdict"], data["claims"]
                problems = [f"\"{c['text']}\": {c['reason']}" for c in claims
                            if c["label"] in ("CONTRADICTED", "NOT_FOUND")]
        except Exception as e:  # fails open: a broken check never stops a post
            logger.warning("FactChecker: %s for %s -- letting it through", type(e).__name__, url)
        self._log(url, stage, post, verdict, claims, usage)
        return {"verdict": verdict, "problems": problems}

    @staticmethod
    def _log(url, stage, post, verdict, claims, usage) -> None:
        try:
            row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url": url, "stage": stage,
                   "verdict": verdict, "post": post, "claims": claims, "usage": usage}
            with open("logs/fact_check.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass
