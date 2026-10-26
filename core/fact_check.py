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
A FAIL gets one correction with these reasons (main_publish.py), written by
Haiku itself since 2026-10-15 (FactChecker.fix; it was the writer model
before), then a second check; a second FAIL drops the candidate.

Haiku broken -> gpt-4o-mini (2026-10-16, AI-89, user: 坏了fallback到4o mini):
when the Haiku call raises, times out, is refused or (for the check) gives an
unreadable answer, the same prompt and message go once to OpenAI's
config.openai.chat_model. A usable answer that is rejected (an empty,
unchanged or note-like correction) is not a failure and does not fall back.
Only when both fail does the check return verdict "ERROR", which the caller
treats like PASS, and the fix return None (the caller drops). No Anthropic
key: the check is off ("OFF"). No OpenAI key: no fallback. Every check is
logged to logs/fact_check.jsonl with the model that answered."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from core.config import AppConfig
from core.openai_client import create_openai_client
from core.claude_usage import log_usage

logger = logging.getLogger(__name__)

# Words a post never has and a note about the post always has: Haiku answering
# the checker instead of correcting (DailyNews, offline 10-15: "ADAspace is not
# stated as the launcher; the article says China launched 10 satellites.").
_NOT_A_POST = re.compile(r"\b(?:the|this) (?:article|post)\b|\bnot stated\b|\b[SP]\d{1,2}\b"
                         # Its own working pasted in (news channels, offline 10-15:
                         # "Wait, I must return the corrected post only. Here it is: ...").
                         r"|\bcorrected post\b|(?:^|\n)\s*Wait\b|\bhere it is\b|\bI (?:should|must|need to) (?:return|output)\b"
                         # Its correction narrated (published 10-19: "Correction: Homeland
                         # Security Investigations agents made the arrests. Let me correct
                         # the first sentence instead."). 0 other hits in 632 published posts.
                         r"|\bCorrection:|\bLet me (?:correct|fix|rewrite|revise|redo)\b",
                         re.I)


def _new_note(text: str, post: str):
    """The first note-like phrase in text that the post did not already carry.
    China Breaks drafts write "The article says ..." themselves; a correction
    keeping that phrase is not Haiku commenting (10-16: two good corrections
    dropped for it)."""
    had = post.lower()
    return next((m for m in _NOT_A_POST.finditer(text) if m.group(0).lower() not in had), None)

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


def _parse(text: str) -> tuple[str, list, list[str]]:
    """(verdict, claims, problems) from either model's JSON answer; raises on
    anything that is not that shape, which the caller treats as a failure."""
    data = json.loads(text)
    verdict, claims = data["verdict"], data["claims"]
    if verdict not in ("PASS", "FAIL", "UNSURE") or not isinstance(claims, list):
        raise ValueError(f"unexpected answer {verdict!r}")
    problems = [f"\"{c['text']}\": {c['reason']}" for c in claims
                if c["label"] in ("CONTRADICTED", "NOT_FOUND")]
    return verdict, claims, problems


class FactChecker:
    def __init__(self, config: AppConfig) -> None:
        fc = config.fact_check
        self.enabled = fc.enabled and bool(fc.api_key)
        self._model = fc.model
        self._effort = fc.effort
        self._system = Path(fc.prompt_file).read_text(encoding="utf-8") if self.enabled else ""
        self._fix_system = (Path("prompts/fact_fix_prompt.txt").read_text(encoding="utf-8")
                            if self.enabled else "")
        self._client = (anthropic.AsyncAnthropic(api_key=fc.api_key, timeout=fc.timeout_seconds, max_retries=1)
                        if self.enabled else None)
        if fc.enabled and not fc.api_key:
            logger.warning("FactChecker: ANTHROPIC_API_KEY not set -- fact check off")
        # The gpt-4o-mini fallback (2026-10-16, AI-89). Built only when the
        # check is on and there is an OpenAI key; anything else means no
        # fallback, never a failed start.
        self._timeout = fc.timeout_seconds
        self._openai_model = config.openai.chat_model
        self._openai = None
        if self.enabled and config.openai.api_key:
            try:
                self._openai = create_openai_client(config)
            except Exception as e:
                logger.warning("FactChecker: no gpt-4o-mini fallback (%s)", type(e).__name__)

    async def check(self, title: str, article: str, post: str, url: str = "", stage: str = "draft") -> dict:
        """{"verdict": PASS|FAIL|UNSURE|ERROR|OFF, "problems": [ "claim: reason", ...]}"""
        if not self.enabled:
            return {"verdict": "OFF", "problems": []}
        user = (f"TODAY: {datetime.now(timezone.utc).date().isoformat()}\n\nARTICLE TITLE: {title}"
                f"\n\nARTICLE:\n{article}\n\nPOST:\n{post}")
        verdict, problems, claims, usage, model = "ERROR", [], [], None, None
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=16000, system=self._system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self._effort, "format": {"type": "json_schema", "schema": _SCHEMA}})
            usage = log_usage(resp, 'fact check')
            if resp.stop_reason == "refusal":
                raise RuntimeError("refusal")
            verdict, claims, problems = _parse(next(b.text for b in resp.content if b.type == "text"))
            model = self._model
        except Exception as e:
            logger.warning("FactChecker: Haiku %s for %s -- %s", type(e).__name__, url,
                           "asking gpt-4o-mini" if self._openai is not None else "no fallback, letting it through")
            if self._openai is not None:
                try:  # fails open: a broken check never stops a post
                    text, usage = await self._openai_answer(self._system, user, json_out=True)
                    verdict, claims, problems = _parse(text)
                    model = self._openai_model
                except Exception as e2:
                    logger.warning("FactChecker: fallback %s for %s -- letting it through", type(e2).__name__, url)
        self._log(url, stage, post, verdict, claims, usage, model)
        return {"verdict": verdict, "problems": problems}

    async def fix(self, title: str, article: str, post: str, problems: list[str]) -> str | None:
        """Claude Haiku's correction of a post this check failed (2026-10-15, user:
        四个新闻频道也应该让 haiku 重写 -- the writer model only writes the first
        draft), or None -- unchanged, a note instead of a post, or Haiku and
        its gpt-4o-mini fallback both failing (2026-10-16, AI-89) -- and the
        caller drops the candidate. The prompt is
        prompts/fact_fix_prompt.txt. Ported from DailyNews b671d57, where on 99
        live drafts the check failed Haiku's corrections passed 90 against 89
        for the writer model's rewrites."""
        if not self.enabled or not problems:
            return None
        user = (f"ARTICLE TITLE: {title}\n\nARTICLE:\n{article}\n\nPOST:\n{post}"
                "\n\nPROBLEMS:\n" + "\n".join(f"- {p}" for p in problems))
        text = None
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=4000, system=self._fix_system,
                messages=[{"role": "user", "content": user}], output_config={"effort": "low"})
            log_usage(resp, "fact check fix")
            if resp.stop_reason == "refusal":
                logger.warning("FactChecker: fix refused")
            else:
                text = next((b.text for b in resp.content if b.type == "text"), "")
        except Exception as e:
            logger.warning("FactChecker: fix failed (%s)", type(e).__name__)
        if text is None:
            # Haiku broken (2026-10-16, AI-89): the same correction once from gpt-4o-mini.
            if self._openai is None:
                return None
            try:
                text, _ = await self._openai_answer(self._fix_system, user, json_out=False)
                logger.info("FactChecker: fix written by %s (Haiku failed)", self._openai_model)
            except Exception as e:
                logger.warning("FactChecker: fallback fix failed (%s)", type(e).__name__)
                return None
        # Paragraphs kept, each one's whitespace tidied; "exports,." after a cut.
        text = "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip()) if p.strip())
        text = re.sub(r"\s*,\s*([.!?])", r"\1", text)
        note = _new_note(text, post)
        if not text or " ".join(text.split()) == " ".join(post.split()) or note:
            logger.warning("FactChecker: fix unusable (%s): %r",
                           "a note, not a post" if note else "empty or unchanged", text[:200])
            return None
        return text

    async def _openai_answer(self, system: str, user: str, json_out: bool) -> tuple[str, dict | None]:
        """One call to config.openai.chat_model with the same system prompt and
        user message (2026-10-16, AI-89). json_out: structured output with the
        check's _SCHEMA, strict (the schema meets OpenAI's strict rules)."""
        kwargs = dict(model=self._openai_model, timeout=self._timeout, messages=[
            {"role": "system", "content": system}, {"role": "user", "content": user}])
        if json_out:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "fact_check", "strict": True, "schema": _SCHEMA}}
        if self._openai_model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = 4000
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 4000
        resp = await self._openai.chat.completions.create(**kwargs)
        if getattr(resp.choices[0].message, "refusal", None):
            raise RuntimeError("refusal")
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
        return resp.choices[0].message.content or "", usage

    @staticmethod
    def _log(url, stage, post, verdict, claims, usage, model=None) -> None:
        try:
            row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "url": url, "stage": stage,
                   "model": model, "verdict": verdict, "post": post, "claims": claims, "usage": usage}
            with open("logs/fact_check.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass
