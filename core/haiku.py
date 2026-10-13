"""Claude Haiku for AM1ST's judges, and what happens when it cannot answer
(2026-10-13).

The user: Haiku comes with a monthly credit that resets at the end of the
month, so a judge goes to Haiku wherever it measured better, and when the
credit runs out the channel must fall back by itself. ask() returns the parsed
JSON answer, or None for any failure; each caller then does what it did before
Haiku: the event judge and the caption stance check ask their old OpenAI model,
the guard judge drops as before, the first-report check clears the article.

An out-of-credit (or billing) error puts every caller on its fallback for an
hour without calling Haiku again; other errors fall back for that one call."""
from __future__ import annotations

import json
import logging
import re
import time

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-5-5"
_COOLDOWN_SECONDS = 3600
_cooldown_until = 0.0
_clients: dict[str, anthropic.AsyncAnthropic] = {}


def available() -> bool:
    return time.time() >= _cooldown_until


def note_error(e: Exception) -> None:
    """Called with any Anthropic error. Out of credit: everyone falls back for an hour."""
    global _cooldown_until
    msg = str(e).lower()
    if getattr(e, "status_code", None) in (400, 402, 403) and any(w in msg for w in ("credit", "billing", "balance")):
        if available():
            logger.warning("Haiku: out of credit (%s) -- every judge on its fallback for %d min", str(e)[:120], _COOLDOWN_SECONDS // 60)
        _cooldown_until = time.time() + _COOLDOWN_SECONDS


def _client(api_key: str) -> anthropic.AsyncAnthropic:
    if api_key not in _clients:
        _clients[api_key] = anthropic.AsyncAnthropic(api_key=api_key, timeout=60, max_retries=1)
    return _clients[api_key]


async def ask(api_key: str, system: str, user: str, schema: dict | None = None) -> dict | None:
    """Haiku at its default thinking (the setting that measured best on every
    judge, 10-13). With a schema the answer is structured output; without one
    the first {...} in the reply is parsed."""
    if not api_key or not available():
        return None
    kw = dict(model=MODEL, max_tokens=16000, system=system, messages=[{"role": "user", "content": user}])
    if schema:
        kw["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    try:
        resp = await _client(api_key).messages.create(**kw)
    except Exception as e:
        note_error(e)
        logger.warning("Haiku: %s -- this call falls back", type(e).__name__)
        return None
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return json.loads(text) if schema else json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        logger.warning("Haiku: unreadable answer -- this call falls back")
        return None
