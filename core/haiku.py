"""Claude Haiku for the channel's judges, and what happens when it cannot answer.

The user (10-13): Haiku comes with a monthly credit that resets at the end of
the month, so a judge goes to Haiku wherever it measured better, and when the
credit runs out the channel must fall back by itself.

The user (10-16, replacing the 10-15 rule): whenever Haiku cannot answer --
out of credit, or broken (timeout, overload, server error -- each retried 3
times -- an unreadable answer, a refusal) -- the caller asks its OpenAI model
instead ("所有的haiku都是坏了，没钱了，就fallback到4o mini。这是生产持续性考虑").
An out-of-credit (or billing) error also puts every caller on OpenAI for an
hour without calling Haiku again (available() is False). Callers with no
OpenAI path (the first-report check) still take their own default.

ask() returns the parsed JSON answer, or None for any failure; a caller tells
the two apart with available()."""
from __future__ import annotations

import json
import logging
import re
import time

import anthropic
from core.claude_usage import log_usage

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
        _clients[api_key] = anthropic.AsyncAnthropic(api_key=api_key, timeout=60, max_retries=3)
    return _clients[api_key]


async def ask(api_key: str, system: str, user: str, schema: dict | None = None, cache: bool = False,
              cache_ttl: str = "5m") -> dict | None:
    """Haiku at its default thinking (the setting that measured best on every
    judge, 10-13). With a schema the answer is structured output; without one
    the first {...} in the reply is parsed. cache=True marks the system prompt
    for Haiku's prompt cache: worth it only for a prompt past ~1,000 tokens
    called several times within a few minutes (user 10-15: use the cache price
    where it reliably saves; under 512 tokens nothing is cached).
    cache_ttl="1h" (2026-10-23, user: 1小时缓存还是5分钟便宜?) for a prompt called
    every cycle: cycles run 10+ minutes apart, past the 5-minute entry, so each
    paid a fresh write (1.25x input). An hour's entry costs 2x to write once and
    every read (0.1x) restarts its clock. Priced on 10-20..10-23's call times."""
    if not api_key or not available():
        return None
    control = {"type": "ephemeral", "ttl": "1h"} if cache_ttl == "1h" else {"type": "ephemeral"}
    sys_block = [{"type": "text", "text": system, "cache_control": control}] if cache else system
    kw = dict(model=MODEL, max_tokens=16000, system=sys_block, messages=[{"role": "user", "content": user}])
    if schema:
        kw["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    try:
        resp = await _client(api_key).messages.create(**kw)
        log_usage(resp, f"haiku {(system or '')[:40]!r}")
    except Exception as e:
        note_error(e)
        logger.warning("Haiku: %s -- %s", type(e).__name__, "this call falls back" if not available() else "the judge takes its default")
        return None
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return json.loads(text) if schema else json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        logger.warning("Haiku: unreadable answer -- the judge takes its default")
        return None
