"""Multi-key fallback wrapper for the OpenAI SDK — added 2026-08-07 at the
user's request ("抓新闻和发布时，需要增加open ai key fallback机制，万一一个key
额度用完，可以换第二个续上"). Every agent that talks to OpenAI (Scorer,
Embedder, Writer, PriorityRanker) previously built its own bare AsyncOpenAI(
api_key=config.openai.api_key) — a single exhausted/rate-limited key would
stall every one of them at once. create_openai_client() is a drop-in
replacement: same .chat.completions.create / .embeddings.create surface,
but backed by config.openai.api_keys (primary + optional fallback), retrying
the next key on RateLimitError before giving up.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)


class FallbackOpenAI:
    """Tries each api key in order on RateLimitError (covers both plain
    429 rate-limiting and OpenAI's "insufficient_quota" — the SDK raises
    the same exception class for both). Only RateLimitError triggers a
    key switch; any other error (bad request, auth error on a genuinely
    malformed key, etc.) surfaces immediately rather than masking a real
    bug behind key-cycling.

    Once a non-primary key succeeds, it's promoted to the front so a
    long-running process doesn't keep re-trying an already-exhausted
    primary key on every single subsequent call for the rest of the day."""

    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError("FallbackOpenAI needs at least one api_key")
        self._clients = [AsyncOpenAI(api_key=key) for key in api_keys]
        self.chat = _ChatProxy(self)
        self.embeddings = _EmbeddingsProxy(self)

    async def _call(self, path: str, *args, **kwargs):
        last_exc: RateLimitError | None = None
        for i, client in enumerate(self._clients):
            method = client
            for part in path.split("."):
                method = getattr(method, part)
            try:
                result = await method(*args, **kwargs)
            except RateLimitError as e:
                last_exc = e
                logger.warning("FallbackOpenAI: key #%d hit RateLimitError (%s), trying next key", i + 1, e)
                continue
            if i > 0:
                logger.warning("FallbackOpenAI: key #%d succeeded, promoting it ahead of earlier key(s)", i + 1)
                self._clients.insert(0, self._clients.pop(i))
            return result
        raise last_exc


class _ChatProxy:
    def __init__(self, parent: FallbackOpenAI) -> None:
        self.completions = _CompletionsProxy(parent)


class _CompletionsProxy:
    def __init__(self, parent: FallbackOpenAI) -> None:
        self._parent = parent

    async def create(self, *args, **kwargs):
        return await self._parent._call("chat.completions.create", *args, **kwargs)


class _EmbeddingsProxy:
    def __init__(self, parent: FallbackOpenAI) -> None:
        self._parent = parent

    async def create(self, *args, **kwargs):
        return await self._parent._call("embeddings.create", *args, **kwargs)


def create_openai_client(config) -> FallbackOpenAI:
    return FallbackOpenAI(config.openai.api_keys)
