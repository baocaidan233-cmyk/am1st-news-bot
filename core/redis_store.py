from __future__ import annotations

import logging

import redis.asyncio as redis

from core.config import AppConfig

logger = logging.getLogger(__name__)


class RedisStore:
    """Exact-duplicate dedup — URL hash only, per-channel key namespace
    (config.redis.key_prefix), 10-day TTL. See standing dedup architecture:
    this is the cheapest layer and runs first, before any embedding/scoring
    cost is spent on a candidate."""

    def __init__(self, config: AppConfig) -> None:
        self._prefix = config.redis.key_prefix
        self._ttl = config.redis.ttl_seconds
        # socket_timeout/socket_connect_timeout default to None (no bound at
        # all) if unset — a stalled connection to Upstash would hang claim_new()
        # forever, freezing the whole ingestion cycle since Layer 1 runs this
        # sequentially over every candidate. Found during a 2026-08-06 code
        # review specifically for "what could hang the whole program."
        self._client = (
            redis.from_url(config.redis.url, decode_responses=True, socket_timeout=10, socket_connect_timeout=10)
            if config.redis.url
            else None
        )

    async def claim_new(self, url_hash: str) -> bool:
        """Atomically checks-and-sets. Returns True if this url_hash hadn't
        been seen in the last ttl_seconds (and is now marked seen), False if
        it's a repeat. Missing REDIS_URL fails open (treats everything as new)
        rather than silently blocking the whole pipeline on a config gap.
        A timed-out connection also fails open (same reasoning) rather than
        raising and aborting the whole cycle over one transient network blip."""
        if self._client is None:
            logger.warning("RedisStore: REDIS_URL not set — dedup disabled, treating all items as new")
            return True
        key = self._prefix + url_hash
        try:
            # SET ... NX returns True only if the key didn't already exist.
            return bool(await self._client.set(key, "1", ex=self._ttl, nx=True))
        except Exception:
            logger.exception("RedisStore: claim_new failed for %s — treating as new (fail open)", url_hash)
            return True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
