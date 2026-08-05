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
        self._client = redis.from_url(config.redis.url, decode_responses=True) if config.redis.url else None

    async def claim_new(self, url_hash: str) -> bool:
        """Atomically checks-and-sets. Returns True if this url_hash hadn't
        been seen in the last ttl_seconds (and is now marked seen), False if
        it's a repeat. Missing REDIS_URL fails open (treats everything as new)
        rather than silently blocking the whole pipeline on a config gap."""
        if self._client is None:
            logger.warning("RedisStore: REDIS_URL not set — dedup disabled, treating all items as new")
            return True
        key = self._prefix + url_hash
        # SET ... NX returns True only if the key didn't already exist.
        return bool(await self._client.set(key, "1", ex=self._ttl, nx=True))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
