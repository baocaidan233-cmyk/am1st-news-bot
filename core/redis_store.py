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


class PostedDupStrikes:
    """Counts how many publish cycles in a row have confirmed one candidate a
    duplicate of already-posted content, so main_publish.py can retire it from
    the pool instead of paying a full extraction + Writer pass on it every
    cycle for the rest of its 24h eligibility window — see
    PublishConfig.posted_dedup_strikes_before_retire for the 2855-verdict
    measurement behind both the change and the threshold.

    Keyed by url_hash like CaptionCache above (not page_id: the same key
    lifetime and the same self-expiry reasoning apply, and the caller already
    has a url_hash in hand for the caption cache). Shares
    caption_ttl_seconds — both need exactly the same property, "outlives the
    candidate's eligibility window, then goes away on its own".

    The count is cumulative within the TTL rather than literally consecutive,
    which is the same thing in practice: the only way a candidate gets a
    "kept" verdict is to be the cycle's winner, and a winner is published and
    flagged sent, so it never comes back to be struck again.

    Fails open by returning 0 on any error or without REDIS_URL: a Redis blip
    must never retire a candidate, only ever fail to retire one."""

    def __init__(self, config: AppConfig) -> None:
        self._prefix = config.redis.dup_strike_prefix
        self._ttl = config.redis.caption_ttl_seconds
        self._client = (
            redis.from_url(config.redis.url, decode_responses=True, socket_timeout=10, socket_connect_timeout=10)
            if config.redis.url
            else None
        )

    async def strike(self, url_hash: str) -> int:
        """Records one duplicate verdict and returns this candidate's running
        total. The EXPIRE is refreshed on every strike, which is correct here:
        the window that matters is the candidate's, and it is re-entering the
        pool each time it gets struck."""
        if self._client is None or not url_hash:
            return 0
        try:
            key = self._prefix + url_hash
            count = await self._client.incr(key)
            await self._client.expire(key, self._ttl)
            return int(count)
        except Exception:
            logger.exception("PostedDupStrikes: strike failed for %s — returning 0, candidate stays in the pool (fail open)", url_hash)
            return 0

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class CaptionCache:
    """Caches Writer.write()'s generated caption by url_hash so the same
    still-unpublished candidate gets an identical post_content (and thus an
    identical embedding) every time it's reconsidered across publish cycles.
    See core/config.py's RedisConfig.caption_ttl_seconds comment for the real
    production bug this closes: without this cache, a re-generated caption's
    natural wording drift moved a real candidate's cosine score against
    posted history from 0.822 to 0.794 three minutes apart, flipping
    same_event()'s verdict and letting a duplicate through. Missing REDIS_URL
    or a transient error both fail open (treated as a cache miss — Writer
    just runs as before), never blocking content generation."""

    def __init__(self, config: AppConfig) -> None:
        self._prefix = config.redis.caption_prefix
        self._ttl = config.redis.caption_ttl_seconds
        self._client = (
            redis.from_url(config.redis.url, decode_responses=True, socket_timeout=10, socket_connect_timeout=10)
            if config.redis.url
            else None
        )

    async def get(self, url_hash: str) -> str | None:
        if self._client is None or not url_hash:
            return None
        try:
            return await self._client.get(self._prefix + url_hash)
        except Exception:
            logger.exception("CaptionCache: get failed for %s — treating as cache miss (fail open)", url_hash)
            return None

    async def set(self, url_hash: str, caption: str) -> None:
        if self._client is None or not url_hash:
            return
        try:
            await self._client.set(self._prefix + url_hash, caption, ex=self._ttl)
        except Exception:
            logger.exception("CaptionCache: set failed for %s — continuing without caching this caption", url_hash)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
