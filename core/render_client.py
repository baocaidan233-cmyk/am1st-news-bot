"""Thin client for the shared headless-Chromium render service (2026-09-04)
— one Chromium process on this VM shared across all news bots that need a
real browser render, instead of each bot (and each of the 3 more channels
expected to land here) launching its own Playwright instance per call. See
that service's own docstring (deployed separately, not part of this repo —
cross-project shared infra) for the concurrency/scheduling rationale.

Both agents/extractor.py's paywall/bot-detection fallback and agents/
rss_fetcher.py's Playwright-fallback path for a handful of RSS feeds that
403 under a plain httpx fetch go through this client — neither imports
playwright directly anymore."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_RENDER_SERVICE_URL = "http://127.0.0.1:8811"


async def render(
    url: str,
    mode: str = "rendered",
    wait_ms: int = 2500,
    timeout_ms: int = 20000,
    cookie: str | None = None,
    extra_headers: dict | None = None,
) -> tuple[int | None, str] | None:
    """Returns (http_status, content) on success, None on any failure
    (service unreachable, timeout, render error) — fail open, same
    convention as every other best-effort network call in this codebase.
    `mode="raw"` returns the actual HTTP response body (for RSS/XML feeds
    — Chromium's own rendered page.content() wraps raw XML in an HTML
    viewer shell, not what feed parsing wants); `mode="rendered"` (default)
    returns the DOM after wait_ms, for real HTML pages needing JS to
    finish (paywall/bot-detection challenges resolving client-side)."""
    payload = {
        "url": url,
        "mode": mode,
        "wait_ms": wait_ms,
        "timeout_ms": timeout_ms,
    }
    if cookie:
        payload["cookie"] = cookie
    if extra_headers:
        payload["extra_headers"] = extra_headers

    try:
        async with httpx.AsyncClient(timeout=(timeout_ms / 1000) + 5) as client:
            resp = await client.post(f"{_RENDER_SERVICE_URL}/render", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.info("render_client: render service reported an error for %s: %s", url, data["error"])
                return None
            return data.get("status"), data.get("content", "")
    except Exception as e:
        logger.info("render_client: failed to reach render service for %s (%s)", url, e)
        return None
