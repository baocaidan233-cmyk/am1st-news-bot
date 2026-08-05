from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx
import trafilatura

from core.alerts import AlertNotifier
from core.config import AppConfig
from core.models import RssSource

logger = logging.getLogger(__name__)

# Same headers as agents/rss_fetcher.py's FEED_HEADERS — a bare httpx client
# gets blocked by some sites' basic bot filters.
FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _find_cookie_source(url: str, sources: list[RssSource]) -> RssSource | None:
    """Matches the article's domain against each configured source's own
    `domain` field — not every source has a paywall cookie, so most matches
    will have an empty cookie, which is fine (extraction is just attempted
    without one)."""
    netloc = urlparse(url).netloc
    for source in sources:
        if source.domain and source.domain in netloc:
            return source
    return None


class Extractor:
    """Full-text extraction — self-contained, no external service.
    Confirmed 2026-08-03: the previously-used internal extract-premium
    service (n8n-svr.gettr.fyi) is unmaintained and broken for most of the
    paywalled sources it was meant to help with (domain not on its
    allowlist, a crashed browser driver, an encoding bug), while offering no
    real advantage over a plain fetch for ordinary sites. This does the same
    job directly: httpx GET (with the matched source's cookie, if any) +
    trafilatura for main-content extraction — confirmed working even for
    two of the eight paywalled sources the old service couldn't handle
    (Epoch Times, SCMP). Sites gated behind real bot-detection/JS challenges
    (NYT, FT, Economist, Bloomberg) aren't solved by this — see the
    Playwright discussion in project memory for that harder tier, not
    attempted here.

    Bug③ fix carried over from the original n8n workflow: a failed
    extraction never silently drops the item — the caller falls back to
    the RSS description, and a Notion @mention alert fires on the matched
    source's row, so a cookie expiry actually gets noticed.

    Called from the publish cycle only (2026-08-05 — moved out of
    ingestion): extracting full text for every ingestion-time candidate
    that merely passed the cheap title+description score gate wasted real
    work on articles that would very likely never be published before
    aging out of the 12h candidate-pool window. Now only the handful of
    candidates actually selected into a publish cycle's batch (up to 5,
    every 30 min) pay this cost — see project_am1st_migration memory.
    """

    def __init__(self, config: AppConfig, alerts: AlertNotifier) -> None:
        self._config = config
        self._alerts = alerts

    async def _fail(self, url: str, source: RssSource | None, reason: str) -> None:
        logger.warning("Extractor: %s for %s", reason, url)
        if source:
            await self._alerts.alert(source.page_id, f"全文抓取失败(可能是cookie失效/反爬拦截): {url}")

    async def extract(self, url: str, sources: list[RssSource]) -> str | None:
        """Returns the extracted main-content text, or None if extraction
        failed — caller decides the fallback (the RSS description)."""
        source = _find_cookie_source(url, sources)
        extraction = self._config.extraction

        headers = dict(FETCH_HEADERS)
        if source and source.cookie:
            headers["cookie"] = source.cookie

        try:
            async with httpx.AsyncClient(timeout=extraction.timeout_seconds, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            await self._fail(url, source, f"fetch failed ({e})")
            return None

        # trafilatura's parsing is CPU-bound, synchronous — offload so it
        # doesn't block the event loop while other candidates are in flight.
        text = await asyncio.to_thread(trafilatura.extract, html)

        if not text or len(text) < extraction.min_text_length:
            await self._fail(url, source, f"extracted only {len(text or '')} chars")
            return None

        return text
