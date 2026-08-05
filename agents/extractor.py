from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx
import trafilatura

from core.alerts import AlertNotifier
from core.config import AppConfig
from core.models import Candidate, RssSource

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
    extraction never silently drops the item — it falls back to the RSS
    description and fires a Notion @mention alert on the matched source's
    row, so a cookie expiry actually gets noticed.
    """

    def __init__(self, config: AppConfig, alerts: AlertNotifier) -> None:
        self._config = config
        self._alerts = alerts

    async def _fail(self, candidate: Candidate, source: RssSource | None, reason: str) -> None:
        logger.warning("Extractor: %s for %s", reason, candidate.url)
        candidate.extraction_failed = True
        candidate.article = candidate.description
        if source:
            await self._alerts.alert(
                source.page_id,
                f"全文抓取失败(可能是cookie失效/反爬拦截): {candidate.url}",
            )

    async def extract(self, candidate: Candidate, sources: list[RssSource]) -> None:
        source = _find_cookie_source(candidate.url, sources)
        extraction = self._config.extraction

        headers = dict(FETCH_HEADERS)
        if source and source.cookie:
            headers["cookie"] = source.cookie

        try:
            async with httpx.AsyncClient(timeout=extraction.timeout_seconds, follow_redirects=True) as client:
                resp = await client.get(candidate.url, headers=headers)
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            await self._fail(candidate, source, f"fetch failed ({e})")
            return

        # trafilatura's parsing is CPU-bound, synchronous — offload so it
        # doesn't block the event loop while other candidates are in flight.
        text = await asyncio.to_thread(trafilatura.extract, html)

        if not text or len(text) < extraction.min_text_length:
            await self._fail(candidate, source, f"extracted only {len(text or '')} chars")
            return

        candidate.article = text
