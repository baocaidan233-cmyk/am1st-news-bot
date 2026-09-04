from __future__ import annotations

import asyncio
import logging
from calendar import timegm
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import feedparser
import httpx
from dateutil import parser as dateutil_parser

from core.config import AppConfig
from core.hashing import sha256_url_hash
from core.models import Candidate, RssSource
from core.render_client import render as render_service_render

logger = logging.getLogger(__name__)

# Confirmed 2026-09-04: these feeds 403 under a plain httpx GET even with
# FEED_HEADERS' real Chrome UA below, but a real Chromium render via the
# shared render service (core/render_client.py) gets through cleanly.
# Three other persistent 403s that day (courant.com, dailynews.com,
# baltimoresun.com — all Tribune Publishing) still 403'd even under a real
# browser render (a harder bot-detection vendor, not just a UA check) —
# deliberately not added here, not worth chasing further. keysnews.com
# 429'd (rate-limited, not blocked) — also not a fit for this list, needs
# a lower poll frequency instead, not a browser render.
_PLAYWRIGHT_FALLBACK_DOMAINS = (
    "army.mil",
    "newsmax.com",
    "state.gov",
    "justthenews.com",
)


def _domain_matches(netloc: str, domain: str) -> bool:
    netloc = netloc[4:] if netloc.startswith("www.") else netloc
    return netloc == domain or netloc.endswith("." + domain)


def _needs_playwright_fallback(url: str) -> bool:
    netloc = urlparse(url).netloc
    return any(_domain_matches(netloc, domain) for domain in _PLAYWRIGHT_FALLBACK_DOMAINS)

# Several sources 403'd under httpx's default "python-httpx/x.x" User-Agent
# (confirmed 2026-08-03 for e.g. Judicial Watch, The Federalist, State Dept,
# Washington Reporter, US Right to Know — all served fine with these headers
# instead). A plain scraper-looking UA is enough to get blocked by some
# sites' basic bot filters; this doesn't touch the harder cases (Tribune
# network sites, army.mil) that 403 even with this.
FEED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_published_utc(entry) -> datetime:
    """feedparser already normalizes *_parsed fields to a UTC struct_time
    regardless of the source feed's own timezone (rule 0.5 — one canonical
    UTC value, nothing stored per-timezone). Falls back to dateutil on the
    raw string, then to "now" if both fail — fail open, never drop an item
    just because its date didn't parse."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime.fromtimestamp(timegm(parsed), tz=timezone.utc)

    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            dt = dateutil_parser.parse(raw)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, OverflowError):
            logger.warning("rss_fetcher: unparseable date %r, keeping item with now() — fail open", raw)

    return datetime.now(timezone.utc)


async def fetch_source(client: httpx.AsyncClient, source: RssSource) -> list[Candidate]:
    try:
        resp = await client.get(source.feed_url)
        resp.raise_for_status()
        content = resp.content
    except Exception:
        if not _needs_playwright_fallback(source.feed_url):
            logger.exception("rss_fetcher: failed to fetch %s (%s)", source.name, source.feed_url)
            return []
        logger.info("rss_fetcher: plain fetch failed for %s, retrying via shared render service", source.name)
        # mode="raw" — the feed's actual response body, not Chromium's own
        # rendered page.content() (which wraps a raw XML response in an
        # HTML viewer shell, not parseable as RSS — confirmed live
        # 2026-09-04 on state.gov/justthenews.com).
        result = await render_service_render(source.feed_url, mode="raw")
        if result is None or result[0] is None or result[0] >= 400:
            logger.warning("rss_fetcher: render-service fallback also failed for %s (%s)", source.name, source.feed_url)
            return []
        content = result[1].encode("utf-8")

    parsed = feedparser.parse(content)
    candidates = []
    for entry in parsed.entries:
        url = entry.get("link", "")
        if not url:
            continue
        candidates.append(
            Candidate(
                url=url,
                url_hash=sha256_url_hash(url),
                source_name=source.name,
                title=entry.get("title", ""),
                description=entry.get("summary", ""),
                published_at=_parse_published_utc(entry),
            )
        )
    return candidates


async def fetch_all(config: AppConfig, sources: list[RssSource]) -> list[Candidate]:
    """Reads every in_use source once per cycle — no rotation (all feeds are
    small enough in number to read in full every 10-minute cycle)."""
    if not sources:
        return []

    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=FEED_HEADERS) as client:
        # All sources read concurrently, once per cycle — no rotation.
        per_source = await asyncio.gather(*(fetch_source(client, s) for s in sources))
    results = [c for batch in per_source for c in batch]

    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.max_publish_age_hours)
    fresh = [c for c in results if c.published_at >= cutoff]
    logger.info(
        "fetch_all: %d raw item(s) from %d source(s), %d within %dh publish-age window",
        len(results), len(sources), len(fresh), config.max_publish_age_hours,
    )
    return fresh
