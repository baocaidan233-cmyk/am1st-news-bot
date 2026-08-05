from __future__ import annotations

import logging

import feedparser
import httpx

logger = logging.getLogger(__name__)

# Google News' own "what's hot in US news right now" ranking — free, no
# API key, no registration. Used purely as read-only context for the
# publish cycle's priority re-rank (agents/priority_ranker.py): we only
# pull the headline titles, never ingest/extract/score/publish from this
# feed itself. This is deliberately a separate, much lighter-weight
# mechanism than the RSS sources in the Notion source table.
#
# NATION (not the "POLITICS" topic) — checked both live 2026-08-05:
# POLITICS pulled in a lot of unrelated world politics (Bangladesh,
# Nigeria, Pakistan); NATION was consistently US-domestic and heavily
# overlapped with this channel's actual themes (Senate primaries, Epstein
# probe fallout, Trump news) — a better match for "trending" here.
TRENDING_FEED_URL = "https://news.google.com/rss/headlines/section/topic/NATION?hl=en-US&gl=US&ceid=US:en"


async def fetch_trending_headlines(limit: int = 15) -> list[str]:
    """Returns up to `limit` current top headline titles from Google News'
    US politics section, or an empty list if the request fails — this is a
    context signal, not a required dependency, so a failure here should
    never block the publish cycle."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(TRENDING_FEED_URL)
            resp.raise_for_status()
    except Exception:
        logger.exception("fetch_trending_headlines: request failed")
        return []

    parsed = feedparser.parse(resp.content)
    return [entry.get("title", "") for entry in parsed.entries[:limit] if entry.get("title")]
