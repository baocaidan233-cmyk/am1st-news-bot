from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from core.config import AppConfig
from core.notion_sources import NOTION_VERSION

logger = logging.getLogger(__name__)


def _plain_title(prop: dict) -> str:
    return "".join(t.get("plain_text", "") for t in prop.get("title", []))


async def fetch_active_hot_topics(config: AppConfig) -> list[str]:
    """Returns the topic text of every currently-live manual hot-topic flag
    for this bot's own channel (config.hot_topics.channel_name) — see
    core/config.py's HotTopicsConfig docstring for the full design.

    "Live" means the shared Notion table's In_Use checkbox is checked AND
    the row's own last_edited_time is within hot_topics.ttl_hours
    (deliberately last_edited_time, not created_time — see that
    docstring), AND this bot's channel_name is one of the tags checked in
    the row's Channel multi-select column.

    Read-only, best-effort — returns [] (never raises) if the table isn't
    configured or the request fails, same fail-open convention as every
    other external read in this codebase. Called once per cycle by
    main.py; this module has no opinion on embeddings, the caller embeds
    the returned texts itself."""
    notion = config.notion
    key = notion.hot_topics_key
    if not key or not notion.hot_topics_db_id:
        return []

    props = notion.hot_topics_props
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=config.hot_topics.ttl_hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    headers = {
        "Authorization": f"Bearer {key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    query = {
        "filter": {
            "and": [
                {"property": props.active, "checkbox": {"equals": True}},
                {"timestamp": "last_edited_time", "last_edited_time": {"on_or_after": cutoff}},
            ]
        }
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"https://api.notion.com/v1/databases/{notion.hot_topics_db_id}/query", headers=headers, json=query,
            )
            resp.raise_for_status()
    except Exception:
        logger.exception("fetch_active_hot_topics: request failed")
        return []

    topics = []
    for page in resp.json().get("results", []):
        page_props = page.get("properties", {})
        channels = {opt.get("name") for opt in page_props.get(props.channel, {}).get("multi_select", [])}
        if config.hot_topics.channel_name not in channels:
            continue
        text = _plain_title(page_props.get(props.name, {}))
        if text:
            topics.append(text)
    return topics
