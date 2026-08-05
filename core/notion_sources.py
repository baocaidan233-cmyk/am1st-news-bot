from __future__ import annotations

import logging

import httpx

from core.config import AppConfig
from core.models import RssSource

logger = logging.getLogger(__name__)

NOTION_VERSION = "2022-06-28"


def _plain_text(prop: dict) -> str:
    """Reads a Notion property value regardless of its type — the source
    table's exact column types (title vs rich_text vs url vs select) aren't
    pinned down yet, so this stays permissive rather than assuming one."""
    kind = prop.get("type")
    if kind in ("title", "rich_text"):
        return "".join(t.get("plain_text", "") for t in prop.get(kind, []))
    if kind == "url":
        return prop.get("url") or ""
    if kind == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    if kind == "checkbox":
        return prop.get("checkbox", False)
    return ""


async def load_rss_sources(config: AppConfig) -> list[RssSource]:
    """Queries the Notion source database fresh every call — no caching, so
    editing/adding a feed row (or flipping in_use) takes effect on the very
    next run_cycle, same reload behavior as the sibling bots' channels.yaml.
    """
    notion = config.notion
    if not notion.api_key or not notion.source_db_id:
        logger.warning("load_rss_sources: NOTION_API_KEY / NOTION_SOURCE_DB_ID not set — no sources loaded")
        return []

    props = notion.source_props
    headers = {
        "Authorization": f"Bearer {notion.api_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {"filter": {"property": props.in_use, "checkbox": {"equals": True}}}

    sources: list[RssSource] = []
    cursor: str | None = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            payload = dict(body)
            if cursor:
                payload["start_cursor"] = cursor
            try:
                resp = await client.post(
                    f"https://api.notion.com/v1/databases/{notion.source_db_id}/query",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
            except Exception:
                logger.exception("load_rss_sources: Notion query failed")
                break

            data = resp.json()
            for row in data.get("results", []):
                p = row.get("properties", {})
                feed_url = _plain_text(p.get(props.feed_url, {}))
                if not feed_url:
                    continue
                sources.append(
                    RssSource(
                        page_id=row.get("id", ""),
                        name=_plain_text(p.get(props.name, {})) or feed_url,
                        feed_url=feed_url,
                        cookie=_plain_text(p.get(props.cookie, {})),
                        domain=_plain_text(p.get(props.domain, {})),
                    )
                )

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    logger.info("load_rss_sources: %d in_use source(s)", len(sources))
    return sources
