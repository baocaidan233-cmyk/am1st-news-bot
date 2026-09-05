from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from core.config import AppConfig
from core.models import Candidate, PublishCandidate
from core.notion_sources import NOTION_VERSION

logger = logging.getLogger(__name__)


def _rich_text(value: str) -> dict:
    # Notion's 2000-char limit is counted in UTF-16 code units (JS string
    # semantics), not Python's code-point-based len()/slicing — a single
    # astral character (some emoji, rare CJK) counts as 2 there but 1 here,
    # so a plain [:2000] slice can still get rejected as "2001". A 1900
    # margin absorbs that without needing to actually count UTF-16 units.
    return {"rich_text": [{"text": {"content": value[:1900]}}]}


async def write_candidate(config: AppConfig, item: Candidate) -> bool:
    """Writes one row to the shared candidate-pool database — called by the
    ingestion cycle (main.py) once an article survives scoring. Full-text
    extraction and content-gen no longer happen at ingestion time (moved to
    the publish cycle, 2026-08-05 — see agents/extractor.py's docstring),
    so props.content/props.post_content are left unset here; the publish
    cycle fills those in-memory for just the small batch it selects, and
    doesn't write them back to this row. send_status is also left unset
    (defaults to false); only the publish cycle ever sets it true, after
    actually posting."""
    notion = config.notion
    if not notion.candidate_key or not notion.candidate_db_id:
        logger.warning("write_candidate: NOTION_CANDIDATE_API_KEY / NOTION_CANDIDATE_DB_ID not set — skipping candidate write")
        return False

    props = notion.candidate_props
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    properties = {
        props.title: {"title": [{"text": {"content": item.title[:1900]}}]},
        props.url: {"url": item.url},
        props.author: _rich_text(item.source_name),
        props.description: _rich_text(item.description),
        props.published_at: {"date": {"start": item.published_at.isoformat()}},
        props.llm_score: {"number": item.llm_score},
        props.llm_comment: _rich_text(item.llm_comment),
        props.url_hash: _rich_text(item.url_hash),
        props.heat_score: {"number": item.heat_score},
        props.event_first_seen_at: {
            "date": {"start": (item.event_first_seen_at or item.published_at).isoformat()}
        },
        props.is_hot: {"checkbox": item.is_hot},
    }
    body = {"parent": {"database_id": notion.candidate_db_id}, "properties": properties}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://api.notion.com/v1/pages", headers=headers, json=body)
            resp.raise_for_status()
        return True
    except httpx.HTTPStatusError as e:
        # Log Notion's actual validation message, not just the terse
        # "400 Bad Request" from raise_for_status — otherwise a rare
        # data-dependent rejection (bad property value, oversized field,
        # odd Unicode from scraped content) is unreproducible after the
        # fact, as one was during 2026-08-05 testing.
        logger.error("write_candidate: Notion write failed for %s — %s", item.url, e.response.text[:500])
        return False
    except Exception:
        logger.exception("write_candidate: Notion write failed for %s", item.url)
        return False


def _plain_text(prop: dict) -> str:
    kind = prop.get("type")
    if kind in ("title", "rich_text"):
        return "".join(t.get("plain_text", "") for t in prop.get(kind, []))
    if kind == "url":
        return prop.get("url") or ""
    if kind == "number":
        return prop.get("number")
    if kind == "checkbox":
        return prop.get("checkbox", False)
    if kind == "date":
        d = prop.get("date")
        return d.get("start") if d else None
    if kind == "created_time":
        return prop.get("created_time")
    return ""


async def query_eligible_candidates(config: AppConfig) -> list[PublishCandidate]:
    """Queries the candidate pool for everything the publish cycle is
    allowed to consider: not yet sent, created within the eligibility
    window, and scored high enough at ingestion time. Sorted by llm_score
    descending, matching the original n8n query — agents/candidate_selector.py
    does the freshness/score tiering on top of this list.

    The original n8n design also filtered on a Notion formula column
    ("former check") for stale "former president Trump"-style phrasing —
    this Python rebuild does that same check as a plain keyword filter in
    agents/candidate_selector.py instead of a Notion formula column, so it
    is deliberately NOT part of this query."""
    notion = config.notion
    if not notion.candidate_key or not notion.candidate_db_id:
        logger.warning("query_eligible_candidates: NOTION_CANDIDATE_API_KEY / NOTION_CANDIDATE_DB_ID not set")
        return []

    props = notion.candidate_props
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=config.publish.candidate_max_age_hours)).isoformat()
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "filter": {
            "and": [
                {"property": props.send_status, "checkbox": {"does_not_equal": True}},
                {"timestamp": "created_time", "created_time": {"after": cutoff}},
                {"property": props.llm_score, "number": {"greater_than_or_equal_to": config.publish.candidate_min_score}},
            ]
        },
        "sorts": [{"property": props.llm_score, "direction": "descending"}],
    }

    rows: list[PublishCandidate] = []
    cursor: str | None = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            payload = dict(body)
            if cursor:
                payload["start_cursor"] = cursor
            try:
                resp = await client.post(
                    f"https://api.notion.com/v1/databases/{notion.candidate_db_id}/query",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
            except Exception:
                logger.exception("query_eligible_candidates: Notion query failed")
                break

            data = resp.json()
            for row in data.get("results", []):
                p = row.get("properties", {})
                try:
                    rows.append(
                        PublishCandidate(
                            page_id=row.get("id", ""),
                            title=_plain_text(p.get(props.title, {})),
                            url=_plain_text(p.get(props.url, {})),
                            author=_plain_text(p.get(props.author, {})),
                            description=_plain_text(p.get(props.description, {})),
                            content=_plain_text(p.get(props.content, {})),
                            post_content=_plain_text(p.get(props.post_content, {})),
                            llm_score=_plain_text(p.get(props.llm_score, {})) or 0.0,
                            llm_comment=_plain_text(p.get(props.llm_comment, {})),
                            url_hash=_plain_text(p.get(props.url_hash, {})),
                            published_at=_plain_text(p.get(props.published_at, {})) or row.get("created_time"),
                            created_at=row.get("created_time"),
                            heat_score=_plain_text(p.get(props.heat_score, {})) or 1.0,
                            event_first_seen_at=_plain_text(p.get(props.event_first_seen_at, {})),
                            is_hot=bool(_plain_text(p.get(props.is_hot, {}))),
                        )
                    )
                except Exception:
                    logger.exception("query_eligible_candidates: skipping malformed row %s", row.get("id"))

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    logger.info("query_eligible_candidates: %d eligible candidate(s)", len(rows))
    return rows


async def has_unpublished_hot_candidate(config: AppConfig) -> bool:
    """Cheap existence check (page_size=1, no pagination) — is there at
    least one manually-flagged-hot candidate (core/hot_topics.py) still
    unsent in the eligibility window? Used by main_publish.py's fast-poll
    loop (config.hot_topics.fast_poll_seconds) to decide whether to cut a
    wait short instead of waiting out the full publish.interval_seconds —
    see that module. Same filter shape as query_eligible_candidates() plus
    is_hot, deliberately NOT reusing that function directly: this runs far
    more often (every fast_poll_seconds vs every interval_seconds) and
    only needs a yes/no, not the full candidate list. Fails open (False)
    if the table isn't configured or the request fails."""
    notion = config.notion
    if not notion.candidate_key or not notion.candidate_db_id:
        return False

    props = notion.candidate_props
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=config.publish.candidate_max_age_hours)).isoformat()
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "page_size": 1,
        "filter": {
            "and": [
                {"property": props.send_status, "checkbox": {"does_not_equal": True}},
                {"property": props.is_hot, "checkbox": {"equals": True}},
                {"timestamp": "created_time", "created_time": {"after": cutoff}},
            ]
        },
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"https://api.notion.com/v1/databases/{notion.candidate_db_id}/query", headers=headers, json=body,
            )
            resp.raise_for_status()
        return bool(resp.json().get("results"))
    except Exception:
        logger.exception("has_unpublished_hot_candidate: Notion query failed")
        return False


async def count_recent_high_score(config: AppConfig, min_score: float, since_hours: float) -> int:
    """Cheap count-only Notion query (page_size capped at 100, no real
    pagination) — how many candidates were added to the pool in the last
    since_hours with llm_score >= min_score, regardless of send_status or
    publish.candidate_max_age_hours eligibility. Used by main_publish.py's
    dynamic publish-cadence scaling (config.dynamic_publish) as a
    real-time "how much strong material is ingestion actually producing
    right now" signal — deliberately NOT query_eligible_candidates(): that
    reflects the unpublished backlog, which shrinks every time something
    gets published and isn't a good proxy for current news volume. The
    100-item cap is a soft bound, not exact counting — dynamic_publish only
    needs a rough bucket (quiet/normal/busy), and the real 2026-09-05
    calibration sample (see DynamicPublishConfig's docstring) never
    exceeded 18 in a 2h window. Fails open (0, i.e. "quiet" — never speeds
    up on a failure) if the table isn't configured or the request fails."""
    notion = config.notion
    if not notion.candidate_key or not notion.candidate_db_id:
        return 0

    props = notion.candidate_props
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "page_size": 100,
        "filter": {
            "and": [
                {"property": props.llm_score, "number": {"greater_than_or_equal_to": min_score}},
                {"timestamp": "created_time", "created_time": {"after": cutoff}},
            ]
        },
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"https://api.notion.com/v1/databases/{notion.candidate_db_id}/query", headers=headers, json=body,
            )
            resp.raise_for_status()
        return len(resp.json().get("results", []))
    except Exception:
        logger.exception("count_recent_high_score: Notion query failed")
        return 0


async def mark_send_status(config: AppConfig, page_id: str) -> bool:
    """Flips send_status to true on the winning candidate — called by the
    publish cycle only after it actually posts to Gettr. While Gettr
    publishing is still deferred (see main_publish.py's _publish_stub),
    nothing calls this yet — a candidate must not be marked sent for
    something that was never actually posted."""
    notion = config.notion
    if not notion.candidate_key:
        logger.warning("mark_send_status: NOTION_CANDIDATE_API_KEY not set — skipping")
        return False

    props = notion.candidate_props
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {"properties": {props.send_status: {"checkbox": True}}}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=headers, json=body)
            resp.raise_for_status()
        return True
    except Exception:
        logger.exception("mark_send_status: Notion update failed for page %s", page_id)
        return False
