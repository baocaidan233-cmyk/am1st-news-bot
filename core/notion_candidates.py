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
    # Only sent when the tagger actually produced a label — Notion rejects a
    # select whose name isn't already an option on the column, and omitting
    # the key entirely leaves the cell empty, which is exactly what "untagged"
    # should look like downstream.
    if item.topic:
        properties[props.topic] = {"select": {"name": item.topic}}
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
    if kind == "select":
        sel = prop.get("select")
        return sel.get("name") if sel else None
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
                {"property": props.extraction_failed, "checkbox": {"does_not_equal": True}},
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
                            topic=_plain_text(p.get(props.topic, {})) or None,
                            extraction_attempts=int(_plain_text(p.get(props.extraction_attempts, {})) or 0),
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
                {"property": props.extraction_failed, "checkbox": {"does_not_equal": True}},
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


async def mark_extraction_failed(config: AppConfig, page_id: str) -> bool:
    """Flips extraction_failed to true — called by main_publish.py's
    run_cycle() the moment full-text extraction fails for a candidate.
    2026-09-05, per the user's explicit request ("把这类读不了全文的文章，
    全部一次之后就放弃，不要一直循环"): a source that fails once (a hard
    paywall, a site that blocks the extractor) essentially never succeeds
    on a later retry, but without this flag the candidate stays eligible
    and gets re-selected into the batch (and re-billed for the same
    extraction attempt) every cycle until it ages out on its own — worse,
    if it also happens to be is_hot=true, it can keep re-triggering
    hot_topics.py's fast-poll early-cycle trigger indefinitely (a real
    2026-09-05 incident: 31 publishes in 2 hours instead of the expected
    ~4, because one permanently-unextractable hot-flagged candidate never
    stopped re-qualifying). query_eligible_candidates() excludes
    extraction_failed=true so this is a one-way, permanent exclusion, not
    a retry-later flag."""
    notion = config.notion
    if not notion.candidate_key:
        logger.warning("mark_extraction_failed: NOTION_CANDIDATE_API_KEY not set — skipping")
        return False

    props = notion.candidate_props
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {"properties": {props.extraction_failed: {"checkbox": True}}}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=headers, json=body)
            resp.raise_for_status()
        return True
    except Exception:
        logger.exception("mark_extraction_failed: Notion update failed for page %s", page_id)
        return False


async def mark_dedup_rejected(config: AppConfig, page_id: str) -> bool:
    """Third user of the same extraction_failed flag — called by
    main_publish.py once a candidate has been confirmed a duplicate of
    already-posted content in posted_dedup_strikes_before_retire consecutive
    cycles. Same problem shape as mark_writer_rejected() below and the same
    resolution: a candidate whose verdict does not change is re-selected
    every 15-45 minutes for its whole 24h window, and unlike the writer case
    this one is the most expensive kind of candidate in the batch, because
    the dedup check runs only AFTER extraction and content generation have
    already been paid for. 172 of the 184 repeatedly-judged candidates in the
    full posted_dedup log were judged identically every time, for 1539 wasted
    cycles — see PublishConfig.posted_dedup_strikes_before_retire.

    Deliberately NOT called on the first verdict; see that config field for
    why one re-roll is kept."""
    return await mark_extraction_failed(config, page_id)


async def mark_writer_rejected(config: AppConfig, page_id: str) -> bool:
    """Flips the same extraction_failed flag as mark_extraction_failed above
    — called by main_publish.py's run_cycle() when Writer.write() returns
    "No comment" for a candidate whose extraction actually succeeded (e.g.
    a source that turns out to be a roundup/digest once the full text is
    seen, or a rehash the RELEVANCY CHECK catches only with the full
    article). 2026-09-08, per the user's request: content_gen_prompt.txt
    is a static prompt, so the same extracted text gets the same "No
    comment" verdict every single time it's re-selected into a batch — a
    real 24h sample had one candidate (a thepiratescove.us roundup
    correctly declined every time) re-scored and re-written 8+ times,
    burning a fresh Writer call each cycle for a foregone conclusion.
    Reuses extraction_failed rather than adding a second Notion property,
    since both cases mean the same thing downstream to
    query_eligible_candidates(): permanently stop reconsidering this
    candidate."""
    return await mark_extraction_failed(config, page_id)


async def recent_published_topic_counts(config: AppConfig) -> dict[str, int]:
    """Counts what subjects this channel has actually published in the last
    topic_mix.window_hours — the "actual share" half of core/topic_mix.py's
    controller.

    Keyed on Notion's own last_edited_time rather than a publish timestamp,
    because there isn't one: mark_send_status() flipping send_status to true
    IS the publish event, and that write is the row's last edit. A row can in
    principle be edited later for another reason (a manual fix), which would
    keep it in the window slightly too long; the controller reads shares, not
    absolute counts, so a handful of stragglers shifts nothing.

    Fails open to {} on any error — core/topic_mix.py reads {} as "not enough
    to act on" and applies no adjustment at all, which is the pre-2026-09-25
    behaviour. Untagged rows are skipped rather than counted under a
    placeholder: they belong to no target, and folding them into one would
    distort every other subject's share."""
    notion = config.notion
    if not notion.candidate_key or not notion.candidate_db_id:
        return {}

    props = notion.candidate_props
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=config.topic_mix.window_hours)).isoformat()
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "filter": {
            "and": [
                {"property": props.send_status, "checkbox": {"equals": True}},
                {"timestamp": "last_edited_time", "last_edited_time": {"after": cutoff}},
            ]
        },
        "page_size": 100,
    }

    counts: dict[str, int] = {}
    cursor: str | None = None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                payload = dict(body)
                if cursor:
                    payload["start_cursor"] = cursor
                resp = await client.post(
                    f"https://api.notion.com/v1/databases/{notion.candidate_db_id}/query",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                for row in data.get("results", []):
                    topic = _plain_text(row.get("properties", {}).get(props.topic, {}))
                    if topic:
                        counts[topic] = counts.get(topic, 0) + 1
                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")
    except Exception:
        logger.exception("recent_published_topic_counts: Notion query failed — continuing with no mix adjustment")
        return {}

    return counts


async def record_extraction_failure(config: AppConfig, page_id: str, attempts_so_far: int) -> bool:
    """Counts one failed full-text extraction, and only gives up on the
    candidate once publish.extraction_max_attempts is reached.

    Replaces calling mark_extraction_failed() on the first failure. That rule
    came from the user on 2026-09-05 and its reasoning was sound for what
    prompted it -- a hard paywall never succeeds later, and one candidate that
    could never be extracted kept re-triggering the is_hot fast lane. But
    measured on 2026-09-26 it was excluding the wrong population: all 303
    flagged candidates were extraction failures rather than writer rejections,
    and re-running extraction on 14 of the flagged 7+ scorers recovered 8,
    including a score-8 story a peer channel took 217 likes on. Over five days
    the flag had permanently dropped 91 of 295 high scorers.

    Both original guarantees survive: the retry is bounded, so a real paywall
    still drops out after a fixed number of cycles, and a failing is_hot
    candidate still stops re-qualifying. Only the flag's timing changes.

    Fails open (returns False) like every other Notion write here; a failed
    write just means this attempt was not counted, and the candidate is
    reconsidered next cycle as it would have been anyway."""
    notion = config.notion
    if not notion.candidate_key:
        logger.warning("record_extraction_failure: NOTION_CANDIDATE_API_KEY not set — skipping")
        return False

    attempts = attempts_so_far + 1
    give_up = attempts >= config.publish.extraction_max_attempts
    props = notion.candidate_props
    properties = {props.extraction_attempts: {"number": attempts}}
    if give_up:
        properties[props.extraction_failed] = {"checkbox": True}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.patch(
                f"https://api.notion.com/v1/pages/{page_id}",
                headers={
                    "Authorization": f"Bearer {notion.candidate_key}",
                    "Notion-Version": NOTION_VERSION,
                    "Content-Type": "application/json",
                },
                json={"properties": properties},
            )
            resp.raise_for_status()
    except Exception:
        logger.exception("record_extraction_failure: Notion write failed for %s", page_id)
        return False

    if give_up:
        logger.info("record_extraction_failure: %s failed extraction %d times — excluding permanently", page_id, attempts)
    else:
        logger.info("record_extraction_failure: %s failed extraction %d/%d — will be reconsidered",
                    page_id, attempts, config.publish.extraction_max_attempts)
    return True
