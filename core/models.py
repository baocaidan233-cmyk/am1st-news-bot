from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class RssSource(BaseModel):
    """One row of the Notion RSS source table."""

    page_id: str  # Notion page id of this row — used to post a targeted @mention alert comment
    name: str
    feed_url: str
    cookie: str = ""  # paid-subscription cookie for this source's domain, if any
    domain: str = ""  # domain the cookie applies to — matched against the article url at extraction time


class Candidate(BaseModel):
    """An RSS entry as fetched, carried through the whole pipeline. Fields
    fill in as the item survives each stage; nothing here is Optional-typed
    away just because an earlier stage hasn't run yet — a field is simply
    empty/zero until its stage sets it."""

    url: str
    url_hash: str = ""
    source_name: str
    title: str
    description: str = ""
    published_at: datetime  # normalized to UTC immediately after parsing

    llm_score: float = 0.0
    llm_comment: str = ""

    # Corroboration/heat scoring (2026-08-06) — filled in main.py's Layer 3,
    # from the same cross-cycle Qdrant query used for dedup, before scoring.
    # heat_score=1.0 (self alone, no corroborating source yet) and
    # event_first_seen_at=None (meaning "just this article's own
    # published_at, nothing earlier found") are the correct defaults for a
    # candidate that hasn't been through that query yet.
    heat_score: float = 1.0
    event_first_seen_at: Optional[datetime] = None

    # Manual breaking-news override (2026-08-31) — set in main.py from
    # EventStore.commit()'s returned hot_until vs now, see
    # core/hot_topics.py / core/config.py's HotTopicsConfig.
    is_hot: bool = False


class PublishCandidate(BaseModel):
    """One row read back from the shared candidate-pool Notion database, as
    seen by the separate publish cycle (main_publish.py) — distinct from
    Candidate, which is the ingestion side's in-flight working object.
    priority_score is 0 until agents/priority_ranker.py fills it in."""

    page_id: str
    title: str
    url: str
    author: str = ""
    description: str = ""
    content: str = ""
    post_content: str
    llm_score: float
    llm_comment: str = ""
    url_hash: str = ""
    published_at: datetime
    created_at: datetime  # Notion's own created_time — the 12h eligibility window is keyed on this, not published_at
    priority_score: float = 0.0

    # Same corroboration/heat fields as Candidate — written at ingestion
    # time, read back here so agents/priority_ranker.py can use
    # event_first_seen_at (when this underlying event was actually first
    # reported) instead of this article's own published_at.
    heat_score: float = 1.0
    event_first_seen_at: Optional[datetime] = None

    # Same manual breaking-news flag as Candidate, read back from Notion —
    # see core/hot_topics.py / core/config.py's HotTopicsConfig.
    # candidate_selector.select_batch() force-includes these regardless of
    # score tier; agents/priority_ranker.py always ranks them above
    # non-hot candidates.
    is_hot: bool = False

    gettr_post_id: Optional[str] = None
