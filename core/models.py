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

    article: str = ""  # full extracted text, or description as fallback
    extraction_failed: bool = False

    post_content: str = ""


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

    gettr_post_id: Optional[str] = None
