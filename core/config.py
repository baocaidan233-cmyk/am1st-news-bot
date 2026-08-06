from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class NotionSourceProps(BaseModel):
    """Defaults match the live "rss_n8n_am1st_news" Notion database's real
    column names, confirmed 2026-08-03 via a live schema read."""

    in_use: str = "in_use"
    feed_url: str = "RSS"
    name: str = "Name"
    cookie: str = "cookie"
    domain: str = "website"


class NotionCandidateProps(BaseModel):
    """Column names of the shared candidate-pool database ("AM1ST-n8n-Channel"
    in the original n8n export). The ingestion cycle writes every article
    that survives scoring/extraction/content-gen here; the separate publish
    cycle reads from it to pick one to post. Confirmed against the original
    n8n "Create a database page" node's field list, 2026-08-04 — see
    project_am1st_migration memory's "two separate workflows" note. This
    replaces the old NotionLogProps/log_db_id, which assumed (incorrectly)
    that ingestion wrote directly to a simple post-publish log."""

    title: str = "title"
    url: str = "url"
    author: str = "author"
    description: str = "description"
    published_at: str = "published_at"
    post_content: str = "post_content"
    llm_score: str = "llm_score"
    llm_comment: str = "llm_comment"
    content: str = "content"
    url_hash: str = "url_hash"
    send_status: str = "send_status"  # checkbox — set true only once the publish cycle actually posts it
    heat_score: str = "heat_score"  # number — corroboration signal, see HeatConfig
    event_first_seen_at: str = "event_first_seen_at"  # date — earliest time any related source was seen, vs published_at's own single-article timestamp


class NotionConfig(BaseModel):
    api_key: str = ""  # env: NOTION_API_KEY — used for source_db_id (and alerts, which live on source rows)
    candidate_api_key: str = ""  # env: NOTION_CANDIDATE_API_KEY — separate integration for candidate_db_id, since the two databases don't have to share one integration's Connections. Falls back to api_key if left blank.
    source_db_id: str = ""  # env: NOTION_SOURCE_DB_ID
    candidate_db_id: str = ""  # env: NOTION_CANDIDATE_DB_ID — shared by both the ingestion and publish cycles
    alert_user_id: str = ""  # env: NOTION_ALERT_USER_ID
    source_props: NotionSourceProps = Field(default_factory=NotionSourceProps)
    candidate_props: NotionCandidateProps = Field(default_factory=NotionCandidateProps)

    @property
    def candidate_key(self) -> str:
        return self.candidate_api_key or self.api_key


class RedisConfig(BaseModel):
    url: str = ""  # env: REDIS_URL (Upstash rediss:// connection string)
    key_prefix: str = "am1st:url_hash:"
    ttl_seconds: int = 864000  # 10 days


class OpenAIConfig(BaseModel):
    api_key: str = ""  # env: OPENAI_API_KEY
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    scoring_prompt_file: str = "prompts/scoring_prompt.txt"
    content_gen_prompt_file: str = "prompts/content_gen_prompt.txt"
    score_threshold: float = 5.0


class DedupConfig(BaseModel):
    semantic_threshold: float = 0.8


class HeatConfig(BaseModel):
    """Corroboration/heat scoring (added 2026-08-06, scoped 2026-08-05 — see
    project_am1st_migration memory). An article's own published_at is a poor
    proxy for how fresh the underlying news event actually is: Reuters can
    break something, CNN rehash it 5h later, CBS rehash it again the next
    day — each with a recent published_at despite covering old news. This
    reuses the same cross-cycle Qdrant query main.py already runs for dedup
    (title+description vs qdrant.collection, within qdrant.cross_cycle_window_hours)
    — one query per candidate, not two — and buckets the returned neighbors
    that fall below the dedup threshold (dedup.semantic_threshold) into a
    lower "related" band to derive two signals: how many distinct sources
    are covering the same event right now (weighted higher for major
    wire/TV outlets), and the earliest time any of them was seen —
    event_first_seen_at, which the ingestion scoring prompt and the
    publish-time priority-rank prompt both use instead of
    hours-since-this-article's-own-timestamp."""

    related_threshold: float = 0.6  # cosine similarity floor for "same event, different source" — below dedup.semantic_threshold on purpose, that band is "duplicate," this one is "corroborating"
    window_hours: int = 24  # how far back a neighbor counts toward heat/event_first_seen_at — narrower than qdrant.cross_cycle_window_hours (72h), which is just how far the query itself reaches
    major_outlets: list[str] = Field(
        default_factory=lambda: [
            "Reuters", "Financial Times", "Wall Street Journal", "CNN", "CBS", "Fox News", "New York Post", "Just The News",
        ]
    )
    major_outlet_weight: float = 2.0  # per-neighbor weight if its source_name is in major_outlets, vs 1.0 for any other corroborating source


class QdrantConfig(BaseModel):
    url: str = ""  # env: QDRANT_URL
    api_key: str = ""  # env: QDRANT_API_KEY
    collection: str = "am1st_embeddings"  # ingestion-side cross-cycle dedup cache (title+description)
    posted_collection: str = "am1st_posting_news_embedding"  # publish-side "already posted" cache (post_content) — separate collection, separate purpose, see core/qdrant_store.py's PostedHistoryStore
    cross_cycle_window_hours: int = 72
    cleanup_retention_days: int = 10


class ExtractionConfig(BaseModel):
    """No external service — see agents/extractor.py's docstring for why
    (the old n8n-svr extract-premium service is unmaintained and broken for
    most paywalled sources). Self-contained httpx+trafilatura instead."""

    timeout_seconds: int = 20
    min_text_length: int = 200  # below this, treat extraction as failed (bot-block/JS-wall pages are usually a few dozen chars)


class GettrConfig(BaseModel):
    user_id: str = ""  # env: GETTR_USER_ID
    user_token: str = ""  # env: GETTR_USER_TOKEN
    api_url: str = "https://gettr.com/api/u/post"


class PublishConfig(BaseModel):
    """Tuning for the separate publish cycle (main_publish.py) — selects and
    posts exactly one candidate per interval. Values below are the ones
    confirmed with the user on 2026-08-04 (see project_am1st_migration
    memory), not guesses: this is a deliberate simplification of the
    original n8n design (which could post an entire surviving batch) down
    to "publish exactly 1, walking down the priority order on duplicates."
    """

    interval_seconds: int = 1800  # 30 minutes
    candidate_min_score: float = 5.0  # Notion query floor — the lower of weekday/weekend_min_score, so both are actually fetched; select_batch applies the day-aware floor on top
    weekday_min_score: float = 6.0  # weekdays: heavier real news volume, prefer this floor first
    weekend_min_score: float = 5.0  # weekends: lighter volume, use this floor directly (also the weekday fallback if 6+ doesn't fill the batch)
    candidate_max_age_hours: int = 12  # candidate pool eligibility window
    fresh_hours: int = 4  # freshness tier line used by the batch-selection cascade
    batch_min: int = 3
    batch_max: int = 10  # was 5; raised 2026-08-05 now that extraction/content-gen only run on the selected batch, not every scored candidate — a bigger batch costs much less than it used to
    priority_rank_prompt_file: str = "prompts/priority_rank_prompt.txt"
    posted_dedup_window_hours: int = 24
    posted_dedup_threshold: float = 0.70  # stricter than the ingestion side's 0.8 — deliberate, per the user: fully autonomous posting should err toward under-posting


class AppConfig(BaseModel):
    notion: NotionConfig = Field(default_factory=NotionConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    heat: HeatConfig = Field(default_factory=HeatConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    gettr: GettrConfig = Field(default_factory=GettrConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)
    max_publish_age_hours: int = 3
    poll_interval_seconds: int = 600
    alert_cooldown_seconds: int = 21600


_ENV_OVERRIDES = {
    ("notion", "api_key"): "NOTION_API_KEY",
    ("notion", "candidate_api_key"): "NOTION_CANDIDATE_API_KEY",
    ("notion", "source_db_id"): "NOTION_SOURCE_DB_ID",
    ("notion", "candidate_db_id"): "NOTION_CANDIDATE_DB_ID",
    ("notion", "alert_user_id"): "NOTION_ALERT_USER_ID",
    ("redis", "url"): "REDIS_URL",
    ("openai", "api_key"): "OPENAI_API_KEY",
    ("qdrant", "url"): "QDRANT_URL",
    ("qdrant", "api_key"): "QDRANT_API_KEY",
    ("gettr", "user_id"): "GETTR_USER_ID",
    ("gettr", "user_token"): "GETTR_USER_TOKEN",
}


def load_config(path: str = "config/config.yaml") -> AppConfig:
    """Load YAML config, then apply environment variable overrides for secrets.

    Env vars always win over the YAML file, so real credentials never need to
    be committed to config.yaml — set them in .env / the deployment environment.
    """
    raw: dict = {}
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    config = AppConfig.model_validate(raw)

    for (section, field), env_var in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            section_obj = getattr(config, section)
            setattr(section_obj, field, value)

    return config
