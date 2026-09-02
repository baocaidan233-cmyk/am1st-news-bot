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
    is_hot: str = "is_hot"  # checkbox — set from the manual hot-topic flag match, see HotTopicsConfig; added 2026-08-31


class NotionHotTopicProps(BaseModel):
    """Column names of the shared "AM1ST热点标记" database (notion.hot_topics_db_id)
    — a small table the user edits directly (2026-08-31), NOT written by
    any bot. Multiple sibling bots read the same table, each filtering to
    its own tag in the `channel` multi-select column via
    HotTopicsConfig.channel_name — see core/hot_topics.py."""

    name: str = "Name"
    channel: str = "Channel"
    active: str = "In_Use"


class NotionConfig(BaseModel):
    api_key: str = ""  # env: NOTION_API_KEY — used for source_db_id (and alerts, which live on source rows)
    candidate_api_key: str = ""  # env: NOTION_CANDIDATE_API_KEY — separate integration for candidate_db_id, since the two databases don't have to share one integration's Connections. Falls back to api_key if left blank.
    hot_topics_api_key: str = ""  # env: NOTION_HOT_TOPICS_API_KEY — separate integration for hot_topics_db_id, since this table is shared across multiple sibling bots' own Notion connections. Falls back to api_key if left blank.
    source_db_id: str = ""  # env: NOTION_SOURCE_DB_ID
    candidate_db_id: str = ""  # env: NOTION_CANDIDATE_DB_ID — shared by both the ingestion and publish cycles
    hot_topics_db_id: str = ""  # env: NOTION_HOT_TOPICS_DB_ID — shared across sibling bots, see NotionHotTopicProps
    alert_user_id: str = ""  # env: NOTION_ALERT_USER_ID
    source_props: NotionSourceProps = Field(default_factory=NotionSourceProps)
    candidate_props: NotionCandidateProps = Field(default_factory=NotionCandidateProps)
    hot_topics_props: NotionHotTopicProps = Field(default_factory=NotionHotTopicProps)

    @property
    def candidate_key(self) -> str:
        return self.candidate_api_key or self.api_key

    @property
    def hot_topics_key(self) -> str:
        return self.hot_topics_api_key or self.api_key


class RedisConfig(BaseModel):
    url: str = ""  # env: REDIS_URL (Upstash rediss:// connection string)
    key_prefix: str = "am1st:url_hash:"
    ttl_seconds: int = 864000  # 10 days


class OpenAIConfig(BaseModel):
    api_key: str = ""  # env: OPENAI_API_KEY
    fallback_api_key: str = ""  # env: OPENAI_API_KEY_FALLBACK — only used if the primary key hits RateLimitError (rate limit or exhausted quota), see core/openai_client.py
    # Every OpenAI-backed call in this codebase (Writer prose, Scorer,
    # PriorityRanker, EventVerifier) shares this one model. A 2026-08-26
    # experiment moved the non-prose judgment calls (Scorer/PriorityRanker/
    # EventVerifier) onto a separate gpt-5-nano field (nano_model) for
    # cost reasons; reverted 2026-09-01 after a live multi-cycle test found
    # real judgment-quality failures in every one of those roles — the
    # 2026-08-26 switch had only verified well-formed output, never actual
    # judgment quality. See agents/scorer.py's docstring for the specific
    # failures found. nano_model removed entirely rather than left as dead
    # config — nothing in this codebase should reference gpt-5-nano again
    # without a fresh, judgment-quality-focused live test first.
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    scoring_prompt_file: str = "prompts/scoring_prompt.txt"
    content_gen_prompt_file: str = "prompts/content_gen_prompt.txt"
    score_threshold: float = 5.0

    @property
    def api_keys(self) -> list[str]:
        """Ordered list handed to core.openai_client.FallbackOpenAI — primary
        first, then the fallback if one is configured and actually different."""
        keys = [self.api_key]
        if self.fallback_api_key and self.fallback_api_key != self.api_key:
            keys.append(self.fallback_api_key)
        return keys


class DedupConfig(BaseModel):
    semantic_threshold: float = 0.8


class EntityVerifierConfig(BaseModel):
    """Second-opinion check on top of EventStore.peek()'s cosine match —
    see project_am1st_migration memory's 2026-08-09 "event identity"
    design note for the full derivation. Validated on real historical
    am1st_events data before implementation: rule tier (entity overlap,
    excluding tokens that have been the core of >=hub_event_count_threshold
    different past events) resolves ~2/3 of cosine-matched candidates
    reliably; the remaining ~1/3 ("AMBIGUOUS" — every shared token is a
    known multi-event hub, e.g. "Trump"/"Senate"/a cabinet official) goes
    to a same-event LLM call. That LLM call intentionally does NOT also
    ask for the update-subtype (CORE_UPDATE/DOWNSTREAM_REACTION/
    RESTATEMENT) in the same prompt — an ablation test found asking both
    at once biases the model toward SAME_EVENT ~30% of the time (it seems
    to pre-assume "same" so it has something to classify), which corrupts
    the one judgment that actually gates merging. Subtype is a separate,
    second call, only made when the first call says SAME_EVENT, purely to
    enrich the logged training record for future distillation — nothing
    downstream reads it yet.

    Deliberately NOT perfect — the user's explicit call (2026-08-09): a
    known residual miss rate (e.g. two different candidates' primary
    races sharing only generic tokens like a state name) is acceptable,
    not worth chasing with more entity-rarity formulas or bigger
    blocklists (see 高余弦相同事件验证器.md's "不应该继续做的事情" list).
    Every rule-tier AMBIGUOUS case and its LLM verdict gets logged
    (event_identity_log_path) specifically so those residual misses
    become future hard-negative training data instead of silently
    recurring forever."""

    hub_event_count_threshold: int = 2  # a shared token counts as real evidence only if it's been the CORE of fewer than this many distinct past events
    pair_cooccur_max: int = 1  # two individually-hub tokens can still count as evidence if this exact PAIR has co-occurred as a joint core in at most this many past events
    min_doc_freq_for_core: int = 2  # a token must appear in at least this many of an event's OWN accumulated articles to join its persisted core_entities (not a ratio — a ratio lets a single one-off token qualify as "core" while an event still only has 1-2 articles, see the same design note)
    hub_key_prefix: str = "am1st:hub:"  # Redis key namespace for the token/pair historical-hub-count index — separate namespace from redis.key_prefix's URL-hash dedup, same REDIS_URL
    same_event_prompt_file: str = "prompts/same_event_prompt.txt"
    update_subtype_prompt_file: str = "prompts/update_subtype_prompt.txt"
    related_event_prompt_file: str = "prompts/related_event_prompt.txt"  # EventVerifier.related_event() — see core/event_identity.py
    log_path: str = "logs/event_identity_decisions.jsonl"

    # Top-K event-candidate verification (2026-08-14 P0 redesign, per the
    # research memo's "难点五" — the single most cosine-similar historical
    # event isn't guaranteed to be the true match; a real match can rank
    # #2+ if the top-ranked candidate is a coincidentally-closer but
    # actually-unrelated event). main.py now walks up to top_k candidates
    # in cosine-descending order, verifying each with the same rule/LLM
    # tiers as before, and only creates a new event once every candidate
    # is rejected (NO_OVERLAP or entity-verifier DIFFERENT_EVENT) — not
    # after just the first one.
    top_k: int = 5

    # Subtype-weighted heat (2026-08-14 P0 — wires up EventVerifier.
    # classify_subtype(), which existed but was never actually called by
    # main.py before this). Applied as a multiplier on a matched cluster's
    # incremental heat contribution (never on a brand-new event's 1.0
    # baseline) — a real new development should move the needle more than
    # an outlet just repeating yesterday's line in different words. Not
    # applied to canonical_title/canonical_summary/timeline state — those
    # stay seed-only/never-rewritten per the user's earlier explicit call
    # (see EventStore.commit()'s docstring); this redesign only touches
    # heat weighting, nothing else.
    subtype_restatement_weight: float = 0.2   # repeats an already-known fact, different wording/outlet — barely moves heat
    subtype_corroboration_weight: float = 1.0  # independent new source confirming the same facts — today's existing behavior, unchanged
    subtype_core_update_weight: float = 1.5    # genuine new fact/decision/status — weighted above plain corroboration

    # 2026-09-02: skip the classify_subtype() LLM call entirely when A/B are
    # this cosine-similar AND name the same places/facilities and the same
    # numbers (see no_conflicting_specifics() in core/event_identity.py) —
    # a live review found the LLM asked to classify byte-identical A/B text
    # still hallucinating a "new development" instead of answering
    # RESTATEMENT. High cosine alone isn't sufficient: "Iran strikes Kuwait
    # base" vs "Iran strikes Qatar base" (or "10 dead" vs "20 dead") can
    # score just as high on cosine while describing a materially different
    # fact — the location/number check guards against that.
    restatement_cosine_floor: float = 0.92

    # IDF-weighted keyword overlap (2026-08-20) — a second, entity-
    # independent lexical signal for verify_compatibility()'s FAIL_OPEN
    # branch (new_tokens from NER came back empty — very short text, or a
    # genuine entity-extraction miss). That branch previously just blindly
    # trusted whatever cosine match it was handed with zero independent
    # check. Ported from North_Korea_News's core/hashing.py (commit
    # 1caea63) — same IDF-over-a-corpus math, no LLM call, no new database
    # (the corpus is this cycle's own batch of candidate titles+
    # descriptions, built in-memory in main.py, not a persisted store).
    # NOTE: this threshold is carried over from North_Korea_News's own
    # 642-real-item validation, NOT yet independently validated against
    # AM1ST's own historical am1st_events data the way every other
    # threshold in this class was (see the class docstring) — treat as a
    # starting point, revisit once real am1st decisions have been logged
    # and reviewed.
    weighted_overlap_threshold: float = 0.15


class HotTopicsConfig(BaseModel):
    """Manual breaking-news override (2026-08-31) — the user, not any
    automatic heat_score threshold, tells the system a specific topic
    matters right now, by adding/editing a row in a small shared Notion
    database (notion.hot_topics_db_id) multiple sibling bots read from,
    each filtering to their own `channel_name` tag in that row's Channel
    multi-select column (core/hot_topics.py).

    A row counts as currently live only while its In_Use checkbox is
    checked AND its own last_edited_time is within ttl_hours —
    deliberately anchored to last_edited_time, not created_time: nudging
    a still-developing topic (toggle the checkbox, edit the title) keeps
    it live without creating a new row, and forgetting to ever uncheck it
    can't leave it live forever. Chosen this way per the user's explicit
    2026-08-31 feedback that relying on someone remembering to uncheck it
    manually every day "肯定会忘记" (they'll definitely forget).

    main.py embeds every currently-live topic text once per ingestion
    cycle and compares against each local cluster's representative
    embedding (match_threshold). A match sets that event's hot_until
    (core/qdrant_store.py's EventStore) — which then flows through
    unchanged to every future corroborating article on the SAME event via
    EventStore.commit()'s usual "carry forward from matched" pattern, so
    later follow-up coverage inherits hot status automatically without
    needing to re-match against the original flag text every time. Not
    yet independently validated against real am1st_events data — a
    starting point, like weighted_overlap_threshold was."""

    channel_name: str = "AM1ST"  # this bot's own tag in the shared table's Channel multi-select column
    # Cosine floor for "this candidate is about a currently-flagged hot
    # topic," on text-embedding-3-small (openai.embedding_model). Calibrated
    # 2026-08-31 with live embedding calls, not guessed: a genuinely related
    # follow-up on the same event ("Trump reacts to X-Y meeting" vs the flag
    # text "X and Y meet at [event]") scored 0.588-0.636, a same-broad-topic
    # but different-event story scored 0.220, an unrelated domestic story
    # scored 0.134 — a first attempt at 0.65 would have MISSED the genuine
    # follow-ups (the whole point of this feature), so 0.5 was chosen
    # instead, comfortably above the unrelated cluster (<=0.22) and below
    # every related score observed. Still just a handful of synthetic
    # examples, not independently validated against real am1st_events data.
    match_threshold: float = 0.5
    ttl_hours: int = 24  # both the Notion flag's own last_edited_time freshness window AND how long a matched event stays "hot" after its last matching commit
    fast_poll_seconds: int = 180  # main_publish.py's short-poll interval while an unpublished is_hot candidate exists, instead of waiting out the full publish.interval_seconds — see main_publish.py


class HeatConfig(BaseModel):
    """Corroboration/heat scoring — event aggregation (redesigned 2026-08-06,
    see project_am1st_migration memory's "event aggregation" note for the
    full history). An article's own published_at is a poor proxy for how
    fresh the underlying news event actually is: Reuters can break
    something, CNN rehash it 5h later, CBS rehash it again the next day —
    each with a recent published_at despite covering old news.

    core/qdrant_store.py's EventStore maintains a dedicated Qdrant
    collection (qdrant.events_collection) where each event is a GROUP of
    points sharing one event_id — not one fixed vector — following the
    standard "topic tracking" pattern from TDT (Topic Detection and
    Tracking) research: a topic/event is represented by a small evolving
    set of representative documents, not a single centroid, so that (a)
    articles using different phrasing than the very first report can still
    be recognized as the same event (mitigates cluster fragmentation), and
    (b) the representative set stays current as a multi-day story evolves
    (mitigates semantic drift). heat_score/first_seen_at/last_updated_at/
    sources are kept in sync across every point of an event via a
    filtered payload update, not stored per-point independently.

    A near-duplicate (score >= dedup.semantic_threshold, dropped from the
    candidate pool) still bumps its matched event's heat tally before being
    dropped — that credit must not be lost just because the article itself
    isn't worth its own candidate-pool row. Only a genuinely new-angle
    match (related_threshold <= score < dedup.semantic_threshold) becomes
    an additional representative point, since a near-duplicate doesn't add
    matching robustness."""

    related_threshold: float = 0.6  # cosine similarity floor for "same event" — below dedup.semantic_threshold on purpose, that band is "duplicate," this one is "corroborating"
    window_hours: int = 240  # 10 days — matches redis.ttl_seconds' 10-day convention; deliberately much wider than qdrant.cross_cycle_window_hours (72h, the plain dedup check's reach) so a multi-day-evolving event doesn't get treated as "expired" and fragmented into a phantom duplicate event
    major_outlets: list[str] = Field(
        default_factory=lambda: [
            "Reuters", "Financial Times", "Wall Street Journal", "CNN", "CBS", "Fox News", "New York Post", "Just The News",
        ]
    )
    major_outlet_weight: float = 2.0  # per-corroborating-source weight if its source_name is in major_outlets, vs 1.0 for any other source


class QdrantConfig(BaseModel):
    url: str = ""  # env: QDRANT_URL
    api_key: str = ""  # env: QDRANT_API_KEY
    collection: str = "am1st_embeddings"  # ingestion-side cross-cycle dedup cache (title+description)
    posted_collection: str = "am1st_posting_news_embedding"  # publish-side "already posted" cache (post_content) — separate collection, separate purpose, see core/qdrant_store.py's PostedHistoryStore
    events_collection: str = "am1st_events"  # event aggregation collection, see HeatConfig/EventStore — a genuinely different kind of thing from the two collections above (a group of points per underlying event, not one point per article)
    cross_cycle_window_hours: int = 72
    cleanup_retention_days: int = 10
    timeout_seconds: int = 15  # AsyncQdrantClient has no timeout by default — a stalled request (real hang observed 2026-08-11 during a live 3-cycle test, no error, no timeout, just stuck) can block run_cycle forever


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
    posted_dedup_window_hours: int = 240  # was 24h — widened 2026-08-07 as a defensive backstop once the ingestion-side EventStore.mark_published() check exists (core/qdrant_store.py); matches heat.window_hours so both "have we already covered this" checks agree on how long an event stays "recent"
    posted_dedup_threshold: float = 0.70  # stricter than the ingestion side's 0.8 — deliberate, per the user: fully autonomous posting should err toward under-posting


class AppConfig(BaseModel):
    notion: NotionConfig = Field(default_factory=NotionConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    entity_verifier: EntityVerifierConfig = Field(default_factory=EntityVerifierConfig)
    hot_topics: HotTopicsConfig = Field(default_factory=HotTopicsConfig)
    heat: HeatConfig = Field(default_factory=HeatConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    gettr: GettrConfig = Field(default_factory=GettrConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)
    max_publish_age_hours: int = 3
    poll_interval_seconds: int = 600
    cycle_timeout_seconds: int = 540  # 9 min — per the user's real n8n experience, a healthy cycle runs ~5min and almost never past 7min; this hard-cuts a stuck cycle so the next one always starts on schedule (main.py and main_publish.py loops both apply this, independently — see 2026-08-12 waterfall/no-external-retrigger discussion)
    alert_cooldown_seconds: int = 21600


_ENV_OVERRIDES = {
    ("notion", "api_key"): "NOTION_API_KEY",
    ("notion", "candidate_api_key"): "NOTION_CANDIDATE_API_KEY",
    ("notion", "hot_topics_api_key"): "NOTION_HOT_TOPICS_API_KEY",
    ("notion", "source_db_id"): "NOTION_SOURCE_DB_ID",
    ("notion", "candidate_db_id"): "NOTION_CANDIDATE_DB_ID",
    ("notion", "hot_topics_db_id"): "NOTION_HOT_TOPICS_DB_ID",
    ("notion", "alert_user_id"): "NOTION_ALERT_USER_ID",
    ("redis", "url"): "REDIS_URL",
    ("openai", "api_key"): "OPENAI_API_KEY",
    ("openai", "fallback_api_key"): "OPENAI_API_KEY_FALLBACK",
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
