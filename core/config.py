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
    extraction_failed: str = "extraction_failed"  # checkbox — set once by main_publish.py the first time full-text extraction fails for this candidate; a permanent exclusion, not a retry-later flag (added 2026-09-05)


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

    # 2026-09-04: verify_compatibility()'s NO_OVERLAP short-circuit
    # (entity overlap == empty set -> confident DIFFERENT_EVENT, no LLM)
    # was found, via a full audit of am1st_events, to wrongly fragment
    # real duplicate events whenever the two articles refer to the same
    # actor/institution through different surface forms with zero shared
    # words (a role vs. a name — "Trump's DHS Boss" vs. "Markwayne
    # Mullin"; an institution vs. its political head — "Treasury
    # Considers..." vs. "Trump Admin Moves To..."). No dictionary or NER
    # fix can close this — the shared word simply doesn't exist in either
    # text. Recommended externally (GPT architecture review, 2026-09-04)
    # and matches this module's own asymmetry principle elsewhere: at
    # this cosine floor, zero overlap is downgraded from NO_OVERLAP to
    # AMBIGUOUS (one extra same_event() LLM call) instead of confidently
    # rejecting — the LLM's world knowledge (who currently holds an
    # office, which agency executes which administration's policy) covers
    # exactly the gap entity-overlap can't. Deliberately NOT set below the
    # existing peek_top_k() candidate floor (heat.related_threshold, 0.6)
    # — this only reconsiders candidates already deemed plausible on
    # semantic grounds.
    #
    # 2026-09-04, recalibrated from an initial 0.75 guess: sampled 12 real
    # zero-overlap pairs per 0.05-wide cosine bucket from 0.60 up, and used
    # same_event() itself as the ground-truth oracle for each (this only
    # tests "is asking the LLM worth it here," not same_event()'s own
    # accuracy, so it's a valid comparison even though same_event() isn't
    # perfect). Actual SAME_EVENT rate despite zero entity overlap:
    # [0.60,0.70)=17%, [0.70,0.75)=50%, [0.75,0.80)=50%, [0.80,0.85)=50%,
    # [0.85,0.90)=58%, [0.90,1.0]=100%. The real jump is at 0.70, not 0.75
    # — 0.75-0.90 is roughly flat at ~50-58%, meaning the original 0.75
    # guess was missing an entire bucket (0.70-0.75) where half of all
    # zero-overlap pairs are genuine duplicates. Moved to 0.70.
    no_overlap_llm_review_floor: float = 0.70

    # IDF-weighted keyword overlap (2026-08-20, REMOVED as a decision gate
    # 2026-09-05) — was a second, entity-independent lexical signal for
    # verify_compatibility()'s FAIL_OPEN branch (new_tokens from NER came
    # back empty), ported from North_Korea_News's core/hashing.py (commit
    # 1caea63) with its threshold (0.15) carried over unvalidated. Once
    # AM1ST's own decisions accumulated enough real text to check (36
    # logged pairs, 2026-09-05), found a real false positive AT that
    # threshold: two installments of a recurring gun-review column (each
    # about a different specific model — a case this session already
    # established must stay separate) shared enough near-identical
    # boilerplate phrasing ("I Am a Retired US Army Special Forces NCO:
    # There Is Nothing Like the...") to score 0.294 — well above 0.15 —
    # and got silently merged as COMPATIBLE with no LLM check at all. Only
    # 2 real observations existed above the old threshold (one of them
    # this false positive), nowhere near enough to trust picking a new
    # number instead. This branch (a rare fallback: both sides' NER came
    # back with zero entities) now always returns AMBIGUOUS instead —
    # negligible added cost, and removes the only place in this class that
    # skipped the LLM without ever having been genuinely data-validated.
    # weighted_overlap() itself (core/hashing.py) is untouched and still
    # computed/logged for visibility, just no longer gates a verdict.


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
    starting point, revisit once enough real matches accumulate to
    check (same caution that led to removing the old
    weighted_overlap_threshold decision gate elsewhere in this class)."""

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
    candidate_max_age_hours: int = 24  # Notion query ceiling — same pattern as candidate_min_score: the WIDER of weekday/weekend_max_age_hours below, so weekend-eligible candidates aren't excluded before select_batch() even sees them; select_batch() applies the actual day-aware ceiling on top
    weekday_max_age_hours: int = 12  # weekdays: heavy real news volume — keep the original "same-day news" freshness rule
    weekend_max_age_hours: int = 24  # weekends (US/Eastern Sat/Sun): real volume is much lower — widened 2026-09-05 per the user's explicit request, so a Friday-night story is still eligible through Saturday instead of the pool running thin/empty
    fresh_hours: int = 4  # freshness tier line used by the batch-selection cascade
    batch_min: int = 3
    batch_max: int = 10  # was 5, raised to 10 on 2026-08-05 (extraction/content-gen only run on the selected batch now, not every scored candidate — a bigger batch costs much less than it used to). Briefly lowered back to 5 on 2026-09-02 after real data showed the winner was always found within the top 3 of the ranked batch — reverted the SAME day: the very first cycle at 5 hit "all candidates were duplicates, nothing to publish" on a day with heavy repeat coverage of a few ongoing stories (Iran, a House socialism vote), and 5 vs 10 wasn't saving much real token cost anyway (per-call cost is small) to be worth the risk of skipped publish cycles.
    priority_rank_prompt_file: str = "prompts/priority_rank_prompt.txt"
    posted_dedup_window_hours: int = 240  # was 24h — widened 2026-08-07 as a defensive backstop once the ingestion-side EventStore.mark_published() check exists (core/qdrant_store.py); matches heat.window_hours so both "have we already covered this" checks agree on how long an event stays "recent"
    posted_dedup_threshold: float = 0.70  # stricter than the ingestion side's 0.8 — deliberate, per the user: fully autonomous posting should err toward under-posting
    max_widen_attempts: int = 3  # 2026-09-05 — how many batch_max-sized chunks of the eligible pool to try before accepting "nothing to publish this cycle" as real, not just "the first batch_max happened to all be duplicates"; each attempt costs a real extraction+content-gen pass, so this isn't unbounded search over the whole pool (which real cycles have seen run into the hundreds)


class DynamicPublishConfig(BaseModel):
    """Automatic publish-cadence scaling (2026-09-05) — distinct from
    hot_topics.py's manual fast lane (a human flags one specific story as
    breaking); this instead reacts to how much genuinely strong material
    the ingestion side is producing right now, with no human involved.
    Signal: count of candidates newly added to the Notion pool in the last
    lookback_hours with llm_score >= hot_score_floor (reusing
    prompts/scoring_prompt.txt's own "8 — Major real-time trigger" band as
    the floor, not an arbitrary new cutoff) — a cheap, existence-count-only
    Notion query, same cost shape as hot_topics.py's
    has_unpublished_hot_candidate().

    Thresholds below are calibrated from a real 33.5-hour sample
    (2026-09-04 00:00 - 2026-09-05 09:29, 956 real ingested candidates,
    resampled at every 30-min tick = 67 slices) of this exact count (>=8,
    trailing 2h): min=0, p25=1, median=3, p75=8, p90=14, max=18 — busy_count
    is the observed p75, quiet_count is the observed p25, so roughly the
    quietest quarter of real history slows down and the busiest quarter
    speeds up, leaving the middle half at the unscaled base interval."""

    hot_score_floor: float = 8.0
    lookback_hours: float = 2.0
    quiet_count: int = 1  # <= this many -> slow down (real p25)
    busy_count: int = 8  # >= this many -> speed up (real p75)
    quiet_scale: float = 1.3  # 30min base -> ~39min
    busy_scale: float = 0.6  # 30min base -> ~18min
    min_interval_seconds: int = 900  # 15 min floor — never faster than this regardless of volume
    max_interval_seconds: int = 2700  # 45 min ceiling — never slower than this regardless of quiet


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
    dynamic_publish: DynamicPublishConfig = Field(default_factory=DynamicPublishConfig)
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
