from __future__ import annotations

import hashlib
import logging
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.config import AppConfig
from core.models import PublishCandidate

logger = logging.getLogger(__name__)

# Which timezone's calendar day decides "weekday vs weekend" — US/Eastern,
# since this is a US-audience channel and that's the standard reference for
# "the US news day," not UTC or wherever this process happens to run.
_DAY_TZ = ZoneInfo("America/New_York")

# Exact phrase list from the original n8n "check former" node (confirmed
# 2026-08-05 by reading its actual JS in v1.4_am1st_notion_to_gettr_auto
# posting.json — not a guess). The real system also had a second,
# independent check ("check former1") reading a Notion formula column's
# precomputed flag — that formula's own definition isn't in the export, so
# it isn't replicated here, but this phrase list is the confirmed one that
# runs in code either way. Checked against title+description+content+
# post_content combined, same as the original. Now serves double duty
# after 2026-08-05's extraction/content-gen move: still catches a stale
# re-synced article, and also the writer's own occasional hallucination
# (content_gen_prompt.txt says "always President Trump," but an LLM can
# still default to "former president" out of training-data habit) once
# post_content is checked here too.
_FORMER_TRUMP_PHRASES = (
    "former president trump",
    "former us president trump",
    "former u.s. president trump",
    "former president donald trump",
    "former us president donald trump",
    "former u.s. president donald trump",
)

# Independent of publish.candidate_min_score (the floor used by the Notion
# query) — the original n8n "batch of top 5" node hardcodes this tier
# boundary at 7 regardless of what the floor is set to.
#
# 8.0 from 2026-10-07, together with the scoring prompt v2.3 (owner-approved).
# Under the old prompt 7 and 8 were reached almost only through heat and held
# ~8% of the pool between them; v2.3 spreads the scale and puts ~28% of the pool
# at 7+ and ~10% at 8+, with 8 the band that separates breakouts (gatewaypundit
# 16% inside it vs 4.6% overall). This line also decides who gets the 24h age
# ceiling and the freshness exemption (agents/priority_ranker.py imports it).
_TIER1_MIN_SCORE = 8.0

# Subjects the channel publishes as a duty, whatever their reader appeal (see key() in select_batch).
DUTY_TOPICS = frozenset(("中国CCP",))


def _is_weekday(now: datetime) -> bool:
    return now.astimezone(_DAY_TZ).weekday() < 5  # Mon=0 ... Sun=6


def is_night(now: datetime, config: AppConfig) -> bool:
    """True inside the overnight window, judged in US/Eastern — same calendar
    reference _is_weekday() uses, for the same reason (this is a US-audience
    channel; "overnight" means overnight for the readers, not for whatever
    timezone this process happens to run in).

    Wraps midnight, so the default start=0/end=7 means 00:00-06:59 ET.
    start == end disables the window entirely. Also imported by
    core/publish_cadence.py, which applies the cadence half of the same gate."""
    pub = config.publish
    start, end = pub.night_start_hour, pub.night_end_hour
    if start == end:
        return False
    hour = now.astimezone(_DAY_TZ).hour
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def filter_former_trump(candidates: list[PublishCandidate]) -> list[PublishCandidate]:
    """Drops anything calling Trump a former president — see module
    docstring for the exact phrase list and why this now runs twice
    (before and after content-gen)."""
    kept = []
    for c in candidates:
        combined = " ".join([c.title, c.description, c.content, c.post_content]).lower()
        if any(phrase in combined for phrase in _FORMER_TRUMP_PHRASES):
            continue
        kept.append(c)
    return kept


def age_ceiling_for(c: PublishCandidate, day_ceiling: float, config: AppConfig) -> float:
    """How old this candidate may be, in hours.

    The top band gets the pool's own outer bound instead of the day-aware
    one. On a weekday that ceiling is 12h, and it was deleting exactly the
    candidates the score is right about.

    Measured. The score emits three values across the live pool -- 5.0, 6.0
    and 8.0, nothing between 6.0 and 8.0 -- so it cannot order within a
    band, but the top band is real: over 716 published posts the 8.0 tier
    breaks out 7.3% of the time against 1.1% at 6.0. A 2026-10-02 audit over
    14 days found 7.0 and 8.0 candidates publish only 48% of the time, 76 of
    them aged past the window unpublished, and only 2.6% of those 76 were
    stories this channel had covered from another source -- the rest were
    simply lost. The channel's second-highest post of all time (277
    engagement, 3.79x the median) was an 8.0 that entered the pool at
    2026-10-01 22:02 and went out by hand 18.1 hours later, because no
    normal cycle could still reach it. On 2026-10-05 three more 8.0
    candidates sat at 22.9 to 24.1 hours, all past the weekday ceiling, and
    went out by hand again on the channel owner's instruction -- publish
    them even though they are late.

    This is not a general relaxation of the freshness rule. It gives one
    band, 2.2% of the pool, the window the whole pool already has at the
    weekend, and it invents no new number: candidate_max_age_hours is the
    Notion query's own ceiling, so nothing older than that is ever fetched.
    """
    if c.llm_score >= _TIER1_MIN_SCORE:
        return max(float(day_ceiling), float(config.publish.candidate_max_age_hours))
    return float(day_ceiling)


def _stable_key(c: PublishCandidate) -> str:
    """A deterministic pseudo-random ordering key, from the candidate's own url.

    Replaces ordering by llm_score inside a tier (2026-09-26). The score still
    decides which tier a candidate lands in, because that is a real editorial
    relevance judgement, but it cannot usefully order candidates within one:
    across 571 published posts the hour-normalised engagement of the 6, 7 and 8
    bands is 1.00, 1.01 and 1.03, and 63% of all ranked candidates carry the
    single value 6.0. Sorting a set that is mostly ties is a stable sort over
    ties -- whatever order Notion returned, which is not an editorial variable
    and was silently deciding a large share of what this channel published.

    Note what the band measurement does NOT say. It compares posts that were
    published, which were already selected; it is not evidence that a random
    6.0 equals a random 8.0. That is why the tiers keep the score and only the
    ordering inside them changes.

    Deterministic rather than random so a candidate keeps its position across
    the cycles it survives, instead of being reshuffled every 15 minutes."""
    return hashlib.sha1((c.url or c.page_id).encode("utf-8")).hexdigest()


def _source_of(c: PublishCandidate) -> str:
    try:
        return urlparse(c.url).netloc.replace("www.", "").lower()
    except Exception:
        return ""


def _fill(batch: list[PublishCandidate], pool, limit: int, topic_cap: int,
          source_cap: int, trace=None, label: str = "") -> None:
    """Appends from `pool` up to `limit`, skipping anything whose subject or
    whose source already holds its cap of slots in this batch.

    The source cap is new (2026-09-26) and is the same idea as the subject cap:
    a batch of ten drawn largely from one outlet is a batch of ten versions of
    that outlet's news judgement. Either cap <= 0 disables it."""
    topics = Counter(c.topic for c in batch if c.topic)
    sources = Counter(_source_of(c) for c in batch if _source_of(c))
    walked: list[str] = []
    skipped: dict[str, str] = {}
    # Materialised so the trace can record how many candidates were COMPETING,
    # not just how many the scan reached. The first version logged n_in as
    # kept+excluded, and `kept` here is the batch itself rather than this
    # pass's input — so "how many aged candidates were in the running" came
    # out as the batch size. Order is unchanged.
    pool = list(pool)
    n_offered = len(pool)
    for c in pool:
        pid = getattr(c, "page_id", "") or getattr(c, "url", "")
        if len(batch) >= limit:
            # Everything from here on was never examined. Recording where the
            # scan stopped is the difference between "looked at and skipped"
            # and "the batch was already full" — the two the previous logs
            # could not tell apart.
            if trace is not None:
                trace.stage(f"fill:{label}", batch, skipped, order=walked,
                            stopped_at=len(walked), capacity_reached=True,
                            n_offered=n_offered, batch_size=len(batch))
            return
        walked.append(pid)
        if topic_cap > 0 and c.topic and topics[c.topic] >= topic_cap:
            skipped[pid] = f"topic_cap:{c.topic}:{topics[c.topic]}/{topic_cap}"
            continue
        src = _source_of(c)
        if source_cap > 0 and src and sources[src] >= source_cap:
            skipped[pid] = f"source_cap:{src}:{sources[src]}/{source_cap}"
            continue
        batch.append(c)
        if c.topic:
            topics[c.topic] += 1
        if src:
            sources[src] += 1
    if trace is not None:
        trace.stage(f"fill:{label}", batch, skipped, order=walked,
                    stopped_at=len(walked), capacity_reached=False,
                    n_offered=n_offered, batch_size=len(batch))


def shortlist(
    candidates: list[PublishCandidate],
    config: AppConfig,
    seen: dict[str, int] | None = None,
    now: datetime | None = None,
) -> list[PublishCandidate]:
    """Cuts the eligible pool down to the set agents/editor.py can actually
    read. Used only when editor.enabled; select_batch() below is untouched and
    remains the path when it is off.

    Applies exactly the same hard filters select_batch() does — the day-aware
    published_at ceiling and the night score gate — and then orders by score,
    breaking ties by how many shortlists the candidate has already appeared on
    and only then by the stable hash.

    That middle term is the whole point. The eligible pool runs at a median of
    130 and a 90th percentile of 296 (measured 2026-09-28), so the cut is
    real, and the score cuts it coarsely but honestly: 7+ is 11% of the pool,
    8 is 6%. Below that it stops working — one sampled moment held 151
    candidates tied at exactly 6.0 — and whatever broke those ties decided
    which of them was ever looked at. That used to be sha1(url), a number that
    never changes, so a candidate it placed below the cut was below it every
    cycle for its whole 12-hour life: 477 of 884 eligible candidates over 72
    hours never entered a single ranking call. Ordering by appearances first
    means never-seen beats already-evaluated-and-passed-over, so the tie group
    rotates instead of freezing.

    `seen` missing or empty is the correct fallback, not a failure: every
    count reads as 0, the term drops out, and the order is the old one."""
    pub = config.publish
    now = now or datetime.now(timezone.utc)
    is_weekday = _is_weekday(now)
    max_age_hours = pub.weekday_max_age_hours if is_weekday else pub.weekend_max_age_hours
    seen = seen or {}

    def hours_old(c: PublishCandidate) -> float:
        return (now - c.published_at).total_seconds() / 3600

    pool = [c for c in candidates
            if hours_old(c) <= age_ceiling_for(c, max_age_hours, config)]
    if is_night(now, config):
        pool = [c for c in pool if c.is_hot or c.llm_score >= pub.night_min_score]
    pool.sort(key=lambda c: (-c.llm_score, seen.get(c.url_hash, 0), _stable_key(c)))
    cut = pool[: config.editor.shortlist_size]
    logger.info(
        "shortlist: %d eligible -> %d after age/night -> %d shortlisted (never-shown %d)",
        len(candidates), len(pool), len(cut), sum(1 for c in cut if seen.get(c.url_hash, 0) == 0),
    )
    return cut


def debt_level(debt_hours: float, config: AppConfig) -> int:
    """2 = badly overdue, 1 = overdue, 0 = served recently.

    Coarse on purpose. A subject 13.2 hours unserved is not meaningfully more
    urgent than one at 12.8, and turning that gap into a decimal would rebuild
    the false precision the old priority_score had."""
    pub = config.publish
    if debt_hours >= pub.topic_debt_high_hours:
        return 2
    if debt_hours >= pub.topic_debt_mid_hours:
        return 1
    return 0


def select_batch(
    candidates: list[PublishCandidate],
    config: AppConfig,
    topic_adjustments: dict[str, float] | None = None,
    topic_debt: dict[str, float] | None = None,
    cycle_token: str = "",
    trace=None,
) -> list[PublishCandidate]:
    """Tiered batch selection — same cascade as the original n8n "batch of
    top 5" node: prefer fresh+high-scoring, progressively relax until at
    least batch_min survive (or give up and just take the newest ones),
    capped at batch_max. `candidates` should already be former-Trump-filtered
    and come from the Notion eligibility query (send_status/age/score gate
    already applied there, at the lower of weekday/weekend_min_score so
    both are actually fetched).

    Weekday/weekend-aware floor (2026-08-05, user request): weekdays see
    much more real news volume, so prefer publish.weekday_min_score first
    and only relax to the lower publish.weekend_min_score if that doesn't
    fill batch_min. Weekends see much less volume, so go straight to
    weekend_min_score — being picky first would usually just waste a tier.

    2026-08-06: the score-based tiers (1-4) now keep cascading until
    batch_max is full, not just until batch_min is met — this used to stop
    as soon as 3 candidates were found, even though batch_max allows 10 and
    extraction/content-gen only runs on this small selected batch (cheap).
    A moderately-scored-but-currently-trending story could end up excluded
    from the batch entirely if 3+ higher-scoring fresh stories already
    existed that cycle — by the time the priority-ranker sees trending
    context (agents/priority_ranker.py), it's too late, that story was
    never in the running. Filling the full batch_max with every genuinely
    scored candidate (not just the top 3) gives the AI re-rank step, which
    DOES see trending_headlines, a real chance to surface it. Tier 5 (the
    pure "newest regardless of score" last resort) still gates on
    batch_min only — it exists for when there's nothing real to pick from
    at all, not to pad out a batch that already has legitimate candidates.

    2026-09-01 — hard, unconditional published_at ceiling added (user's
    explicit "iron rule": only same-day news, publish nothing rather than
    something stale). Confirmed live: a 3-day-old CNBC article that only
    entered the candidate pool THAT DAY (query_eligible_candidates()'s own
    eligibility window is keyed on Notion's created_time — when this bot
    first saw it — not the article's own published_at) sailed through
    tier 3 below, which only checks llm_score, not freshness at all, and
    got published as if it were breaking. Every tier below, including the
    hot-topic force-include and the batch_min last-resort fallback, now
    only ever draws from `candidates` after this filter — none of them can
    bypass it.

    2026-09-05 — this ceiling is now day-aware, same weekday/weekend split
    as the score floor above: weekday_max_age_hours (12h) keeps the
    original same-day rule; weekend_max_age_hours (24h, widened per the
    user's explicit request) reflects real lower weekend news volume, so a
    Friday-night story is still eligible through Saturday instead of the
    batch running under batch_min or empty. query_eligible_candidates()'s
    own Notion-query ceiling (config.publish.candidate_max_age_hours) is
    deliberately the WIDER of the two (24h) so weekend-eligible candidates
    are never excluded before reaching this actual day-aware filter — same
    "fetch the superset, then apply the real floor here" pattern as
    candidate_min_score/weekday_min_score/weekend_min_score above. On a
    genuinely slow news day this can still leave the batch under
    batch_min, even empty — that's the accepted trade-off, not a bug to
    work around."""
    pub = config.publish
    now = datetime.now(timezone.utc)
    is_weekday = _is_weekday(now)
    preferred_floor = pub.weekday_min_score if is_weekday else pub.weekend_min_score
    fallback_floor = pub.weekend_min_score
    max_age_hours = pub.weekday_max_age_hours if is_weekday else pub.weekend_max_age_hours

    def hours_old(c: PublishCandidate) -> float:
        return (now - c.published_at).total_seconds() / 3600

    if trace is not None:
        trace.note(now=now.isoformat(), is_weekday=is_weekday,
                   preferred_floor=preferred_floor, fallback_floor=fallback_floor,
                   max_age_hours=max_age_hours, fresh_hours=pub.fresh_hours,
                   tier1_min_score=_TIER1_MIN_SCORE, batch_max=pub.batch_max,
                   batch_min=pub.batch_min)
        trace.stage("returned", candidates, {})
    _before = candidates
    candidates = [c for c in candidates
                  if hours_old(c) <= age_ceiling_for(c, max_age_hours, config)]
    if trace is not None:
        kept_ids = {getattr(c, "page_id", "") for c in candidates}
        trace.stage("age_ceiling", candidates,
                    {(getattr(c, "page_id", "") or ""): "older_than_%.0fh:%.1f" % (age_ceiling_for(c, max_age_hours, config), hours_old(c))
                     for c in _before if (getattr(c, "page_id", "") or "") not in kept_ids})

    # 2026-09-17 — overnight quality gate; see PublishConfig.night_min_score
    # for the decision-log evidence behind the window and the threshold.
    # Applied as a hard filter on `candidates` here, exactly like the
    # published_at ceiling above and for the same reason: every tier below —
    # including the weekday->weekend floor relaxation and the batch_min
    # "newest overall, regardless of score" last resort — draws only from this
    # filtered list, so none of them can quietly relax the floor back down.
    # is_hot is exempt on purpose: a manually-flagged breaking story must
    # still be publishable overnight, the same guarantee the force-include
    # below already makes against the score tiers.
    if is_night(now, config):
        _b = candidates
        candidates = [c for c in candidates if c.is_hot or c.llm_score >= pub.night_min_score]
        if trace is not None:
            kept_ids = {getattr(c, "page_id", "") for c in candidates}
            trace.stage("night_gate", candidates,
                        {(getattr(c, "page_id", "") or ""): "night_min_score:%.1f" % c.llm_score
                         for c in _b if (getattr(c, "page_id", "") or "") not in kept_ids})

    fresh = [c for c in candidates if hours_old(c) <= pub.fresh_hours]
    if trace is not None:
        fresh_ids = {getattr(c, "page_id", "") for c in fresh}
        trace.stage("fresh", fresh,
                    {(getattr(c, "page_id", "") or ""): "not_fresh:%.1fh" % hours_old(c)
                     for c in candidates if (getattr(c, "page_id", "") or "") not in fresh_ids})
        trace.stage("tier1_input", [c for c in candidates if c.llm_score >= _TIER1_MIN_SCORE],
                    {(getattr(c, "page_id", "") or ""): "below_tier1:%.1f" % c.llm_score
                     for c in candidates if c.llm_score < _TIER1_MIN_SCORE})

    tcap = config.topic_mix.per_batch_cap if config.topic_mix.enabled else 0
    scap = pub.per_batch_source_cap

    # Ordering inside every tier. This is where the subject mix has to act,
    # and until 2026-09-28 it did not: the adjustment was passed in, used only
    # for the per-batch cap, and left to agents/priority_ranker.py.
    #
    # That was wrong about which step decides the output. 96.5% of publishes
    # finish on the first widen attempt (1278 of 1324 measured 2026-09-28,
    # against 43 on the second and 3 on the third), so the ~9 candidates this
    # function returns ARE the candidate set for that post -- widening is a
    # failure fallback, not a wider search, and the rest of the pool is never
    # looked at. Ordering those nine by a hash made the published subject mix
    # a random sample of the pool's, which is what the 7-day audit measured:
    # total variation between pool and published of 13.6%, 外交与战争 running
    # at 17.4% against a 9% target while 选举诚信 and 媒体与审查, the only two
    # subjects with a measured engagement edge, sat under theirs. The ranker
    # cannot fix that downstream -- it only ever sees these nine.
    #
    # The key is the controller's own signed adjustment, so a subject already
    # over its target sorts last by construction and cannot take a batch; an
    # empty adj (controller dormant below min_window_posts) makes every key 0.0
    # and this is exactly the previous behaviour. The score tiers still decide
    # who is eligible, the per-batch cap of 3 and the uncapped refill below are
    # untouched, and _stable_key breaks ties -- which is most pairs, and see
    # its docstring for why llm_score cannot be what breaks them.
    # Ordering inside a tier. Subject coverage debt comes first: measured
    # 2026-09-28, 犯罪治安 had 42 eligible candidates waiting against 28.4
    # hours unpublished and 政府腐败与浪费 24 against 21.3, while the previous
    # key — the mix adjustment, identical for every candidate of one subject,
    # then sha1(url), which never changes — could not see either. A starved
    # subject stayed starved however much of it was in the pool.
    #
    # Within-batch diversity is the existing per-subject cap of 3, not a decay
    # on this key: sorted() evaluates the key once per candidate before _fill
    # takes anything, so a decay applied during the fill would never be seen.
    # Keeping them separate is also correct on its own terms — publication
    # debt is what the channel has actually published, and selecting into a
    # batch is not publishing.
    # Ordering inside a tier is a hash and nothing else, as of 2026-10-05.
    #
    # It used to be (-subject debt level, -subject mix adjustment, hash). Both
    # subject terms are gone because neither was ever the channel owner's
    # instruction. The mix targets were fitted to n=573 past posts and then
    # labelled editorial policy; sixteen of the eighteen rested on differences
    # that were not significant, and across 571 published posts only one
    # bucket reached significance at all -- which itself disappears once what
    # the reader gets is controlled for (p 0.011 -> 0.37), while the reverse
    # does not. Subject debt was the better-built half, an observed "hours
    # since we last ran this subject" with no target in it, but it still
    # partitions by the same eighteen categories, and the owner did not ask
    # for that partition to order anything either.
    #
    # What is left is honest rather than good: nothing available can order
    # the middle of the pool. The score cannot -- 97.8% of candidates carry
    # 5.0 or 6.0 -- and that is why tier1 is now filled score-first over the
    # whole top band (see the _fill below) and why the mid-band is explicitly
    # a draw rather than a formula pretending to rank it.
    #
    # 2026-10-06, owner-approved trial: within a tier, a candidate whose story
    # carries a reader appeal (agents/appeal_tagger.py) goes ahead of one that
    # does not, and the hash only orders inside each group. The tiers above are
    # untouched, so this never lifts a candidate past a higher score band; it
    # decides which of the many equal-score candidates make the batch the
    # ranker picks from. Unlabelled (None) sorts with "no appeal". Off unless
    # publish.appeal_order, and then the key is the hash alone, as before.
    #
    # 2026-10-08, owner: duty stories go as if they carried an appeal. Of 13
    # Trump-Xi stories where the two leaders actually dealt with each other, the
    # appeal labeller called 8 "none" -- plain hard news ("trade truce extended
    # through January", "Xi lays out terms to avoid military conflict") gives the
    # reader no line to repeat -- so they sorted behind every appeal story in
    # their band and could wait out their whole life in the pool. The owner's
    # rule is that such stories go out even when engagement is low. "Duty" is the
    # subject label agents/topic_tagger.py already gave the candidate at pool
    # entry (中国CCP), not a new word list.
    appeal_first = config.publish.appeal_order

    def key(c: PublishCandidate):
        h = (_stable_key(c) if not cycle_token else hashlib.sha1(
            (cycle_token + (c.url or c.page_id)).encode("utf-8")).hexdigest())
        return (0 if (c.appeal or c.topic in DUTY_TOPICS) else 1, h) if appeal_first else h

    batch: list[PublishCandidate] = []
    # tier1 draws from `candidates`, not from `fresh`. The two filters are
    # different questions and conflating them is what 87b0e15 half-fixed:
    # that change gave the top band the pool's own 24h ceiling, but tier1 --
    # the only place the batch is filled score-first -- was still gated on
    # fresh_hours, so a 7.0 at five hours could reach the batch only through
    # tier3, which runs at all only when tier1 and tier2 leave room and
    # orders by subject debt and a hash with no score term in it. Widening
    # the outer ceiling alone moved those candidates from deleted to queued
    # in the tier that starves, which is why the 2026-10-05 recovery still
    # had to go out by hand.
    #
    # Measured the same day: the pool held six candidates at 7.0 or above
    # (three 7.0, three 8.0) and tier1 could see three of them. The other
    # three were 19 to 24 hours old. Over the previous 14 days 76 top-band
    # candidates aged out unpublished and only 2.6% of them were stories
    # covered from another source; the channel's second-highest post ever
    # (277 engagement, 3.79x) was an 8.0 that sat 18.1 hours and went out by
    # hand. The ceiling that applies here is still age_ceiling_for()'s, so
    # nothing enters that the filter above already rejected.
    _fill(batch, sorted((c for c in candidates if c.llm_score >= _TIER1_MIN_SCORE), key=key),
          pub.batch_max, tcap, scap, trace, "tier1")

    if len(batch) < pub.batch_max:
        picked_ids = {c.page_id for c in batch}
        _fill(batch, sorted(
            (c for c in fresh if preferred_floor <= c.llm_score < _TIER1_MIN_SCORE and c.page_id not in picked_ids),
            key=key), pub.batch_max, tcap, scap, trace, "tier2_fresh_mid")

    if len(batch) < pub.batch_max:
        picked_ids = {c.page_id for c in batch}
        _fill(batch, sorted(
            (c for c in candidates if c.llm_score >= preferred_floor and c.page_id not in picked_ids),
            key=key), pub.batch_max, tcap, scap, trace, "tier3_aged")

    if is_weekday and len(batch) < pub.batch_max:
        # Weekday-only extra fallback: still room in the batch, so relax
        # down to the weekend's lower floor before giving up on score entirely.
        picked_ids = {c.page_id for c in batch}
        _fill(batch, sorted(
            (c for c in candidates if fallback_floor <= c.llm_score < preferred_floor and c.page_id not in picked_ids),
            key=key), pub.batch_max, tcap, scap, trace, "tier4_weekday_relax")

    # Same cascade re-run with the subject cap off, so the cap can only ever
    # change WHICH candidates fill a batch, never leave the batch short and
    # cause a skipped cycle. Only reachable when the cap actually bound.
    if (tcap > 0 or scap > 0) and len(batch) < pub.batch_max:
        picked_ids = {c.page_id for c in batch}
        _fill(batch, sorted(
            (c for c in candidates if c.llm_score >= fallback_floor and c.page_id not in picked_ids),
            key=key), pub.batch_max, 0, 0, trace, "tier5_caps_off")

    if len(batch) < pub.batch_min:
        # Last resort: newest overall, regardless of score.
        batch = sorted(candidates, key=hours_old)[: pub.batch_min]

    # Manual hot-topic force-include (2026-08-31, core/hot_topics.py) — a
    # candidate the user has explicitly flagged as breaking must never be
    # silently excluded from the batch just because its llm_score tier
    # didn't make the cut above; by the time agents/priority_ranker.py sees
    # it, it's too late (same reasoning as the 2026-08-06 tier-cascade
    # change above, applied to a stronger signal). Evicts the current
    # lowest-scored non-hot member if the batch is already full, rather
    # than growing past batch_max.
    picked_ids = {c.page_id for c in batch}
    missing_hot = [c for c in candidates if c.is_hot and c.page_id not in picked_ids]
    for c in missing_hot:
        if len(batch) < pub.batch_max:
            batch.append(c)
            continue
        evict_idx = min(
            (i for i, b in enumerate(batch) if not b.is_hot),
            key=lambda i: batch[i].llm_score,
            default=None,
        )
        if evict_idx is not None:
            batch[evict_idx] = c

    return batch[: pub.batch_max]
