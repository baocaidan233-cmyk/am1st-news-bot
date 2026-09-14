from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from agents.embedder import Embedder
from agents.trending import fetch_trending_headlines
from core.config import AppConfig
from core.hashing import cosine_similarity
from core.notion_candidates import query_eligible_candidates

logger = logging.getLogger(__name__)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _load_and_prune_calibration(path: Path, window_days: float) -> list[dict]:
    """Reads the rolling calibration log, drops entries older than
    window_days, and rewrites the file with only the kept entries (bounds
    file size -- this runs once per publish cycle, every ~15-39min, so an
    unpruned log would grow without limit). Fails open to [] on any read
    error, same convention as every other best-effort read in this
    codebase; an empty history just means the caller falls back to its
    own bootstrap defaults."""
    cutoff = time.time() - window_days * 86400
    kept: list[dict] = []
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("ts", 0) >= cutoff:
                        kept.append(row)
    except Exception:
        logger.exception("_load_and_prune_calibration: read failed -- continuing with empty history")
        return []

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row) + "\n")
    except Exception:
        logger.exception("_load_and_prune_calibration: rewrite failed -- continuing anyway")

    return kept


def _append_calibration_sample(path: Path, backlog: int, heat: float, trending: float) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "backlog": backlog, "heat": heat, "trending": trending}) + "\n")
    except Exception:
        logger.exception("_append_calibration_sample: write failed -- continuing without recording this sample")


def _urgency(value: float, low: float, high: float, *, log_space: bool) -> float:
    """Maps value into [0, 1] against a [low, high] reference band,
    clamped at both ends. log_space=True compares ln(1+value) instead of
    the raw value -- for backlog/heat, both right-skewed count-like
    signals where a linear scale leaves the typical case indistinguishable
    from quiet (see DynamicPublishConfig docstring)."""
    if log_space:
        value, low, high = math.log1p(value), math.log1p(low), math.log1p(high)
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


async def compute_dynamic_interval(config: AppConfig) -> float:
    """Automatic cadence scaling -- see DynamicPublishConfig docstring
    for the full design (three noisy-OR-combined signals: backlog, heat,
    trending; rolling self-calibrated reference bands). Runs its own
    query_eligible_candidates() call rather than reusing run_cycle()s --
    a second cheap Notion query is simpler and lower-risk than threading
    that list through run_cycle()s several early-return paths, and this
    function already ran its own independent query before this redesign
    too (the old count_recent_high_score)."""
    dp = config.dynamic_publish

    try:
        candidates = await query_eligible_candidates(config)
    except Exception:
        logger.exception("compute_dynamic_interval: query_eligible_candidates failed -- failing open to max_interval_seconds")
        return float(dp.max_interval_seconds)

    backlog = len(candidates)
    heat = max((c.heat_score for c in candidates), default=0.0)

    trending = 0.0
    try:
        headlines = await fetch_trending_headlines()
        if headlines and candidates:
            embedder = Embedder(config)
            trending_embeddings = [await embedder.embed(h) for h in headlines if h]
            if trending_embeddings:
                top = sorted(candidates, key=lambda c: c.llm_score, reverse=True)[: dp.trending_check_top_k]
                for c in top:
                    cand_embedding = await embedder.embed(f"{c.title}\n{c.description}"[:2000])
                    best = max(cosine_similarity(cand_embedding, e) for e in trending_embeddings)
                    trending = max(trending, best)
    except Exception:
        logger.exception("compute_dynamic_interval: trending signal failed -- continuing without it")

    calib_path = Path(dp.calibration_log_path)
    history = _load_and_prune_calibration(calib_path, dp.calibration_window_days)
    _append_calibration_sample(calib_path, backlog, heat, trending)

    if len(history) >= dp.calibration_min_samples:
        backlog_low = _percentile([h["backlog"] for h in history], 0.10)
        backlog_high = _percentile([h["backlog"] for h in history], 0.90)
        heat_low = _percentile([h["heat"] for h in history], 0.10)
        heat_high = _percentile([h["heat"] for h in history], 0.90)
        trending_low = _percentile([h["trending"] for h in history], 0.10)
        trending_high = _percentile([h["trending"] for h in history], 0.90)
        source = "calibrated"
    else:
        backlog_low, backlog_high = dp.backlog_low_default, dp.backlog_high_default
        heat_low, heat_high = dp.heat_low_default, dp.heat_high_default
        trending_low, trending_high = dp.trending_low_default, dp.trending_high_default
        source = "bootstrap"

    u_backlog = _urgency(backlog, backlog_low, backlog_high, log_space=True)
    u_heat = _urgency(heat, heat_low, heat_high, log_space=True)
    u_trending = _urgency(trending, trending_low, trending_high, log_space=False)

    combined = 1 - (1 - u_backlog) * (1 - u_heat) * (1 - u_trending)
    span = dp.max_interval_seconds - dp.min_interval_seconds
    interval = dp.max_interval_seconds - combined * span
    interval = max(dp.min_interval_seconds, min(dp.max_interval_seconds, interval))

    logger.info(
        "compute_dynamic_interval: backlog=%d(u=%.2f) heat=%.1f(u=%.2f) trending=%.3f(u=%.2f) [%s, n=%d] -> combined=%.2f interval=%.0fs",
        backlog, u_backlog, heat, u_heat, trending, u_trending, source, len(history), combined, interval,
    )
    return interval
