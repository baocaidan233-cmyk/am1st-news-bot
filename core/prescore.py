"""Embedding pre-score (2026-09-29) — skips the Scorer, and everything
between it and here, for a candidate whose title is very unlikely to reach
the pool gate.

Ported from Leading News, which shipped this the same day. Two things are
different here and both were measured on this channel's own data, not
assumed:

1. **Where it sits.** Leading News scores early, so its pre-score only saves
   the scoring call. This channel scores at main.py's Layer 4, after the
   og:description backfill (an HTTP fetch per candidate), the intra-batch
   clustering embed and the event-store dedup (which asks an LLM judge
   whenever the rule layer is unsure). A candidate dropped here costs none
   of that. On 2026-09-28 this channel made 4397 LLM calls of which 1879
   were scoring; the rest are mostly that judge.

2. **The uninformative-title pass-through**, which Leading News's design doc
   raises as its open question 2 and leaves unanswered. It is worth more
   here than the model is. Measured on 1730 labelled titles from this
   channel (2026-09-29): the model ranks fine -- out-of-fold AUC 0.84 for
   score>=5, 0.86 for >=6, 0.89 for >=7 -- but the threshold, which is the
   lowest out-of-fold probability among rows at the gate, was pinned at
   0.0991 by "So Long, Minocqua Brewing", and right behind it sat "Open The
   Books Articles", "Interesting…" (which the Scorer gave a 6.0) and
   "Morning Minute: Run Through the Tape". Those are not stories the model
   misjudged. They are titles that cannot be judged, on which the Scorer was
   reading the description, which this layer never sees. Passing a title of
   four words or fewer straight through costs 2.3% of candidates and moves
   the threshold to 0.1314: skips go from 4.5% to 8.6% with, still, nothing
   at 5, 6 or 7 lost.

Everything else is Leading News's design and is kept deliberately: the
threshold is the minimum rather than a percentile, every safety layer can
only keep a candidate, and a missing or unreadable model file skips nothing.

The audit cohort is a fixed share of ALL candidates, chosen by url_hash
before the model looks, that is always scored. Their verdicts are recorded
but not acted on, which is the only way to measure P(skipped | reaches the
gate) without the selection bias that makes a live filter's miss rate
unknowable -- a skipped candidate has no score, so it can never appear in
any other estimate of what skipping costs."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def log_prescore_decision(path: str, record: dict) -> None:
    """One JSON line per verdict. The audit cohort's rows are the whole point
    of keeping this: tools/prescore_report.py joins them by URL to the score
    the Scorer went on to give, and that join is the only way to see what a
    skip would have cost. Best-effort — a logging failure must never take the
    ingestion cycle down."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"logged_at": int(time.time()), **record}, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("log_prescore_decision: could not write %s", path)

# Matches a word for the length test. Kept the same shape as the tokeniser in
# core/title_guard.py so "cancer-free" and "Trump’s" each count once.
_WORD = re.compile(r"[\wÀ-ɏ'’-]+")


def title_word_count(title: str) -> int:
    return len(_WORD.findall(title or ""))


class PreScorer:
    def __init__(self, config) -> None:
        cfg = config.prescore
        self.enabled = False
        self.audit_rate = cfg.audit_rate
        self.min_title_words = cfg.min_title_words
        self._ref: np.ndarray | None = None
        self.ood_cut: float | None = None
        self.version = "?"
        if not cfg.enabled:
            return
        try:
            path = Path(cfg.model_file)
            model = json.loads(path.read_text(encoding="utf-8"))
            self._w = np.asarray(model["w"], dtype=np.float32)
            self._b = float(model["b"])
            self.threshold = float(model["threshold"])
            self.version = model.get("version", "?")
            if model.get("ood_ref_file") and model.get("ood_cut") is not None:
                ref = np.load(path.parent / model["ood_ref_file"]).astype(np.float32) / 127.0
                self._ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
                self.ood_cut = float(model["ood_cut"])
            self.enabled = True
            logger.info("PreScorer: loaded %s (threshold %.4f, trained on %s rows, OOD reference %s)",
                        self.version, self.threshold, model.get("n_train"),
                        "on" if self._ref is not None else "OFF")
        except Exception:
            logger.exception("PreScorer: model %s not usable — pre-score disabled, every candidate is scored",
                             cfg.model_file)

    def probability(self, title_embedding) -> float:
        z = float(np.dot(self._w, np.asarray(title_embedding, dtype=np.float32))) + self._b
        return 1.0 / (1.0 + math.exp(-z))

    def in_audit(self, url_hash: str) -> bool:
        """Deterministic per candidate, and decided before the model looks."""
        return int(hashlib.sha1(url_hash.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < self.audit_rate

    def nearest_similarity(self, title_embedding) -> float | None:
        if self._ref is None:
            return None
        v = np.asarray(title_embedding, dtype=np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        return float(np.max(self._ref @ v))

    def decide(self, url_hash: str, title: str, title_embedding) -> tuple[str, dict]:
        """Returns (decision, detail). Everything except "skip" is scored.

        "pass"              the model does not want to skip it
        "audit_would_skip"  audit cohort; the model would have skipped it
        "audit_pass"        audit cohort; the model would not have
        "keep_short_title"  too few words to judge a story by
        "keep_ood"          nothing in training looks like this
        "skip"              not scored
        """
        p = self.probability(title_embedding)
        detail: dict = {"prescore_p": round(p, 4)}
        would_skip = p < self.threshold
        if self.in_audit(url_hash):
            return ("audit_would_skip" if would_skip else "audit_pass"), detail
        if not would_skip:
            return "pass", detail
        # The title is what this layer judges on, so a title with nothing in
        # it is a reason to defer to the Scorer, which can see the description.
        if title_word_count(title) <= self.min_title_words:
            return "keep_short_title", detail
        sim = self.nearest_similarity(title_embedding)
        if sim is not None:
            detail["nn_sim"] = round(sim, 4)
            if sim < self.ood_cut:
                return "keep_ood", detail
        return "skip", detail
