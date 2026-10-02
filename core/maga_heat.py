"""How hot a story is on the big MAGA accounts, right now, by their engagement.

This is NOT the consensus signal maga_pulse_shadow.py already collects. That one
counts how many distinct channels are running a story, and counting channels was
measured against our own engagement and came back null: 0.95x [0.79, 1.10], and
we already hold 100% of the >=3-channel stories, 34 of 37 published before the
consensus even formed. Counting coverage tells us nothing we did not have.

What the accounts' own ENGAGEMENT tells us is a different question, and it was
never asked until 2026-10-02. Asked that day over the 40 highest-engagement
posts on 13 big accounts, channel-normalised, matched word-for-word against our
own candidate titles (cosine >= 0.90, so the pairing is checkable by eye rather
than taken on trust):

    we published it                           4
    it was in our pool and we did not publish 3   (one at score 7.0)
    IT NEVER REACHED THE SCORER               9   <-- dropped as a duplicate
                                                      before anything scored it

Those nine are the finding. They passed the pre-filter, then our own dedup threw
them away before the scorer saw them, and a candidate with no score cannot be
recovered by anything downstream:

    3.84x  Hegseth orders military cyber agencies to defend midterms
    3.75x  Ted Cruz Blasts Jack Smith, "Egregious Abuse of Power"
    3.68x  Sen. Blackburn Personally Sues Jack Smith Over Subpoenaed Phone
    3.64x  Justice Department Indicts 10 Non-Citizens for Illegally Voting
    3.51x  Virginia Releases Illegal Alien Accused of Crimes Against Children
    3.25x  Immigration From Muslim-Majority Countries Drops to All-Time Low
    2.65x  Education Department formally ends Biden-era Title IX regulations

Dedup is not wrong by its own standard — another outlet's version of each had
been seen. It is wrong about what this audience wants, because the big accounts
run several angles of a story the audience is on and each one earns.

Normalisation is per-account and not optional. The 13 accounts differ by more
than an order of magnitude in reach, so a raw count would just rank the biggest
account's routine posts above everyone else's best. `adj` is a post's total over
the median total of its own channel.

Reads the cache maga_pulse_shadow.py already writes every hour. It does not
fetch, so it costs nothing and cannot fail on Gettr being slow; if the timer
stops, the heat index goes stale and `heat()` returns 0.0 rather than guessing.
"""

from __future__ import annotations

import base64
import json
import logging
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# GlobalPulse is deliberately absent from the accounts maga_pulse collects, and
# the reason applies here too: 125 posts/day, 91% video reposts. Its topic mix
# is a volume strategy, not evidence of what the audience wants.
DEFAULT_CACHE = "logs/.maga_pulse_cache.json"


class MagaHeat:
    """heat(embedding) -> (adj, channel, body) for the hottest matching story.

    `adj` is channel-normalised engagement, so 3.0 means "three times the median
    post on the account that ran it". 0.0 means no match, which is the answer for
    most candidates and is meant to be.
    """

    def __init__(self, cache_path: str = DEFAULT_CACHE, window_hours: float = 36.0,
                 min_adj: float = 1.5, match_cosine: float = 0.62,
                 coverage_hours: float = 24.0) -> None:
        self.coverage_hours = coverage_hours
        self.window_hours = window_hours
        self.min_adj = min_adj
        self.match_cosine = match_cosine
        self._V: np.ndarray | None = None
        self._meta: list[dict] = []
        self._RV: np.ndarray | None = None
        self._rch: list[str] = []
        self.loaded_at = 0.0
        self._path = Path(cache_path)
        self.reload()

    def reload(self) -> None:
        """Re-read the cache. Cheap enough to call once per cycle, and has to be
        called again at some point because the timer rewrites the file hourly."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("MagaHeat: cannot read %s (%s) — heat signal is off",
                           self._path, type(e).__name__)
            self._V, self._meta = None, []
            return
        posts = [p for p in (raw.get("posts") or {}).values()
                 if p.get("vec") and (p.get("body") or "").strip()]
        if not posts:
            self._V, self._meta = None, []
            return
        by_channel: dict[str, list[int]] = defaultdict(list)
        for p in posts:
            p["_tot"] = (p.get("lk") or 0) + (p.get("cm") or 0) + (p.get("sh") or 0)
            by_channel[p.get("ch") or "?"].append(p["_tot"])
        medians = {c: (st.median(v) or 1.0) for c, v in by_channel.items()}
        cutoff = time.time() - self.window_hours * 3600
        hot = []
        for p in posts:
            if (p.get("cdate") or 0) / 1000.0 < cutoff:
                continue
            adj = p["_tot"] / medians[p.get("ch") or "?"]
            if adj < self.min_adj:
                continue
            p["_adj"] = adj
            hot.append(p)
        if not hot:
            logger.info("MagaHeat: no post above %.1fx in the last %.0fh",
                        self.min_adj, self.window_hours)
            self._V, self._meta = None, []
            self.loaded_at = time.time()
            # coverage 仍然可用 —— 它不依赖互动
            cov_cutoff = time.time() - self.coverage_hours * 3600
            recent = [p for p in posts if (p.get("cdate") or 0) / 1000.0 >= cov_cutoff]
            if recent:
                RV = np.stack([np.frombuffer(base64.b64decode(p["vec"]), dtype=np.float32)
                               for p in recent])
                self._RV = RV / np.maximum(np.linalg.norm(RV, axis=1, keepdims=True), 1e-9)
                self._rch = [p.get("ch") or "?" for p in recent]
            else:
                self._RV, self._rch = None, []
            return
        # coverage 用的索引：近期全部帖，不按互动筛。互动要 36-48 小时才长出来
        # （实测中位 adj：0-6h 0.37 / 12-24h 0.49 / 24-48h 1.00 / 48h+ 1.40），
        # 所以「大号在这条上挣了多少」只能说明两天前的事。而「大号现在正在发这条」
        # 是即时可得的，这才是能用在选稿决策上的那一半。
        cov_cutoff = time.time() - self.coverage_hours * 3600
        recent = [p for p in posts if (p.get("cdate") or 0) / 1000.0 >= cov_cutoff]
        if recent:
            RV = np.stack([np.frombuffer(base64.b64decode(p["vec"]), dtype=np.float32)
                           for p in recent])
            self._RV = RV / np.maximum(np.linalg.norm(RV, axis=1, keepdims=True), 1e-9)
            self._rch = [p.get("ch") or "?" for p in recent]
        else:
            self._RV, self._rch = None, []

        V = np.stack([np.frombuffer(base64.b64decode(p["vec"]), dtype=np.float32)
                      for p in hot])
        self._V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
        self._meta = [{"adj": p["_adj"], "ch": p.get("ch") or "?",
                       "body": " ".join((p.get("body") or "").split())[:160]}
                      for p in hot]
        self.loaded_at = time.time()
        logger.info("MagaHeat: %d hot stories from %d accounts in the last %.0fh "
                    "(>=%.1fx their own median)",
                    len(hot), len({m["ch"] for m in self._meta}),
                    self.window_hours, self.min_adj)

    @property
    def ready(self) -> bool:
        return self._V is not None and len(self._meta) > 0

    def heat(self, embedding) -> tuple[float, str, str]:
        """(adj, channel, body) of the hottest matching story, or (0.0, "", "")."""
        if not self.ready:
            return 0.0, "", ""
        try:
            v = np.asarray(embedding, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n <= 0:
                return 0.0, "", ""
            sims = self._V @ (v / n)
        except Exception:
            logger.debug("MagaHeat: bad embedding")
            return 0.0, "", ""
        # The hottest match above the cosine floor, not the closest one: a
        # candidate can match two angles of one story and the louder angle is
        # the better evidence of what the audience is on.
        best = (0.0, "", "")
        for i in np.where(sims >= self.match_cosine)[0]:
            m = self._meta[int(i)]
            if m["adj"] > best[0]:
                best = (m["adj"], m["ch"], m["body"])
        return best

    def coverage(self, embedding) -> tuple[int, list[str]]:
        """(how many DISTINCT big accounts are running this story now, which).

        Available the moment they post, which is the whole reason it exists
        alongside heat(): engagement needs 36-48 hours to be readable and news
        does not wait that long. Counting coverage was measured against our own
        engagement in 2026-09 and came back null, and that measurement stands —
        but it answers a different question than this is for. "Five big MAGA
        accounts are running this story right now" is a reason not to throw a
        candidate away as a duplicate whether or not it predicts our likes.
        """
        if self._RV is None or not self._rch:
            return 0, []
        try:
            v = np.asarray(embedding, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n <= 0:
                return 0, []
            sims = self._RV @ (v / n)
        except Exception:
            return 0, []
        chans = {self._rch[int(i)] for i in np.where(sims >= self.match_cosine)[0]}
        return len(chans), sorted(chans)
