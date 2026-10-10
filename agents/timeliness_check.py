"""Timeliness check before the writer (2026-10-10). Replaces agents/
staleness_checker.py's FRESH / OPINION / STALE call.

What changed and why:
  - The old call ran only once the event store had seen the story 72 hours
    earlier. The first report of a stale story is usually not in our feeds, or
    the event store files it as a different event, so it seldom ran: 9 of the
    200 posts of 10/06-10/09 were stale and none had been stopped.
  - The old call let the model judge freshness. Here the model only copies
    (prompts/timeliness_extract_prompt.txt) and core/news_time.py decides,
    because "when did this first become public" is date arithmetic and
    reading "told Blaze News" as a first report, both of which code does
    reliably.
  - OPINION is gone (user, 2026-10-10): commentary on an old event with
    nothing new is not published, rather than published as analysis.

One gpt-4o-mini call per candidate, and only for candidates code cannot
already clear as a first report. The copied developments are cached per
url_hash for the life of the process, so a candidate that waits in the pool
is re-judged every cycle -- its age keeps growing -- without a second call.
Logged to logs/timeliness.jsonl. Any failure leaves the story fresh.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.config import AppConfig
from core.news_time import decide, first_disclosure
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "timeliness.jsonl")
_lock = threading.Lock()


def _log(row: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        with _lock, open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row},
                                ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.debug("timeliness: could not write %s", _LOG, exc_info=True)


class TimelinessCheck:
    def __init__(self, config: AppConfig, cache_size: int = 2000) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._prompt = Path(config.openai.timeliness_extract_prompt_file).read_text(encoding="utf-8")
        self._window = config.publish.timeliness_window_hours
        self._cache: OrderedDict[str, list] = OrderedDict()
        self._cache_size = cache_size

    async def _developments(self, url_hash: str, title: str, body: str) -> Optional[list]:
        if url_hash in self._cache:
            self._cache.move_to_end(url_hash)
            return self._cache[url_hash]
        resp = await self._client.chat.completions.create(
            model=self._model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": self._prompt},
                      {"role": "user", "content": f"{title}\n\n{body[:6000]}"}])
        devs = json.loads(_FENCE.sub("", resp.choices[0].message.content or ""))["developments"] or []
        self._cache[url_hash] = devs
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return devs

    async def check(self, url: str, url_hash: str, title: str, body: str, published_at: datetime,
                    now: Optional[datetime] = None) -> dict:
        """The decision from core/news_time.decide(). On any failure, a fresh
        verdict with the error as its reason: a story is never stopped because
        the check broke."""
        now = now or datetime.now(timezone.utc)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        devs = None
        try:
            # A first report needs no model: it is news as of its own date.
            if not first_disclosure(title, body, url):
                devs = await self._developments(url_hash, title, body)
            result = decide(devs or [], title, body, url, published_at, now, self._window)
        except Exception as e:
            logger.warning("timeliness: check failed for %s (%s) — treating as fresh", url, e)
            result = {"verdict": "fresh", "reason": "check failed: %s" % str(e)[:200], "news_at": None, "age_hours": None}
        _log({"url": url, "title": title[:200], "published": published_at.isoformat(), **result, "developments": devs})
        return result
