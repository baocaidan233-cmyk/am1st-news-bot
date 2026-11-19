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

2026-10-13 (AI-23): an article that gives no time for its headline's news
(news_at None) used to stop here as fresh; agents/first_report_check.py now
looks for an earlier report of the same fact first.

2026-11-19 (AI-23): the other way round too. A dated story that relays
another outlet's reporting, or documents someone obtained, can be an old
matter revealed only now -- user, 10-09: 过去的事件刚刚爆料出来，那不算旧闻，
这种不能误伤. 11-11..11-19 the date rule stopped three such stories: an Erie
Democratic chairman's July Notice to Appear dug up by the Erie Times-News, a
San Francisco complaint of March 3 that NBC Bay Area brought out, VA figures
for 2025 "obtained by The Daily Signal". The model had read each as the date
of the news. Such a stale verdict now stands only if the first-report search
finds an earlier report of the same fact; a failed search keeps it stale.
Only for a date 30+ days back that the sentence does not give as the day it
was made public: on the 59 dated stale verdicts of 11-11..11-19, asking for
every relayed story released 6 that were only late (a sentencing on Tuesday,
remarks on Wednesday), or public from the start ("figures released in
September 2025", "the audit, released in August 2020"), next to the 4 real ones.
With the two limits, exactly the 4.
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

from agents.first_report_check import FirstReportCheck
from core.config import AppConfig
from core.news_time import _norm, decide, first_disclosure, names_this_outlet
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


# The article passes on someone else's digging: records, documents, a report.
_REVEAL = re.compile(r"\b(?:obtained by|according to (?:new |newly |detailed )?(?:records|documents|court (?:records|documents|filings)|"
                     r"emails|filings|reporting)|(?:records|documents|filings|emails) (?:show|showed|reveal|revealed|obtained)|"
                     r"newly (?:released|obtained|unsealed|surfaced)|unsealed|leaked|(?:detailed |new )?reporting (?:from|by))\b", re.I)
# A first-report answer that means "nobody reported this earlier".
_NOT_EARLIER = ("no earlier report of the same fact found", "same fact reported earlier, but this article adds a major development")


_REVEAL_MIN_DAYS = 30
# The sentence gives its date as the day the thing was made public.
_MADE_PUBLIC = re.compile(r"\b(?:released|published|announced|unveiled|reported|posted|aired|made public|"
                          r"said in a statement|told reporters|declared|disclosed)\b", re.I)


def _relays_revelation(devs: list, body: str, url: str, news_at: Optional[datetime] = None,
                       now: Optional[datetime] = None) -> bool:
    """Whether a stale-by-date story may be an old matter just brought out by
    someone else's reporting. Not when the headline's date is that report's
    own: "Moonshot AI looked to rent Blackwell chips, CNBC reported Nov. 5th";
    not when the date is recent (the story is late, not revealed); not when the
    sentence says that date is when it came out."""
    head = devs[0] if devs and isinstance(devs[0], dict) else {}
    if news_at is not None and now is not None and (now - news_at).days < _REVEAL_MIN_DAYS:
        return False
    if _MADE_PUBLIC.search(head.get("what") or ""):
        return False
    outlet = head.get("cited_outlet")
    if outlet and not names_this_outlet(outlet, url):
        what = _norm(head.get("what") or "")
        return not (_norm(outlet) in what and "report" in what)
    return bool(_REVEAL.search((body or "")[:4000]))


class TimelinessCheck:
    def __init__(self, config: AppConfig, cache_size: int = 2000) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._prompt = Path(config.openai.timeliness_extract_prompt_file).read_text(encoding="utf-8")
        self._window = config.publish.timeliness_window_hours
        self._cache: OrderedDict[str, list] = OrderedDict()
        self._cache_size = cache_size
        self._first_report = FirstReportCheck(config)

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
        first_report = None
        if (result["verdict"] == "stale" and result.get("news_at") is not None and self._first_report.enabled
                and _relays_revelation(devs or [], body, url, datetime.fromisoformat(result["news_at"]), now)):
            first_report = await self._first_report.check(url, url_hash, title, body, now)
            if first_report.get("reason") in _NOT_EARLIER:
                result = {**result, "verdict": "fresh",
                          "reason": "dated %s, but no earlier report of it: brought out only now" % result["reason"].split(" is at least")[0]}
        elif result["verdict"] == "fresh" and result.get("news_at") is None:
            first_report = await self._first_report.check(url, url_hash, title, body, now)
            if first_report["verdict"] == "stale":
                result = {"verdict": "stale", "reason": "no time in the article; " + first_report["reason"],
                          "news_at": None, "age_hours": (first_report.get("evidence") or {}).get("age_hours")}
        _log({"url": url, "title": title[:200], "published": published_at.isoformat(), **result, "developments": devs,
              "first_report": first_report})
        return result
