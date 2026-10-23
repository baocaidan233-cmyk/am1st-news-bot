"""First-report check for articles that give no time (2026-10-13, Linear AI-23).

agents/timeliness_check.py clears an article as fresh when it carries no time
for its headline's news. 10-10..10-12, 23 of the 39 posts cleared that way
were old news or commentary on old news. This looks for an earlier report of
the same fact, published 48 hours or more before now:

  1. our own stores first (AM1ST and the other channels' Qdrant collections),
     each hit dated by its own article's time, never by its event's
     first_seen_at, which a merged article inherits;
  2. if none of those is the same fact, Google News, searched with the
     headline's words up to two days ago, each result opened for its first
     paragraph (a headline alone is never evidence).

Claude Haiku compares headline + first paragraph on both sides
(prompts/first_report_prompt.txt): same specific fact or not, and whether ours
adds a major development by the user's 10-12 standard. Code checks both quotes
are really in the texts and decides: stale only for a same fact, quotes found,
no major development. Anything else, any failure, leaves the article fresh.

Measured offline before going live: 13 of 47 old posts stopped, 0 of 37 fresh
ones (7 of them newly revealed old matters) wrongly stopped, on the 10-10..10-12
leak sample and a held-out 10-12/13 sample. Default thinking: low effort and no
thinking each stopped a new Trump statement in the held-out sample. Most misses
have no same-fact candidate at all; this is a partial fix.

The result is cached per url_hash for the life of the process, like the
timeliness extraction it extends."""
from __future__ import annotations

import html as htmlmod
import json
import logging
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote, quote_plus

import anthropic
import httpx
import trafilatura
from qdrant_client import AsyncQdrantClient

from agents.embedder import Embedder
from core import haiku
from core.config import AppConfig
from core.same_news import lead_paragraph
from core.claude_usage import log_usage


def first_paragraphs(body: str, n: int = 3, limit: int = 1500) -> str:
    """Headline + first paragraph could not tell a new round from the old
    one (10-14: a new batch of DOJ voting charges, a bill formally introduced
    after its September announcement, a second undercover video were all
    stopped as old). The first three paragraphs carry the names and dates
    that tell them apart."""
    out = []
    for para in (body or "").split("\n"):
        p = " ".join(para.split())
        if len(p) >= 80:
            out.append(p)
        if len(out) == n:
            break
    return "\n".join(out)[:limit]

logger = logging.getLogger(__name__)

# collection -> (time field of the article itself, text fields)
_STORES = {
    "am1st_embeddings": ("publishedAt", ("content",)),
    "dailynews_embeddings": ("published_at_ts", ("title", "description")),
    "china_breaks_scrapped_embeddings": ("publishedAt", ("content",)),
    "am1st_events": ("published_at", ("text", "representative_text")),
    "chinabreaks_events": ("published_at", ("text", "representative_text")),
    "leadingnews_test_events": ("published_at", ("text", "representative_text")),
    "marketwatcher_events": ("published_at", ("text", "representative_text")),
}
_STORE_FLOOR = 0.60
# A listing page is not a report of anything: a tag page dated by its newest
# item stood in as the "earlier report" of the Omar raid (10-13).
_LISTING = re.compile(r"/(tag|tags|category|categories|topic|topics|author|authors|search|page)/", re.I)
_MAX_CANDIDATES = 3
_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9"}
_THIN = re.compile(r"The post .{0,200} appeared first on .*", re.S)
_STOP = set("a an the of to in on for and or but with by at from as is are was were be been it its this that these those after "
            "before over into about just new now says said say will would could can has have had not no why how what who his her "
            "their our your than more most watch video breaking exclusive report reports".split())
_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["candidates", "new_fact_quote"],
    "properties": {
        "candidates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "relation", "our_quote", "their_quote"],
            "properties": {"id": {"type": "integer"},
                           "relation": {"type": "string", "enum": ["SAME_FACT", "DIFFERENT", "UNSURE"]},
                           "our_quote": {"type": "string"}, "their_quote": {"type": "string"}}}},
        "new_fact_quote": {"type": "string"},
    },
}


def _clean(s: str) -> str:
    return " ".join(_THIN.sub("", htmlmod.unescape(re.sub(r"<[^>]+>", " ", s or ""))).split())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"[‘’“”]", "", (s or "").lower()))


def _strip_title(body: str, title: str) -> str:
    lines = (body or "").split("\n")
    return "\n".join(lines[1:]) if lines and _norm(lines[0])[:60] == _norm(title)[:60] else body


def _keywords(title: str) -> str:
    words = [w for w in re.sub(r"[^\w\s'-]", " ", re.sub(r"\[.*?\]|\(.*?\)", " ", title)).split()
             if w.lower() not in _STOP and len(w) > 2]
    return " ".join(words[:8])


class FirstReportCheck:
    def __init__(self, config: AppConfig, cache_size: int = 2000) -> None:
        key = config.publish.first_report_api_key
        self.enabled = config.publish.first_report_check and bool(key) and bool(config.qdrant.url)
        self._window = config.publish.timeliness_window_hours
        self._model = config.publish.first_report_model
        self._system = Path(config.publish.first_report_prompt_file).read_text(encoding="utf-8") if self.enabled else ""
        self._claude = anthropic.AsyncAnthropic(api_key=key, timeout=60, max_retries=3) if self.enabled else None
        self._qdrant = (AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key or None,
                                          timeout=config.qdrant.timeout_seconds) if self.enabled else None)
        self._embedder = Embedder(config) if self.enabled else None
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size
        if config.publish.first_report_check and not key:
            logger.warning("FirstReportCheck: FIRST_REPORT_ANTHROPIC_API_KEY not set -- first-report check off")

    async def _store_candidates(self, title: str, lead: str, url: str, cutoff: float) -> list[dict]:
        vec = await self._embedder.embed(f"{title}\n{lead}")
        own = url.split("?")[0].rstrip("/")
        hits, seen = [], set()
        for col, (tf, fields) in _STORES.items():
            try:
                res = await self._qdrant.query_points(col, query=vec, limit=6, with_payload=True)
            except Exception as e:
                logger.info("FirstReportCheck: %s query failed (%s)", col, type(e).__name__)
                continue
            for h in res.points:
                u = (h.payload.get("url") or "").split("?")[0].rstrip("/")
                try:
                    t = float(h.payload.get(tf))
                except (TypeError, ValueError):
                    continue
                if h.score < _STORE_FLOOR or u == own or t > cutoff or u in seen or _LISTING.search(u):
                    continue
                seen.add(u)
                text = _clean(" \n".join(str(h.payload.get(k)) for k in fields if h.payload.get(k)))[:700]
                hits.append({"src": col, "score": round(h.score, 3), "url": h.payload.get("url", ""), "t": t, "text": text})
        hits.sort(key=lambda h: -h["score"])
        hits = hits[:_MAX_CANDIDATES]
        # A store keeps only headline + summary: open each hit for its first paragraphs.
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            for h in hits:
                try:
                    page = await client.get(h["url"], headers=_HDR)
                    paras = first_paragraphs(trafilatura.extract(page.text, include_comments=False) or "") if page.status_code == 200 else ""
                except Exception:
                    paras = ""
                if paras:
                    h["text"] = _clean(h["text"].split("\n")[0] + "\n" + paras)[:1700]
        return hits

    @staticmethod
    async def _resolve(client: httpx.AsyncClient, url: str) -> Optional[str]:
        """The publisher's URL behind a news.google.com link (China Breaks'
        agents/extractor.py _resolve_google_news, 2026-10-03)."""
        m = re.search(r"news\.google\.com/(?:rss/articles|articles|read)/([^?/]+)", url)
        if not m:
            return url
        gid = m.group(1)
        page = await client.get(f"https://news.google.com/rss/articles/{gid}", headers=_HDR)
        sg = re.search(r'data-n-a-sg="([^"]+)"', page.text)
        ts = re.search(r'data-n-a-ts="([^"]+)"', page.text)
        if not (sg and ts):
            return None
        req = ["garturlreq", [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None, None, None, 0, 1],
                              "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0], gid, int(ts.group(1)), sg.group(1)]
        resp = await client.post("https://news.google.com/_/DotsSplashUi/data/batchexecute",
                                 content="f.req=" + quote(json.dumps([[["Fbv4je", json.dumps(req)]]])),
                                 headers={**_HDR, "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"})
        inner = json.loads(resp.text.split("\n\n")[1])[:-2]
        real = json.loads(inner[0][2])[1]
        return real if isinstance(real, str) and real.startswith("http") else None

    async def _search_candidates(self, title: str, now: datetime, cutoff: float) -> list[dict]:
        before = (now - timedelta(hours=self._window)).date().isoformat()
        q = quote_plus(f"{_keywords(title)} before:{before}")
        out = []
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en", headers=_HDR)
            for item in re.findall(r"<item>(.*?)</item>", r.text, re.S)[:_MAX_CANDIDATES]:
                head = htmlmod.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", (re.search(r"<title>(.*?)</title>", item, re.S) or [None, ""])[1]))
                link = (re.search(r"<link>(.*?)</link>", item, re.S) or [None, ""])[1]
                pub = (re.search(r"<pubDate>(.*?)</pubDate>", item) or [None, ""])[1]
                try:
                    t = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    continue
                if t > cutoff:
                    continue
                lead, real = "", None
                try:
                    real = await self._resolve(client, link)
                    if real:
                        page = await client.get(real, headers=_HDR)
                        if page.status_code == 200:
                            lead = first_paragraphs(trafilatura.extract(page.text, include_comments=False) or "")
                except Exception:
                    pass
                if lead and not _LISTING.search(real or link):  # a headline alone is never evidence
                    out.append({"src": "google_news", "url": real or link, "t": t, "text": _clean(f"{head}\n{lead}")[:700]})
        return out

    async def _judge(self, ours: str, cands: list[dict]) -> dict:
        user = f"OUR ARTICLE:\n{ours}\n\n" + "\n\n".join(f"EARLIER ARTICLE {i + 1}:\n{c['text']}" for i, c in enumerate(cands))
        resp = await self._claude.messages.create(
            model=self._model, max_tokens=16000, system=self._system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}})
        usage = log_usage(resp, 'first-report check')
        if resp.stop_reason == "refusal":
            return {"stale": False, "same": [], "raw": None, "usage": usage}
        v = json.loads(next(b.text for b in resp.content if b.type == "text"))
        same = []
        for c in v["candidates"]:
            i = c["id"] - 1
            if c["relation"] != "SAME_FACT" or not (0 <= i < len(cands)):
                continue
            ok = (len(_norm(c["our_quote"]).split()) >= 3 and _norm(c["our_quote"]) in _norm(ours)
                  and _norm(c["their_quote"]) in _norm(cands[i]["text"]))
            same.append({"cand": i, "quotes_ok": ok})
        stale = any(x["quotes_ok"] for x in same) and not v["new_fact_quote"].strip()
        return {"stale": stale, "same": same, "raw": v, "usage": usage}

    async def check(self, url: str, url_hash: str, title: str, body: str, now: Optional[datetime] = None) -> dict:
        """{"verdict": "stale" | "fresh", "reason": str, "evidence": {...} | None, ...}. Fails open."""
        if not self.enabled:
            return {"verdict": "fresh", "reason": "first-report check off"}
        if not haiku.available():  # out of credit: cleared as before this check existed, not cached
            return {"verdict": "fresh", "reason": "Haiku out of credit"}
        if url_hash in self._cache:
            self._cache.move_to_end(url_hash)
            return self._cache[url_hash]
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=self._window)).timestamp()
        lead = lead_paragraph(_strip_title(body, title))
        ours = f"{title}\n{first_paragraphs(_strip_title(body, title))}"
        result = {"verdict": "fresh", "reason": "no earlier report of the same fact found", "evidence": None, "stages": []}
        try:
            for stage in ("store", "search"):
                cands = (await self._store_candidates(title, lead, url, cutoff) if stage == "store"
                         else await self._search_candidates(title, now, cutoff))
                if not cands:
                    continue
                j = await self._judge(ours, cands)
                result["stages"].append({"stage": stage, "candidates": [{k: c[k] for k in ("src", "url", "t")} for c in cands],
                                         "verdict": j["raw"], "usage": j["usage"]})
                if j["stale"]:
                    hit = cands[next(x["cand"] for x in j["same"] if x["quotes_ok"])]
                    age = round((now.timestamp() - hit["t"]) / 3600, 1)
                    result.update(verdict="stale", evidence={"url": hit["url"], "src": hit["src"], "age_hours": age},
                                  reason=f"already reported {age:.0f}h ago ({hit['url'][:120]})")
                    break
                if any(x["quotes_ok"] for x in j["same"]):
                    result["reason"] = "same fact reported earlier, but this article adds a major development"
                    break
        except Exception as e:
            haiku.note_error(e)
            logger.warning("FirstReportCheck: %s for %s -- treating as fresh", type(e).__name__, url)
            result = {"verdict": "fresh", "reason": "first-report check failed: %s" % type(e).__name__,
                      "evidence": None, "stages": result.get("stages", [])}
        self._cache[url_hash] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result
