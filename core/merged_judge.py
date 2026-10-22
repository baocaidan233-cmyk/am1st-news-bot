"""Ingestion's merged judge (Linear AI-69, user 10-16..10-21).

One call per article that cleared the score gate answers three questions
(prompts/merged_judge_prompt.txt, prompt v8 of the offline tests):

  1. which event in the event store the article's main news belongs to, or a
     new one;
  2. compared with OUR POSTS ONLY, does it tell our readers anything new
     ("重复 = 读者从我们帖子里已经知道了"). NOTHING_NEW and NOT_WORTH are
     dropped (user 10-20), but only when the answer names the post that
     already said it (covered_by, user 10-20): no named post, no drop;
  3. is the same update already waiting in Notion, not yet posted? Then the
     article is only recorded in the event store and does not get its own
     Notion row (user 10-19: one row per update, not per event).

What the judge is shown is found by code, no model involved:
  - our posts of the last window_hours: top post_top_k by cosine between this
    article and each post's source article (>= post_cos_floor), plus top
    post_top_k by keyword TF-IDF (>= post_kw_floor). A 0.60 floor found the
    matching post for 99 of 149 audited repeats; these two lists together found
    149 (10-16).
  - candidate events: top event_top_k events by cosine (>= event_cos_floor),
    plus the events those posts sit in, each shown by up to 3 filed articles
    and our posts in it. On the diesel-deal storyline the main event reached
    the candidates for 22/24 articles instead of 8 (10-16).
  - the Notion queue: event-store points published within queue_hours that
    are in Notion (in_notion is not False) and not posted, top by cosine plus
    top by keyword, plus articles added earlier in this same cycle.

Measured offline (10-16..10-20, Haiku default effort): real developments
wrongly dropped 0/15 in five runs; never-posted stories wrongly dropped 0-1/72;
same update recognised 45/50; user's spot checks 12/12 and 60-article event
audit 59/60. A replay of 10-20's live feeds: 778 judged, 147 dropped, 119 only
recorded in the event store, median 4.1 s per call.

Haiku broken or out of credit -> the same prompt on openai.chat_model
(gpt-4o-mini), user 10-16 and 10-20 ("等真用上它兜底的时候再说"). Both failing
-> the article is kept, in no event (a new one): failing open loses no news."""
from __future__ import annotations

import collections
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import haiku
from core.config import AppConfig
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

_RELATIONS = ["NOTHING_NEW", "NOT_WORTH", "LOW_VALUE", "HAS_NEW", "CANNOT_TELL", "NO_POSTS"]
_DROP = ("NOTHING_NEW", "NOT_WORTH")
_MAX_EVENTS = 15
_MAX_QUEUE = 10
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["candidate_main_news", "event", "event_reason", "relation", "new_info", "reason",
                 "same_update_as", "same_update_reason", "covered_by"],
    "properties": {
        "candidate_main_news": {"type": "string"},
        "event": {"type": "string", "enum": [f"E{i}" for i in range(1, _MAX_EVENTS + 1)] + ["NEW"]},
        "event_reason": {"type": "string"},
        "relation": {"type": "string", "enum": _RELATIONS},
        "new_info": {"type": "string"},
        "reason": {"type": "string"},
        "same_update_as": {"type": "string", "enum": [f"Q{i}" for i in range(1, _MAX_QUEUE + 1)] + ["NONE"]},
        "same_update_reason": {"type": "string"},
        "covered_by": {"type": "array", "items": {"type": "string"}},
    },
}

_STOP = set(
    "that this with from have will said says were been their they them about after into over more than what "
    "when which while would could should there these those also just only other some such your president trump "
    "trumps news says told week".split())
_TAG = re.compile(r"<[^>]+>")


def _toks(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z'-]{2,}", (text or "").lower()) if w not in _STOP]


def _flat(text: str) -> str:
    return " ".join(_TAG.sub(" ", text or "").split())


def _post_text(content: str) -> str:
    return _flat((content or "").split("https://")[0])[:500]


def _split(text: str) -> tuple[str, str]:
    head, _, rest = (text or "").partition("\n")
    return head.strip()[:200], _flat(rest)[:400]


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


class _Tfidf:
    """Keyword similarity over a small set of texts (our ~600 posts, or the queue)."""

    def __init__(self, docs: dict):
        counts = {k: collections.Counter(_toks(t)) for k, t in docs.items()}
        self._df = collections.Counter()
        for c in counts.values():
            self._df.update(set(c))
        self._n = max(1, len(counts))
        self._vecs = {k: self._weigh(c) for k, c in counts.items()}
        self._norms = {k: math.sqrt(sum(x * x for x in v.values())) or 1.0 for k, v in self._vecs.items()}

    def _weigh(self, c: collections.Counter) -> dict:
        return {w: (1 + math.log(n)) * math.log(self._n / (1 + self._df.get(w, 0)) + 1) for w, n in c.items()}

    def scores(self, text: str) -> dict:
        q = self._weigh(collections.Counter(_toks(text)))
        qn = math.sqrt(sum(x * x for x in q.values())) or 1.0
        return {k: sum(q[w] * v.get(w, 0) for w in q) / (qn * self._norms[k]) for k, v in self._vecs.items()}


def _top(scores: dict, k: int, floor: float) -> list:
    return [key for key in sorted(scores, key=lambda key: -scores[key])[:k] if scores[key] >= floor]


def _hm(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M")


def render(candidate: dict, events: list[dict], extra_posts: list[dict], queue: list[dict]) -> str:
    """The judge's input, laid out exactly as in the offline tests (rel_run_v8.py)."""
    n = 0
    lines = ["NEW ARTICLE", f"Headline: {candidate['headline']}", f"Summary: {candidate['summary'] or '(none)'}", ""]
    for i, e in enumerate(events, 1):
        lines.append(f"E{i}")
        for a in e["articles"]:
            lines.append(f"  Filed article: {a['headline']} | {a['summary']}")
        if e["our_posts"]:
            for p in e["our_posts"]:
                n += 1
                lines.append(f"  OUR POST P{n} ({p['time']} UTC): {p['post']}")
        else:
            lines.append("  OUR POSTS: none")
        lines.append("")
    if extra_posts:
        lines.append("OUR RECENT POSTS FOUND BY SEARCH (not filed under the events above; most may be unrelated)")
        for p in extra_posts:
            n += 1
            lines.append(f"  OUR POST P{n} ({p['time']} UTC): {p['post']}")
        lines.append("")
    if queue:
        lines.append("OUR QUEUE (articles waiting to be posted, not posted yet; found by search, most may be unrelated)")
        for i, q in enumerate(queue, 1):
            lines.append(f"  Q{i}: {q['headline']} | {q['summary']}")
    else:
        lines.append("OUR QUEUE: nothing similar")
    lines.append("")
    return "\n".join(lines)


@dataclass
class Verdict:
    action: str                # "drop" | "fold" (event store only) | "keep"
    event_id: str | None       # existing event chosen, or None for a new one
    relation: str
    model: str
    fold_url: str = ""
    record: dict = field(default_factory=dict)  # everything that goes to the decision log


class MergedJudge:
    def __init__(self, config: AppConfig, event_store, posted_store, qdrant_store, stuck_versions=None) -> None:
        self._cfg = config.merged_judge
        self._events = event_store
        self._posted = posted_store
        self._articles = qdrant_store
        self._stuck = stuck_versions
        self._prompt = Path(self._cfg.prompt_file).read_text(encoding="utf-8")
        self._key = self._cfg.api_key
        self._openai = create_openai_client(config)
        self._fallback_model = config.openai.chat_model
        self.ready = False
        if not self._key:
            logger.warning("MergedJudge: FIRST_REPORT_ANTHROPIC_API_KEY not set -- every call goes to %s", self._fallback_model)

    async def prepare_cycle(self) -> bool:
        """Loads once per cycle what every article is compared with. False (and
        the cycle keeps everything, logged) when any of it cannot be read."""
        self.ready = False
        try:
            posts = await self._posted.recent_posts(self._cfg.window_hours)
            vecs = await self._articles.vectors_for(sorted({p["urlHash"] for p in posts if p.get("urlHash")}))
            post_events = await self._events.event_ids_for_urls([p["url"] for p in posts if p.get("url")])
            queue = await self._events.recent_points(self._cfg.queue_hours)
        except Exception:
            logger.exception("MergedJudge: could not load posts / event store -- the judge is off this cycle")
            return False
        self._posts = {p["urlHash"]: p for p in posts if p.get("urlHash")}
        self._post_vec = {h: _unit(v) for h, v in vecs.items() if h in self._posts}
        self._post_event = post_events
        self._post_tfidf = _Tfidf({h: f"{p.get('title') or ''} {_post_text(p.get('content'))}" for h, p in self._posts.items()})
        posted_urls = {p["url"] for p in posts if p.get("url")}
        self._queue = [{**q, "vec": _unit(q["vec"])} for q in queue
                       if q.get("in_notion", True) is not False and q.get("url") not in posted_urls]
        self.ready = True
        logger.info("MergedJudge: %d posts (%d with vectors), %d queued articles", len(self._posts), len(self._post_vec), len(self._queue))
        return True

    def note_added(self, url: str, text: str, embedding, in_notion: bool, event_id: str) -> None:
        """An article this cycle has just kept: later articles in the same cycle
        see it in the queue (when it went to Notion)."""
        if self.ready and in_notion:
            self._queue.append({"url": url, "text": text, "vec": _unit(embedding), "event_id": event_id, "in_notion": True})

    def forget(self, url: str) -> None:
        """A queued row replaced by a stronger version this cycle (main.py,
        10-22): later articles must not fold into it."""
        if self.ready:
            self._queue = [q for q in self._queue if q.get("url") != url]

    def posts_since(self, title: str, description: str, embedding, since: float) -> list[dict]:
        """Our posts sent after `since` that this same search would show the
        judge for this article (cosine to the post's source article >=
        post_cos_floor, or keywords >= post_kw_floor, top post_top_k of each),
        searched among those later posts only. main_publish.py (2026-10-21,
        AI-24, user 10-21): an article judged at ingestion is asked again only
        when one of these exists -- the judge already saw everything older."""
        later = {h for h, p in self._posts.items() if p.get("sentAt", 0) > since}
        if not later:
            return []
        cfg = self._cfg
        text = f"{(title or '').strip()[:200]} {_flat(description)[:400]}"
        v = _unit(embedding)
        cos = {h: float(self._post_vec[h] @ v) for h in later if h in self._post_vec}
        kw = {h: s for h, s in self._post_tfidf.scores(text).items() if h in later}
        found = dict.fromkeys(_top(cos, cfg.post_top_k, cfg.post_cos_floor) + _top(kw, cfg.post_top_k, cfg.post_kw_floor))
        return [self._posts[h] for h in found]

    async def judge(self, url: str, title: str, description: str, embedding) -> Verdict:
        cand = {"headline": (title or "").strip()[:200], "summary": _flat(description)[:400]}
        text = f"{cand['headline']} {cand['summary']}"
        v = _unit(embedding)
        cfg = self._cfg

        # Our posts: by cosine to their source article, and by keywords.
        cos = {h: float(pv @ v) for h, pv in self._post_vec.items()}
        found = list(dict.fromkeys(_top(cos, cfg.post_top_k, cfg.post_cos_floor)
                                   + _top(self._post_tfidf.scores(text), cfg.post_top_k, cfg.post_kw_floor)))
        # Candidate events: nearest events, plus the events those posts sit in.
        nearest = await self._events.top_events(embedding, cfg.event_top_k, cfg.event_cos_floor)
        event_ids = list(dict.fromkeys([e for e, _ in nearest]
                                       + [self._post_event[self._posts[h]["url"]] for h in found
                                          if self._posts[h].get("url") in self._post_event]))[:_MAX_EVENTS]
        articles = await self._events.event_articles(event_ids, per_event=3) if event_ids else {}
        by_event = collections.defaultdict(list)
        for h, p in sorted(self._posts.items(), key=lambda kv: kv[1].get("sentAt", 0)):
            ev = self._post_event.get(p.get("url"))
            if ev in event_ids:
                by_event[ev].append(p)
        events, shown = [], set()
        for e in event_ids:
            ops = [{"time": _hm(p["sentAt"]), "post": _post_text(p.get("content"))} for p in by_event.get(e, [])]
            shown |= {p["urlHash"] for p in by_event.get(e, [])}
            events.append({"articles": [dict(zip(("headline", "summary"), _split(t))) for t in articles.get(e, [])],
                           "our_posts": ops, "_id": e})
        extra = [self._posts[h] for h in found if h not in shown]
        extra_posts = [{"time": _hm(p["sentAt"]), "post": _post_text(p.get("content"))}
                       for p in sorted(extra, key=lambda p: p.get("sentAt", 0))]
        n_posts = sum(len(e["our_posts"]) for e in events) + len(extra_posts)

        # The Notion queue: nearest by cosine and by keywords.
        qitems = [q for q in self._queue if q.get("url") != url]
        qcos = {i: float(q["vec"] @ v) for i, q in enumerate(qitems)}
        qkw = _Tfidf({i: q.get("text", "") for i, q in enumerate(qitems)}).scores(text) if qitems else {}
        picked = list(dict.fromkeys(_top(qcos, cfg.post_top_k, cfg.post_cos_floor)
                                    + _top(qkw, cfg.post_top_k, cfg.post_kw_floor)))[:_MAX_QUEUE]
        queue = [{**dict(zip(("headline", "summary"), _split(qitems[i].get("text", "")))), "_url": qitems[i].get("url", "")}
                 for i in picked]

        user = render(cand, events, extra_posts, queue)
        started = time.monotonic()
        answer, model = None, "none"
        if self._key:
            answer = await haiku.ask(self._key, self._prompt, user, SCHEMA, cache=True)
            model = haiku.MODEL if answer is not None else model
        if answer is None:
            answer = await self._ask_openai(user)
            model = self._fallback_model if answer is not None else "none"
        seconds = round(time.monotonic() - started, 1)

        record = {"url": url, "title": cand["headline"], "model": model, "seconds": seconds,
                  "n_events": len(events), "n_posts": n_posts, "n_queue": len(queue)}
        if not answer or answer.get("relation") not in _RELATIONS:
            record.update({"action": "keep", "relation": "ERROR"})
            return Verdict("keep", None, "ERROR", model, record=record)

        relation = answer["relation"]
        covered = [p for p in answer.get("covered_by") or []
                   if re.fullmatch(r"P\d+", p or "") and 1 <= int(p[1:]) <= n_posts]
        ev = answer.get("event", "NEW")
        event_id = events[int(ev[1:]) - 1]["_id"] if ev and ev != "NEW" and int(ev[1:]) <= len(events) else None
        same = answer.get("same_update_as", "NONE")
        fold_url = queue[int(same[1:]) - 1]["_url"] if same and same != "NONE" and int(same[1:]) <= len(queue) else ""
        if fold_url and self._stuck is not None:
            # A queued version the publish cycle cannot publish is not the
            # version of this update (core/redis_store.StuckVersions).
            from core.hashing import sha256_url_hash
            if await self._stuck.reason(sha256_url_hash(fold_url)):
                fold_url = ""

        if relation in _DROP and covered:
            action = "drop"
        elif fold_url:
            action = "fold"
        else:
            action = "keep"
        record.update({"action": action, "relation": relation, "event": ev, "event_id": event_id,
                       "same_update_as": same, "fold_url": fold_url, "covered_by": covered,
                       "released_by_check": relation in _DROP and not covered,
                       "main_news": answer.get("candidate_main_news", "")[:200], "new_info": answer.get("new_info", "")[:300],
                       "reason": answer.get("reason", "")[:400], "event_reason": answer.get("event_reason", "")[:300],
                       "same_update_reason": answer.get("same_update_reason", "")[:300]})
        return Verdict(action, event_id, relation, model, fold_url, record)

    async def _ask_openai(self, user: str) -> dict | None:
        try:
            resp = await self._openai.chat.completions.create(
                model=self._fallback_model, temperature=0, max_tokens=1200,
                response_format={"type": "json_schema", "json_schema": {"name": "merged_judge", "strict": True, "schema": SCHEMA}},
                messages=[{"role": "system", "content": self._prompt}, {"role": "user", "content": user}])
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception:
            logger.exception("MergedJudge: %s could not answer either", self._fallback_model)
            return None


def log_verdict(config: AppConfig, record: dict) -> None:
    try:
        with open(config.merged_judge.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"logged_at": int(time.time()), **record}, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("MergedJudge: could not write the decision log")
