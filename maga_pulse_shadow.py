#!/usr/bin/env python3
"""Shadow-mode MAGA cross-channel consensus signal. Observes only — this
process is not imported by main.py or main_publish.py, writes nothing to
Notion, and changes no selection or ranking decision. Removing it changes
production output by exactly zero bytes.

Why: on 2026-09-19 every content signal invented from reading our own top
posts came back null (subject fame p=0.89, enemy p=0.07, all writer copy
metrics p=1.00). The one thing that did separate was external and
observable rather than invented -- whether several MAGA news channels were
independently covering the same story. Measured over 09-15..09-19 on five
news-oriented channels (GlobalPulse deliberately excluded: 217 posts/day,
93% video reposts, its topic mix is a volume strategy and not evidence of
what the audience wants):

    topics >=2 channels covered   39   -> in our candidate pool 27 (69%)
    topics >=3 channels covered    5   -> in our pool            5 (100%)
    topics >=4 channels covered    3   -> in our pool            3 (100%)
    single-channel control       334   -> in our pool          156 (47%)

So fetching is NOT the bottleneck for genuinely-consensus stories; we hold
them. Of those 27 we published 10, and their scores were {5: 6, 6: 17,
7: 2, 8: 2} -- the scorer does not recognise a story that five MAGA
channels are simultaneously running. That is the hypothesis this collects
evidence for or against.

n was 39 topics. Too small to act on, and today already produced three
signals that looked real at small n and evaporated under a proper test, so
this accumulates before anything touches scoring.

What a run does:
  1. pull the newest posts from the five channels into a rolling cache
  2. pull candidates created in the last CANDIDATE_WINDOW_H from Notion
     (read-only, and deliberately NOT query_eligible_candidates() -- the
     unpublished and below-threshold ones are the whole point)
  3. embed both sides (cached, so each post/candidate is embedded once)
  4. for each candidate record how many DISTINCT channels are running a
     matching story, plus the top matches with their raw cosines

Writes logs/maga_consensus_shadow.jsonl, one row per candidate per CHANGE
(same convention as engagement_collector.py -- re-emitting hundreds of
unchanged rows hourly would bury the signal in its own noise). Because
consensus builds over hours, the change rows are also a record of WHEN a
story reached n channels, which a single end-of-day pull cannot recover.

Deliberately stores the top matches with raw cosine rather than just a
verdict at MATCH_THRESHOLD: the 0.62 threshold came from one afternoon's
calibration on one window and will want re-tuning, and re-tuning has to be
possible offline from this file without re-fetching a window that Gettr's
200-post cap has by then already dropped.

API notes (see reference_gettr_undocumented_api): posts endpoint needs
fp=f_uo, `aux.post` is a dict keyed by post id rather than a list, and
offset >= 200 returns [] -- which is exactly why this has to run on a
timer and accumulate instead of being re-derived later.
"""
from __future__ import annotations

import array
import asyncio
import base64
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from core.config import load_config
from core.hashing import cosine_similarity
from core.openai_client import create_openai_client

CHANNELS = ["gatewaypundit", "newsmax", "rsbnetwork", "jfradioshow", "stevebannon"]

# Calibrated 2026-09-19 against hand-read cluster pairs on text-embedding-3-
# small, same model the dedup layers use. 0.62 is well below dedup's 0.85
# "same story" bar on purpose: two channels covering one event write nothing
# alike (a Bannon monologue line vs a Newsmax headline), so this only has to
# find "same subject matter", not "same article".
MATCH_THRESHOLD = 0.62
POST_WINDOW_H = 96        # how far back the channel-post cache reaches
CANDIDATE_WINDOW_H = 30   # publish.candidate_max_age_hours (24) plus margin
PAIR_MAX_GAP_H = 48       # a channel post this much older/newer than the candidate is a different news cycle
TOP_MATCHES = 5
POSTS_PER_CHANNEL = 60    # newest only; the cache supplies the older tail
MIN_BODY_CHARS = 25       # a bare link or "LIVE:" with no body embeds to noise

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
OUT_PATH = LOG_DIR / "maga_consensus_shadow.jsonl"
CACHE_PATH = LOG_DIR / ".maga_pulse_cache.json"
INCL = urllib.parse.quote("posts|stats|userinfo|shared|liked")
UA = {"User-Agent": "Mozilla/5.0"}


def pack(vec: list[float]) -> str:
    """float32 + base64. The cache holds ~1500 vectors of 1536 dims; as JSON
    numbers that is ~100MB rewritten hourly, as packed floats it is ~9MB and
    loses nothing (the source values are float32 anyway)."""
    return base64.b64encode(array.array("f", vec).tobytes()).decode("ascii")


def unpack(blob: str) -> list[float]:
    arr = array.array("f")
    arr.frombytes(base64.b64decode(blob))
    return list(arr)


def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text or "").strip()


def fetch_channel_posts(handle: str, want: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    offset = 0
    while len(rows) < want:
        url = (f"https://api.gettr.com/u/user/{handle}/posts"
               f"?offset={offset}&max=20&dir=fwd&incl={INCL}&fp=f_uo")
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.load(resp).get("result", {})
        listing = result.get("data", {}).get("list", [])
        if not listing:
            break
        aux = result.get("aux", {})
        posts = aux.get("post") or {}
        stats = aux.get("s_pst") or {}
        for item in listing:
            pid = (item.get("activity") or {}).get("tgt_id") or item.get("_id")
            post = posts.get(pid)
            if not isinstance(post, dict) or pid in seen:
                continue
            seen.add(pid)
            s = stats.get(pid) or {}
            body = strip_urls(f"{post.get('ttl') or ''} {post.get('txt') or ''}")
            rows.append({
                "id": pid, "ch": handle, "cdate": post.get("cdate"),
                "body": body[:600],
                "lk": s.get("lkbpst", 0), "cm": s.get("cm", 0), "sh": s.get("shbpst", 0),
            })
        offset += len(listing)
        time.sleep(0.4)
    return rows


def _plain(prop: dict):
    kind = prop.get("type")
    value = prop.get(kind)
    if isinstance(value, list):
        return "".join(x.get("plain_text", "") for x in value)
    if isinstance(value, dict):
        return value.get("start") or ""
    return value


async def fetch_candidates(config) -> list[dict]:
    """Read-only Notion query. Intentionally unfiltered on send_status,
    llm_score and extraction_failed: the question is whether consensus
    stories are being scored low and dropped, so the dropped ones have to
    be in the sample."""
    notion = config.notion
    props = notion.candidate_props
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=CANDIDATE_WINDOW_H)).isoformat()
    headers = {
        "Authorization": f"Bearer {notion.candidate_key}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    rows: list[dict] = []
    cursor = None
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            body = {"filter": {"timestamp": "created_time",
                               "created_time": {"on_or_after": cutoff}},
                    "page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            resp = await client.post(
                f"https://api.notion.com/v1/databases/{notion.candidate_db_id}/query",
                headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            for page in data.get("results", []):
                p = page["properties"]
                title = _plain(p.get(props.title, {})) or ""
                if not title:
                    continue
                rows.append({
                    "id": page["id"],
                    "url": _plain(p.get(props.url, {})) or "",
                    "title": title[:200],
                    "text": f"{title} {(_plain(p.get(props.description, {})) or '')[:600]}",
                    "score": (p.get(props.llm_score) or {}).get("number"),
                    "heat": (p.get(props.heat_score) or {}).get("number"),
                    "sent": bool((p.get(props.send_status) or {}).get("checkbox")),
                    "published_at": _plain(p.get(props.published_at, {})) or "",
                    "created": page["created_time"][:19],
                })
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
    return rows


def load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"posts": {}, "cands": {}, "last": {}}


def to_unix(value) -> float:
    """Notion dates are ISO strings, Gettr cdate is epoch ms."""
    if isinstance(value, (int, float)):
        return value / 1000.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


async def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    load_dotenv(ROOT / ".env")
    config = load_config()
    client = create_openai_client(config)
    cache = load_cache()

    # --- 1. channel posts -------------------------------------------------
    posts: dict[str, dict] = {
        k: v for k, v in cache.get("posts", {}).items()
        if to_unix(v.get("cdate")) >= now - POST_WINDOW_H * 3600
    }
    fetched = 0
    for handle in CHANNELS:
        try:
            for row in fetch_channel_posts(handle, POSTS_PER_CHANNEL):
                if to_unix(row["cdate"]) < now - POST_WINDOW_H * 3600:
                    continue
                if len(strip_urls(row["body"])) < MIN_BODY_CHARS:
                    continue
                fetched += 1
                if row["id"] not in posts:
                    posts[row["id"]] = row
        except Exception as e:
            # One dead channel must not take the run down -- the whole point
            # is an unbroken hourly series, and a gap is worse than a channel
            # missing from one hour's denominator.
            print(f"channel {handle} failed: {e}", file=sys.stderr)

    # --- 2. candidates ----------------------------------------------------
    try:
        cands = await fetch_candidates(config)
    except Exception as e:
        print(f"notion query failed: {e}", file=sys.stderr)
        return 1

    # --- 3. embed whatever is new ----------------------------------------
    sem = asyncio.Semaphore(8)

    async def embed(text: str) -> list[float] | None:
        async with sem:
            try:
                r = await client.embeddings.create(
                    model=config.openai.embedding_model, input=text[:2000])
                return r.data[0].embedding
            except Exception as e:
                print(f"embed failed: {e}", file=sys.stderr)
                return None

    cand_vec_cache = cache.get("cands", {})
    need_posts = [p for p in posts.values() if "vec" not in p]
    need_cands = [c for c in cands if c["id"] not in cand_vec_cache]
    if need_posts:
        for p, v in zip(need_posts, await asyncio.gather(*[embed(p["body"]) for p in need_posts])):
            if v:
                p["vec"] = pack(v)
    if need_cands:
        for c, v in zip(need_cands, await asyncio.gather(*[embed(c["text"]) for c in need_cands])):
            if v:
                cand_vec_cache[c["id"]] = pack(v)

    live_posts = [p for p in posts.values() if "vec" in p]
    post_vecs = [(p, unpack(p["vec"])) for p in live_posts]

    # --- 4. match ---------------------------------------------------------
    last = cache.get("last", {})
    written = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for c in cands:
            blob = cand_vec_cache.get(c["id"])
            if not blob:
                continue
            cvec = unpack(blob)
            ctime = to_unix(c["published_at"]) or to_unix(c["created"])
            matches = []
            for p, pvec in post_vecs:
                if ctime and abs(to_unix(p["cdate"]) - ctime) > PAIR_MAX_GAP_H * 3600:
                    continue
                sim = cosine_similarity(cvec, pvec)
                if sim >= MATCH_THRESHOLD:
                    matches.append((sim, p))
            matches.sort(key=lambda m: -m[0])
            channels = sorted({p["ch"] for _, p in matches})
            sig = [len(channels), len(matches)]
            if last.get(c["id"]) == sig:
                continue
            last[c["id"]] = sig
            f.write(json.dumps({
                "ts": now,
                "cand_id": c["id"],
                "url": c["url"],
                "title": c["title"],
                "llm_score": c["score"],
                "heat_score": c["heat"],
                "sent": c["sent"],
                "published_at": c["published_at"],
                "created": c["created"],
                "n_channels": len(channels),
                "n_posts": len(matches),
                "channels": channels,
                "max_sim": round(matches[0][0], 4) if matches else 0.0,
                "top": [{"ch": p["ch"], "sim": round(s, 4), "lk": p["lk"],
                         "cdate": p["cdate"], "body": p["body"][:120]}
                        for s, p in matches[:TOP_MATCHES]],
            }, ensure_ascii=False) + "\n")
            written += 1

    # --- 5. persist -------------------------------------------------------
    live_cand_ids = {c["id"] for c in cands}
    CACHE_PATH.write_text(json.dumps({
        "posts": posts,
        "cands": {k: v for k, v in cand_vec_cache.items() if k in live_cand_ids},
        "last": {k: v for k, v in last.items() if k in live_cand_ids},
    }), encoding="utf-8")

    consensus = sum(1 for c in cands if last.get(c["id"], [0])[0] >= 2)
    print(f"{fetched} channel posts fetched ({len(live_posts)} in window), "
          f"{len(cands)} candidates, {consensus} with >=2 channels, "
          f"{written} changed rows written")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
