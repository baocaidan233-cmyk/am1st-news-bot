#!/usr/bin/env python3
"""What are we refusing, in bulk, and is any of it piling up?

    .venv/bin/python tools/reject_report.py --days 3

The one failure that actually costs this channel is not a bad post. It is a
story MAGA is talking about that our scorer keeps turning away, because nothing
downstream can recover a candidate that never got a score. That happened:
the Lindsay Clancy retrial was refused 66 times, 44 of them at exactly 4.0
against a 5.0 floor, and none were published, because an ordinary murder case
was on none of the retired prompt's 21 Core Themes.

A single refusal is invisible and should be. Ten refusals of the same story in
one day is a different object, and this is the report that makes that object
visible. It reads logs/main.log* (the rejection line carries URL, score, the
scorer's own reasons and the title) and clusters the titles by embedding — no
list of topics to be on, no model asked to judge anything, nothing to keep
up to date. A grievance nobody anticipated shows up as a cluster of refusals,
which is the whole point: it does not need to be on a list first.

Cost: embeddings only, cached on disk by title. A full history of ~10,000
refused titles is about $0.003; a day's increment is under a tenth of a cent.
That is deliberate — this report replaces a shadow scorer that cost $0.86/day
to answer the same question by paying a second model to re-judge everything.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(".env")
from core.config import load_config
from core.openai_client import create_openai_client

# "run_cycle: <url> scored 4.0, below threshold — <reasons> — <title>"
# The reasons group is optional: lines written before 2026-10-02 have only a
# title, and this report is still readable across that change.
LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d) \d\d:\d\d:\d\d,\d+ INFO main: run_cycle: (\S+) "
    r"scored ([\d.]+), below threshold — (.+)$")
CACHE = "logs/.reject_embeddings.json"


def _lines(path: str):
    op = gzip.open if path.endswith(".gz") else open
    try:
        with op(path, "rt", errors="replace") as fh:
            for line in fh:
                yield line.rstrip("\n")
    except OSError:
        return


def refused(days: int) -> list[dict]:
    """Most recent refusal per URL. Keyed by URL because the same candidate is
    re-offered across cycles and counting it twice would invent a cluster."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    out: dict[str, dict] = {}
    for path in sorted(glob.glob("logs/main.log*")):
        for line in _lines(path):
            m = LINE.match(line)
            if not m or m.group(1) < cutoff:
                continue
            day, url, score, tail = m.group(1), m.group(2), float(m.group(3)), m.group(4)
            # The reasons field, when present, is the scorer's own llm_comment
            # and is separated from the title by the same " — " it uses.
            reasons, title = ("", tail)
            if " — " in tail:
                head, rest = tail.split(" — ", 1)
                if "=" in head:
                    reasons, title = head, rest
            out[url] = {"day": day, "url": url, "score": score,
                        "reasons": reasons, "title": title.strip()}
    return [r for r in out.values() if len(r["title"]) > 10]


def published_sources() -> set[str]:
    """Source URLs that did reach Gettr. A story refused once and published
    later is not a miss, and leaving it in would pad every cluster."""
    seen = set()
    try:
        with open("logs/engagement_snapshots.jsonl", encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("src"):
                    seen.add(d["src"])
    except FileNotFoundError:
        pass
    return seen


async def embed(titles: list[str]) -> np.ndarray:
    cache: dict[str, list[float]] = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            cache = {}
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for t in titles]
    todo = [(k, t) for k, t in zip(keys, titles) if k not in cache]
    if todo:
        client = create_openai_client(load_config("config/config.yaml"))
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            resp = await client.embeddings.create(
                model="text-embedding-3-small", input=[t for _, t in chunk])
            for (k, _), d in zip(chunk, sorted(resp.data, key=lambda x: x.index)):
                cache[k] = d.embedding
        try:
            json.dump(cache, open(CACHE, "w", encoding="utf-8"))
        except Exception:
            pass
    V = np.array([cache[k] for k in keys], dtype=np.float32)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)


def cluster(V: np.ndarray, sim: float) -> list[int]:
    """Greedy single-pass assignment against cluster centroids. Deliberately
    not a dedup decision — 2026-09 measured that no cosine threshold separates
    真重复 from 同事件不同角度 on this corpus, so this only has to group a story
    with its own variants well enough for a human to recognise it."""
    centers: list[np.ndarray] = []
    counts: list[int] = []
    assign: list[int] = []
    for i in range(len(V)):
        v = V[i]
        best, score = -1, -1.0
        for ci, cv in enumerate(centers):
            s = float(v @ cv)
            if s > score:
                score, best = s, ci
        if score >= sim:
            assign.append(best)
            n = counts[best]
            centers[best] = (centers[best] * n + v) / (n + 1)
            centers[best] /= max(float(np.linalg.norm(centers[best])), 1e-9)
            counts[best] = n + 1
        else:
            centers.append(v.copy())
            counts.append(1)
            assign.append(len(centers) - 1)
    return assign


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--sim", type=float, default=0.78)
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()

    rows = refused(a.days)
    if not rows:
        print("窗口内没有被拒的候选")
        return
    pub = published_sources()
    rows = [r for r in rows if r["url"] not in pub]
    days = sorted({r["day"] for r in rows})
    print(f"\n{len(rows)} 条被打分拒掉、且从未发布 — {days[0]} 到 {days[-1]}")
    print(f"  分数分布: {dict(sorted(Counter(r['score'] for r in rows).items()))}")
    per = Counter(r["day"] for r in rows)
    print("  每天: " + "  ".join(f"{d[5:]} {per[d]}" for d in days))

    V = await embed([r["title"] for r in rows])
    assign = cluster(V, a.sim)
    groups: dict[int, list[dict]] = defaultdict(list)
    for r, k in zip(rows, assign):
        groups[k].append(r)
    print(f"  聚成 {len(groups)} 簇（余弦 {a.sim}）\n")

    today = days[-1]
    multi = [g for g in groups.values() if len(g) > 1]
    # Sorted by total size. The first draft of this sorted by "how many are
    # still being refused today", on the theory that a cluster that stopped is
    # over — and running it immediately showed why that is wrong: it put
    # today's two-item clusters above the Lindsay Clancy cluster's 15, burying
    # the exact case this report exists for. 13 refusals yesterday and one more
    # today is the bigger loss than two today. Still-active clusters are
    # flagged rather than reordered.
    multi.sort(key=len, reverse=True)
    print(f"=== 反复被拒的故事（{len(multi)} 簇有 2 条以上）===")
    print("    排序按总量；仍在累积的另行标注\n")
    for g in multi[:a.show]:
        n_today = sum(1 for r in g if r["day"] == today)
        g.sort(key=lambda r: r["day"])
        spread = Counter(r["day"][5:] for r in g)
        scores = sorted({r["score"] for r in g})
        flag = "  ← 今天仍在拒" if n_today else ""
        print(f"  {len(g):>3} 条  今天 {n_today:>2}  分数 {scores}{flag}")
        print(f"       逐日 {dict(spread)}")
        print(f"       {g[0]['title'][:88]}")
        if g[0]["reasons"]:
            print(f"       打分器的判断: {g[0]['reasons'][:88]}")
        print()

    print("=== 被拒的稿里，打分器给的理由分布 ===")
    with_reasons = [r for r in rows if r["reasons"]]
    if not with_reasons:
        print("  这批日志行还没带判断字段（2026-10-02 之后写的才有）")
    else:
        print(f"  {len(with_reasons)}/{len(rows)} 条带判断")
        # The grievance is appended after a "|" inside the LAST field, so a
        # field has to be cut at the "|" before counting it. Without that,
        # "who=none" and "who=none | 无诉求" count as two different values and
        # one value appears as two rows — a table that is wrong in exactly the
        # way that produces a wrong conclusion later.
        def fields(reasons: str) -> list[str]:
            return [p.split("|")[0].strip() for p in reasons.split(";")]

        for key in ("d=", "who="):
            vals = Counter(f for r in with_reasons for f in fields(r["reasons"])
                           if f.startswith(key))
            print(f"    {dict(vals.most_common())}")
        hard = Counter(f for r in with_reasons for f in fields(r["reasons"])
                       if f.startswith("DROP="))
        if hard:
            print(f"    硬拒: {dict(hard.most_common())}")
        nog = sum(1 for r in with_reasons if "无诉求" in r["reasons"])
        print(f"    其中「无诉求」{nog} 条 ({100*nog/len(with_reasons):.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
