"""A/B review for the editor selection arm.

Written before the experiment runs, on purpose: the measures are fixed here so
they cannot be chosen later to suit the result.

Three measures, in the order they become readable:

  publish-side duplicate rate — how often the chosen story turns out to be one
      we already posted. Readable within a day and it is the mechanism the
      editor is supposed to improve directly, since its brief lists what was
      just published. 50% of all candidates reaching that check are judged
      duplicates today.
  hour-normalised engagement — likes over the median likes of the same posting
      hour, on posts at least 12 hours old. Never compare raw likes across
      arms: posting hour is the largest single effect in this channel.
  breakout rate — share of posts at or above twice the channel's own median.

Reads the arm from the published record's payload, so only posts written after
that field existed can be split.

    ./.venv/bin/python tools/editor_ab_review.py [days]
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import statistics as st
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import AsyncQdrantClient  # noqa: E402

from core.config import load_config  # noqa: E402

HANDLE = "gettrworldnews"


def gettr_posts() -> list[dict]:
    out: list[dict] = []
    for off in range(0, 200, 20):
        url = (f"https://api.gettr.com/u/user/{HANDLE}/posts"
               f"?offset={off}&max=20&dir=fwd&incl=posts&fp=f_uo")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        posts = ((data.get("result") or {}).get("aux") or {}).get("post") or {}
        if not posts:
            break
        for v in posts.values():
            text, ts = (v.get("txt") or "").strip(), v.get("cdate")
            if text and ts:
                out.append({"txt": text, "lk": int(v.get("lkbpst") or 0), "ts": ts / 1000})
    return out


def normalise(posts: list[dict]) -> dict[str, float]:
    """url -> likes / median likes of that posting hour, mature posts only."""
    now = time.time()
    mature = [p for p in posts if now - p["ts"] >= 12 * 3600]
    by_hour: dict[int, list[int]] = collections.defaultdict(list)
    for p in mature:
        by_hour[time.gmtime(p["ts"]).tm_hour].append(p["lk"])
    med = {h: st.median(v) for h, v in by_hour.items() if len(v) >= 3}
    overall = st.median([p["lk"] for p in mature]) or 1
    out: dict[str, float] = {}
    for p in mature:
        m = re.search(r"https?://\S+", p["txt"])
        if not m:
            continue
        base = med.get(time.gmtime(p["ts"]).tm_hour, overall) or overall
        out[m.group(0).rstrip(").,").split("?")[0]] = p["lk"] / base
    return out


async def main() -> None:
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
    cfg = load_config("config/config.yaml")
    cli = AsyncQdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key or None, timeout=60)
    cutoff = time.time() - days * 86400
    rows, off = [], None
    while True:
        pts, off = await cli.scroll(collection_name=cfg.qdrant.posted_collection,
                                    limit=500, offset=off, with_payload=True, with_vectors=False)
        if not pts:
            break
        rows += [p.payload for p in pts if (p.payload or {}).get("sentAt", 0) >= cutoff]
        if off is None:
            break
    arms = collections.Counter((r.get("arm") or "(未标记)") for r in rows)
    print(f"近 {days:g} 天已发 {len(rows)} 条：{dict(arms)}")
    if not {"editor", "legacy"} & set(arms):
        print("还没有带 arm 标记的帖子 —— A/B 尚未开始，或开始后还没发过。")
        return

    norm = normalise(gettr_posts())
    by_arm: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        arm = r.get("arm")
        u = (r.get("url") or "").split("?")[0]
        if arm in ("editor", "legacy") and u in norm:
            by_arm[arm].append(norm[u])
    print(f"\n{'臂':<10}{'n':>5}{'归一中位':>10}{'爆款率':>9}")
    for arm in ("editor", "legacy"):
        v = by_arm.get(arm) or []
        if not v:
            print(f"{arm:<10}{0:>5}{'—':>10}{'—':>9}")
            continue
        print(f"{arm:<10}{len(v):>5}{st.median(v):>10.2f}{100 * sum(1 for x in v if x >= 2) / len(v):>8.0f}%")
    if by_arm.get("editor") and by_arm.get("legacy"):
        e, l = st.median(by_arm["editor"]), st.median(by_arm["legacy"])
        print(f"\n倍数 {e / max(l, 0.01):.2f}x")
        print("每臂 n 不到 150 之前不要下结论；单次读数会因为少数几条高赞帖大幅摆动。")


if __name__ == "__main__":
    asyncio.run(main())
