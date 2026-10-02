"""A/B review for the editor selection arm.

Written before the experiment runs, on purpose: the measures are fixed here so
they cannot be chosen later to suit the result.

Three measures, in the order they become readable:

  publish-side duplicate rate — how often the chosen story turns out to be one
      we already posted. Readable within a day and it is the mechanism the
      editor is supposed to improve directly, since its brief lists what was
      just published. 50% of all candidates reaching that check are judged
      duplicates today.
  hour-normalised engagement TOTAL — likes + comments + reposts, over the median
      total of the same posting hour, on posts at least 12 hours old. Never
      compare raw totals across arms: posting hour is the largest single effect
      in this channel, worth about 2.2x between the best and worst.
  breakout rate — share of posts at or above twice the channel's own median.

2026-10-02: the engagement measure was likes ONLY until today, which would have
made this A/B not count. The optimisation target on this channel is the SUM of
all three and nothing else — not one of them weighted above the others. The three
are split out only to explain why a total moved, never as the result. (An earlier
draft of this comment said "comments especially"; that is wrong and the user
corrected it: "评论不是我最在意的，engage 总数我最在意。三个值都重要。")

Fixing it meant changing where the numbers come from, not just adding two fields:
measured against the live endpoint this file used, aux.post carries lkbpst and
shbpst but `cm` is None and aux.s_pst comes back empty, so comments are not
available there at all. They are in
logs/engagement_snapshots.jsonl, which the hourly collector has been writing all
along, so that is the source now — and it is the same estimator every other
measurement on this channel uses: the highest reading at or before a fixed 12h
age, divided by the median of that UTC hour.

Reads the arm from the published record's payload, so only posts written after
that field existed can be split.

    ./.venv/bin/python tools/editor_ab_review.py [days]
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import AsyncQdrantClient  # noqa: E402

from core.config import load_config  # noqa: E402



SNAPSHOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "logs", "engagement_snapshots.jsonl")


def engagement_by_source() -> dict[str, float]:
    """source url -> hour-normalised (likes + comments + reposts).

    Same estimator as every other engagement measurement on this channel: the
    highest total among readings at or before a fixed 12h age, so a post read
    at 3h is not compared against one read at 30h; a post with no reading
    inside that window falls back to its first reading; posts whose cdate
    predates the earliest snapshot are excluded, because the collector cannot
    have seen them young.
    """
    rows = []
    try:
        with open(SNAPSHOTS, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        print(f"no snapshots at {SNAPSHOTS} — is the engagement collector timer on?")
        return {}
    if not rows:
        return {}
    earliest = min(r["ts"] for r in rows)
    best: dict[str, dict] = {}
    first: dict[str, dict] = {}
    for r in rows:
        pid, cdate = r.get("post_id"), (r.get("cdate") or 0) / 1000.0
        if not pid or cdate < earliest or not r.get("src"):
            continue
        total = (r.get("lk") or 0) + (r.get("cm") or 0) + (r.get("sh") or 0)
        first.setdefault(pid, {"cdate": cdate, "src": r["src"], "total": total})
        if (r["ts"] - cdate) / 3600.0 > 12:
            continue
        cur = best.setdefault(pid, {"cdate": cdate, "src": r["src"], "total": -1})
        if total >= cur["total"]:
            cur["total"] = total
    for pid, r in first.items():
        best.setdefault(pid, r)
    mature = [v for v in best.values() if time.time() - v["cdate"] >= 12 * 3600]
    if not mature:
        return {}
    by_hour: dict[int, list[int]] = collections.defaultdict(list)
    for v in mature:
        by_hour[time.gmtime(v["cdate"]).tm_hour].append(v["total"])
    med = {h: st.median(x) for h, x in by_hour.items() if len(x) >= 3}
    overall = st.median([v["total"] for v in mature]) or 1
    out: dict[str, float] = {}
    for v in mature:
        base = med.get(time.gmtime(v["cdate"]).tm_hour, overall) or overall
        out[v["src"].split("?")[0]] = v["total"] / base
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

    norm = engagement_by_source()
    if not norm:
        print('没有可用的互动读数 —— 无法比较')
        return
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
