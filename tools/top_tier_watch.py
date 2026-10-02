#!/usr/bin/env python3
"""Top-tier candidates in the pool that are ageing out unpublished.

    .venv/bin/python tools/top_tier_watch.py [--min-score 7.0]

The channel's top band earns the most and goes out least. Measured 2026-10-02
over 14 days: candidates at 7.0 and 8.0 publish 48% of the time, 76 of them aged
past the publish window unpublished, and only 2.6% of those 76 were stories we
had already covered from another source. The mechanism is in the ranking log — a
candidate that loses a batch gets older, the freshness penalty grows with age
(median 1.63 at the 8.0 tier against 1.10 at 6.0), and being older makes losing
the next batch likelier. One 8.0 candidate lost three batches running with its
penalty going 1.02 -> 2.51 -> 2.60.

Why a watch rather than a fix to the ranker: a story has to be caught while it
is still publishable. The three best examples found today were 7.0, 7.0 and 7.5
on a beat this channel measures 2.35x and 1.94x on — Minnesota welfare and voter
fraud, Dearborn — and by the time they were found they were 42 to 52 hours old,
which the freshness rule puts out of reach. Nothing recovers those. The only
useful instrument is one that shows them with hours left, not days.

Pair with tools/publish_one.py, which publishes one by URL through every gate
main_publish.py runs.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(".env")
from core.config import load_config                      # noqa: E402
from core.notion_candidates import query_eligible_candidates  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=7.0)
    a = ap.parse_args()

    cfg = load_config("config/config.yaml")
    ceiling = cfg.publish.candidate_max_age_hours
    pool = await query_eligible_candidates(cfg)
    now = datetime.now(timezone.utc)
    rows = []
    for c in pool:
        if (c.llm_score or 0) < a.min_score:
            continue
        age = (now - c.published_at).total_seconds() / 3600
        rows.append((ceiling - age, age, c))
    rows.sort()
    live = [r for r in rows if r[0] > 0]
    print(f"\n候选池 {len(pool)} 条，其中 >={a.min_score:.1f} 分的 {len(rows)} 条，"
          f"还在 {ceiling}h 窗口内的 {len(live)} 条")
    if not live:
        print("窗口内没有顶档稿 —— 要么都发了，要么都已经过期")
        return
    print(f"\n{'剩余':>7} {'龄':>7} {'分':>5}  标题")
    print("-" * 96)
    for left, age, c in live:
        mark = " ←急" if left < 4 else ""
        print(f"{left:6.1f}h {age:6.1f}h {c.llm_score:5.1f}  {(c.title or '')[:62]}{mark}")
        # Full URL, never truncated: it is meant to be copied into
        # publish_one.py, and twice today a URL rebuilt from a truncated
        # display pointed at a different article than the one intended.
        print(f"{'':>22}{c.url}")
    print(f"\n发其中一条：.venv/bin/python tools/publish_one.py <url> --dry")


if __name__ == "__main__":
    asyncio.run(main())
