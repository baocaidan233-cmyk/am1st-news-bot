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
    # Three different clocks, and the first version of this file used the
    # wrong one. candidate_max_age_hours (24h) is only the Notion QUERY
    # ceiling. Selection is tighter and tiered:
    #   fresh_hours (4h)            tier 1, which is the >=7.0 tier, draws
    #                               ONLY from candidates younger than this
    #   weekday_max_age_hours (12h) everything is dropped past this on a
    #   weekend_max_age_hours (24h) weekday, 24h at the weekend
    # Reporting 24h told the operator a 19h-old 7.0 had "4.1h left" when its
    # top-tier eligibility had ended 15 hours earlier and its eligibility of
    # any kind was already over. Everything recovered by hand on 2026-10-02
    # was 13-21h old: not expiring, long gone, and only publishable because
    # tools/publish_one.py bypasses select_batch.
    pub = cfg.publish
    tier1 = pub.fresh_hours
    today_ceiling = (pub.weekday_max_age_hours
                     if datetime.now(timezone.utc).weekday() < 5
                     else pub.weekend_max_age_hours)
    pool = await query_eligible_candidates(cfg)
    now = datetime.now(timezone.utc)
    rows = []
    for c in pool:
        if (c.llm_score or 0) < a.min_score:
            continue
        age = (now - c.published_at).total_seconds() / 3600
        rows.append((tier1 - age, age, c))
    rows.sort()
    in_tier1 = [r for r in rows if r[1] <= tier1]
    selectable = [r for r in rows if tier1 < r[1] <= today_ceiling]
    gone = [r for r in rows if r[1] > today_ceiling]
    print(f"\n候选池 {len(pool)} 条，>={a.min_score:.1f} 分的 {len(rows)} 条")
    print(f"  tier1 可见（<{tier1}h，这是 >=7.0 档唯一的入口）  {len(in_tier1)} 条")
    print(f"  仅低档可选（{tier1}-{today_ceiling}h）                    {len(selectable)} 条")
    print(f"  已出局（>{today_ceiling}h，正常周期发不了）              {len(gone)} 条")
    if not rows:
        return
    print(f"\n{'tier1剩余':>10} {'龄':>7} {'分':>5}  状态   标题")
    print("-" * 100)
    for left, age, c in rows:
        if age <= tier1:
            state, mark = "tier1", (" ←急" if left < 1 else "")
        elif age <= today_ceiling:
            state, mark = "仅低档", ""
        else:
            state, mark = "已出局", ""
        shown = f"{left:9.1f}h" if age <= tier1 else f"{'—':>10}"
        print(f"{shown} {age:6.1f}h {c.llm_score:5.1f}  {state}  {(c.title or '')[:56]}{mark}")
        # Full URL, never truncated: it is meant to be copied into
        # publish_one.py, and twice today a URL rebuilt from a truncated
        # display pointed at a different article than the one intended.
        print(f"{'':>22}{c.url}")
    print(f"\n发其中一条：.venv/bin/python tools/publish_one.py <url> --dry")


if __name__ == "__main__":
    asyncio.run(main())
