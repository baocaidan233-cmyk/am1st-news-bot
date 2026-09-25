# -*- coding: utf-8 -*-
"""One command that produces the 题材配比控制器 review.

Written 2026-09-25, the day the controller shipped (353af32), while the
baseline numbers were still in hand. Run it on the VM:

    cd ~/AM1ST && ./.venv/bin/python tools/topic_mix_review.py

Read-only: engagement_snapshots.jsonl, the ranking decision log, and a
Notion query. It writes nothing and touches no service.

Every number here is computed the way the baseline was, and the three
choices that matters most are:
  * mature = older than 12h. 12h is ~93% of final engagement; comparing a
    fresh post against a settled one is the single easiest way to get a
    wrong answer out of this data.
  * breakout = at or above 2x the channel's OWN median in the window, not
    an absolute like 100 likes. The median moves with the news cycle.
  * the before/after cut uses each post's likes at a FIXED 12h age, from
    the hourly snapshots, not its final value -- the post-change sample is
    systematically younger, and final values would read that as a loss.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import random
import statistics as st
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHIP_TS = dt.datetime(2026, 9, 25, 16, 13, tzinfo=dt.timezone.utc).timestamp()

# Recorded 2026-09-25 from n=534 covering 09-16..09-25, before the controller
# could have had any effect. See memory project_am1st_topic_mix.
BASE_N = 534
BASE_BREAKOUT = 0.054
BASE_NORM_MEDIAN = 1.00
BASE_TOPIC_BREAKOUT = {
    "媒体与审查": (23, 0.217), "移民边境ICE": (100, 0.080), "选举诚信": (39, 0.077),
    "司法武器化": (16, 0.062), "其他": (154, 0.058), "外交与战争": (56, 0.036),
    "中国CCP": (27, 0.0), "经济通胀关税": (23, 0.0), "共和党内部与人物": (26, 0.0),
    "枪权": (13, 0.0), "犯罪治安": (19, 0.0),
}

LOG = "logs/engagement_snapshots.jsonl"
PRD = "logs/priority_rank_decisions.jsonl"


def q(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] * (c - k) + v[c] * (k - f)


def rate_ci(hits, n, it=6000):
    if not n:
        return 0.0, 0.0
    s = sorted(sum(random.random() < hits / n for _ in range(n)) / n for _ in range(it))
    return s[int(0.025 * it)], s[int(0.975 * it)]


def load_posts():
    """Final value per post, mature only, with hour-normalised ratio."""
    by = {}
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
            except Exception:
                continue
            p = x.get("post_id")
            if not p or not x.get("cdate"):
                continue
            if p not in by or x["ts"] > by[p]["ts"]:
                by[p] = x
    now = time.time()
    posts = [x for x in by.values() if (now - x["cdate"] / 1000) > 12 * 3600]
    for x in posts:
        d = dt.datetime.fromtimestamp(x["cdate"] / 1000, dt.timezone.utc)
        x["hr"], x["t"] = d.hour, x["cdate"] / 1000
    if not posts:
        return []
    hm = {h: (st.median([y["lk"] for y in posts if y["hr"] == h]) or 1)
          for h in {x["hr"] for x in posts}}
    med = st.median([x["lk"] for x in posts]) or 1
    for x in posts:
        x["r"] = x["lk"] / hm[x["hr"]]
        x["hi"] = 1 if x["lk"] >= 2 * med else 0
    return posts


def fixed_age_series(age_h=12.0):
    """Likes at a fixed post age, from the hourly snapshots — the only way to
    compare two periods whose posts differ in how long they have been up."""
    by = collections.defaultdict(list)
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get("cdate"):
                by[x["post_id"]].append(x)
    out = []
    for xs in by.values():
        xs.sort(key=lambda z: z["ts"])
        c = xs[0]["cdate"] / 1000
        ok = [z for z in xs if z["ts"] - c >= age_h * 3600]
        if not ok:
            continue
        z = min(ok, key=lambda z: z["ts"] - c)
        if z["ts"] - c > (age_h + 2.5) * 3600:   # snapshot gap too wide to trust
            continue
        d = dt.datetime.fromtimestamp(c, dt.timezone.utc)
        out.append({"t": c, "hr": d.hour, "lk": z["lk"]})
    if out:
        hm = {h: (st.median([y["lk"] for y in out if y["hr"] == h]) or 1)
              for h in {y["hr"] for y in out}}
        for y in out:
            y["r"] = y["lk"] / hm[y["hr"]]
    return out


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main():
    posts = load_posts()
    if not posts:
        print("engagement_snapshots.jsonl 里没有成熟帖 — 采集器停了？")
        return
    after = [x for x in posts if x["t"] >= SHIP_TS]
    days = (max(x["t"] for x in posts) - SHIP_TS) / 86400

    section("1. 总体 — 上线后 vs 基线")
    n = len(after)
    hits = sum(x["hi"] for x in after)
    print(f"  上线至今 {days:.1f} 天，成熟帖 {n} 条")
    if n < 60:
        print(f"  ⚠ 样本太少（{n} 条），下面的数字只能看方向，别下结论")
    if n:
        lo, hi = rate_ci(hits, n)
        print(f"  爆款率   {hits/n:>6.1%}  [{lo:.1%}, {hi:.1%}]   基线 {BASE_BREAKOUT:.1%} (n={BASE_N})")
        print(f"  归一中位 {st.median([x['r'] for x in after]):>6.2f}                 基线 {BASE_NORM_MEDIAN:.2f}")
        print(f"  赞中位   {st.median([x['lk'] for x in after]):>6.0f}")
        print("\n  两个都要看：中位数最差的源(daily beast 0.78)爆款率最高(12.5%)，")
        print("  只看中位数会砍掉产爆款的东西。")

    section("2. 控制器是否真的在工作")
    try:
        from core.config import load_config
        cfg = load_config()
    except Exception as e:
        print(f"  读配置失败：{e}")
        return
    try:
        import asyncio
        from core.notion_candidates import recent_published_topic_counts
        counts = asyncio.run(recent_published_topic_counts(cfg))
    except Exception as e:
        print(f"  查 Notion 失败：{e}")
        counts = {}
    tot = sum(counts.values())
    print(f"  24h 窗口内带标签的已发帖：{tot} 条（min_window_posts={cfg.topic_mix.min_window_posts}）")
    if tot < cfg.topic_mix.min_window_posts:
        print("  → 控制器仍在休眠，还没开始调整。若已上线数天仍为此状态，查摄取端标注是否在跑。")
    else:
        print("  → 控制器已激活")
        print(f"\n  {'题材':<16}{'实际':>8}{'目标':>8}{'偏差':>8}{'基线爆款率':>11}")
        for t, target in sorted(cfg.topic_mix.targets.items(), key=lambda z: -z[1]):
            act = counts.get(t, 0) / tot
            b = BASE_TOPIC_BREAKOUT.get(t)
            btxt = f"{b[1]:.1%}(n={b[0]})" if b else "—"
            print(f"  {t:<16}{act:>8.1%}{target:>8.1%}{act-target:>+8.1%}{btxt:>11}")

    section("3. 供给风险 — 媒体与审查会不会被榨干")
    per_day = counts.get("媒体与审查", 0) if tot else 0
    print(f"  最近 24h 发了 {per_day} 条媒体与审查（池子里约 27 条/天可选）")
    if per_day >= 12:
        print("  ⚠ 已经在挑该题材最差的一半了 — 把 config.yaml 里 targets.媒体与审查 调低")
    elif per_day:
        print(f"  取用率约 {per_day/27:.0%}，仍在安全区")

    section("4. 09-23 那两个改动（K 1.5→1.0、night_end_hour 7→8）")
    fx = fixed_age_series()
    cut = dt.datetime(2026, 9, 23, 16, 37, tzinfo=dt.timezone.utc).timestamp()
    A = [x["r"] for x in fx if x["t"] < cut]
    B = [x["r"] for x in fx if x["t"] >= cut]
    if len(A) >= 30 and len(B) >= 30:
        print(f"  固定 12h 龄取值：前 n={len(A)} 归一中位 {st.median(A):.2f}   后 n={len(B)} 归一中位 {st.median(B):.2f}")
        print(f"  倍数 {st.median(B)/max(st.median(A), .01):.3f}x")
        print("  09-25 中期读数是 0.905x（n=33），没测出收益。若仍 <0.95，考虑把 K 回滚到 1.5。")
    else:
        print(f"  样本不足（前 {len(A)} / 后 {len(B)}）")

    section("5. 健康检查")
    try:
        cand = [json.loads(l) for l in open(PRD, encoding="utf-8") if l.strip()]
        cand = [r for r in cand if r.get("check_type") != "publish_outcome"]
        tagged = sum(1 for r in cand if r.get("topic"))
        print(f"  今日排序决策 {len(cand)} 条，其中带 topic 的 {tagged} 条 ({tagged/max(len(cand),1):.0%})")
        adj = [r.get("topic_adjustment") or 0 for r in cand]
        if adj:
            print(f"  topic_adjustment 范围 [{min(adj):+.2f}, {max(adj):+.2f}]（上限 ±{cfg.topic_mix.max_adjustment}）")
        if tagged and tagged / max(len(cand), 1) < 0.5:
            print("  ⚠ 半数以上候选未标注 — 存量候选还没换完，或标注器在报错")
    except Exception as e:
        print(f"  读排序日志失败：{e}")
    print()


if __name__ == "__main__":
    main()
