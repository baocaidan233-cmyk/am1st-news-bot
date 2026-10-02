#!/usr/bin/env python3
"""What core/maga_heat.py would do, before it is wired to anything.

Two numbers decide whether this is safe to put in the pipeline, and they pull in
opposite directions:

  what it protects — candidates our dedup threw away before the scorer saw them,
      that the big accounts were running hot. Those are the stories this exists
      for and each one is a post we did not make.
  what it lets through — candidates it would protect that really are the same
      story as something we already published. Dedup exists to stop those, and
      every one this waves past is a repeat post on the channel.

Run at several thresholds, pick where the second number is acceptable.

    .venv/bin/python tools/maga_heat_dryrun.py
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from dotenv import load_dotenv
load_dotenv(".env")
from core.maga_heat import MagaHeat

norm = lambda u: (u or "").split("?")[0].rstrip("/")
POOL = re.compile(r"added to candidate pool: (\S+) \(score=([\d.]+)")
REJ = re.compile(r"run_cycle: (\S+) scored ([\d.]+), below threshold")


def pipeline_state() -> tuple[dict, set, set]:
    pooled, scored = {}, set()
    for path in sorted(glob.glob("logs/main.log*")):
        op = gzip.open if path.endswith(".gz") else open
        try:
            with op(path, "rt", errors="replace") as fh:
                for line in fh:
                    m = POOL.search(line)
                    if m:
                        pooled[norm(m.group(1))] = float(m.group(2))
                        scored.add(norm(m.group(1)))
                        continue
                    m = REJ.search(line)
                    if m:
                        scored.add(norm(m.group(1)))
        except OSError:
            pass
    published = set()
    try:
        with open("logs/engagement_snapshots.jsonl", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("src"):
                    published.add(norm(r["src"]))
    except FileNotFoundError:
        pass
    return pooled, scored, published


def main() -> None:
    work = os.path.expanduser("~/scorer_v2_work")
    emb_path = os.path.join(work, "prescore_emb.npy")
    if not os.path.exists(emb_path):
        print(f"need candidate title embeddings at {emb_path}")
        sys.exit(1)
    cand, seen = [], set()
    for line in open("logs/prescore_decisions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["title"] in seen:
            continue
        seen.add(r["title"])
        cand.append(r)
    labels = json.load(open(os.path.join(work, "prescore_v2_labels.json")))
    cand = [r for r in cand if labels.get(r["title"])]
    V = np.load(emb_path)
    if len(V) != len(cand):
        print(f"embedding count {len(V)} != candidate count {len(cand)}")
        sys.exit(1)
    pooled, scored, published = pipeline_state()
    print(f"候选 {len(cand)} 条；其中到过打分器 {sum(1 for r in cand if norm(r['url']) in scored)}，"
          f"入过池 {sum(1 for r in cand if norm(r['url']) in pooled)}，"
          f"发出去 {sum(1 for r in cand if norm(r['url']) in published)}\n")

    for window, min_adj, cos in ((36.0, 1.5, 0.62), (36.0, 2.0, 0.62),
                                 (36.0, 2.5, 0.62), (36.0, 2.0, 0.70),
                                 (36.0, 2.5, 0.70), (24.0, 2.5, 0.70)):
        mh = MagaHeat(window_hours=window, min_adj=min_adj, match_cosine=cos)
        if not mh.ready:
            print(f"窗口{window:.0f}h 热度>={min_adj} 余弦>={cos}: 索引为空")
            continue
        protect, waste, already = [], 0, 0
        for r, v in zip(cand, V):
            adj, ch, body = mh.heat(v)
            if adj <= 0:
                continue
            u = norm(r["url"])
            if u in published:
                already += 1
            elif u in scored:
                waste += 1          # 打分器看过了，保不保都不改变什么
            else:
                protect.append((adj, ch, r["title"], body))
        print(f"窗口{window:.0f}h 热度>={min_adj:.1f} 余弦>={cos:.2f}: "
              f"索引 {len(mh._meta)} 条热故事 | 命中候选 {len(protect)+waste+already} "
              f"(其中没到打分器 {len(protect)}，到过打分器 {waste}，已发 {already})")
    print()

    mh = MagaHeat(window_hours=36.0, min_adj=2.0, match_cosine=0.70)
    print("=" * 96)
    print("按 热度>=2.0 / 余弦>=0.70 列出「大号在跑、我们的打分器却从没见过」的候选")
    print("=" * 96)
    rows = []
    for r, v in zip(cand, V):
        adj, ch, body = mh.heat(v)
        if adj > 0 and norm(r["url"]) not in scored:
            rows.append((adj, ch, r["title"], body))
    for adj, ch, title, body in sorted(rows, key=lambda x: -x[0])[:20]:
        print(f"\n  {adj:.2f}x {ch}")
        print(f"    我们的稿: {title[:86]}")
        print(f"    大号那条: {body[:86]}")
    print(f"\n共 {len(rows)} 条")


if __name__ == "__main__":
    main()
