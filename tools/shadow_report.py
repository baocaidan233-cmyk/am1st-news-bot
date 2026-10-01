#!/usr/bin/env python3
"""Compare the production scorer with the shadow scorer on real traffic.

Reads logs/scorer_shadow.jsonl, which agents/scorer_shadow.py appends one row
to per scored candidate. Prints where the two disagree, which is the only part
worth a human's time: agreement tells you nothing about which is right.

    .venv/bin/python tools/shadow_report.py --hours 24

The free-text `g` field is clustered too, so a grievance nobody anticipated
shows up as a cluster rather than having to be on a list first — that is the
whole reason it is free text and not an enum.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path: str, hours: float) -> list[dict]:
    cutoff = time.time() - hours * 3600
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("ts", 0) >= cutoff:
                    rows.append(r)
    except FileNotFoundError:
        print(f"no shadow log at {path} — is openai.shadow_scoring_enabled on?")
        sys.exit(1)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--log", default="logs/scorer_shadow.jsonl")
    ap.add_argument("--gate", type=float, default=5.0)
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()

    rows = load(a.log, a.hours)
    if not rows:
        print("nothing in the window")
        return
    span = (max(r["ts"] for r in rows) - min(r["ts"] for r in rows)) / 3600
    print(f"{len(rows)} candidates over {span:.1f}h\n")

    old_pass = sum(1 for r in rows if r["production_score"] >= a.gate)
    new_pass = sum(1 for r in rows if r["shadow_score"] >= a.gate)
    print(f"  production passes {old_pass:5d}  {old_pass/len(rows):6.1%}")
    print(f"  shadow     passes {new_pass:5d}  {new_pass/len(rows):6.1%}")

    only_new = [r for r in rows if r["shadow_score"] >= a.gate > r["production_score"]]
    only_old = [r for r in rows if r["production_score"] >= a.gate > r["shadow_score"]]
    both = sum(1 for r in rows if min(r["production_score"], r["shadow_score"]) >= a.gate)
    print(f"  both {both} | only shadow {len(only_new)} | only production {len(only_old)}")
    print(f"  they disagree on {(len(only_new)+len(only_old))/len(rows):.1%} of candidates\n")

    print("=== shadow would publish, production threw away ===")
    for r in sorted(only_new, key=lambda x: -x["shadow_score"])[:a.show]:
        print(f"  旧{r['production_score']:.1f} 新{r['shadow_score']:.1f}  d={r['d']:<9} who={r['who']:<9} {r['title'][:66]}")
        if r["g"] != "none":
            print(f"      g: {r['g'][:80]}")

    print("\n=== production would publish, shadow threw away ===")
    for r in sorted(only_old, key=lambda x: -x["production_score"])[:a.show]:
        print(f"  旧{r['production_score']:.1f} 新{r['shadow_score']:.1f}  d={r['d']:<9} who={r['who']:<9} drop={r['drop']:<10} {r['title'][:52]}")

    print("\n=== shadow hard rejections (the blacklist, not the score) ===")
    for k, v in Counter(r["drop"] for r in rows if r["drop"] != "none").most_common():
        print(f"  {v:>4}  {k}")
    for r in [x for x in rows if x["drop"] == "anti_admin"][:8]:
        print(f"    anti_admin (production gave it {r['production_score']:.1f}): {r['title'][:72]}")

    print("\n=== field distribution ===")
    for key in ("d", "who"):
        print(f"  {key}: {dict(Counter(r[key] for r in rows).most_common())}")
    print(f"  shadow scores: {dict(sorted(Counter(r['shadow_score'] for r in rows).items()))}")
    passing = [r for r in rows if r["shadow_score"] >= a.gate]
    if passing:
        c = Counter(r["shadow_score"] for r in passing)
        print(f"  among passing, biggest tie: {max(c.values())/len(passing):.0%} at {max(c, key=c.get)}")

    print("\n=== grievances named most often (free text, no list) ===")
    named = [r["g"] for r in rows if r["g"] != "none"]
    print(f"  {len(named)}/{len(rows)} candidates carry one")
    for g, n in Counter(named).most_common(12):
        print(f"  {n:>4}  {g[:84]}")


if __name__ == "__main__":
    main()
