#!/usr/bin/env python3
"""Compare the scorer that decides with the scorer that only watches.

Reads logs/scorer_shadow.jsonl, which agents/scorer_shadow.py appends one row to
per scored candidate. Prints only where the two disagree — agreement tells you
nothing about which one is right.

    .venv/bin/python tools/shadow_report.py --hours 24

The log spans a role swap on 2026-10-02 and rows from either side are readable
here, because the column names never changed meaning: `production_score` is
always whatever decided that cycle, `shadow_score` always whatever only watched.
What swapped is which prompt sits behind each name, and each row written since
says so in its own `scorers` field.

  rows before the swap   production = 21-theme prompt, shadow = v2 fields
                         (these rows carry g/d/who/drop)
  rows after  the swap   production = v2 fields, shadow = 21-theme prompt
                         (these rows carry production_comment/shadow_comment)

So the half of the report that matters flips with the swap. Before, the question
was "what does the new prompt want that we are not publishing". Now it is the
reverse and more important one: WHAT DOES THE NEW PROMPT THROW AWAY. v2 rejects
27% of what we used to publish, and the median of that 27% is 0.86x engagement,
which is why the cutover was made — but a median says nothing about any single
story, and the failure that actually costs this channel is killing one that
becomes a hot topic. That is what the "只有退役 prompt 想发" section is for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

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


def is_post_swap(r: dict) -> bool:
    """A row written after 2026-10-02. Detected by shape, not by timestamp, so
    the two eras stay distinguishable even if the log is ever reordered."""
    return "shadow_comment" in r or r.get("scorers", "").startswith("prod=v2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--log", default="logs/scorer_shadow.jsonl")
    ap.add_argument("--gate", type=float, default=5.0)
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()

    allrows = load(a.log, a.hours)
    if not allrows:
        print("nothing in the window")
        return
    rows = [r for r in allrows if is_post_swap(r)]
    legacy = len(allrows) - len(rows)
    if legacy:
        print(f"（窗口里有 {legacy} 行是 10-02 对调之前写的，已跳过——那时候两列的含义是反的）")
    if not rows:
        print("窗口里没有对调之后的行")
        return

    span = (max(r["ts"] for r in rows) - min(r["ts"] for r in rows)) / 3600
    print(f"\n{len(rows)} 条候选，跨 {span:.1f} 小时")
    print("  生产 = v2 四判断（只喂标题）   影子 = 退役的 21 主题 prompt（喂全量输入）\n")

    new_pass = sum(1 for r in rows if r["production_score"] >= a.gate)
    old_pass = sum(1 for r in rows if r["shadow_score"] >= a.gate)
    print(f"  新打分器（在决策）放行 {new_pass:5d}  {new_pass/len(rows):6.1%}")
    print(f"  退役 prompt（只观察）放行 {old_pass:5d}  {old_pass/len(rows):6.1%}")

    only_new = [r for r in rows if r["production_score"] >= a.gate > r["shadow_score"]]
    only_old = [r for r in rows if r["shadow_score"] >= a.gate > r["production_score"]]
    both = sum(1 for r in rows if min(r["production_score"], r["shadow_score"]) >= a.gate)
    print(f"  两边都放 {both} | 只有新的放 {len(only_new)} | 只有退役的放 {len(only_old)}")
    print(f"  分歧率 {(len(only_new)+len(only_old))/len(rows):.1%}\n")

    # 这一节是整份报告的重点：新打分器杀掉、但退役 prompt 想发的。
    # 看的时候只问一件事：这里面有没有哪条后来变成了热点。
    print("=== 只有退役 prompt 想发（新打分器杀掉的）===")
    print("    ↑ 唯一要看的问题：这里面有没有后来变成热点的\n")
    for r in sorted(only_old, key=lambda x: -x["shadow_score"])[:a.show]:
        print(f"  新{r['production_score']:.1f} 旧{r['shadow_score']:.1f}  {r['title'][:72]}")
        if r.get("production_comment"):
            print(f"      新打分器的判断: {r['production_comment'][:96]}")

    print("\n=== 只有新打分器想发（退役 prompt 会丢掉的）===")
    for r in sorted(only_new, key=lambda x: -x["production_score"])[:a.show]:
        print(f"  新{r['production_score']:.1f} 旧{r['shadow_score']:.1f}  {r['title'][:72]}")
        if r.get("production_comment"):
            print(f"      {r['production_comment'][:96]}")

    print("\n=== 新打分器的硬拒 ===")
    drops = [r for r in rows if "DROP=" in (r.get("production_comment") or "")]
    if not drops:
        print("  窗口内没有硬拒")
    else:
        kinds = Counter(c["production_comment"].split("DROP=")[1].split(";")[0].strip()
                        for c in drops)
        for k, v in kinds.most_common():
            print(f"  {v:>4}  {k}")
        print("\n  硬拒里退役 prompt 本来会发的（这些就是误杀的全部代价）:")
        costly = [r for r in drops if r["shadow_score"] >= a.gate]
        if not costly:
            print("    一条都没有——这批硬拒全是退役 prompt 也不要的，零成本")
        for r in sorted(costly, key=lambda x: -x["shadow_score"]):
            kind = r["production_comment"].split("DROP=")[1].split(";")[0].strip()
            print(f"    [{kind}] 退役 prompt 给 {r['shadow_score']:.1f}: {r['title'][:68]}")

    print("\n=== 分数分布 ===")
    for key, lab in (("production_score", "新（在决策）"), ("shadow_score", "旧（只观察）")):
        print(f"  {lab}: {dict(sorted(Counter(r[key] for r in rows).items()))}")
    for key, lab in (("production_score", "新"), ("shadow_score", "旧")):
        p = [r for r in rows if r[key] >= a.gate]
        if p:
            c = Counter(r[key] for r in p)
            print(f"  {lab}: 放行的帖里 {max(c.values())/len(p):.0%} 并列在 {max(c, key=c.get)} 分")

    print("\n=== 新打分器写下的诉求（自由文本，没有清单）===")
    named = [r["production_comment"].split("|", 1)[1].strip()
             for r in rows if "|" in (r.get("production_comment") or "")]
    named = [g for g in named if g and g != "无诉求"]
    print(f"  {len(named)}/{len(rows)} 条有诉求，{len(set(named))} 个不同说法")
    for g, n in Counter(named).most_common(14):
        print(f"  {n:>4}  {g[:84]}")


if __name__ == "__main__":
    main()
