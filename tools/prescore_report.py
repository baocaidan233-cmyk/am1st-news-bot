"""What the pre-score is actually skipping, and what that costs.

The number that matters is P(the model would skip it | it reaches the gate),
and it cannot be read off the live skips: a skipped candidate is never
scored, so it has no label, so nothing about it can be checked afterwards.
The audit cohort exists for exactly this. It is a fixed share of candidates,
chosen by url_hash before the model looks, that is scored whatever the model
says, with the model's unused verdict recorded. Joining those rows to the
score they went on to get is an unbiased estimate of the miss rate.

The upper bound is Clopper-Pearson, one-sided, 95%. With no miss seen it is
1 - 0.05^(1/n), so about 1% at n=300 and about 3% at n=100. Read it as the
thing the data can rule out, not as the rate itself.

Run:  ./.venv/bin/python tools/prescore_report.py [--hours 168]
"""
from __future__ import annotations

import argparse
import datetime
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

_SCORED = re.compile(
    r"INFO main: run_cycle: (\S+) scored ([\d.]+), below threshold")
_POOLED = re.compile(
    r"INFO main: run_cycle: added to candidate pool: (\S+) \(score=([\d.]+)")


def _lines(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            yield line


def scores_by_url() -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted(glob.glob("logs/main.log*")):
        for line in _lines(path):
            for pattern in (_SCORED, _POOLED):
                m = pattern.search(line)
                if m:
                    out[m.group(1)] = float(m.group(2))
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--gate", type=float, default=None)
    args = ap.parse_args()

    from core.config import load_config
    config = load_config("config/config.yaml")
    gate = args.gate if args.gate is not None else config.openai.score_threshold

    since = time.time() - args.hours * 3600
    rows = []
    for path in sorted(glob.glob(config.prescore.log_path + "*")):
        for line in _lines(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("logged_at", 0) >= since:
                rows.append(r)
    if not rows:
        sys.exit(f"no pre-score decisions in the last {args.hours:.0f}h — is prescore.enabled on?")

    counts = Counter(r["decision"] for r in rows)
    total = len(rows)
    skipped = counts.get("skip", 0)
    span = (max(r["logged_at"] for r in rows) - min(r["logged_at"] for r in rows)) / 3600
    print(f"{total} candidates over {span:.1f}h "
          f"({datetime.datetime.utcfromtimestamp(min(r['logged_at'] for r in rows)):%m-%d %H:%M} UTC on)")
    for decision, n in counts.most_common():
        print(f"  {decision:<18} {n:>6}  {100 * n / total:5.1f}%")
    print(f"\nscoring calls skipped: {skipped}/{total} = {100 * skipped / total:.1f}%")
    kept_by_a_gate = sum(n for d, n in counts.items() if d.startswith("keep_"))
    if kept_by_a_gate:
        print(f"  (a safety layer kept {kept_by_a_gate} the model wanted to skip)")

    scores = scores_by_url()
    audit = [r for r in rows if r["decision"].startswith("audit_")]
    labelled = [(r, scores[r["url"]]) for r in audit if r["url"] in scores]
    print(f"\naudit cohort: {len(audit)} candidates, {len(labelled)} of them matched to a score")
    if not labelled:
        print("  nothing to measure yet — the join needs main.log to still hold those cycles")
        return
    at_gate = [(r, s) for r, s in labelled if s >= gate]
    missed = [(r, s) for r, s in at_gate if r["decision"] == "audit_would_skip"]
    print(f"  reached the {gate:.0f} gate: {len(at_gate)}")
    if at_gate:
        rate = len(missed) / len(at_gate)
        if missed:
            print(f"  the model would have skipped {len(missed)} of them — miss rate {100 * rate:.2f}%")
            for r, s in sorted(missed, key=lambda x: -x[1])[:10]:
                print(f"    {s:.1f}  p={r.get('prescore_p')}  {r.get('title', '')[:70]}")
        else:
            bound = 1 - 0.05 ** (1 / len(at_gate))
            print(f"  none of them would have been skipped — one-sided 95% upper bound {100 * bound:.2f}%")
            print(f"  (need ~300 at the gate with no miss to put it under 1%; {len(at_gate)} so far)")
    for label in ("audit_pass", "audit_would_skip"):
        vals = [s for r, s in labelled if r["decision"] == label]
        if vals:
            vals.sort()
            print(f"  {label}: n={len(vals)} median {vals[len(vals) // 2]:.1f} max {max(vals):.1f} "
                  f"at-gate share {100 * sum(1 for v in vals if v >= gate) / len(vals):.1f}%")


if __name__ == "__main__":
    main()
