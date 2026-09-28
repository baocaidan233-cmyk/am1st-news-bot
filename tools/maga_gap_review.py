"""What MAGA's big accounts get engagement on, against what this channel has,
picks and publishes — plus what the scorer threw away before any of that.

Two questions, one tool, because the first one's answer is meaningless without
the second. A subject that is big for the benchmark accounts and thin in our
pool can mean two different things: nobody feeds us that kind of story, or the
scorer rejects it at the door. Those have opposite fixes — add sources, or fix
the scorer — and only the rejected-log side can tell them apart. Rejected
candidates are never written to Notion, so the log line is the only record
they exist at all.

Methodology, fixed here so it is not re-chosen per run:
  - Benchmark engagement is normalised per account against that account's own
    median. Raw likes across accounts with 19k and 7.4M followers compare
    nothing.
  - The comparison is on subject mix, not on surface features. Post length,
    emoji, hashtags and the rest were measured on this same benchmark and did
    not separate good posts from bad ones within an account.
  - Rejected candidates are classified from their URL slug, which is usually
    the headline. Slugs shorter than 18 characters are skipped rather than
    guessed at.

    ./.venv/bin/python tools/maga_gap_review.py [hours] [rejected_sample]
"""
from __future__ import annotations

import asyncio
import collections
import glob
import gzip
import json
import os
import random
import re
import statistics as st
import sys
import time
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from agents.topic_tagger import TOPICS, _PROMPT as TOPIC_PROMPT  # noqa: E402
from core.config import load_config  # noqa: E402
from core.notion_candidates import query_eligible_candidates, recent_published_topic_counts  # noqa: E402
from core.openai_client import create_openai_client  # noqa: E402

BENCHMARK = os.path.expanduser("~/maga_benchmark/posts.jsonl")
OURS = {"gettrworldnews"}
LOG_GLOB = "logs/main.log*"

cfg = load_config("config/config.yaml")
client = create_openai_client(cfg)
_sem = asyncio.Semaphore(8)


async def tag(text: str) -> str | None:
    if len(text) < 18:
        return None
    async with _sem:
        try:
            resp = await client.chat.completions.create(
                model=cfg.openai.chat_model, temperature=0, max_tokens=24,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": TOPIC_PROMPT},
                          {"role": "user", "content": text[:500]}])
            t = json.loads(resp.choices[0].message.content or "{}").get("t")
            return t if t in TOPICS else None
        except Exception:
            return None


def log_lines() -> list[str]:
    out: list[str] = []
    for path in sorted(glob.glob(LOG_GLOB)):
        try:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
                out += f.readlines()
        except Exception:
            continue
    return out


def slug_of(url: str) -> str:
    try:
        path = unquote(urlparse(url).path)
    except Exception:
        return ""
    parts = [p for p in path.split("/") if len(p) > 18]
    text = re.sub(r"[-_]+", " ", " ".join(parts[-2:]))
    text = re.sub(r"\.(html?|php|aspx?|shtml)$", "", text)
    return " ".join(re.sub(r"\b\d{5,}\b", " ", text).split())[:200]


def benchmark_top(hours: float, n: int) -> list[dict]:
    cutoff = time.time() - hours * 3600
    rows = []
    with open(BENCHMARK, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("cdate") or 0) / 1000 < cutoff or r.get("author") in OURS:
                continue
            text = (r.get("txt") or "")[:260] or (r.get("link_title") or "")[:260]
            if text:
                rows.append({"a": r.get("author"), "lk": int(r.get("like") or 0), "txt": text})
    per: dict[str, list[int]] = collections.defaultdict(list)
    for r in rows:
        per[r["a"]].append(r["lk"])
    med = {a: (st.median(v) or 1) for a, v in per.items()}
    for r in rows:
        r["n"] = r["lk"] / med[r["a"]]
    rows.sort(key=lambda r: -r["n"])
    print(f"大号近 {hours:g}h 帖 {len(rows)} 条（{len(per)} 个账号），取归一化最高的 {min(n, len(rows))} 条")
    return rows[:n]


def share(counter: collections.Counter, key: str) -> float:
    total = sum(counter.values())
    return 100 * counter[key] / total if total else 0.0


async def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 72.0
    rej_n = int(sys.argv[2]) if len(sys.argv) > 2 else 250

    top = benchmark_top(hours, 45)
    big = collections.Counter(t for t in await asyncio.gather(*[tag(r["txt"]) for r in top]) if t)

    pool = collections.Counter(c.topic for c in await query_eligible_candidates(cfg) if c.topic)
    published = collections.Counter(await recent_published_topic_counts(cfg))

    lines = log_lines()
    # Prefers the logged title, falling back to the URL slug for lines written
    # before the title was added. The slug is much the weaker signal — it sends
    # 78% of a sample to 其他 — so the two are counted separately below.
    rejected: list[tuple[str, bool]] = []
    for line in lines:
        m = re.search(r"run_cycle: (\S+) scored [\d.]+, below threshold(?: — (.*))?$", line.rstrip())
        if not m:
            continue
        title = (m.group(2) or "").strip()
        rejected.append((title, True) if title else (slug_of(m.group(1)), False))
    accepted = sum(1 for line in lines if "added to candidate pool:" in line)
    print(f"打分器日志：入池 {accepted} 条，被拒 {len(rejected)} 条"
          f"（拒绝率 {100 * len(rejected) / max(1, accepted + len(rejected)):.0f}%）\n")

    random.seed(3)
    sample = [r for r in random.sample(rejected, min(rej_n, len(rejected))) if len(r[0]) >= 18]
    from_title = sum(1 for _, ok in sample if ok)
    print(f"  被拒样本 {len(sample)} 条，其中 {from_title} 条有真标题、"
          f"{len(sample) - from_title} 条只能用 URL slug 猜")
    slugs = [t for t, _ in sample]
    rej_tags = await asyncio.gather(*[tag(t) for t in slugs])
    rej = collections.Counter(t for t in rej_tags if t)

    keys = sorted(set(big) | set(pool) | set(published) | set(rej), key=lambda k: -share(big, k))
    print("%-16s%10s%10s%10s%12s" % ("题材", "大号高赞", "我们池子", "我们已发", "打分器拒掉"))
    for k in keys:
        print("%-16s%9.0f%%%9.0f%%%9.0f%%%11.0f%%"
              % (k, share(big, k), share(pool, k), share(published, k), share(rej, k)))

    print("\n=== 大号强、我们弱的题材，是稿源不够还是打分器拒掉 ===")
    for k in keys:
        gap = share(big, k) - share(published, k)
        if gap < 5:
            continue
        verdict = ("打分器拒掉了不少 —— 先查打分器" if share(rej, k) >= share(pool, k)
                   else "被拒的也不多 —— 是稿源不够")
        print(f"  {k}：大号 {share(big, k):.0f}% / 已发 {share(published, k):.0f}% / "
              f"池中 {share(pool, k):.0f}% / 被拒中 {share(rej, k):.0f}% → {verdict}")
        for s, t in zip(slugs, rej_tags):
            if t == k:
                print(f"     被拒样本· {s[:100]}")
                break


if __name__ == "__main__":
    asyncio.run(main())
