"""For each story the benchmark accounts did well with: did we have it, and did
we publish it?

The unit is the story, not the subject mix. A subject-share comparison says our
pool holds 5% corruption against their 16%, which sounds like a supply problem
and may not be one — the NY Post story on the US asking China to execute
fentanyl suppliers sat in the pool, scored, tagged and extracted, for sixteen
consecutive cycles and aged out unpublished. Having it in the pool is worth
nothing. So every benchmark post that performed lands in exactly one bucket:

  命中   we had it and published it
  漏掉   we had it and did not publish it   <- selection failure, and the
                                              benchmark account's own
                                              engagement is the evidence it
                                              was worth publishing
  没有   it never reached our pool           <- supply failure

That split is the whole point: the two failures have opposite fixes, and the
only stories we can prove we were wrong to skip are the ones somebody else
published to a real audience.

Matching is cosine over the benchmark post against our candidates' title +
description within +/-36h, then one LLM confirmation on the best match above a
floor. Cosine alone cannot decide this — measured repeatedly here, a genuine
reprint and a different angle on the same event sit in the same band — so it
only decides what to ask about.

    ./.venv/bin/python tools/maga_gap_review.py [days] [top_n]
"""
from __future__ import annotations

import asyncio
import collections
import datetime as dt
import json
import math
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

import httpx  # noqa: E402

from agents.embedder import Embedder  # noqa: E402
from core.config import load_config  # noqa: E402
from core.openai_client import create_openai_client  # noqa: E402

BENCHMARK = os.path.expanduser("~/maga_benchmark/posts.jsonl")
OURS = {"gettrworldnews"}
COSINE_FLOOR = 0.62          # below this, not worth an LLM call
WINDOW_H = 36.0

cfg = load_config("config/config.yaml")
client = create_openai_client(cfg)
embedder = Embedder(cfg)
_sem = asyncio.Semaphore(8)

_SAME = """Do these two describe the same underlying news event?

The first is a post from another channel. The second is a headline our own
newsroom had available. Same event means the same thing happened to the same
people: a report of an incident and a statement about that same incident count
as the same event; two similar but separate incidents do not, and two different
people acting on one broad topic do not.

Reply with JSON: {"same": true} or {"same": false}"""


async def same_event(a: str, b: str) -> bool:
    async with _sem:
        try:
            r = await client.chat.completions.create(
                model=cfg.openai.chat_model, temperature=0, max_tokens=12,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": _SAME},
                          {"role": "user", "content": f"A: {a[:400]}\n\nB: {b[:400]}"}])
            return bool(json.loads(r.choices[0].message.content or "{}").get("same"))
        except Exception:
            return False


async def embed(text: str) -> list[float] | None:
    async with _sem:
        try:
            return await embedder.embed(text[:2000])
        except Exception:
            return None


def cos(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1
    nb = math.sqrt(sum(x * x for x in b)) or 1
    return dot / (na * nb)


def benchmark_top(days: float, n: int) -> list[dict]:
    cutoff = time.time() - days * 86400
    rows = []
    with open(BENCHMARK, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ts = (r.get("cdate") or 0) / 1000
            if ts < cutoff or r.get("author") in OURS:
                continue
            text = (r.get("txt") or "").strip()[:400] or (r.get("link_title") or "")[:400]
            if len(text) > 25:
                rows.append({"a": r.get("author"), "lk": int(r.get("like") or 0), "txt": text, "ts": ts,
                             # Whether the post points at an article at all.
                             # Our pipeline ingests RSS articles, so a native
                             # video or screenshot post is not something more
                             # feeds would ever give us — it is a different
                             # ingestion path, and lumping the two together
                             # turns an architecture question into a fake
                             # sourcing one.
                             "article": bool(r.get("link")) or bool(r.get("link_title")),
                             "media": bool(r.get("vid")) or bool(r.get("imgs"))})
    per = collections.defaultdict(list)
    for r in rows:
        per[r["a"]].append(r["lk"])
    med = {a: (st.median(v) or 1) for a, v in per.items()}
    for r in rows:
        r["n"] = r["lk"] / med[r["a"]]
    rows.sort(key=lambda r: -r["n"])
    print(f"大号近 {days:g} 天帖 {len(rows)} 条（{len(per)} 个账号），取归一化最高的 {min(n, len(rows))} 条")
    return rows[:n]


async def our_candidates(days: float) -> list[dict]:
    H = {"Authorization": "Bearer " + cfg.notion.candidate_key,
         "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days + 1)).isoformat()
    out, cur = [], None
    async with httpx.AsyncClient(timeout=60) as cli:
        while True:
            body = {"filter": {"property": "published_at", "date": {"after": since}}, "page_size": 100}
            if cur:
                body["start_cursor"] = cur
            d = (await cli.post(f"https://api.notion.com/v1/databases/{cfg.notion.candidate_db_id}/query",
                                headers=H, json=body)).json()
            for pg in d.get("results", []):
                p = pg["properties"]

                def tx(q):
                    t = (q or {}).get("type")
                    return "".join(b.get("plain_text", "") for b in (q.get(t) or [])) if t in ("rich_text", "title") else ""
                pa = ((p.get("published_at") or {}).get("date") or {}).get("start")
                title = tx(p.get("title"))
                if not title or not pa:
                    continue
                out.append({
                    "title": title, "desc": tx(p.get("description"))[:300],
                    "url": (p.get("url") or {}).get("url") or "",
                    "score": (p.get("llm_score") or {}).get("number"),
                    "sent": bool((p.get("send_status") or {}).get("checkbox")),
                    "ts": dt.datetime.fromisoformat(pa.replace("Z", "+00:00")).timestamp(),
                })
            cur = d.get("next_cursor")
            if not d.get("has_more"):
                break
    print(f"我们近 {days + 1:g} 天候选 {len(out)} 条（已发 {sum(1 for c in out if c['sent'])} 条）\n", flush=True)
    return out


async def main() -> None:
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    top = benchmark_top(days, top_n)
    cands = await our_candidates(days)
    tv = await asyncio.gather(*[embed(f"{c['title']}\n{c['desc']}") for c in cands])
    bv = await asyncio.gather(*[embed(r["txt"]) for r in top])

    async def resolve(r, v):
        near = [(cos(v, tv[i]), i) for i, c in enumerate(cands)
                if tv[i] and abs(c["ts"] - r["ts"]) <= WINDOW_H * 3600]
        near.sort(reverse=True)
        for sim, i in near[:3]:
            if sim < COSINE_FLOOR:
                break
            if await same_event(r["txt"], f"{cands[i]['title']}. {cands[i]['desc']}"):
                return r, cands[i], sim
        return r, None, (near[0][0] if near else 0.0)

    results = await asyncio.gather(*[resolve(r, v) for r, v in zip(top, bv)])

    hit = [(r, c) for r, c, _ in results if c and c["sent"]]
    miss = [(r, c) for r, c, _ in results if c and not c["sent"]]
    absent = [(r, s) for r, c, s in results if not c]
    n = len(results)
    print("=" * 76)
    print("大号跑得好的 %d 条，我们的处理：" % n)
    print("  命中（池里有、也发了）  %2d 条 (%.0f%%)" % (len(hit), 100 * len(hit) / n))
    print("  漏掉（池里有、没发）    %2d 条 (%.0f%%)   ← 选稿问题" % (len(miss), 100 * len(miss) / n))
    art = [x for x in absent if x[0]["article"]]
    native = [x for x in absent if not x[0]["article"]]
    print("  没有（池里根本没有）    %2d 条 (%.0f%%)" % (len(absent), 100 * len(absent) / n))
    print("      其中带文章外链      %2d 条   ← 真·稿源问题，加源能解" % len(art))
    print("      其中原生社交内容    %2d 条   ← 视频/截图/自述，RSS 管道拿不到" % len(native))
    if miss:
        print("\n=== 漏掉的（他们发了有反响，我们手里有却没发）===")
        for r, c in sorted(miss, key=lambda x: -x[0]["n"]):
            print("\n  [%s 归一 %.1f / %d赞]  %s" % (r["a"], r["n"], r["lk"], r["txt"][:110].replace("\n", " ")))
            print("     我们手里： %s分  %s" % (c["score"], c["title"][:95]))
    if absent:
        print("\n=== 池子里根本没有的 ===")
        for label, group in (("带文章外链（加源能解）", art), ("原生社交内容（管道拿不到）", native)):
            print("\n  -- %s --" % label)
            for r, _ in sorted(group, key=lambda x: -x[0]["n"])[:8]:
                print("  [%s 归一 %.1f]  %s" % (r["a"], r["n"], r["txt"][:100].replace("\n", " ")))


if __name__ == "__main__":
    asyncio.run(main())
