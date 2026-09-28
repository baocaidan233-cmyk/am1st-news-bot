"""Missing-slot regression eval: does the writer put back a specific the
source does not contain?

The accidents this guards against share one shape. The article gives a
language slot that wants a concrete value -- a year, a name -- without giving
the value, and the model fills it from parametric memory. Waiting for that to
happen in production and then re-running the one bad article proves nothing:
the failure is stochastic and low-frequency, so a handful of reruns can only
detect an enormous effect.

So the cases are manufactured instead. Take articles this channel really
published from, delete one specific, and check whether it comes back. Every
case has a known correct answer, the buckets are separable, and the whole set
can be re-run against any prompt or gate change.

Read-only: it re-fetches sources and calls the writer, and publishes nothing.

    ./.venv/bin/python tools/missing_slot_eval.py [n_articles]

Baseline, 2026-09-28, 30 articles, three runs, taken right after the writer
started receiving resolved dates (core/date_context.py):

    missing_year      0%, 5%, 0%   of 22 cases
    unnamed_person    7%, 11%, 11% of 27 cases

Read that as a range, never as one number: the same set moved 0-5 points
between runs at temperature 0, so a single run cannot tell a real change
from the spread. Run it at least three times either side of any prompt or
gate change.

One known confound in unnamed_person: every hit so far has been "President
Trump" in an article that is unmistakably about him even with the name
deleted. Recovering a name the remaining text identifies beyond doubt is not
the same failure as inventing one, and this bucket cannot separate them --
which is an argument for reading it alongside the far cleaner missing_year,
not on its own.
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import AsyncQdrantClient  # noqa: E402

from agents.extractor import Extractor  # noqa: E402
from agents.writer import Writer  # noqa: E402
from core.config import load_config  # noqa: E402
from core.notion_sources import load_rss_sources  # noqa: E402

_YEAR = re.compile(r"\b(19[89]\d|20[0-5]\d)\b")
_PERSON = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")
_ROLES = ("the official", "the lawmaker", "the executive", "the spokesperson")
# Two capitalised words is not enough to be a person. Masking an institution
# and then counting its reappearance as a hallucinated name scored 1 of the 2
# hits in the first run ("Supreme Court").
_NOT_A_PERSON = re.compile(
    r"\b(Supreme|White|United|New|Wall|Fox|Federal|National|Justice|State|Homeland|Middle"
    r"|Democratic|Republican|European|Social|Health|Treasury|Defense|Border|Capitol"
    r"|Silicon|Hong|South|North|East|West|Saudi|Great|High|Big|Air|Space|Second|First)\b")


class _NoAlerts:
    async def alert(self, page_id, message):
        return None


def case_missing_year(src: str):
    """Delete every instance of the article's dominant year."""
    years = _YEAR.findall(src)
    if not years:
        return None
    year = max(set(years), key=years.count)
    masked = re.sub(r"\s*\b" + year + r"\b", "", src)
    return {"bucket": "missing_year", "removed": year, "source": masked}


def case_unnamed_person(src: str):
    """Replace one full name, everywhere, with a generic role."""
    names = _PERSON.findall(src)
    if not names:
        return None
    people = {f"{a} {b}" for a, b in names if not _NOT_A_PERSON.match(f"{a} {b}")}
    if not people:
        return None
    full = max(people, key=lambda n: src.count(n))
    if src.count(full) < 2:
        return None
    role = random.choice(_ROLES)
    masked = src.replace(full, role).replace(full.split()[1], role)
    return {"bucket": "unnamed_person", "removed": full, "source": masked}


def scored(case: dict, caption: str) -> bool:
    """True when the writer put the deleted specific back."""
    if case["bucket"] == "missing_year":
        return case["removed"] in caption
    surname = case["removed"].split()[-1]
    return bool(re.search(r"\b" + re.escape(surname) + r"\b", caption))


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg = load_config()
    cli = AsyncQdrantClient(url=cfg.qdrant.url, api_key=cfg.qdrant.api_key or None, timeout=60)
    rows, off = [], None
    while len(rows) < 600:
        pts, off = await cli.scroll(collection_name=cfg.qdrant.posted_collection,
                                    limit=500, offset=off, with_payload=True, with_vectors=False)
        if not pts:
            break
        rows += [p.payload for p in pts]
        if off is None:
            break
    rows = [r for r in rows if r.get("url")]
    rows.sort(key=lambda r: r.get("sentAt") or 0, reverse=True)
    rows = rows[:n]

    ex, srcs = Extractor(cfg, _NoAlerts()), await load_rss_sources(cfg)
    sem = asyncio.Semaphore(3)

    async def fetch(r):
        async with sem:
            await asyncio.sleep(random.uniform(0.3, 1.2))
            try:
                return r, await ex.extract(r["url"], srcs)
            except Exception:
                return r, None

    got = [(r, s) for r, s in await asyncio.gather(*[fetch(r) for r in rows]) if s and len(s) > 600]
    print(f"取到原文 {len(got)}/{len(rows)}", flush=True)

    writer = Writer(cfg)
    pub = datetime.now(timezone.utc)
    cases = []
    for r, src in got:
        for build in (case_missing_year, case_unnamed_person):
            c = build(src)
            if c:
                c["url"] = r["url"]
                cases.append(c)

    wsem = asyncio.Semaphore(4)

    async def run(c):
        async with wsem:
            try:
                cap = await writer.write(c["url"].split("/")[-1][:80].replace("-", " "),
                                         c["source"], published_at=pub)
            except Exception as exc:
                return c, None, repr(exc)[:60]
            return c, cap, None

    results = await asyncio.gather(*[run(c) for c in cases])
    buckets: dict[str, list[bool]] = {}
    shown = 0
    for c, cap, err in results:
        if err or not cap:
            continue
        bad = scored(c, cap)
        buckets.setdefault(c["bucket"], []).append(bad)
        if bad and shown < 6:
            shown += 1
            print(f"\n--- [{c['bucket']}] 补回了 {c['removed']!r}  {c['url'][:70]}")
            print(f"    {cap[:180]}")

    print("\n%-18s%8s%8s%9s" % ("桶", "用例", "补回", "补回率"))
    for b, v in sorted(buckets.items()):
        print("%-18s%8d%8d%8.0f%%" % (b, len(v), sum(v), 100 * sum(v) / max(1, len(v))))


if __name__ == "__main__":
    asyncio.run(main())
