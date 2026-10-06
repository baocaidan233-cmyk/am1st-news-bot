"""Label the candidates already in the pool with the yes/no reader appeal, once,
when publish.appeal_order is first turned on (2026-10-06). New candidates are
labelled at pool entry by main.py; without this, everything already eligible
would sit unlabelled -- i.e. ordered as "no appeal" -- until it aged out.

  ./.venv/bin/python tools/appeal_backfill.py            # label and write
  ./.venv/bin/python tools/appeal_backfill.py --dry-run  # label, write nothing,
                                                         # and show the batch
                                                         # select_batch would build
                                                         # with and without it

One gpt-4o-mini call per unlabelled eligible candidate (a few hundred at most).
Candidates that already have a label are skipped, so a re-run costs nothing."""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")

from agents.appeal_tagger import AppealTagger  # noqa: E402
from agents.candidate_selector import filter_former_trump, select_batch  # noqa: E402
from core.config import load_config  # noqa: E402
from core.notion_candidates import query_eligible_candidates  # noqa: E402
from core.redis_store import AppealLabels  # noqa: E402


async def main() -> None:
    dry = "--dry-run" in sys.argv[1:]
    config = load_config("config/config.yaml")
    store = AppealLabels(config)
    tagger = AppealTagger(config)
    try:
        cands = filter_former_trump(await query_eligible_candidates(config))
        have = await store.get_many([c.url_hash for c in cands])
        todo = [c for c in cands if c.url_hash not in have]
        print("eligible %d, already labelled %d, to label %d%s" % (
            len(cands), len(have), len(todo), " (dry run: nothing written)" if dry else ""))
        sem = asyncio.Semaphore(8)

        async def one(c):
            async with sem:
                return await tagger.tag(c)

        got = await asyncio.gather(*(one(c) for c in todo))
        labels = dict(have)
        for c, a in zip(todo, got):
            if a is None:
                continue
            labels[c.url_hash] = a
            if not dry:
                await store.set(c.url_hash, a)
        print("labelled: %d yes, %d no, %d failed" % (
            sum(a is True for a in got), sum(a is False for a in got), sum(a is None for a in got)))

        if dry:
            for c in cands:
                c.appeal = labels.get(c.url_hash)
            base = config.model_copy(deep=True)
            base.publish.appeal_order = False
            on = config.model_copy(deep=True)
            on.publish.appeal_order = True
            for name, cfg in (("hash only (now)", base), ("appeal first", on)):
                b = select_batch(cands, cfg, cycle_token="backfill-check")
                print("\n== %s: batch of %d, appeal yes %d / no %d / unlabelled %d" % (
                    name, len(b), sum(c.appeal is True for c in b), sum(c.appeal is False for c in b),
                    sum(c.appeal is None for c in b)))
                for c in b:
                    print("   %.1f  %-4s %s" % (c.llm_score, {True: "yes", False: "no"}.get(c.appeal, "-"), c.title[:90]))
            pool = [c for c in cands if c.appeal is not None]
            print("\npool: %d labelled, %.0f%% appeal" % (len(pool), 100 * sum(c.appeal for c in pool) / max(1, len(pool))))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
