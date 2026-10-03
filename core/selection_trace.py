"""Per-cycle record of where every eligible candidate went. Observation only.

Written because eight rounds of eliminating hypotheses one at a time did not
find why a fresh 8.0 candidate appeared in zero of thirteen consecutive
batches. Ruled out by reading code and logs: Notion pagination (complete),
Notion sort order (score-descending, so an 8.0 is on page one), the
extraction_failed filter, the "former president Trump" phrase filter (all six
phrases require president adjacent to trump), the per-subject cap (3, and the
batches held one of that subject), the per-source cap (3, and that source held
at most one), the within-tier hash (cycle_token is live, so the order is
reshuffled every cycle), and the 4h tier-1 window (the article was 25 minutes
old when it entered the pool, and 4 hours covered all thirteen batches).

None of those explains it, and the existing logs cannot: priority_rank_decisions
records the candidates that ENTERED a batch and nothing about the set they were
drawn from. So "72 of 389 8.0 candidates never appear in it" only means they
have no batch record — they may have entered a tier, been ranked, and lost on
position. That metric is named never_selected_in_observed_cycles for that
reason, and never_offered_to_ranker is NOT something today's logs can measure.

The invariant this records, at every stage:

    set in = set kept  ∪  set excluded,  with an empty intersection
    and nothing unaccounted for

A stage that drops a candidate with no reason recorded is the bug, and that is
what this is built to surface rather than deduce.

Deliberately NOT in Notion and NOT read by any model: this is engineering
observability, ~400 candidates × ~60 cycles a day. It goes to a local JSONL,
one row per cycle holding stable ids plus a reason per excluded candidate, with
the id→url map written once per cycle. Normal use is aggregate counts; a single
candidate's path is expanded only when something looks wrong.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = "logs/selection_trace.jsonl"


class SelectionTrace:
    """Collects one cycle's stages, then writes a single row.

    Every method fails open. A trace that cannot be written must never change
    what the channel publishes, which is the same contract the shadow scorer
    had, and the reason this is a separate object rather than fields threaded
    through select_batch's return value.
    """

    def __init__(self, path: str = DEFAULT_PATH, enabled: bool = True) -> None:
        self.enabled = enabled
        self._path = Path(path)
        self.reset()

    def reset(self) -> None:
        self._stages: list[dict] = []
        self._urls: dict[str, str] = {}
        self._meta: dict = {}
        self._t0 = time.time()

    def note(self, **kv) -> None:
        """Cycle-level facts: the clocks in force, the floors, the caps."""
        if self.enabled:
            self._meta.update(kv)

    def stage(self, name: str, kept, excluded: dict | None = None,
              order: list | None = None, **extra) -> None:
        """One stage. `kept` is what survives, `excluded` maps id -> reason.

        `order` is the ids in the order the stage actually walked them, which is
        what distinguishes "never looked at because the batch filled" from
        "looked at and skipped" — the two the current logs cannot tell apart.
        """
        if not self.enabled:
            return
        try:
            kept_ids = [self._id(c) for c in kept]
            self._stages.append({
                "stage": name,
                "n_in": len(kept_ids) + len(excluded or {}),
                "kept": kept_ids,
                "excluded": dict(excluded or {}),
                "order": list(order) if order is not None else None,
                **extra,
            })
        except Exception:
            logger.debug("SelectionTrace: could not record stage %s", name)

    def _id(self, c) -> str:
        """Stable id — the Notion page id, which survives url normalisation and
        is what every other stage keys on. The url is stored once, separately,
        because repeating it per stage is most of the row's size."""
        pid = getattr(c, "page_id", "") or getattr(c, "url", "")
        url = getattr(c, "url", "")
        if pid and url:
            self._urls.setdefault(pid, url)
        return pid

    def flush(self, selected=None) -> None:
        if not self.enabled or not self._stages:
            self.reset()
            return
        try:
            row = {
                "ts": int(self._t0),
                "meta": self._meta,
                "stages": self._stages,
                "selected": [self._id(c) for c in (selected or [])],
                "urls": self._urls,
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("SelectionTrace: could not write %s", self._path)
        finally:
            self.reset()
