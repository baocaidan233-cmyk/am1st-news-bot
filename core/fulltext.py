"""Full text fetched once, at ingestion (2026-10-21, Linear AI-29).

User, 10-21: an article whose text cannot be fetched is not worth scoring --
fetch the full text in the summary step, where the articles without a summary
need the first paragraph anyway; an article that fails is tried once more an
hour later and then given up. Measured the same day: 23% of the articles
reaching the scorer had no fetchable text, and the publish cycle lost 1.1 of
every 10 candidates it picked to the same failure.

Two files under ingestion.fulltext_dir, both on VM-02's own disk (main.py and
main_publish.py run side by side there):
  <url_hash>.txt  the extracted text, read by main_publish.py instead of
                  fetching the page again; pruned after fulltext_keep_hours.
  retry.json      articles waiting for their second try:
                  {url_hash: {"due": unix seconds, "candidate": Candidate}}.
                  Only main.py reads or writes it."""
from __future__ import annotations

import json
import logging
import os
import time

from core.models import Candidate

logger = logging.getLogger(__name__)

_RETRY = "retry.json"


def _dir(config) -> str:
    return config.ingestion.fulltext_dir


def save(config, url_hash: str, text: str) -> None:
    try:
        os.makedirs(_dir(config), exist_ok=True)
        tmp = os.path.join(_dir(config), url_hash + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, os.path.join(_dir(config), url_hash + ".txt"))
    except OSError:
        logger.exception("fulltext: could not save %s", url_hash)


def load(config, url_hash: str) -> str | None:
    """The text saved at ingestion, or None (an older article, or a cache miss)."""
    if not url_hash:
        return None
    try:
        with open(os.path.join(_dir(config), url_hash + ".txt"), encoding="utf-8") as fh:
            return fh.read() or None
    except OSError:
        return None


def prune(config) -> int:
    """Deletes saved texts older than ingestion.fulltext_keep_hours."""
    cutoff = time.time() - config.ingestion.fulltext_keep_hours * 3600
    removed = 0
    try:
        entries = list(os.scandir(_dir(config)))
    except OSError:
        return 0
    for e in entries:
        if e.name.endswith(".txt") and e.stat().st_mtime < cutoff:
            try:
                os.remove(e.path)
                removed += 1
            except OSError:
                pass
    return removed


def _read_retry(config) -> dict:
    try:
        with open(os.path.join(_dir(config), _RETRY), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write_retry(config, waiting: dict) -> None:
    try:
        os.makedirs(_dir(config), exist_ok=True)
        tmp = os.path.join(_dir(config), _RETRY + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(waiting, fh)
        os.replace(tmp, os.path.join(_dir(config), _RETRY))
    except OSError:
        logger.exception("fulltext: could not write the retry list")


def add_retry(config, candidates: list[Candidate]) -> None:
    if not candidates:
        return
    waiting = _read_retry(config)
    due = time.time() + config.ingestion.fulltext_retry_minutes * 60
    for c in candidates:
        waiting[c.url_hash] = {"due": due, "candidate": c.model_dump(mode="json")}
    _write_retry(config, waiting)


def take_due(config) -> list[Candidate]:
    """Removes and returns the articles whose second try is due."""
    waiting = _read_retry(config)
    now = time.time()
    due = [h for h, w in waiting.items() if w.get("due", 0) <= now]
    if not due:
        return []
    out = []
    for h in due:
        try:
            out.append(Candidate.model_validate(waiting.pop(h)["candidate"]))
        except Exception:
            logger.exception("fulltext: unreadable retry entry %s, dropped", h)
    _write_retry(config, waiting)
    return out
