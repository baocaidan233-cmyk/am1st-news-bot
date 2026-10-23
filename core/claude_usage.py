"""One log line per Claude call with its token use, cache reads and writes
included (2026-10-23, user: 写上啊。不然以后怎么复盘), in the same shape as
Market Watcher's core/haiku.py, so cache hits and spend can be read off the
logs per step. Never raises: a call that worked is not failed by its log line."""
from __future__ import annotations

import logging

logger = logging.getLogger("claude_usage")


def log_usage(resp, label: str) -> dict:
    """{"in", "out", "cache_read", "cache_write"} of this response, logged."""
    try:
        u = resp.usage
        d = {"in": u.input_tokens or 0, "out": u.output_tokens or 0,
             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
             "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
        logger.info("Claude usage [%s] model=%s in=%d cache_read=%d cache_write=%d out=%d",
                    label, getattr(resp, "model", ""), d["in"], d["cache_read"], d["cache_write"], d["out"])
        return d
    except Exception:
        return {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
