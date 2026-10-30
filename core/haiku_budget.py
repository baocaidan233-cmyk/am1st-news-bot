"""Daily Claude budget across every bot on both VMs (2026-10-30, user: 「haiku每天
总开销在5块以内」, past it 「切换到luna」, and 「不要给DN单独划线」: one line for
all channels together).

Imported once by each production entry script (main.py, main_publish.py), it
wraps the Anthropic SDK's messages.create for the whole process:

  * after every answer, the call's cost at Haiku 5.5 list prices is added to
    this VM's total for the UTC day, kept in one small file that every bot on
    the VM shares (~/.haiku_budget/<day>.json, kept 30 days);
  * every SYNC_SECONDS one bot on the VM that has REDIS_URL (the news
    channels, DailyNews) reports the VM's total to the shared Redis and reads
    back the other VM's, so both VMs judge against the two together;
  * once the two VMs' total for the day reaches DAILY_LIMIT_USD, a call fails
    before it is sent, with the same out-of-credit error Anthropic returns, so
    every judge takes the path it already takes when the credit runs out: an
    hour on openai.fallback_model (gpt-6-luna), then Haiku is tried again and
    fails again until the UTC day turns.

The other VM's number is at most SYNC_SECONDS old; with Redis unreachable each
VM goes on its own total. Tests and one-off scripts do not import this, so they
neither count toward the budget nor get cut by it; the test key has its own
monthly limit ($50). Never fails a call for its own sake: a broken counter lets
calls through."""
from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import logging
import os
import socket
import time

import anthropic
import httpx
from anthropic.resources.messages import AsyncMessages, Messages

logger = logging.getLogger("haiku_budget")

DAILY_LIMIT_USD = 5.00
SYNC_SECONDS = 600
_DIR = os.path.expanduser("~/.haiku_budget")
_HOST = socket.gethostname()
_KEY = "haiku_budget"  # Redis hash per UTC day: host -> that VM's total
# Haiku 5.5, $ per million tokens (10-26). A cache write the usage does not
# split into 5 min / 1 h is priced as 1 h, the dearer one.
_IN, _CACHE_READ, _WRITE_5M, _WRITE_1H, _OUT = 0.10, 0.01, 0.125, 0.20, 0.50
_warned_day = ""


def _day() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _path(day: str) -> str:
    return os.path.join(_DIR, f"{day}.json")


def _read(day: str) -> dict:
    try:
        with open(_path(day), encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            return json.load(f)
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def _locked(day: str):
    """The day's record under an exclusive lock; changes are written back."""
    os.makedirs(_DIR, exist_ok=True)
    with open(_path(day), "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        if not raw.strip():
            _prune()
        d = json.loads(raw) if raw.strip() else {"usd": 0.0, "calls": 0}
        yield d
        f.seek(0)
        f.truncate()
        json.dump(d, f)


def _prune(keep_days: int = 30) -> None:
    """Day files older than keep_days go, once a day, when the new day's file starts."""
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=keep_days)).strftime("%Y-%m-%d")
    for name in os.listdir(_DIR):
        if name.endswith(".json") and name[:10] < cutoff:
            try:
                os.remove(os.path.join(_DIR, name))
            except OSError:
                pass


def spent(day: str | None = None) -> float:
    """Both VMs' Claude spend for the UTC day, in dollars (the other VM's as of the last sync)."""
    d = _read(day or _day())
    return float(d.get("usd", 0.0)) + float(d.get("others", 0.0))


def _redis_url() -> str:
    url = os.environ.get("REDIS_URL", "")
    if url:
        return url
    for p in (".env", "config/.env"):
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("REDIS_URL="):
                        return line.split("=", 1)[1].strip().strip("\"'")
        except OSError:
            pass
    return ""


def _sync(day: str) -> None:
    """At most once per SYNC_SECONDS per VM: this VM's total to Redis, the other's back."""
    url = _redis_url()
    if not url:
        return
    try:
        with _locked(day) as d:
            if time.time() - d.get("synced_at", 0) < SYNC_SECONDS:
                return
            d["synced_at"] = time.time()  # claimed before the network call, so one bot syncs
            mine = d.get("usd", 0.0)
        import redis
        r = redis.Redis.from_url(url, socket_timeout=3, socket_connect_timeout=3)
        key = f"{_KEY}:{day}"
        pipe = r.pipeline()
        pipe.hset(key, _HOST, f"{mine:.6f}")
        pipe.expire(key, 3 * 86400)
        pipe.hgetall(key)
        totals = pipe.execute()[2]
        others = sum(float(v) for k, v in totals.items() if k.decode() != _HOST)
        with _locked(day) as d:
            d["others"] = others
    except Exception as e:
        logger.warning("haiku_budget: could not sync with the other VM (%s) -- going on this VM's total", type(e).__name__)


def _cost(usage) -> float:
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    split = getattr(usage, "cache_creation", None)
    w5 = (getattr(split, "ephemeral_5m_input_tokens", 0) or 0) if split else 0
    w1 = write - w5
    return ((usage.input_tokens or 0) * _IN + cache_read * _CACHE_READ + w5 * _WRITE_5M + w1 * _WRITE_1H
            + (usage.output_tokens or 0) * _OUT) / 1e6


def _add(resp) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    try:
        with _locked(_day()) as d:
            d["usd"] = d.get("usd", 0.0) + _cost(usage)
            d["calls"] = d.get("calls", 0) + 1
    except Exception as e:
        logger.warning("haiku_budget: could not count a call (%s)", type(e).__name__)


def _check() -> None:
    global _warned_day
    day = _day()
    _sync(day)
    total = spent(day)
    if total < DAILY_LIMIT_USD:
        return
    if _warned_day != day:
        _warned_day = day
        logger.warning("haiku_budget: $%.2f spent on %s UTC by both VMs, past the $%.2f daily line -- Claude calls "
                       "fail as out of credit and go to the fallback model until the day turns", total, day, DAILY_LIMIT_USD)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    raise anthropic.BadRequestError(
        f"Your credit balance is too low: daily budget ${DAILY_LIMIT_USD:.2f} reached "
        f"(${total:.2f} spent on {day} UTC by both VMs; local limit, request not sent)",
        response=httpx.Response(400, request=request), body=None)


_create = Messages.create
_acreate = AsyncMessages.create


def _wrapped(self, *args, **kwargs):
    _check()
    resp = _create(self, *args, **kwargs)
    _add(resp)
    return resp


async def _awrapped(self, *args, **kwargs):
    _check()
    resp = await _acreate(self, *args, **kwargs)
    _add(resp)
    return resp


if not getattr(Messages.create, "_haiku_budget", False):
    _wrapped._haiku_budget = _awrapped._haiku_budget = True
    Messages.create = _wrapped
    AsyncMessages.create = _awrapped
