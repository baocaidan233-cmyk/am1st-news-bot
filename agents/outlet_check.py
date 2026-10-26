"""Our caption checked against other outlets' reports of the same news
(2026-10-23, Linear AI-86; user: 那你先补上，参考DN的代码。不要照抄).

core/fact_check.py holds the caption to its own article, so an error the
article itself makes goes out with it. DailyNews checks its posts against
what other outlets print (its agents/caption_verify.py, AI-83); this is that
check fitted to AM1ST.

Evidence, published in the three days up to now: other outlets' articles in
AM1ST's event store (title and summary), then Google News searched with the
caption's first sentence, each result opened and dated by its own page
(agents/first_report_check.py's helpers). The caption's own outlet is never
evidence for itself. One Claude Haiku call -- gpt-4o-mini when Haiku cannot
answer (core/haiku.py, user 10-16) -- lists the statements the reports
support or contradict, each with a quote from both sides.

What differs from DailyNews. Offline on 151 AM1ST posts (10-23, DailyNews's
check as it is) 5 came back contradicted and none was our error: "more than
1,500%" against 1,600% and "nearly $42 million" against $40 million (readings
that agree), "strikes down part of the ban" against "threw out the ban" (ours
the more exact), a rally schedule the outlets gave differently, a cocaine
haul that looks like another seizure. The outlets AM1ST reads round and
restate each other far more than the wires DailyNews reads, so the judge's
label alone decides nothing. Code decides (contradiction()): both quotes must
be verbatim in their texts, as in DailyNews, and they must differ on
something checkable -- figures no reading reconciles (compared in code, with
"more than", "nearly", "about" read as ranges), a day that two time zones do
not explain (DailyNews's rule), a name the report gives in place of ours, or
a denial the report makes and we do not.

The fix is chosen in code too, where DailyNews corrects first and deletes
when the correction kept the words: the disputed figure, day or name in our
own article means the article and the outlet disagree, and the detail is
deleted (Caidan, 10-15, for DailyNews: 原文本身错了就删掉细节); not in our
article means the writer got it wrong, and FactChecker.fix corrects it from
the article. main_publish.py sends a changed caption back through the fact
check and drops one that still carries the disputed words. Anything failing
here leaves the caption as it was."""
from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote_plus, urlparse

import httpx
import trafilatura

import agents.first_report_check as frc
from agents.embedder import Embedder
from core import haiku
from core.claude_usage import log_usage
from core.config import AppConfig
from core.fact_check import _new_note
from core.openai_client import create_openai_client
from core.qdrant_store import EventStore
from core.time_words import EASTERN, _outlet_tz, _us_event

logger = logging.getLogger(__name__)

_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "outlet_check.jsonl")

_JUDGE = """You check a short news post that we published against OTHER outlets' reports of the same news (headline and the start of each article). The reports are evidence; the post is what is being checked.

For each factual statement in the post that at least one report also speaks to, decide:
- SUPPORTED: a report states the same thing.
- CONTRADICTED: a report gives an incompatible account of the same thing at the same moment: a different number, person, place, date or outcome, or says it did not happen, was denied or retracted.
Leave out statements no report speaks to. A report about a different event, or a different occasion of the same kind of event, is not evidence. A figure a later report updates (a death toll that rises) is not CONTRADICTED, nor are two figures that measure different things. Wording, emphasis and the post's own characterisation are not checked.

For each statement copy an exact short quote from the post and an exact short quote from the report, verbatim.
verdict: CONTRADICTED if any statement is CONTRADICTED, else SUPPORTED if any is SUPPORTED, else NO_EVIDENCE."""

_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["checks", "verdict"], "properties": {
    "checks": {"type": "array", "items": {"type": "object", "additionalProperties": False,
               "required": ["post_quote", "report", "report_quote", "label", "reason"],
               "properties": {"post_quote": {"type": "string"}, "report": {"type": "integer"},
                              "report_quote": {"type": "string"},
                              "label": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED"]},
                              "reason": {"type": "string"}}}},
    "verdict": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "NO_EVIDENCE"]}}}

_DELETE = """You edit a short English news post that is about to be published. It has two parts separated by a blank line.
You get the post and details in it that other outlets dispute.

Delete each listed detail. Do not replace it with another figure, name or day, and add no new fact. Keep every other word exactly as it is and keep each sentence grammatical; if a sentence means nothing without the detail, delete that sentence. Keep the blank line between the two parts.
Return only the edited post, nothing else."""

# --------------------------------------------------------------------- days
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_DAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
                  r"|\b(?:%s)[a-z]*\.?\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:%s)[a-z]*\b" % (_MONTHS, _MONTHS), re.I)


# ------------------------------------------------------------------ figures
# A figure is read as the range of values its wording allows; two figures
# agree when their ranges meet. "more than 1,500%" and "1,600%" agree,
# "nearly $42 million" and "$40 million" agree, "1,309 pounds" and "nearly
# 1,800 pounds" do not. Exact figures allow half a percent for rounding.
_ABOVE = r"more than|over|at least|upwards of|above|exceeding|topped|topping"
_NEARLY = r"nearly|almost|just under|close to"
_BELOW = r"under|less than|fewer than|below|up to"
_ABOUT = r"about|around|roughly|approximately|an estimated|some"
_FIG = re.compile(
    r"(?:\b(?P<q>%s|%s|%s|%s)\s+)?\$?(?<![\d.])(?P<n>\d+(?:,\d{3})*(?:\.\d+)?)(?::(?P<min>[0-5]\d))?"
    r"(?P<ampm>\s*[ap]\.?m\b\.?)?(?P<pct>\s*(?:%%|percent\b))?"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion)\b)?" % (_ABOVE, _NEARLY, _BELOW, _ABOUT), re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def _figures(text: str) -> list[dict]:
    out = []
    for m in _FIG.finditer(text or ""):
        v = float(m.group("n").replace(",", ""))
        if m.group("min") or m.group("ampm"):
            kind, v = "time", v + int(m.group("min") or 0) / 60
        elif m.group("pct"):
            kind = "pct"
        else:
            kind = "n"
            v *= _SCALE.get((m.group("scale") or "").lower(), 1)
        q = (m.group("q") or "").lower()
        lo, hi = ((v, math.inf) if re.fullmatch(_ABOVE, q) else (0.9 * v, v) if re.fullmatch(_NEARLY, q)
                  else (0, v) if re.fullmatch(_BELOW, q) else (0.9 * v, 1.1 * v) if re.fullmatch(_ABOUT, q)
                  else (0.995 * v, 1.005 * v))
        written = m.group("n") + (":" + m.group("min") if m.group("min") else "") + (m.group("ampm") or "")
        out.append({"kind": kind, "lo": lo, "hi": hi, "text": written.strip()})
    return out


def _figure_conflict(ours: str, theirs: str) -> list[str]:
    """Our figures that no figure of theirs (of the same kind) agrees with,
    when the two quotes have comparable figures at all. Dates are left to
    _day_conflict and its time-zone rule."""
    a, b = _figures(_DAY.sub(" ", ours)), _figures(_DAY.sub(" ", theirs))
    pairs = [(x, y) for x in a for y in b if x["kind"] == y["kind"]]
    if not pairs or any(x["lo"] <= y["hi"] and y["lo"] <= x["hi"] for x, y in pairs):
        return []
    return sorted({x["text"] for x, _ in pairs})


def _days(text: str) -> list[tuple[str, int, str]]:
    out = []
    for m in _DAY.finditer(text or ""):
        if m.group(1):
            out.append(("wd", _WEEKDAYS.index(m.group(1).lower()), m.group(0)))
        else:
            out.append(("md", int(m.group(2) or m.group(3)), m.group(0)))
    return out


def _day_conflict(ours: str, theirs: str, post: str, src_url: str, ev_url: str) -> list[str]:
    """Our days when both quotes name days of the same form and none match --
    except a one-day gap that two time zones explain (DailyNews's rule: our
    day is US Eastern for a US event, else the article's outlet's zone). An
    outlet core/time_words.py does not place abroad counts as Eastern here:
    almost every outlet AM1ST reads is American, and DailyNews's reading of
    "zone unknown" as "maybe another zone" would excuse every one-day gap."""
    a, b = _days(ours), _days(theirs)
    pairs = [(x, y) for x in a for y in b if x[0] == y[0]]
    if not pairs or any(x[1] == y[1] for x, y in pairs):
        return []
    one_apart = all((abs(x[1] - y[1]) % 7 in (1, 6)) if x[0] == "wd" else abs(x[1] - y[1]) == 1 for x, y in pairs)
    if one_apart:
        i = post.lower().find(ours.lower())
        us = i >= 0 and _us_event(post, i, i + len(ours))
        zone_a, zone_b = (EASTERN if us else (_outlet_tz(src_url) or EASTERN)), (_outlet_tz(ev_url) or EASTERN)
        now = datetime.now(timezone.utc)
        if now.astimezone(zone_a).utcoffset() != now.astimezone(zone_b).utcoffset():
            return []
    return sorted({x[2] for x, _ in pairs})


# -------------------------------------------------------------------- names
_NOT_NAMES = {"the", "a", "an", "president", "breaking", "just", "in", "new", "former", "and", "of"} | set(_WEEKDAYS)


def _names(text: str) -> set[str]:
    """Capitalised words, not at the start of a sentence or quote."""
    out, text = set(), text or ""
    for m in re.finditer(r"\b[A-Z][a-zA-Z'-]+", text):
        before = text[:m.start()].rstrip()
        w = m.group(0)
        if (before and before[-1] not in ".!?:\"“‘'(" and w.lower() not in _NOT_NAMES
                and not re.fullmatch(r"(?:%s)[a-z]*" % _MONTHS, w.lower())):
            out.add(w)
    return out


def _name_conflict(ours: str, theirs: str) -> list[str]:
    """Our names when the report puts another name where ours stands."""
    a, b = _names(ours), _names(theirs)
    return sorted(a - b) if (a - b) and (b - a) else []


_DENIAL = re.compile(r"\b(?:not|never|no|denied|denies|deny|false|retracted|didn't|did not|wasn't|hasn't)\b", re.I)


def contradiction(c: dict, post: str, report: dict, src_url: str) -> Optional[dict]:
    """{"kind", "ours": [the words of ours in dispute]} or None. See the module docstring."""
    pq, rq = c["post_quote"], c["report_quote"]
    for kind, ours in (("figure", _figure_conflict(pq, rq)),
                       ("day", _day_conflict(pq, rq, post, src_url, report["url"])),
                       ("name", _name_conflict(pq, rq))):
        if ours:
            return {"kind": kind, "ours": ours}
    if _DENIAL.search(rq) and not _DENIAL.search(pq):
        return {"kind": "denial", "ours": [pq]}
    return None


# -------------------------------------------------------------------- utils
def _host(url: str) -> str:
    return urlparse(url or "").netloc.lower().removeprefix("www.")


def _words(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", re.sub(r"(?<=\d),(?=\d)|[‘’“”\"']", "", (s or "").lower())).split())


def _has(text: str, words: str) -> bool:
    w = _words(words)
    return bool(w) and re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", _words(text)) is not None


def in_article(con: list[dict], title: str, article: str) -> bool:
    """Every disputed figure, day and name is in our own article too."""
    src = f"{title}\n{article}"
    return all(_has(src, w) for x in con for w in x["ours"])


def still_disputed(post: str, con: list[dict]) -> bool:
    return any(_has(post, w) for x in con for w in x["ours"])


def problems(con: list[dict]) -> list[str]:
    return [f"\"{x['post_quote']}\" is not what the article says; another outlet ({_host(x['report_url'])}) "
            f"reports \"{x['report_quote']}\". Use the article's own reading." for x in con]


def _log(row: dict) -> None:
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row},
                               ensure_ascii=False) + "\n")
    except Exception:
        pass


class OutletCheck:
    def __init__(self, config: AppConfig, event_store: EventStore, embedder: Embedder) -> None:
        oc = config.outlet_check
        self.enabled = oc.enabled and bool(oc.api_key)
        self._key = oc.api_key
        self._days_back = oc.days_back
        self._max_google = oc.max_google
        self._store = event_store
        self._embedder = embedder
        self._openai_model = config.openai.chat_model
        self._openai = None
        if self.enabled and config.openai.api_key:
            try:
                self._openai = create_openai_client(config)
            except Exception as e:
                logger.warning("OutletCheck: no gpt-4o-mini fallback (%s)", type(e).__name__)

    # ------------------------------------------------------------- evidence
    async def _from_store(self, post: str, src_url: str, t: float) -> list[dict]:
        res = await self._store._client.query_points(self._store._collection, query=await self._embedder.embed(post),
                                                     limit=15, with_payload=["url", "text", "published_at"])
        out: list[dict] = []
        for h in res.points:
            p = h.payload or {}
            u = p.get("url") or ""
            try:
                pt = float(p.get("published_at"))
            except (TypeError, ValueError):
                continue
            if (h.score < 0.6 or not (t - self._days_back * 86400 <= pt <= t) or _host(u) == _host(src_url)
                    or any(_host(o["url"]) == _host(u) for o in out)):
                continue
            out.append({"url": u, "text": frc._clean(p.get("text") or "")[:700]})
            if len(out) == 3:
                break
        return out

    async def _from_google(self, post: str, src_url: str, t: float, skip: list[dict]) -> list[dict]:
        day = datetime.fromtimestamp(t, timezone.utc).date()
        first = re.split(r"(?<=[.!?])\s+", post.strip())[0]
        q = quote_plus(f"{frc._keywords(first)} after:{day - timedelta(days=self._days_back)} before:{day + timedelta(days=1)}")
        out: list[dict] = []
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en", headers=frc._HDR)
            for item in re.findall(r"<item>(.*?)</item>", r.text, re.S)[:self._max_google + 4]:
                head = frc.htmlmod.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "",
                                                   (re.search(r"<title>(.*?)</title>", item, re.S) or [None, ""])[1]))
                link = (re.search(r"<link>(.*?)</link>", item, re.S) or [None, ""])[1]
                try:
                    real = await frc.FirstReportCheck._resolve(client, link)
                    if not real or _host(real) == _host(src_url) or any(_host(o["url"]) == _host(real) for o in out + skip):
                        continue
                    page = await client.get(real, headers=frc._HDR)
                    if page.status_code != 200:
                        continue
                    page_day = frc._page_day(page.text)
                    body = frc._clean(" ".join((trafilatura.extract(page.text, include_comments=False) or "").split("\n")))
                except Exception:
                    continue
                if page_day is None or not (day - timedelta(days=self._days_back) <= page_day <= day) or not body:
                    continue
                out.append({"url": real, "text": frc._clean(f"{head}\n{body[:1500]}")[:1600]})
                if len(out) == self._max_google:
                    break
        return out

    async def _judge(self, user: str) -> tuple[Optional[dict], str]:
        d = await haiku.ask(self._key, _JUDGE, user, schema=_SCHEMA, cache=True, cache_ttl="1h")
        if d is not None:
            return d, haiku.MODEL
        if self._openai is None:
            return None, ""
        try:  # Haiku cannot answer: the same question once to gpt-4o-mini (user 10-16)
            resp = await self._openai.chat.completions.create(
                model=self._openai_model, temperature=0, max_tokens=4000,
                messages=[{"role": "system", "content": _JUDGE}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "outlet_check", "strict": True, "schema": _SCHEMA}})
            return json.loads(resp.choices[0].message.content or ""), self._openai_model
        except Exception as e:
            logger.warning("OutletCheck: fallback %s", type(e).__name__)
            return None, ""

    # ---------------------------------------------------------------- check
    async def check(self, post: str, url: str) -> dict:
        """{"verdict": CONTRADICTED|SUPPORTED|NO_EVIDENCE|ERROR|OFF, "contradicted": [...]}. Fails open."""
        if not self.enabled:
            return {"verdict": "OFF", "contradicted": []}
        t = datetime.now(timezone.utc).timestamp()
        result: dict = {"verdict": "ERROR", "contradicted": [], "evidence": []}
        try:
            ev: list[dict] = []
            try:
                ev += await self._from_store(post, url, t)
            except Exception as e:
                logger.info("OutletCheck: store search failed (%s) for %s", type(e).__name__, url)
            try:
                ev += await self._from_google(post, url, t, ev)
            except Exception as e:
                logger.info("OutletCheck: Google search failed (%s) for %s", type(e).__name__, url)
            result["evidence"] = [e["url"] for e in ev]
            if not ev:
                result["verdict"] = "NO_EVIDENCE"
            else:
                user = f"POST:\n{post}\n\n" + "\n\n".join(f"REPORT {i + 1} ({_host(e['url'])}):\n{e['text']}"
                                                           for i, e in enumerate(ev))
                d, result["model"] = await self._judge(user)
                if d is not None:
                    con = []
                    for c in d.get("checks", []):
                        k = c.get("report", 0) - 1
                        if c.get("label") != "CONTRADICTED" or not (0 <= k < len(ev)):
                            continue
                        if not (_words(c["post_quote"]) and _has(post, c["post_quote"])
                                and _has(ev[k]["text"], c["report_quote"])):
                            continue
                        why = contradiction(c, post, ev[k], url)
                        if why:
                            con.append({**c, **why, "report_url": ev[k]["url"]})
                    result["checks"] = d.get("checks", [])
                    result["contradicted"] = con
                    result["verdict"] = ("CONTRADICTED" if con else "SUPPORTED"
                                         if any(c.get("label") == "SUPPORTED" for c in result["checks"]) else "NO_EVIDENCE")
        except Exception as e:
            logger.warning("OutletCheck: %s for %s -- letting it through", type(e).__name__, url)
            result["verdict"] = "ERROR"
        _log({"url": url, "stage": "check", "post": post, **result})
        return result

    # ---------------------------------------------------------------- delete
    async def delete(self, post: str, con: list[dict], url: str = "") -> Optional[str]:
        """Haiku cuts the disputed details (gpt-4o-mini when Haiku fails), or None."""
        user = "POST:\n" + post + "\n\nDETAILS TO DELETE:\n" + "\n".join(f"- \"{w}\"" for x in con for w in x["ours"])
        text, model = None, haiku.MODEL
        if haiku.available():
            try:
                resp = await haiku._client(self._key).messages.create(
                    model=haiku.MODEL, max_tokens=4000, system=_DELETE,
                    messages=[{"role": "user", "content": user}], output_config={"effort": "low"})
                log_usage(resp, "outlet check delete")
                if resp.stop_reason != "refusal":
                    text = next((b.text for b in resp.content if b.type == "text"), "")
            except Exception as e:
                haiku.note_error(e)
                logger.warning("OutletCheck: delete failed (%s)", type(e).__name__)
        if text is None and self._openai is not None:
            try:
                resp = await self._openai.chat.completions.create(
                    model=self._openai_model, temperature=0, max_tokens=1000,
                    messages=[{"role": "system", "content": _DELETE}, {"role": "user", "content": user}])
                text, model = resp.choices[0].message.content or "", self._openai_model
            except Exception as e:
                logger.warning("OutletCheck: fallback delete failed (%s)", type(e).__name__)
        if not text:
            return None
        text = "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip()) if p.strip())
        text = re.sub(r"\s*,\s*([.!?])", r"\1", text)
        if not text or " ".join(text.split()) == " ".join(post.split()) or _new_note(text, post):
            return None
        _log({"url": url, "stage": "delete", "post": text, "model": model})
        return text
