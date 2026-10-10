"""Is this still news? The code half of agents/timeliness_check.py (2026-10-10).

The user's rules (2026-10-09 / 10-10):
  - old or new is decided by when the thing FIRST BECAME PUBLIC, not when it
    happened. An old event revealed for the first time is news, and must never
    be stopped;
  - what is stopped is a story first reported 48 hours or more before we
    publish, with nothing new in the article;
  - this is a news channel: commentary on an old event with nothing new is not
    published either (the 2026-09-05 OPINION pass-through is gone).

A model copies the developments and the article's own time words
(prompts/timeliness_extract_prompt.txt); everything here is code, so it can be
tested without a model. A story is only stopped on evidence. Anything this
cannot read -- no time, a vague time, a sentence the model made up -- leaves
the story fresh.

Measured on AM1ST's own posts (Claude-labelled, the four doubtful ones by the
user): 10/06-10/09, 179 posts with full text -> all 13 stale ones stopped, one
wrongly (a Missouri arrest on a Friday that local TV only reported 10/06-07:
"happened" dates are when it happened, and nothing in the article says when it
became public). Held out, 10/09-10/10, 58 posts -> 3 stopped, all stale.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
_MON = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_WD = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _day_end(d: date) -> datetime:
    """A day-only time is read as the last minute of that day, so a story is
    never called older than the most generous reading of its own words. Read
    as noon instead, the 10/06-10/09 replay stopped 4 more posts, one of them
    Axios's first report of Emmer's closed-door remarks."""
    return datetime.combine(d, time(23, 59), EASTERN)


def latest_moment(phrase: Optional[str], anchor: Optional[date]) -> Optional[datetime]:
    """The latest moment the phrase can mean, counting from the article's own
    publication day; None if it is unreadable, vague, or a plan."""
    if not phrase or not anchor:
        return None
    p = phrase.lower()
    # A data cut-off is not when the data came out: "Nigeria's external debt
    # stood at $54.5bn as of June 30" was the Debt Office's release of 10/08.
    if re.search(r"\b(as of|as at|at the end of|by the end of|through the end of)\b|截至|截止|по состоянию на|станом на|към края", p):
        return None
    # Nor is a point of comparison: gold "above $4,200 for the first time
    # since Oct. 2" (现货黄金…为10月2日以来首次) is today's move.
    if re.search(r"\bsince\b|以来|以後|以后|с начала|начиная с|від початку|откакто", p):
        return None
    if re.search(r"\b(will|next|beginning|starting|later this|due to|scheduled|upcoming|until)\b", p):
        return None
    # Month-day, taking the latest when the phrase names several or a range:
    # "from October 4 to 8" is the 8th (the SCO exercise ended on the 8th,
    # and reading the 4th stopped that post). The day must not run into more
    # digits: "September 2026" is not September 20.
    found = []
    for m in re.finditer(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?!\d)(?:st|nd|rd|th)?"
                         r"(?:\s*(?:-|\u2013|\u2014|to|through|and)\s*(\d{1,2})(?!\d))?(?:,?\s*(\d{4}))?", p):
        day = int(m.group(3) or m.group(2))
        y = int(m.group(4)) if m.group(4) else anchor.year
        try:
            d = date(y, _MON[m.group(1)], day)
        except ValueError:
            continue
        # A month-day with no year is last year's only when it is far ahead:
        # "Tuesday, Oct. 13" on Oct. 8 is a rally being planned, and pushing
        # it back a year stopped that post in the first replay.
        if not m.group(4) and d > anchor + timedelta(days=120):
            d = date(y - 1, d.month, d.day)
        found.append(d)
    if found:
        d = max(found)
        return None if d > anchor else _day_end(d)
    # A month with its year ("in August 2023") is the end of that month. A
    # month alone stays too vague (below).
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(\d{4})\b", p)
    if m:
        y, mo = int(m.group(2)), _MON[m.group(1)]
        d = (date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1))
        return None if d > anchor else _day_end(d)
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", p)
    if m:
        try:
            d = date(*map(int, m.groups()))
        except ValueError:
            return None
        return None if d > anchor else _day_end(d)
    if re.search(r"\b(today|this (morning|afternoon|evening)|tonight|earlier today|hours ago|just now)\b", p):
        return _day_end(anchor)
    if re.search(r"\b(yesterday|last night|overnight)\b", p):
        return _day_end(anchor - timedelta(days=1))
    if re.search(r"\b(this weekend|over the weekend|this week)\b", p):
        return _day_end(anchor)
    if re.search(r"\blast week\b", p):
        # Only an estimate. Emmer's donor remarks, first reported by Axios on
        # 10/06, were "last week".
        return None
    for k, w in enumerate(_WD):
        if re.search(r"\b" + w + r"\b", p):
            back = (anchor.weekday() - k) % 7
            if "last " + w in p and back == 0:
                back = 7
            return _day_end(anchor - timedelta(days=back))
    # "earlier this month", "in August", a bare year: too vague to stop a
    # story on. "Signing a contract in August" headed an AP story that was
    # news because AP had just obtained the records.
    return None


# --- this outlet is the first to publish it -------------------------------

_EXCLUSIVE = re.compile(r"\bEXCLUSIVE\b|\bExclusive\s*[|:—-]|\bSCOOP\b|\bScoop\s*[|:—-]")
_OWN_INTERVIEW = re.compile(r"专访|專訪")  # "VOA专访俞大㵢": the outlet's own interview, first public in this article

_FIRST_ON = re.compile(r"\bfirst on ((?:[A-Z][\w.&'-]*\s?){1,3})", re.I)
_TOLD = re.compile(r"\b(?:told|provided to|obtained by|shared with|confirmed to|in an interview with)\s+(?:the\s+)?((?:[A-Z][\w.&'-]*\s?){1,4})")
_WE = re.compile(r"\b(we asked|told us|asked by this outlet|this outlet has learned|reached by)\b", re.I)
_OWN_VERB = (r"(?:recently\s+)?(?:reviewed|obtained|visited|found|learned|analy[sz]ed|asked|reached out|spoke|"
             r"interviewed|requested|confirmed|first reported|reported exclusively)")
_OWN_SUBJECT = re.compile(r"((?:The\s+)?(?:[A-Z][\w.&'-]*\s){1,3})" + _OWN_VERB)


def _host_word(url: str) -> str:
    h = re.sub(r"[^a-z]", "", urlparse(url).netloc.lower().replace("www.", "").split(".")[0])
    return h[3:] if h.startswith("the") and len(h) > 6 else h


def names_this_outlet(outlet: str, url: str) -> bool:
    """Whether a name in the text is the outlet whose article this is: told
    Blaze News on theblaze.com, The Post on nypost.com. Containment either
    way, not a shared prefix -- a prefix made "the California Post", quoted by
    californiaglobe.com, read as the Globe itself."""
    o = re.sub(r"[^a-z]", "", re.sub(r"^\s*the\s+", "", (outlet or "").lower()))
    h = _host_word(url)
    return len(o) >= 3 and bool(h) and (o in h or h in o)


def first_disclosure(title: str, body: str, url: str) -> bool:
    """The article says it is where this became public: EXCLUSIVE / SCOOP in
    the headline or opening it, FIRST ON <this outlet>, told / obtained by
    <this outlet>, <this outlet> visited / reviewed ..., we asked. User,
    2026-10-10: scoop and "first on Fox" count the same as an exclusive."""
    if _EXCLUSIVE.search(title or "") or _EXCLUSIVE.match((body or "").lstrip()[:40]) or _OWN_INTERVIEW.search(title or ""):
        return True
    text = f"{title}\n{(body or '')[:4000]}"
    if _WE.search(text):
        return True
    for rx in (_TOLD, _FIRST_ON, _OWN_SUBJECT):
        for m in rx.finditer(text):
            if names_this_outlet(m.group(1), url):
                return True
    return False


# "posted this week by Garland Favorito", "newly declassified": the thing came
# out later than it happened (the Fulton voter video, 09/28 meeting, posted 10/07).
_LATE = re.compile(r"\b(posted|released|surfaced|published|made public) (this week|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"
                   r"|\b(newly|just) (released|declassified|unsealed|obtained|surfaced)\b", re.I)

# "According to a Reuters exclusive published Friday": the article itself dates
# when someone first published it (the Prince / Congo ambush of 09/17).
_DISCLOSED = re.compile(
    r"\b(?:exclusive|first reported|reported|revealed|disclosed|published|broke the news|aired)\b[^.\n]{0,40}?\b("
    r"today|yesterday|last night|this (?:morning|afternoon|week)|(?:on |late |early )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|(?:on )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2})\b", re.I)


_PRIVATE = re.compile(r"closed[- ]door|behind closed doors|private (?:meeting|briefing|call|dinner|fundraiser)|"
                      r"according to (?:a |two |three |several )?(?:person|people|sources?|officials?) (?:familiar|with (?:direct )?knowledge)|"
                      r"leaked|secretly|не для печати|закрыт\w* (?:встреч|заседани)|闭门|閉門|知情人士", re.I)


def _norm(t: str) -> str:
    t = (t or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def decide(developments: list[dict], title: str, body: str, url: str, published_at: datetime,
           now: datetime, window_hours: float = 48) -> dict:
    """{"verdict": "fresh" | "stale", "reason": str, "news_at": iso | None,
    "age_hours": float | None}. published_at is the article's own time; it
    anchors "Monday" and "yesterday"."""
    def out(verdict, reason, t=None):
        age = round((now - t).total_seconds() / 3600, 1) if t else None
        return {"verdict": verdict, "reason": reason, "news_at": t.isoformat() if t else None, "age_hours": age}

    anchor = published_at.astimezone(EASTERN).date()
    if first_disclosure(title, body, url):
        return out("stale" if (now - published_at) >= timedelta(hours=window_hours) else "fresh",
                   "this article is the first report", published_at)
    devs = [d for d in (developments or [])[:3] if isinstance(d, dict)]
    if not devs:
        return out("fresh", "no development copied")
    source = _norm(f"{title}\n{body}")
    # The headline's development decides. If its sentence is not in the
    # article the answer is not evidence, and the next development is not
    # promoted in its place: on Blakeman's 10/06 rally those were two
    # background arrests from September.
    head = devs[0]
    if not head.get("what") or _norm(head["what"]).rstrip(".") not in source:
        return out("fresh", "headline sentence not found in the article")
    times = []
    for d in devs:
        # A closed-door meeting or a source's account has no public moment of
        # its own: it became public when someone reported it. The F-35 parts
        # story dated Pentagon officials' closed-door briefing of 09/29, which
        # Bloomberg reported days later; Emmer's donor remarks were the same.
        if _PRIVATE.search(d.get("what") or ""):
            times.append(None)
            continue
        when = d.get("when")
        if not isinstance(when, str) or _norm(when) not in source:
            when = None
        elif re.search(r"(?:since|as of|as at)\s+(?:the\s+)?" + re.escape(_norm(when)) + "|" + re.escape(_norm(when)) + r"\s*(?:以来|以後|以后)",
                       _norm(d.get("what") or "")):
            when = None   # the model copied only the date out of "since Oct. 2" / "10月2日以来"
        times.append(latest_moment(when, anchor))
    # A plan in the headline sentence: a bare weekday is the coming one
    # ("Sefcovic will head to Beijing on Thursday" is next Thursday's talks,
    # not last Thursday's), so it is no evidence of age. A plan with a full
    # date that has already passed still counts: "the case will be heard by
    # the Supreme Court on Oct. 5", published on Oct. 10, is a piece written
    # before its own event.
    if times[0] is not None and re.search(r"\b(will|is set to|are set to|is due to|plans to|is scheduled to)\b|将于|将在",
                                          head.get("what") or "") \
            and not re.search(r"\d", head.get("when") or ""):
        times[0] = None
    relayed = head.get("cited_outlet") and not names_this_outlet(head["cited_outlet"], url)
    # Something that happened, known only through someone else's report that
    # the article does not date, became public when that report came out --
    # not when it happened: Shi Tingfu's death in prison on 09/22, "according
    # to the Washington-based nonprofit"; the Missouri arrest of a Friday,
    # which KSDK reported days later. A dated "said" development (the KCRA
    # report "on Friday night") is that report's time and still counts.
    if times[0] is not None and head.get("kind") == "happened" and relayed \
            and not any(x is not None and d.get("kind") == "said" for d, x in list(zip(devs, times))[1:]):
        times[0] = None
    relay = times[0] is not None and relayed
    if not relay and _LATE.search(body[:4000]):
        return out("fresh", "the article says it only now came out")
    own = [n for n in {_host_word(url)} if len(n) >= 4]
    for m in _DISCLOSED.finditer(body[:6000]):
        before = re.sub(r"[^a-z]", "", body[max(0, m.start() - 40):m.start()].lower())
        if any(n in before for n in own):
            continue  # "As The Gateway Pundit reported Tuesday": its own earlier story
        e = latest_moment(m.group(1), anchor)
        if e and now - e < timedelta(hours=window_hours):
            return out("fresh", "the article says it was first reported %r" % m.group(0)[:60], e)
    t = times[0]
    if t is None:
        return out("fresh", "no readable time for the headline's development")
    if now - t >= timedelta(hours=window_hours):
        # Something said in the window is a new development, so not "nothing
        # new": Joe Abraham's remarks of 10/09 under a story the model headed
        # with the January 2025 crash. Said, not happened: the Tillis post was
        # rescued by "appeared on The View Tuesday".
        for d, x in list(zip(devs, times))[1:]:
            if x and d.get("kind") == "said" and now - x < timedelta(hours=window_hours):
                return out("fresh", "another development is new: %r" % d.get("when"), x)
        return out("stale", "%r is at least %.0fh before now" % (head.get("when"), (now - t).total_seconds() / 3600), t)
    return out("fresh", "%r is within the window" % head.get("when"), t)
