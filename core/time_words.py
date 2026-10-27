"""Every day a post names is a day in US Eastern, and one the source gives.

Caidan, 2026-10-13: 写美国的事情，全部美东时间。非美国的事情，原文写几号，周几，
就用原文信息. This supersedes the all-Eastern rule described below: a day is
moved to the Eastern publication date only for a US event (_us_event: a US
person, office or place before the day or in the eight words after it) or
beside an Eastern clock the writer put there; any other event keeps the
source's own day and its local clock. Same as DailyNews f235a63, where the
editor had put "Tuesday" back on the People's Bank of China's approval of
Visa that the Eastern rule made Monday.

Ported unchanged from DailyNews core/time_words.py (7efe07d, read-only, 2026-10-11)
at the user's request: 时效性不能错 -- learn DailyNews' time handling. Its
notes below are DailyNews' own and are kept as written; this channel's replay
is recorded in the commit that wired it in.

Caidan, 2026-10-11: 所有时间都是美东时间. One rule replaces the special
cases this module grew one at a time:

    Each weekday or date in a post is matched, as a calendar day, to a day the
    source names. It is then written as that day in Eastern, in the source's
    own form (原文写周几就写周几, 2026-10-05). A day the source does not name
    is cut, with its preposition.

"As that day in Eastern" is the source's own day except in one case, the
only one that can be settled from the text: the source's day is later than
the Eastern publication date and the post reports it as past. Something
already reported happened before it was published, so it happened on the
Eastern publication date -- whatever zone the text is written in. A Korea
Times story out at 11:41 Sunday Seoul time says "on Sunday"; in Eastern
that is Saturday, and Saturday is what the post says.

Why not convert every day from the outlet's zone. Replayed on 812 drafts
2026-10-11: the zone a text is written in is not the outlet's. The
Guardian's US desk, Anadolu's Americas desk, Korea Times and Asia Times
running wire copy all write Washington's "Friday" for a Washington event.
Converting from the outlet's zone cut 180 days from East Asian posts
against 61 before, most of them right as written. A day the source gives
for an earlier day, or a day yet to come, therefore stands as written.

Also: a wire dateline is when the agency filed, not when anything happened
(core/dateline.py); a day the source gives only there does not count. A
bare 日 (财联社 "古特雷斯9日") is a date in the month it was written in.

2026-10-12, three DailyNews posts an editor corrected (ported from DailyNews
the same day):

- A day moved to Eastern takes its local clock with it. TASS: missiles
  fired "on Monday ... at about 9:20 a.m. local time"; the post said
  Sunday (right: 20:20 Eastern) and kept "9:20 a.m. local time" -- a
  Sunday morning in Seoul that never happened. When a day in the post is
  the Eastern day of a later local one, every clock not marked Eastern is
  cut, and so are the local "early", "morning", "local time" around it.
- 今天 in a dated Chinese wire story is the dateline's day. 中央社
  "（…上海12日電）中國外交部今天上午…": the post's Sunday was cut as a
  day the source never gives, and the post went out with no day at all.
- 财联社 writes "【headline】财联社10月12日电，近日…". The 【】 in front
  hid the dateline, so its 12日 licensed "completed on October 11" for a
  data centre the source said was finished 近日 (recently).

Not covered: the right day on the wrong person. 2026-10-11 Axios had "Trump
said Friday"; the post gave Friday to Fitzpatrick, whose day the source
never gives. It is a day the source names, so it stands.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from core.wire_dateline import _CNA, wire_dateline

EASTERN = ZoneInfo("America/New_York")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# The source's own language. Latin-script words are matched as whole words;
# the rest as substrings, which is how those scripts write them.
_WD_LATIN = {
    0: ("montag", "lundi", "lunes", "pazartesi"),
    1: ("dienstag", "mardi", "martes", "salı"),
    2: ("mittwoch", "mercredi", "miércoles", "miercoles", "çarşamba"),
    3: ("donnerstag", "jeudi", "jueves", "perşembe"),
    4: ("freitag", "vendredi", "viernes", "cuma"),
    5: ("samstag", "sonnabend", "samedi", "sábado", "sabado", "cumartesi"),
    6: ("sonntag", "dimanche", "domingo", "pazar"),
}
for _i, _ws in {0: ("luni", "segunda-feira", "segunda", "lunedì", "lunedi"),
                1: ("marți", "marti", "terça-feira", "terça", "terca", "martedì", "martedi"),
                2: ("miercuri", "quarta-feira", "quarta", "mercoledì", "mercoledi"),
                3: ("joi", "quinta-feira", "quinta", "giovedì", "giovedi"),
                4: ("vineri", "sexta-feira", "sexta", "venerdì", "venerdi"),
                5: ("sâmbătă", "sambata", "sabato"),
                6: ("duminică", "duminica", "domenica")}.items():
    _WD_LATIN[_i] = _WD_LATIN[_i] + _ws
_WD_OTHER = {
    0: ("الاثنين", "الإثنين", "周一", "星期一", "週一", "禮拜一", "礼拜一", "月曜", "понедельник"),
    1: ("الثلاثاء", "周二", "星期二", "週二", "禮拜二", "礼拜二", "火曜", "вторник"),
    2: ("الأربعاء", "الاربعاء", "周三", "星期三", "週三", "禮拜三", "礼拜三", "水曜", "сред"),
    3: ("الخميس", "周四", "星期四", "週四", "禮拜四", "礼拜四", "木曜", "четверг"),
    4: ("الجمعة", "周五", "星期五", "週五", "禮拜五", "礼拜五", "金曜", "пятниц"),
    5: ("السبت", "周六", "星期六", "週六", "禮拜六", "礼拜六", "土曜", "суббот"),
    6: ("الأحد", "الاحد", "周日", "星期日", "星期天", "週日", "禮拜天", "礼拜天", "日曜", "воскресень"),
}
for _i, _ws in {0: ("понеділ", "понеделник"), 1: ("вівтор",), 2: ("серед", "сряда"), 3: ("четвер", "четвъртък"),
                4: ("п'ятниц", "п’ятниц", "петък"), 5: ("субот", "събота"), 6: ("неділ", "неделя")}.items():
    _WD_OTHER[_i] = _WD_OTHER[_i] + _ws

_MONTHS = {
    # English
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    # German
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5, "juni": 6,
    "juli": 7, "oktober": 10, "okt": 10, "dezember": 12, "dez": 12,
    # French
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10,
    "novembre": 11, "décembre": 12, "decembre": 12,
    # Spanish
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    # Arabic, both the Egyptian and the Levantine names
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4, "مايو": 5,
    "يونيو": 6, "يوليو": 7, "أغسطس": 8, "اغسطس": 8, "سبتمبر": 9, "أكتوبر": 10,
    "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    "كانون الثاني": 1, "شباط": 2, "آذار": 3, "نيسان": 4, "أيار": 5, "حزيران": 6,
    "تموز": 7, "آب": 8, "أيلول": 9, "تشرين الأول": 10, "تشرين الثاني": 11,
    "كانون الأول": 12,
    # Added for China Breaks / Leading News / Market Watcher sources (2026-10-11):
    # Russian and Ukrainian genitive, Bulgarian, Romanian, Portuguese, Italian.
    # Without them a correct "October 9" from "9 октября" was cut.
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
    "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    "січня": 1, "лютого": 2, "березня": 3, "квітня": 4, "травня": 5, "червня": 6, "липня": 7,
    "серпня": 8, "вересня": 9, "жовтня": 10, "листопада": 11, "грудня": 12,
    "януари": 1, "февруари": 2, "март": 3, "април": 4, "май": 5, "юни": 6, "юли": 7,
    "август": 8, "септември": 9, "октомври": 10, "ноември": 11, "декември": 12,
    "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4, "iunie": 6, "iulie": 7,
    "septembrie": 9, "octombrie": 10, "noiembrie": 11,
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "maio": 5, "junho": 6, "julho": 7,
    "setembro": 9, "outubro": 10, "dezembro": 12,
    "gennaio": 1, "febbraio": 2, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7,
    "settembre": 9, "ottobre": 10, "dicembre": 12,
}
_MONTH_ALT = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))

# Where each outlet's own "today" is. Only the outlets this channel reads;
# an outlet not listed is never re-dated, only checked.
_OUTLET_TZ = {
    "scmp.com": "Asia/Hong_Kong", "asiatimes.com": "Asia/Hong_Kong",
    "koreatimes.co.kr": "Asia/Seoul", "focustaiwan.tw": "Asia/Taipei",
    "cna.com.tw": "Asia/Taipei", "digitimes.com": "Asia/Taipei",
    "cls.cn": "Asia/Shanghai", "dailynk.jp": "Asia/Tokyo", "thediplomat.com": "Asia/Tokyo",
    "abc.net.au": "Australia/Sydney",
    "arabnews.com": "Asia/Riyadh", "trend.az": "Asia/Baku", "tass.com": "Europe/Moscow",
    "aa.com.tr": "Europe/Istanbul", "politico.eu": "Europe/Brussels",
    "theguardian.com": "Europe/London", "independent.co.uk": "Europe/London",
    "middleeasteye.net": "Europe/London", "middleeastmonitor.com": "Europe/London",
    "alaraby.co.uk": "Europe/London", "bbc.com": "Europe/London",
    "dailymail.com": "Europe/London",
}

_FUTURE = re.compile(
    r"\b(will|would|is set to|are set to|is scheduled|are scheduled|scheduled for|plans? to|"
    r"planned for|next|upcoming|is due|are due|due to|expected to|ahead of|through|until|"
    r"to be held|to visit|to meet|to hold|to attend|to travel)\b", re.IGNORECASE)

_POST_WD = re.compile(r"\b(%s)\b" % "|".join(WEEKDAYS))
_EN_MON = (r"(January|February|March|April|May|June|July|August|September|October|November|"
           r"December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept\.?|Sep\.?|Oct\.?|"
           r"Nov\.?|Dec\.?)")
_POST_MD = re.compile(_EN_MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?"
                      r"(?:\s*(?:-|–|to|and|through)\s*(\d{1,2})(?:st|nd|rd|th)?)?"
                      r"(?:,?\s+(\d{4}))?\b")
_POST_DM = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _EN_MON + r"(?:,?\s+(\d{4}))?\b")

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _month(name: str) -> Optional[int]:
    return _MONTHS.get(name.lower().rstrip("."))


def _source_has_weekday(source: str, idx: int) -> bool:
    if re.search(r"\b%s\b" % WEEKDAYS[idx], source, re.IGNORECASE):
        return True
    low = source.lower()
    if any(re.search(r"(?<![\w])%s(?![\w])" % re.escape(w), low) for w in _WD_LATIN[idx]):
        return True
    return any(w in source for w in _WD_OTHER[idx])


def _instant(published_at) -> Optional[dt.datetime]:
    if published_at is None:
        return None
    if isinstance(published_at, dt.datetime):
        t = published_at
    else:
        raw = str(published_at).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return None                     # a day with no hour cannot be moved
        try:
            t = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _outlet_tz(url: str) -> Optional[ZoneInfo]:
    host = urlparse(url or "").netloc.lower()
    for domain, tz in _OUTLET_TZ.items():
        if host == domain or host.endswith("." + domain):
            return ZoneInfo(tz)
    return None


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("! ", 0, start), text.rfind("? ", 0, start))
    right = min([i for i in (text.find(". ", end), text.find("! ", end), text.find("? ", end)) if i != -1]
                or [len(text)])
    return text[left + 1:right]


def _cut(post: str, token: str) -> str:
    """Remove the token with its preposition, touching nothing else."""
    t = re.escape(token)
    m = None
    for pat in (r"(?i)(\s*,)?\s*\b(?:on|this|last|since|by|until|from|through|as of|early|late)\s+%s\b(,)?" % t,
                r"\b%s['’]s\s+" % t,
                r"(\s*,)?\s*\b%s\b" % t):
        m = re.search(pat, post)
        if m:
            break
    if not m:
        return post
    before, after = post[:m.start()], post[m.end():]
    # "Bank on October 2, making it" keeps its comma; ", on Tuesday," as an
    # aside goes whole.
    if m.re.groups == 2 and m.group(2) and not m.group(1):
        after = "," + after
    at_sentence_start = not before.strip() or re.search(r"[.!?]\s*$", before)
    if at_sentence_start:
        after = after.lstrip(" ,")
        after = after[:1].upper() + after[1:]
        joined = before.rstrip() + (" " if before.strip() else "") + after
    else:
        # Keep one space between the words either side; let punctuation close up.
        joined = before.rstrip() + ("" if re.match(r"\s*[.,;:!?]", after) else " ") + after.lstrip()
    return re.sub(r" {2,}", " ", joined).strip()


# "【headline】财联社10月12日电，近日…" / "新华社北京10月12日电（记者…）". 日电 followed
# by 话 is a phone call (10月12日电话会谈), hence the lookahead.
_CN_WIRE = re.compile(r"[\u4e00-\u9fff]{2,12}?(?P<mo>\d{1,2})月(?P<day>\d{1,2})日[电電](?=[，,：:（(\s]|$)")

# The day a dated Chinese wire story calls 今天 is its dateline's; 昨天 the one
# before. Only with a dateline: without one there is nothing to count from.
# Not quoted words, and not English "today": replayed 2026-10-12, Trend's
# Aliyev story has "today" only in his speech, and the speech was "held
# yesterday"; "the ministry announced today" dates the announcement, not the
# signing the post is about. CNA's 今天 was the news day in every case.
_REL_DAYS = (
    (0, re.compile(r"今天|今日(?!头条|頭條)")),
    (-1, re.compile(r"昨天|昨日")),
)
_QUOTED = re.compile(r"「[^」]*」|『[^』]*』|“[^”]*”|\"[^\"\n]*\"")


def _without_dateline(source: str) -> tuple[str, Optional[tuple[Optional[int], int]]]:
    """The source with its wire dateline taken out, and the (month, day) the
    dateline carried (month None for 中央社's day-only one). "WASHINGTON,
    October 6. /TASS/." is when the agency filed, not when anything happened:
    Trump's Nebraska rally was the evening of October 5 in Eastern, TASS filed
    it on the 6th, and the draft said "a Nebraska rally on October 6" -- a date
    the source did carry, so step 1 let it stand. A date the source gives only
    in its dateline does not license a date in the post; the dateline only
    says which day the story's own 今天 is."""
    title, sep, body = (source or "").partition("\n")
    if not sep:
        title, body = "", title
    head = body[:200]
    filed = None
    m = wire_dateline(head)
    if m:
        filed = (_month(m.group("mon")), int(m.group("day")))
    else:
        m = _CNA.search(head)
        if m:
            filed = (None, int(m.group("day")))
        else:
            m = _CN_WIRE.search(head)
            if m:
                filed = (int(m.group("mo")), int(m.group("day")))
    if m:
        body = body[:m.start()] + body[m.end():]
    return f"{title}{sep}{body}", filed


def _filed_date(filed: tuple[Optional[int], int], ref: dt.date) -> Optional[dt.date]:
    """The dateline's day as a date: a day alone is this month's, or last
    month's when this month has not reached it yet."""
    mo, d = filed
    if mo:
        return _year_for(mo, d, ref)
    for back in (0, 1):
        y, m = (ref.year, ref.month - back) if ref.month - back else (ref.year - 1, 12)
        try:
            day = dt.date(y, m, d)
        except ValueError:
            continue
        if day <= ref + dt.timedelta(days=1):
            return day
    return None


# A clock in the post, with what hangs on it: "at about 9:20 a.m. local time
# (0020 GMT)". A 24-hour clock counts only with a zone or "hours" after it --
# "3:10" alone may be a ratio or a score.
_ZONE = (r"(?:local time|(?:[A-Z][a-z]+(?: [A-Z][a-z]+)?) time|GMT|UTC|KST|JST|MSK|HKT|CST|BST|CET|CEST"
         r"|AEST|AEDT|SGT|PHT)")
_CLOCK = re.compile(
    r"(?:\b(?:at|around|about|by|from|until|after|before|shortly after|shortly before|just after|just before)\s+)?"
    r"(?:(?:about|around|approximately|roughly|nearly|some)\s+)?"
    r"(?:\b\d{1,2}(?:[:.]\d{2})?\s*(?:a\.m\.|p\.m\.|am\b|pm\b|AM\b|PM\b)(?:\s+" + _ZONE + r")?"
    r"|\b\d{1,2}:?\d{2}\s*(?:hours|hrs|h)?\s+" + _ZONE + r")"
    r"(?:\s*\([^()]{0,40}\))?")
_EASTERN_MARK = re.compile(r"\b(?:ET|EST|EDT|Eastern|Washington time|New York time)\b")


def _remove_span(post: str, start: int, end: int) -> str:
    """The post without post[start:end], its sentence and commas closed up."""
    before, after = post[:start], post[end:]
    if not before.strip() or re.search(r"[.!?]\s*$", before):
        after = after.lstrip(" ,")
        after = after[:1].upper() + after[1:]
        joined = before.rstrip() + (" " if before.strip() else "") + after
    else:
        joined = before.rstrip() + ("" if re.match(r"\s*[.,;:!?]", after) else " ") + after.lstrip()
    joined = re.sub(r",\s*,", ",", joined)
    joined = re.sub(r",(\s*[.;:!?])", r"\1", joined)
    return re.sub(r" {2,}", " ", joined).strip()


def _drop_local_clock(post: str, shifted: list[str], changes: list) -> str:
    """A day in the post is the Eastern day of a later local one: the local
    morning, "local time" and clock that came with it are no longer true of
    it, so they go. A clock marked Eastern stays."""
    for s in shifted:
        m = re.search(r"(?:\b(?:early|late)\s+)?%s(?:\s+(?:morning|afternoon|evening|night))?"
                      r"(?:\s*\(?\s*%s\s*\)?)?" % (re.escape(s), _ZONE), post)
        if m and m.group(0) != s:
            post = post[:m.start()] + s + post[m.end():]
            changes.append(f"{m.group(0)!r} -> {s!r} (a local part of day, on an Eastern day)")
    while True:
        for m in _CLOCK.finditer(post):
            if not _EASTERN_MARK.search(m.group(0) + post[m.end():m.end() + 12]):
                break
        else:
            return post
        changes.append(f"cut {m.group(0).strip()!r} (a local clock, on an Eastern day)")
        post = _remove_span(post, m.start(), m.end())


# "Guterres on 9 condemned" -- 财联社's 9日 copied as a bare number, live on
# 2026-10-09. Also "on the 9th". Only "on" with a bare day, or a preposition
# with an ordinal: "rose by 9 points" and "by 11:13 AM" are not dates.
_POST_BARE_DAY = re.compile(
    r"\b(?:on (?:the )?(\d{1,2})(?:st|nd|rd|th)?|(?:since|until|by) the (\d{1,2})(?:st|nd|rd|th))\b"
    r"(?!\s*(?:%|percent|per cent|[:.,]\d|" + _EN_MON + r"))"
    # "sanctions on 9 officials" is a count, whatever the source's 日 says.
    r"(?!\s+(?:[a-z]+s|people|men|women|children|personnel|staff)\b)")


@dataclass
class Day:
    local: dt.date              # as the source writes it, in the outlet's zone
    forms: set                  # "wd" and/or "md": how the source writes it


def _year_for(mo: int, d: int, ref: dt.date) -> Optional[dt.date]:
    """The date a month and day with no year mean: this year's, unless that
    is more than ten months off (a December story's "January 3")."""
    for y in (ref.year, ref.year + 1, ref.year - 1):
        try:
            day = dt.date(y, mo, d)
        except ValueError:
            continue
        if -300 <= (day - ref).days <= 60 or y != ref.year:
            return day
    return None


def _bare_dates(bare: set, ref: Optional[dt.date]) -> dict:
    """Each day the source writes with no month (古特雷斯9日), as the date it
    means -- only when that can be told: a day up to the day after
    publication is this month's. A later one is not settled either way:
    中央社 writes "20日" for a meeting later this month as readily as a
    Shanghai desk on the 1st writes "30日" for yesterday, and a first version
    that took the latest date on or before publication turned a CNA story's
    upcoming "October 20" into "September 20"."""
    out = {}
    if ref is None:
        return out
    for d in bare:
        if 1 <= d <= ref.day + 1:
            try:
                out[d] = dt.date(ref.year, ref.month, d)
            except ValueError:
                pass
    return out


def _date_mentions(source: str) -> tuple[set[tuple[int, int]], set[int]]:
    """(month, day) pairs the source writes, and days it writes with no month
    beside them (Chinese 20日, the first end of 4至9日)."""
    s = source.translate(_ARABIC_DIGITS)
    pairs: set[tuple[int, int]] = set()
    bare: set[int] = set()
    rng = r"(?:\s*(?:-|–|—|to|and|through|bis|au|al|a|y|و|至|到|-)\s*(\d{1,2})\.?)?"
    for m in re.finditer(r"(?i)(?<![\w])(%s)\.?\s+(\d{1,2})(?:st|nd|rd|th)?%s" % (_MONTH_ALT, rng), s):
        mo = _month(m.group(1))
        if mo:
            pairs.add((mo, int(m.group(2))))
            if m.group(3):
                pairs.add((mo, int(m.group(3))))
    for m in re.finditer(r"(?i)(?<!\d)(\d{1,2})(?:st|nd|rd|th|er|\.)?%s\s*(?:of\s+|de\s+)?(%s)(?![\w])"
                         % (rng, _MONTH_ALT), s):
        mo = _month(m.group(3))
        if mo:
            pairs.add((mo, int(m.group(1))))
            if m.group(2):
                pairs.add((mo, int(m.group(2))))
    for m in re.finditer(r"(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?(?:\s*(?:至|到|-|—|~)\s*(\d{1,2})\s*(?:日|号))?", s):
        pairs.add((int(m.group(1)), int(m.group(2))))
        if m.group(3):
            pairs.add((int(m.group(1)), int(m.group(3))))
    for m in re.finditer(r"(?<![\d月])(\d{1,2})\s*(?:日|号)", s):
        bare.add(int(m.group(1)))
    for m in re.finditer(r"(?<![\d月])(\d{1,2})\s*(?:至|到|-|—|~)\s*\d{1,2}\s*(?:日|号)", s):
        bare.add(int(m.group(1)))
    for m in re.finditer(r"\b\d{4}-(\d{2})-(\d{2})\b", s):
        pairs.add((int(m.group(1)), int(m.group(2))))
    # Numeric dates (added 2026-10-11): "09.10.2026" (ПРАЙМ), "9/24-30" (a
    # Rasmussen tweet), Vietnamese "9/10". Day and month order differ by
    # country, so both readings are kept: an extra reading can only spare a
    # day from being cut, never cut one.
    for m in re.finditer(r"(?<![\d.])(\d{1,2})[./](\d{1,2})(?:[./](?:\d{4}|\d{2}))?(?:\s*[-–]\s*(\d{1,2}))?(?![\d])", s):
        a, b, c = int(m.group(1)), int(m.group(2)), m.group(3)
        for mo, d in ((b, a), (a, b)):
            if 1 <= mo <= 12 and 1 <= d <= 31:
                pairs.add((mo, d))
                if c and int(c) <= 31:
                    pairs.add((mo, int(c)))
    return pairs, bare


def _weekday_dates(w: int, ref: dt.date) -> tuple[dt.date, dt.date]:
    """The last day named w on or before ref, and the next on or after it."""
    return (ref - dt.timedelta(days=(ref.weekday() - w) % 7),
            ref + dt.timedelta(days=(w - ref.weekday()) % 7))


def _source_days(source: str, ref: dt.date) -> dict[dt.date, Day]:
    """Every day the source names, as a date in its own time zone."""
    days: dict[dt.date, Day] = {}

    def add(day: Optional[dt.date], form: str) -> None:
        if day:
            days.setdefault(day, Day(day, set())).forms.add(form)

    pairs, bare = _date_mentions(source)
    for mo, d in pairs:
        add(_year_for(mo, d, ref), "md")
    for day in _bare_dates(bare, ref).values():
        add(day, "md")
    for w in range(7):
        if _source_has_weekday(source, w):
            for day in _weekday_dates(w, ref):
                add(day, "wd")
    return days


def _eastern(day: dt.date, local_ref: dt.date, east_ref: dt.date, future: bool) -> dt.date:
    """The source's day, in Eastern: the Eastern publication date when the
    source's day is later than that and already past, else as written."""
    if not future and day > east_ref and day <= local_ref:
        return east_ref
    return day


# A US person, office or place. "House" only as the White House or the
# chamber: Diaoyutai State Guesthouse and Seoul's Blue House are not.
# A currency is not a place: "US$133.53 billion" in a Chinese customs story
# moved Beijing's Wednesday to Tuesday (DailyNews, 2026-10-14).
_US_MARK = re.compile(
    r"\b(?:U\.S\.(?!\s*dollars?)|US(?!\$|\s*dollars?)|USA|United States|Americans?|Washington|White House|the House|House Speaker"
    r"|House of Representatives|Pentagon|Congress|congressional|Senate|senators?|congressman|congresswoman"
    r"|lawmakers|Capitol|State Department|Treasury|Justice Department|DOJ|FBI|CIA|DHS|ICE|Federal Reserve"
    r"|the Fed|Centcom|CENTCOM|Trump|Vance|Rubio|Hegseth|Bessent|Leavitt|Lutnick|Greer|Witkoff|Pigott"
    r"|Bondi|Noem|New York|California|Texas|Florida)\b")


def _us_event(post: str, start: int, end: int) -> bool:
    """Whether the day at post[start:end] belongs to a US event: a US mark
    in its sentence before the day -- the subject, normally. Only when the
    day opens the sentence ("On Monday, President Trump said") is the subject
    after it, and then the eight words after it count. Not the rest of the
    sentence: "The People's Bank of China on Monday approved Visa, giving the
    US payments giant a year" is a Chinese event."""
    left = max(post.rfind(". ", 0, start), post.rfind("! ", 0, start), post.rfind("? ", 0, start))
    before = post[left + 1:start]
    if _US_MARK.search(before):
        return True
    if len(before.split()) > 3:
        return False
    after = " ".join(re.split(r"\s+", post[end:end + 120].strip())[:8])
    return bool(_US_MARK.search(after))


def _written(day: dt.date, form: str, token: str) -> str:
    if form == "wd":
        return WEEKDAYS[day.weekday()]
    if re.search(r"\b\d{4}\b", token):
        return f"{day.strftime('%B')} {day.day}, {day.year}"
    return f"{day.strftime('%B')} {day.day}"


def _bare_day_in_post(post: str, days_of: dict, changes: list) -> str:
    """A day the post writes with no month, when the source writes that same
    day with no month: given its month, so the step after can check it."""
    def fix(m):
        if re.search(_EN_MON + r"\.?\s*$", post[:m.start()]):
            return m.group(0)
        d = int(m.group(1) or m.group(2))
        if d not in days_of:
            return m.group(0)
        prep = m.group(0).split(" ", 1)[0]
        new = f"{prep} {_written(days_of[d], 'md', '')}"
        changes.append(f"{m.group(0)!r} -> {new!r} (the source's {d}日)")
        return new
    return _POST_BARE_DAY.sub(fix, post)


def _post_tokens(post: str) -> list[tuple[str, str, object]]:
    tokens: list[tuple[str, str, object]] = []   # (kind, text, value)
    for m in _POST_WD.finditer(post):
        tokens.append(("wd", m.group(0), WEEKDAYS.index(m.group(1))))
    for m in _POST_MD.finditer(post):
        if re.search(r"\bLong\s+$", post[max(0, m.start() - 6):m.start()]):
            continue          # "a Long March 12 rocket" (长征十二号) is not March 12
        days = [int(m.group(2))] + ([int(m.group(3))] if m.group(3) else [])
        tokens.append(("md", m.group(0).rstrip(), (_month(m.group(1)), days)))
    for m in _POST_DM.finditer(post):
        tokens.append(("md", m.group(0).rstrip(), (_month(m.group(2)), [int(m.group(1))])))
    return tokens


def _named_without_hour(kind: str, value, source: str, pairs: set) -> bool:
    """With no publication hour nothing can be dated: the day only has to be
    one the source writes."""
    if kind == "wd":
        return _source_has_weekday(source, value)
    mo, ds = value
    return all((mo, d) in pairs for d in ds)


def fix_time_words(post: str, source: str, published_at=None, url: str = "") -> tuple[str, list[str]]:
    """The post with every weekday and date it gives as the source's day, in
    Eastern. Returns (post, changes); changes are human-readable, for the log."""
    if not post:
        return post, []
    changes: list[str] = []
    source, filed = _without_dateline(source or "")
    inst = _instant(published_at)
    tz = _outlet_tz(url) or EASTERN
    if inst is None:
        pairs, _ = _date_mentions(source)
        for kind, text, value in _post_tokens(post):
            if text in post and not _named_without_hour(kind, value, source, pairs):
                post = _cut(post, text)
                changes.append(f"cut {text!r} (not in source)")
        return post, changes

    local_ref = inst.astimezone(tz).date()
    east_ref = inst.astimezone(EASTERN).date()
    _, bare = _date_mentions(source)
    days_of = _bare_dates(bare, local_ref)
    post = _bare_day_in_post(post, days_of, changes)
    src = _source_days(source, local_ref)
    anchor = _filed_date(filed, local_ref) if filed else None
    if anchor:
        unquoted = _QUOTED.sub("", source)
        for shift, rel in _REL_DAYS:
            if rel.search(unquoted):
                day = anchor + dt.timedelta(days=shift)
                src.setdefault(day, Day(day, set())).forms.update(("wd", "md"))
    shifted: list[str] = []

    # A span across months ("From February 28 to August 22, 2026") is kept or
    # cut whole (added 2026-10-11): cutting each end on its own left "To , the
    # group executed..." in the replay.
    def _known(tok_m) -> bool:
        mo, d = _month(tok_m.group(1)), int(tok_m.group(2))
        return bool(mo) and (_year_for(mo, d, local_ref) in src or
                             any(_eastern(x.local, local_ref, east_ref, False) == _year_for(mo, d, east_ref) for x in src.values()))
    span_re = re.compile(r"(?i)(?:,\s*)?\b(?:from|between)\s+(" + _POST_MD.pattern + r")\s+(?:to|and|through|until)\s+(" + _POST_MD.pattern + r")(,)?")
    for m in list(span_re.finditer(post)):
        a, b = _POST_MD.match(m.group(1)), _POST_MD.match(m.group(6))
        if a and b and not (_known(a) and _known(b)):
            whole = m.group(0)
            start = post.find(whole)
            before, after = post[:start], post[start + len(whole):]
            if not before.strip() or re.search(r"[.!?]\s*$", before):
                after = after.lstrip(" ,"); after = after[:1].upper() + after[1:]
            sep = "" if before.endswith("\n") or not before.strip() else " "
            post = re.sub(r" {2,}", " ", (before.rstrip(" ") + sep + after.lstrip(" ")).strip())
            changes.append(f"cut {whole.strip(' ,')!r} (a span the source does not give)")
    for kind, text, value in _post_tokens(post):
        if text not in post:
            continue
        i = post.find(text)
        future = bool(_FUTURE.search(_sentence_around(post, i, i + len(text))))
        if kind == "md" and len(value[1]) > 1:
            # A range: each end must be the source's; left as written.
            mo, ds = value
            if all(_year_for(mo, d, local_ref) in src or d in bare for d in ds):
                continue      # bare 日 ends (中央社 "8日至9日") stand, as a single bare day does
            post = _cut(post, text)
            changes.append(f"cut {text!r} (not a day the source gives)")
            continue

        def as_date(ref: dt.date) -> Optional[dt.date]:
            if kind == "wd":
                last, nxt = _weekday_dates(value, ref)
                return nxt if future else last
            return _year_for(value[0], value[1][0], ref)

        # The source's day the token is: read as the source's local day (the
        # writer copied it) or as an Eastern day (the writer converted it).
        day = src.get(as_date(local_ref))
        if day is None:
            east = as_date(east_ref)
            day = next((d for d in src.values()
                        if _eastern(d.local, local_ref, east_ref, future) == east), None)
        if day is None and kind == "md" and value[1][0] in bare:
            # The source's bare 日: its own month if that can be told (a
            # draft's "September 9" for 财联社's 9日 is put right), else the
            # day stands -- "将于20日" may be this month or next.
            if value[1][0] not in days_of:
                continue
            day = src[days_of[value[1][0]]]
        if day is None:
            post = _cut(post, text)
            changes.append(f"cut {text!r} (not a day the source gives)")
            continue
        # A non-US day is the source's own -- unless the writer put an Eastern
        # clock beside it ("Sunday at 8:20 p.m. ET"), which only the Eastern
        # day agrees with. A window, not _sentence_around: "8:20 p.m. ET" has
        # a ". " inside it.
        near = post[max(0, i - 80):i + len(text) + 80]
        eastern_clock = any(_EASTERN_MARK.search(m.group(0) + near[m.end():m.end() + 12])
                            for m in _CLOCK.finditer(near))
        target = (_eastern(day.local, local_ref, east_ref, future)
                  if eastern_clock or _us_event(post, i, i + len(text)) else day.local)
        meant = as_date(local_ref) if as_date(local_ref) == day.local else as_date(east_ref)
        form = kind if kind in day.forms else sorted(day.forms)[0]
        if target != day.local:
            shifted.append(text)        # whoever moved it, the writer or this
        if target == meant and form == kind:
            continue
        new = _written(target, form, text)
        post = post.replace(text, new, 1)
        # "Tuesday, Oct. 27" whose source gives only the weekday: one
        # "Tuesday", not "Tuesday, Tuesday" (2026-10-27, gpt-6-luna test).
        post = re.sub(r"\b(%s),?\s+\1\b" % re.escape(new), r"\1", post, count=1)
        if target != day.local:
            shifted[-1] = new
        why = ("the source's day, in Eastern (a US event)" if target != day.local
               else "the source's day, as the source writes it")
        changes.append(f"{text!r} -> {new!r} ({why})")
    if shifted:
        post = _drop_local_clock(post, shifted, changes)
    return post, changes


_MIN_SOURCE_CHARS = 300


def fix_caption_time_words(post: str, title: str, body: str, published_at=None, url: str = "") -> tuple[str, list[str]]:
    """fix_time_words() for a caption about to be published (2026-10-11).

    Does nothing when there is no source to check against: a cached caption
    reused without extracting the article again (Market Watcher), a body too
    short to be the article, or a "No comment" refusal. With no source, every
    day in the post would read as "not a day the source gives" and be cut.
    """
    if not post or post.strip().lower().startswith("no comment"):
        return post, []
    if not body or len(body) < _MIN_SOURCE_CHARS:
        return post, []
    return fix_time_words(post, f"{title}\n{body}", published_at, url)
