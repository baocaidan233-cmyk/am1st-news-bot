"""Digests, rolling live pages and link lists — the channel does not run them.

User decision, carried over from Leading News and re-measured here: a post
has to be about one thing. A page that is several unrelated stories cannot be
written into one caption without the caption picking one of them and dropping
the rest, or describing the page instead of the news.

It cost a post on 2026-09-29: "Trump posts map showing 'Trump Strait' off
Iran, and other Mideast developments" is an ms.now omnibus whose body opens
"Here is a look at the latest news from the Middle East on Saturday" and then
runs Houthi missiles, Iraqi sanctions exemptions and West Bank springs under
separate headings. Fox's /live-news/ page went out the same way, with
"Coverage for this event has ended" sitting in the middle of it.

Every pattern here was checked against this channel's own material by reading
the bodies, because the version this was ported from would have been wrong in
both directions:

- Leading News's `^LIVE:` is written for rolling pages like "Australia news
  live:". Here LIVE: usually means a single event being streamed, and "LIVE:
  President Trump Holds a Rally in Mobile, AL" was the highest-scoring
  candidate in the whole 1730-row sample at 8.0. That exemption is gone as of
  2026-10-06, by the channel owner's decision: every live or stream page is
  treated as a roundup, single event or not ("不能为了一两条的好互动，让频道
  冒着发拼盘新闻的风险"). Gateway Pundit's "Watch Live: ... and More!" pages
  are a multi-story show behind a single "CLICK HERE TO WATCH RIGHT NOW!", and
  most other outlets' watch-live pages are a video with no article for the
  extractor to find. Hence _LIVE_STREAM, checked before any exemption.
- Matching "briefing" in a URL caught /briefing-room/, a place in the White
  House, so the URL rule does not look for it.
- "things you need to know" reads as a digest and is not one: on Market
  Watcher it flagged an 8.0 about Madagascar's cocoa industry (5048 characters
  on one subject) and a 6.0 Bitcoin week-ahead (6474 characters, its "key
  points" all the same story). Dropped.
- "Breitbart Business Digest" is a column name: 5937 characters on the Fed's
  split over AI investment, its subheadings arguing one case. Hence
  _SINGLE_SUBJECT_COLUMNS. The Dow Jones "Auto & Transport Roundup: Market
  Talk" and AP "Editorial Roundup: United States" really are lists, and
  "Monday Executive News Summary" really is 9123 characters of unrelated items
  ending in a bare list of outbound links.
- The Guardian appends "– Europe live" to its rolling world blog and files it
  under /world/live/: 15021 characters, opening on an Estonian arson case that
  has nothing to do with its own headline about Odesa. Hence the live-section
  URL rule.

Measured on this channel's own material (2026-09-29): 12 of 1730 scored
candidates (0.7%), 2 of which would otherwise have entered the pool, and 3 of
197 already-published articles, each verified by reading its body.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

# Live and stream pages of any kind, single event or not (owner, 2026-10-06).
# Title: a leading "LIVE:" / "LIVE NOW:" / "WATCH LIVE:", "watch live" anywhere,
# "livestream", or Gateway Pundit's "... and More!" show title. URL: a
# watch-live or livestream slug, Epoch TV's /live-now-..., RSBN's /video/live-...
# Not a bare "live-" slug: /live-nation-sued/ is a story about Live Nation.
_LIVE_STREAM_TITLE = re.compile(
    r"^\W*(?:watch\s+)?live(?:\s+now)?\s*[:|–-]|\bwatch\s+live\b|\blive\s*stream\b|\band\s+more!",
    re.IGNORECASE)
_LIVE_STREAM_URL = re.compile(r"/(?:[^/]*watch-live|[^/]*live-?stream|live-now-)[^/]*(?=/|$)|/video/live-", re.IGNORECASE)

# A whole programme and podcast episodes: owner, 2026-10-12, on "NTD Evening
# News Full Broadcast (Oct. 10)" -- 拼盘新闻，播客，am1st不需要. Checked before the
# single-event exemption, like the live pages: "Full Episode" interviews are
# still a show. Title: "Full Broadcast / Episode / Show", "Podcast:", "Ep. 62",
# "Episode 12". URL: a /podcast(s)/ or /episode(s)/ section, a full-broadcast
# slug. Not "podcast" anywhere in a title: "Trump on Theo Von's podcast" is a
# story about what he said. On this channel's ingestion log since 2026-09-16:
# 327 items, every one read: programmes and podcast episodes (Epoch TV/NTD full
# broadcasts, Texas Scorecard and NYT "The Headlines" podcasts, LifeSiteNews
# episodes, The Spectator's "Ep. N", Liberty Nation "Full Episode", 404 Media
# and NH Journal podcasts) plus TV-episode recaps (Decider, Page Six), no news
# story among them; one went out on 2026-10-06 (NTD Evening News Full
# Broadcast, Oct. 6).
# 2026-10-13: NTD's daily shows also run without "Full Broadcast", as the day's
# headlines strung together and the show's name ("JD Vance to host campaign
# rally in Texas; federal agents search Rep. Omar's home…: NTD Good Morning
# (Oct. 13)"), which reached the candidate pool. Named shows only: an NTD clip
# of one story carries no show name.
_PROGRAMME_TITLE = re.compile(
    r"\bfull\s+(broadcast|episode|show)\b|^\W*(video\s*(&|and)\s*)?podcast\s*[:|–-]|\bep\.?\s*\d+\b|\bepisode\s+\d+\b|"
    r"\bNTD\s+(Good\s+Morning|Evening\s+News|Newsroom|News\s+Today)\b",
    re.IGNORECASE)
_PROGRAMME_URL = re.compile(r"/(podcasts?|episodes?)/|full-(broadcast|episode|show)|"
                            r"\bntd-(good-morning|evening-news|newsroom|news-today)\b", re.IGNORECASE)

# One named event covered live or in full is a single subject, whatever the
# title calls itself. No longer exempts a live stream: _LIVE_STREAM_* runs first.
_SINGLE_EVENT = re.compile(
    r"(?i)\b(rally|speech|remarks|address|announcement|press conference|news conference|"
    r"briefing room|hearing|testimony|interview|debate|summit|signing|roundtable|"
    r"ceremony|launch|vote|votes|verdict|ruling)\b")

# Columns whose brand name contains a digest word but which run one subject.
# A keep-list, so a name missing from it costs one story rather than letting a
# real digest through; add to it when one shows up.
_SINGLE_SUBJECT_COLUMNS = ("breitbart business digest",)

_TITLE_RE = re.compile(
    r"(\b(morning|evening|midday|weekly|weekend|daily|nightly)\s+"
    r"(briefing|brief|digest|round-?up|wrap|rundown|summary)\b"
    r"|\b(news|war|executive|business)\s+(briefing|digest|round-?up|wrap|rundown|summary)\b"
    r"|\bnews\s+(in brief|summary)\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
    r"(briefing|summary|round-?up|digest)\b"
    r"|\bnews live\s*[:|–-]|\bliveblog\b|\blive blog\b|\bas it happened\b"
    r"|\blive updates?\s*[:|–-]|–\s*(europe|us|uk|world|business|markets?|politics|sport)\s+live\b"
    r"|\band other [\w'’\s]{0,24}(news|stories|headlines|developments|updates|items)\b"
    r"|\bweek in review\b|\bthe week in\b|\btop (stories|headlines)\b|\bnews quiz\b"
    r"|\bopen thread\b|\bmorning minute\b"
    r"|\broundup\s*:|\bdigest\s*:|\bwrap\s*:)", re.IGNORECASE)

# "briefing" is deliberately absent: the White House briefing room is a place.
_URL_RE = re.compile(
    r"/[^/]*(live-?blog|live-?updates?|live-news|newsblog|live-?ticker|digest|round-?up)[^/]*(?=/|$)",
    re.IGNORECASE)
# A whole section served under /live/ is the rolling page, not one story.
_URL_LIVE_SECTION = re.compile(r"/(business|markets?|politics|world|news|sport)/live/", re.IGNORECASE)
# One entry of a live blog on its own page is a single story; the rolling page
# that collects them is not.
_LIVE_ENTRY_RE = re.compile(r"/(live-blog-update|liveblog_entry|liveblog-entry)/", re.IGNORECASE)


# Date-stamped issues (owner, 2026-10-22: 需要加). A numeric date in the title
# -- "JUICE🍊— 10.22.2026 —", "Pro-Life 10.20.26", "In The Mailbox: 10.22.26",
# "Tuesday's Mini-Report, 9.29.26", "... Rally in Grand Island, NE – 10/05/26"
# -- marks one day's issue, episode or stream, not a story. Every title that
# reached the pre-score since 09-29 with such a date: 130, of which 129 were
# newsletters, podcast and show episodes, livestreams and a crossword; the one
# story quoted a date ("'Remember 7.10.23': In Israel, Hostage Square..."), so
# a date that touches a quote mark does not count. Nor, since the rule went to
# CB, LN and MW the same day, does a date outside this year and last: Zeit's
# "Angriff auf Israel 7.10.2023: ..." is a story about the 2023 attack, and
# "PHP 8.4.27" is a version number.
_DATED_TITLE = re.compile(r"(?<![\w'‘’\"“”])\d{1,2}[./]\d{1,2}[./]((?:20)?\d{2})(?![\w'‘’\"“”])")


def _dated_issue(title: str) -> bool:
    this_year = datetime.now(timezone.utc).year
    for m in _DATED_TITLE.finditer(title or ""):
        year = int(m.group(1))
        if year < 100:
            year += 2000
        if this_year - 1 <= year <= this_year:
            return True
    return False


def roundup_rule(title: str, url: str) -> str | None:
    """The name of the rule this candidate breaks, or None."""
    text = title or ""
    lowered = text.lower()
    if _dated_issue(text):
        return "dated_issue"
    if _LIVE_STREAM_TITLE.search(text):
        return "live_stream"
    if _LIVE_STREAM_URL.search(urlparse(url or "").path or ""):
        return "live_stream_url"
    if _PROGRAMME_TITLE.search(text) or _PROGRAMME_URL.search(urlparse(url or "").path or ""):
        return "programme_or_podcast"
    if any(name in lowered for name in _SINGLE_SUBJECT_COLUMNS):
        return None
    if _SINGLE_EVENT.search(text):
        return None
    if _TITLE_RE.search(text):
        return "title_roundup"
    path = urlparse(url or "").path or ""
    if _LIVE_ENTRY_RE.search(path):
        return None
    if _URL_RE.search(path):
        return "url_roundup"
    if _URL_LIVE_SECTION.search(path):
        return "url_live_section"
    return None

# Newsletters (owner, 2026-10-20: "in today's newsletter, 就是拼盘简报").
# Their headline reads like one story, so the title and URL rules above miss
# them; the summary or the URL gives the page away. WSJ's newsletter items
# open their summary with "Plus: ..." or "Plus, ...", FT's say "Also in
# today's newsletter, ...", TechCabal's "In today's edition: ...", and Fox
# News, CNBC, The Texan and WSJ's What's News carry "newsletter" in the URL.
# Measured 2026-10-20 on the last ~10 days of ingested articles: 19 on AM1ST (WSJ Politics Newsletter, Logistics Report, Morning Risk Report and What's News; FT newsletters; Fox News' Campus Radicals and Antisemitism Exposed newsletters; The Texan's The 40),
# every one a newsletter, none a single story.
_NEWSLETTER_SUMMARY = re.compile(
    r"^\W*plus\s*[:,]\s"
    r"|\bin\s+(?:today|this\s+week|this\s+morning|tonight)['’]?s?\s+[\w&\s]{0,40}?(?:newsletter|edition|issue)\b",
    re.IGNORECASE)
_NEWSLETTER_URL = re.compile(r"newsletter", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def newsletter_rule(description: str, url: str) -> str | None:
    """The name of the newsletter rule this candidate breaks, or None."""
    if _NEWSLETTER_SUMMARY.search(_TAG.sub(" ", description or "")):
        return "newsletter_summary"
    if _NEWSLETTER_URL.search(urlparse(url or "").path or ""):
        return "newsletter_url"
    return None
