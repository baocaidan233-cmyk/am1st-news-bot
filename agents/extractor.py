from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx
import trafilatura

from core.alerts import AlertNotifier
from core.config import AppConfig
from core.models import RssSource
from core.render_client import render as render_service_render

logger = logging.getLogger(__name__)


def _main_text(html: str) -> str | None:
    """The article body only. trafilatura keeps the comment section by default
    (include_comments=True), and that is reader text, not the story: on
    2026-10-06 a Conservative Treehouse article came back at 35,830 characters
    with its readers' posts in it, including a link to a different outlet's
    version of the story, all of which the writer could have quoted as fact.
    Channel owner: reader comments must not be scraped."""
    return trafilatura.extract(html, include_comments=False)

# Same headers as agents/rss_fetcher.py's FEED_HEADERS — a bare httpx client
# gets blocked by some sites' basic bot filters.
FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Domains confirmed gated behind real bot-detection/JS challenges that a
# plain httpx GET can't get past (README's "已知限制" list, 2026-08-06). A
# real Chromium render is a meaningfully heavier cost than one HTTP request
# — same tradeoff as China_Scandal_News/agents/headless_scraper.py — so it's
# only attempted for domains actually confirmed to need it, and only as a
# fallback after the cheap plain fetch already came back empty/too-thin.
_BROWSER_REQUIRED_DOMAINS = (
    # 2026-09-28: returns 403 to httpx on every header set tried, including a
    # full desktop Chrome fingerprint, but renders normally under headless
    # Chromium — bot detection rather than a paywall, which is exactly what
    # this tier is for. Found when a breaking Fauci/Ebola story had already
    # burned two of its three extraction attempts against the 403.
    "justthenews.com",
    "nytimes.com",
    "ft.com",
    "economist.com",
    "bloomberg.com",
    "washingtonpost.com",
)

# These sites use real bot-detection vendors (DataDome/PerimeterX-class),
# not just a basic UA check — confirmed 2026-08-11: a bare Playwright
# render against a live nytimes.com article was itself served a DataDome
# CAPTCHA page. Routed through the shared render service (2026-09-04,
# core/render_client.py) rather than this module launching its own
# Playwright instance per call — stealth measures now live there.

# Confirmed 2026-08-11 on a real FT article: an expired paywall cookie
# doesn't make the fetch fail — it "succeeds" with the site's own marketing
# copy for the paywall itself, which is long enough to clear
# min_text_length and would otherwise pass through as if it were the real
# article. Tried building an automated login-based cookie refresher first
# (same Playwright approach as the bot-detection fallback above), but
# NYT/WaPo's blocks happen before auth is even checked, and FT's own login
# page is itself gated behind a Cloudflare Turnstile challenge that never
# resolved in headless testing — automating past that would mean building
# a captcha-bypass tool, which isn't something to build. So this stays
# detection-only: recognize the teaser text and alert, same channel as
# every other extraction failure, rather than silently treating marketing
# copy as if it were the article.
#
# The three exact phrases this used to hold matched the three walls they were
# copied from and nothing else. Washington Examiner's reads "Join Washington
# Examiner for unlimited access", which misses "try unlimited access" by one
# word, so five of its articles published between 2026-09-27 and 09-29 were
# written from their opening two paragraphs. One did real damage: the op-ed
# behind "Trump dines with Xi while the Pentagon starves the tech that could
# defeat him" argues across 7382 characters that the villain is the Pentagon's
# cost-plus procurement bureaucracy, and 1060 characters reached the writer --
# the hook, one thesis sentence, and the subscription pitch. Everything naming
# the real target sat behind the wall, so the only story left to write was
# that the president is at a banquet while the military starves. The channel
# published an attack on him and the user deleted it.
#
# Neither signal works alone. Measured on 197 real published articles
# (2026-09-29): teasers run 726-2460 characters while legitimate articles
# start at 342 (Instapundit's blog format), so no length cut separates them --
# 2500 catches every teaser and 22.5% of the real articles with it. Wording
# alone flags long pieces that merely close with a marketing footer. Together
# they are exact: a pitch within the last 400 characters AND a body under 2500
# caught 8 of 8 teasers with 0 of 189 false positives, and that holds for
# every tail window from 300 to 600 and every cap from 2500 to 3000, so it is
# not a tuned coincidence.
_PAYWALL_TEASER_SIGNALS = (
    "unlimited access",
    "subscribe to unlock",
    "already a member",
    "sign in to continue",
    "create a free account",
    "register to continue",
    "this article is for subscribers",
    "start your free trial",
    "become a member",
    "complete digital access to quality",
)
_TEASER_TAIL_CHARS = 400
_TEASER_MAX_LENGTH = 2500


def _looks_like_paywall_teaser(text: str) -> bool:
    """Whether this is the top of an article plus the wall, not the article.

    The pitch has to sit at the END, where the text was cut off. A full
    article that closes with a subscription footer is still a full article,
    which is what the length bound is for."""
    if len(text) >= _TEASER_MAX_LENGTH:
        return False
    tail = text[-_TEASER_TAIL_CHARS:].lower()
    return any(signal in tail for signal in _PAYWALL_TEASER_SIGNALS)


def _domain_matches(netloc: str, domain: str) -> bool:
    """True host-boundary match — netloc IS domain, or is a genuine
    subdomain of it. A plain substring check (`domain in netloc`) matched
    "ft.com" against "joehoft.com" in real testing (2026-08-11): FT's real
    subscription cookie got sent to an unrelated site, and that site got
    routed through the heavy Playwright fallback for nothing."""
    return netloc == domain or netloc.endswith("." + domain)


def _find_cookie_source(url: str, sources: list[RssSource]) -> RssSource | None:
    """Matches the article's domain against each configured source's own
    `domain` field — not every source has a paywall cookie, so most matches
    will have an empty cookie, which is fine (extraction is just attempted
    without one)."""
    netloc = urlparse(url).netloc
    for source in sources:
        if source.domain and _domain_matches(netloc, source.domain):
            return source
    return None


def _needs_browser(url: str) -> bool:
    netloc = urlparse(url).netloc
    return any(_domain_matches(netloc, domain) for domain in _BROWSER_REQUIRED_DOMAINS)


class Extractor:
    """Full-text extraction — self-contained, no external service.
    Confirmed 2026-08-03: the previously-used internal extract-premium
    service (n8n-svr.gettr.fyi) is unmaintained and broken for most of the
    paywalled sources it was meant to help with (domain not on its
    allowlist, a crashed browser driver, an encoding bug), while offering no
    real advantage over a plain fetch for ordinary sites. This does the same
    job directly: httpx GET (with the matched source's cookie, if any) +
    trafilatura for main-content extraction — confirmed working even for
    two of the eight paywalled sources the old service couldn't handle
    (Epoch Times, SCMP).

    2026-08-11: added a second tier for the remaining sites gated behind
    real bot-detection/JS challenges (NYT, FT, Economist, Bloomberg, WaPo —
    see _BROWSER_REQUIRED_DOMAINS), once the VM was upgraded to actually
    support running headless Chromium. Only these confirmed domains ever
    reach it, and only after the plain httpx attempt already came back
    empty or too-thin — every other source still pays just one HTTP
    request, same as before.

    Bug③ fix carried over from the original n8n workflow: a failed
    extraction never silently drops the item — the caller falls back to
    the RSS description, and a Notion @mention alert fires on the matched
    source's row, so a cookie expiry actually gets noticed.

    Called from the publish cycle only (2026-08-05 — moved out of
    ingestion): extracting full text for every ingestion-time candidate
    that merely passed the cheap title+description score gate wasted real
    work on articles that would very likely never be published before
    aging out of the 12h candidate-pool window. Now only the handful of
    candidates actually selected into a publish cycle's batch (up to 5,
    every 30 min) pay this cost — see project_am1st_migration memory.
    """

    def __init__(self, config: AppConfig, alerts: AlertNotifier) -> None:
        self._config = config
        self._alerts = alerts

    async def _fail(self, url: str, source: RssSource | None, reason: str, alert_message: str | None = None) -> None:
        logger.warning("Extractor: %s for %s", reason, url)
        if source and source.cookie:
            # 2026-09-01: paid paywall cookies are expired and not being
            # renewed (user decision) — extraction failure on a
            # cookie-configured source is now a permanent, expected state,
            # not something to page on. Still logged above, just not
            # @mentioned in Notion.
            logger.info("Extractor: suppressing alert for %s — known-expired paywall cookie", url)
            return
        if source:
            await self._alerts.alert(source.page_id, alert_message or f"全文抓取失败(可能是cookie失效/反爬拦截): {url}")

    async def _fetch_plain(self, url: str, headers: dict) -> str | None:
        try:
            async with httpx.AsyncClient(
                timeout=self._config.extraction.timeout_seconds, follow_redirects=True
            ) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.text
        except Exception as e:
            logger.info("Extractor: plain fetch failed for %s (%s)", url, e)
            return None

    async def _fetch_browser(self, url: str, headers: dict) -> str | None:
        """Real Chromium render via the shared render service (2026-09-04)
        — only reached for _BROWSER_REQUIRED_DOMAINS, after the plain fetch
        already proved insufficient. No longer launches its own Playwright
        instance per call; see core/render_client.py."""
        result = await render_service_render(
            url,
            mode="rendered",
            wait_ms=2500,
            timeout_ms=self._config.extraction.timeout_seconds * 1000,
            cookie=headers.get("cookie"),
        )
        if result is None:
            return None
        _status, content = result
        return content

    async def extract(self, url: str, sources: list[RssSource],
                      attempts: int = 0) -> str | None:
        """Returns the extracted main-content text, or None if extraction
        failed — caller decides the fallback (the RSS description).

        `attempts` is how many times this candidate has already failed
        extraction. Past extraction.browser_after_attempts the headless
        browser is tried even for a domain not on the required list, because
        by then the plain fetch has demonstrated it does not work for this
        URL and the only thing left to change is the client.

        The domain list still exists and still fires on the first attempt:
        for a site known to answer 403 to httpx, spending two failures
        rediscovering that is pure waste. The counter is for the sites we do
        not know about yet — justthenews.com was one until a breaking story
        had already burned two of its three attempts against a 403."""
        source = _find_cookie_source(url, sources)
        extraction = self._config.extraction

        headers = dict(FETCH_HEADERS)
        if source and source.cookie:
            headers["cookie"] = source.cookie

        html = await self._fetch_plain(url, headers)
        text = await asyncio.to_thread(_main_text, html) if html else None

        too_thin = not text or len(text) < extraction.min_text_length
        # A recognised teaser is its own reason to render, whatever the domain
        # and whatever attempt this is. Nothing else in this function can turn
        # a truncated article back into the article, and the detector above
        # measured 0 false positives on 189 real ones, so this cannot send
        # anything here that was already whole. Washington Examiner's op-ed
        # came back at 1060 characters plain and 7382 rendered.
        truncated = bool(text) and _looks_like_paywall_teaser(text)
        used_browser = False
        browser_worth_trying = (
            _needs_browser(url)
            or (extraction.browser_after_attempts >= 0
                and attempts >= extraction.browser_after_attempts)
        )
        if (too_thin and browser_worth_trying) or truncated:
            used_browser = True
            html = await self._fetch_browser(url, headers)
            # trafilatura's parsing is CPU-bound, synchronous — offload so
            # it doesn't block the event loop while other candidates are in
            # flight.
            text = await asyncio.to_thread(_main_text, html) if html else None

        if not text or len(text) < extraction.min_text_length:
            reason = f"extracted only {len(text or '')} chars" if html else "fetch failed"
            if used_browser:
                reason += " (after browser retry, attempt %d)" % (attempts + 1)
            await self._fail(url, source, reason)
            return None

        if _looks_like_paywall_teaser(text):
            # Still the wall after rendering, so the article is genuinely out
            # of reach. Dropping is the whole point: a caption written from
            # the top of an article is not a shorter version of that article,
            # it is a different claim, and on 2026-09-29 it was the opposite
            # one. Publishing off title+description is already refused in
            # main_publish.py for the same reason.
            await self._fail(
                url,
                source,
                f"extracted text is a paywall teaser, not the article "
                f"({len(text)} chars{', even after rendering' if used_browser else ''})",
                alert_message=f"抓到的内容像是付费墙提示文案，cookie可能已过期，需要手动更新: {url}",
            )
            return None

        return text
