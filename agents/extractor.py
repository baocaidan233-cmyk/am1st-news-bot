from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx
import trafilatura
from playwright.async_api import async_playwright

from core.alerts import AlertNotifier
from core.config import AppConfig
from core.models import RssSource

logger = logging.getLogger(__name__)

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
    "nytimes.com",
    "ft.com",
    "economist.com",
    "bloomberg.com",
    "washingtonpost.com",
)

# These sites use real bot-detection vendors (DataDome/PerimeterX-class),
# not just a basic UA check — confirmed 2026-08-11: a bare Playwright
# render against a live nytimes.com article was itself served a DataDome
# CAPTCHA page. Same stealth measures as
# China_Scandal_News/agents/headless_scraper.py's _STEALTH_INIT_SCRIPT,
# reused here rather than reinvented.
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
window.chrome = window.chrome || { runtime: {} };
"""
_VIEWPORT = {"width": 1366, "height": 900}

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
_PAYWALL_TEASER_SIGNALS = (
    "subscribe to unlock this article",
    "try unlimited access",
    "complete digital access to quality",
)


def _looks_like_paywall_teaser(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in _PAYWALL_TEASER_SIGNALS)


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
        """Real Chromium render — only reached for _BROWSER_REQUIRED_DOMAINS,
        after the plain fetch already proved insufficient. One launch per
        call rather than a shared long-lived browser: this path is rare
        (a handful of paywall domains, only within the small publish-cycle
        batch), so the launch cost isn't worth the extra lifecycle
        management a persistent instance would need — same call shape as
        China_Scandal_News/agents/headless_scraper.py, including its
        stealth init script (see _STEALTH_INIT_SCRIPT above)."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=headers["User-Agent"],
                        viewport=_VIEWPORT,
                        locale="en-US",
                    )
                    try:
                        await context.add_init_script(_STEALTH_INIT_SCRIPT)
                        if "cookie" in headers:
                            await context.set_extra_http_headers({"cookie": headers["cookie"]})
                        page = await context.new_page()
                        try:
                            await page.goto(
                                url,
                                wait_until="domcontentloaded",
                                timeout=self._config.extraction.timeout_seconds * 1000,
                            )
                            # DataDome-class challenges resolve client-side a
                            # couple seconds after load — give the page a
                            # moment before reading content back out.
                            await page.wait_for_timeout(2500)
                            return await page.content()
                        finally:
                            await page.close()
                    finally:
                        await context.close()
                finally:
                    await browser.close()
        except Exception as e:
            logger.info("Extractor: browser fetch failed for %s (%s)", url, e)
            return None

    async def extract(self, url: str, sources: list[RssSource]) -> str | None:
        """Returns the extracted main-content text, or None if extraction
        failed — caller decides the fallback (the RSS description)."""
        source = _find_cookie_source(url, sources)
        extraction = self._config.extraction

        headers = dict(FETCH_HEADERS)
        if source and source.cookie:
            headers["cookie"] = source.cookie

        html = await self._fetch_plain(url, headers)
        text = await asyncio.to_thread(trafilatura.extract, html) if html else None

        used_browser = False
        if (not text or len(text) < extraction.min_text_length) and _needs_browser(url):
            used_browser = True
            html = await self._fetch_browser(url, headers)
            # trafilatura's parsing is CPU-bound, synchronous — offload so
            # it doesn't block the event loop while other candidates are in
            # flight.
            text = await asyncio.to_thread(trafilatura.extract, html) if html else None

        if not text or len(text) < extraction.min_text_length:
            reason = f"extracted only {len(text or '')} chars" if html else "fetch failed"
            if used_browser:
                reason += " (after browser retry)"
            await self._fail(url, source, reason)
            return None

        if _looks_like_paywall_teaser(text):
            await self._fail(
                url,
                source,
                f"extracted text is a paywall teaser, not the article ({len(text)} chars)",
                alert_message=f"抓到的内容像是付费墙提示文案，cookie可能已过期，需要手动更新: {url}",
            )
            return None

        return text
