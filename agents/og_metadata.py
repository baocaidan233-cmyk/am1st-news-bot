"""Fetches Open Graph metadata (title/description/image) for a Gettr
link-preview card — see agents/gettr_publisher.py's docstring for why
AM1ST's posts showed no preview until this existed: Gettr's post payload
needs explicit dsc/previmg/prevsrc/ttl fields, which nothing in this
project populated before 2026-08-06.

Ported from DN_Video_Scraper_Agent/agents/publish_agent.py's
_fetch_og_metadata/_fetch_og_via_caps_gettr (confirmed working there,
against real Gettr posts) — adapted from aiohttp to this project's httpx,
and trimmed of that project's site-specific bits (cls.cn's JSON API shape,
Google News URL resolution) that don't apply to AM1ST's RSS sources."""

from __future__ import annotations

import html as html_module
import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

_META_TAG_RE = re.compile(r"<meta\b[^>]+>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'\b(property|name|content)\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
_NEXT_DATA_RE = re.compile(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL)
_IMG_TAG_RE = re.compile(r'<img\b[^>]+\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", text)).strip()


def _extract_og(html: str, prop_name: str, name_variants: tuple = ()) -> str | None:
    """Scans every <meta> tag regardless of attribute order — some sites
    write name="og:title" before content=, others after."""
    for tag in _META_TAG_RE.finditer(html):
        attrs = dict(_ATTR_RE.findall(tag.group(0)))
        key = attrs.get("property") or attrs.get("name") or ""
        if key.lower() == prop_name.lower() or key.lower() in name_variants:
            val = _strip_html(html_module.unescape(attrs.get("content", "")))
            if val:
                return val
    return None


def _extract_next_data_image(html: str) -> str | None:
    """Next.js SSR sites often skip plain <meta> OG tags in favor of an
    embedded __NEXT_DATA__ JSON blob."""
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        j = json.loads(m.group(1))
        page_props = j.get("props", {}).get("pageProps", {})
        article = page_props.get("article") or {}
        return article.get("image") or page_props.get("image")
    except Exception:
        return None


def _extract_fields(body: str, url: str) -> dict:
    prev_img = _extract_og(body, "og:image", ("twitter:image", "twitter:image:src"))
    prev_ttl = _extract_og(body, "og:title", ("twitter:title",))
    prev_desc = _extract_og(body, "og:description", ("twitter:description",))
    prev_src = _extract_og(body, "og:url") or url

    if not prev_img:
        prev_img = _extract_next_data_image(body)
    if not prev_img:
        # <img src> fallback — picks up a masthead/brand image when a page
        # has no OG tags at all, rather than posting with no image at all.
        for m in _IMG_TAG_RE.finditer(body):
            src = m.group(1)
            if src.startswith("data:") or not src.startswith("http"):
                continue
            low = src.lower()
            if any(x in low for x in ("icon", "avatar", "logo-small", "spinner", "loading", "favicon")):
                continue
            prev_img = src
            break

    return {"prev_img": prev_img, "prev_ttl": prev_ttl, "prev_desc": prev_desc, "prev_src_link": prev_src}


async def _fetch_via_caps_gettr(client: httpx.AsyncClient, url: str) -> dict:
    """Gettr's own OG-scraping proxy — bypasses the bot-detection many news
    sites use against a plain scripted request. Almost certainly the same
    endpoint Gettr's own web/app calls when a user pastes a URL into the
    post box, which is why this is tried first."""
    try:
        resp = await client.get(
            f"https://caps.gettr.com/{url}",
            headers={"origin": "https://gettr.com", "referer": "https://gettr.com/"},
            timeout=8,
        )
        if resp.status_code != 200:
            return {}
        body = resp.text
    except Exception as e:
        logger.debug("og_metadata: caps.gettr.com failed for %s: %s", url, e)
        return {}

    fields = _extract_fields(body, url)
    if not fields["prev_img"]:
        return {}  # no image found via the proxy — caller falls back to a direct fetch
    return fields


async def _fetch_direct(client: httpx.AsyncClient, url: str) -> dict:
    try:
        resp = await client.get(
            url,
            headers={"user-agent": _USER_AGENT, "accept": "text/html,application/xhtml+xml,*/*"},
            timeout=15,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return {}
        body = resp.text
    except Exception as e:
        logger.debug("og_metadata: direct fetch failed for %s: %s", url, e)
        return {}
    return _extract_fields(body, url)


async def fetch_link_preview(url: str) -> dict:
    """Returns {"prev_img", "prev_ttl", "prev_desc", "prev_src_link"} for
    Gettr's link-preview card fields — any value may be None if not found.
    Never raises — a failed lookup just means no preview card on this post,
    not a blocked publish (agents/gettr_publisher.py omits any field that
    comes back empty)."""
    if not url:
        return {}
    async with httpx.AsyncClient() as client:
        fields = await _fetch_via_caps_gettr(client, url)
        if not fields:
            fields = await _fetch_direct(client, url)
    return fields
