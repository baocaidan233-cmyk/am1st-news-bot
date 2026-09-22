"""Decides whether a post needs our own rendered card instead of Gettr's link
preview, and produces the attachable media if so.

One entry point, called from main_publish.py right before publishing. It is
the only module that knows the trigger, so changing "when do we post a card"
never touches the renderer, the uploader or the publish path.

The trigger is deliberately "the preview Gettr would draw is broken", not
"this story would look better as a card". Those are different questions and
only the first one is settled: 190 of 200 live posts carried a preview image
that actually fetches, and the 10 that did not showed no engagement penalty
(like median 40 against 42). The second question -- whether a card beats a
working preview -- is unmeasured on this channel and would need a real A/B,
because a format only produces data once it is actually published.

Fails open in every direction. A validation error, a render error and an
upload failure all return None, which publishes the post exactly as it would
have been published before this module existed. Losing the card is a cosmetic
regression; losing the post is not.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from agents.headline_card import make_card
from agents.media_uploader import MediaUploader
from core.config import AppConfig

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


async def preview_image_is_usable(config: AppConfig, image_url: str | None) -> tuple[bool, str]:
    """Fetches the preview image the way Gettr's own card renderer would have
    to, and reports whether it is a real photo. Returns (usable, reason).

    Checked against the 200-post window on 2026-09-22; every failure mode
    below is one that actually occurred, not a hypothetical:
      no previmg (2) / URL returns text-html (2) / too small (2) /
      HTTP 403 (1) / read timeout (1) / control character in the URL (1).
    """
    if not image_url:
        return False, "no previmg"
    if not image_url.startswith(("http://", "https://")):
        return False, "previmg is not an absolute URL"
    if any(ord(c) < 32 for c in image_url):
        return False, "previmg contains a control character"
    try:
        async with httpx.AsyncClient(timeout=config.poster.validate_timeout_seconds,
                                     follow_redirects=True, headers=_UA) as client:
            # A ranged GET rather than HEAD: several of the real failures came
            # from hosts that answer HEAD with a 200 and then serve an HTML
            # error page to the actual GET.
            resp = await client.get(image_url, headers={**_UA, "Range": "bytes=0-65535"})
            if resp.status_code not in (200, 206):
                return False, f"HTTP {resp.status_code}"
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                return False, f"content-type {content_type[:32]!r}"
            declared = resp.headers.get("Content-Range", "")
            total = 0
            if declared and "/" in declared:
                try:
                    total = int(declared.rsplit("/", 1)[1])
                except ValueError:
                    total = 0
            size = total or int(resp.headers.get("Content-Length") or 0) or len(resp.content)
            if size < config.poster.min_image_bytes:
                return False, f"only {size} bytes"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:40]}"
    return True, ""


async def build_card_media(config: AppConfig, title: str, source_url: str, dry_run: bool = False) -> dict | None:
    """Renders a headline card for `title` and uploads it. None on any failure."""
    tag = config.poster.tag_text
    domain = re.sub(r"^www\.", "", urlsplit(source_url).netloc)
    out_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            out_path = tmp.name
        make_card(title, out_path, tag_text=tag, attribution=domain)
        media = await MediaUploader(config, dry_run=dry_run).upload_photo(out_path)
        if media:
            logger.info("poster: attached a rendered card for %s", source_url)
        return media
    except Exception:
        logger.exception("poster: could not build a card for %s — publishing without one", source_url)
        return None
    finally:
        if out_path:
            Path(out_path).unlink(missing_ok=True)


async def card_for_broken_preview(config: AppConfig, title: str, source_url: str,
                                  image_url: str | None, dry_run: bool = False) -> dict | None:
    """The one call main_publish.py makes. None means "publish as usual"."""
    if not config.poster.enabled:
        return None
    usable, reason = await preview_image_is_usable(config, image_url)
    if usable:
        return None
    logger.info("poster: preview unusable for %s (%s) — rendering a card instead", source_url, reason)
    if not title:
        logger.info("poster: no headline to draw for %s — publishing without a card", source_url)
        return None
    return await build_card_media(config, title, source_url, dry_run=dry_run)
