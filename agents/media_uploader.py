"""Uploads a rendered card to Gettr's CDN so it can be attached to a post.

Ported 2026-09-22 from russia_news/agents/media_uploader.py, narrowed to
photos: this channel has no video path, and carrying the video branch would
mean carrying its 300s write timeout and m3u8/duration plumbing for a case
that cannot occur.

Gettr's upload flow is a four-step dance against a separate host
(upload.gettr.com), none of it documented:
  1. GET  /media/get_upload_channel      -> a GCS resumable-init URL + a
     notify URL to call once the bytes are up.
  2. POST that init URL (x-goog-resumable: start) -> a `Location` header,
     which is the real upload session URL.
  3. PUT  the raw bytes to that Location.
  4. GET  the notify URL with the uploaded location -> the media metadata
     (ori/screen/...) the publish payload needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from core.config import AppConfig

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class MediaUploader:
    def __init__(self, config: AppConfig, dry_run: bool = False) -> None:
        self._config = config
        self._dry_run = dry_run

    async def upload_photo(self, media_path: str) -> dict | None:
        """Returns Gettr's media metadata dict, or None if upload is not
        configured or fails. Never raises: a failed upload must degrade to
        publishing the post without a card, not lose the post."""
        gettr = self._config.gettr
        path = Path(media_path)
        if not path.exists():
            logger.warning("MediaUploader: file not found: %s", media_path)
            return None

        if self._dry_run:
            logger.info("[dry-run] would upload %s to %s", path.name, gettr.media_upload_host)
            return {"ori": f"dry-run://{path.name}", "screen": f"dry-run://{path.name}"}

        # Deliberately a separate branch from dry-run: the sibling bot once
        # returned the same fake dry-run:// dict for a real deployment with
        # blanked credentials, and the publisher happily embedded that literal
        # string as an image URL in a real post instead of surfacing the
        # misconfiguration.
        if not gettr.user_id or not gettr.user_token:
            logger.error("MediaUploader: GETTR_USER_ID/GETTR_USER_TOKEN not set — cannot upload %s "
                         "(a real misconfiguration, not dry-run)", path.name)
            return None

        timeout = httpx.Timeout(30.0, read=60.0, write=120.0, pool=30.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                channel = await client.get(
                    f"{gettr.media_upload_host}/media/get_upload_channel",
                    params={"scene": "getter"},
                    headers={"filename": path.name, "authorization": gettr.user_token,
                             "userid": gettr.user_id, "user-agent": _USER_AGENT},
                )
                channel.raise_for_status()
                data = channel.json()
                init_url = (data.get("gcs") or data.get("gcp") or {}).get("url")
                notify_url = data.get("notify_url")
                if not init_url or not notify_url:
                    logger.error("MediaUploader: get_upload_channel missing gcs/gcp url or notify_url: %s", data)
                    return None

                init = await client.post(init_url, headers={"x-goog-resumable": "start", "content-type": "image/jpeg"},
                                         json={"unuse": 0})
                init.raise_for_status()
                location = init.headers.get("location")
                if not location:
                    logger.error("MediaUploader: no Location header from GCS init")
                    return None

                put = await client.put(location, headers={"content-type": "image/jpeg"}, content=path.read_bytes())
                put.raise_for_status()

                location_no_query = urlunsplit(urlsplit(location)._replace(query=""))
                notify_base = (notify_url if notify_url.startswith("http")
                               else f"{gettr.media_upload_host}/{notify_url.lstrip('/')}")
                notify = await client.get(
                    notify_base, params={"uploadedurl": location_no_query, "result": "ok"},
                    headers={"authorization": gettr.user_token, "userid": gettr.user_id,
                             "origin": "https://gettr.com"},
                )
                notify.raise_for_status()
                media = notify.json()
                if media.get("message") == "ERR_UPLOAD_FAILURE" or not (media.get("ori") or media.get("screen")):
                    logger.error("MediaUploader: upload failed, notify response: %s", media)
                    return None
                return media
        except Exception as e:
            logger.error("MediaUploader: upload failed for %s: %s", media_path, e)
            return None
