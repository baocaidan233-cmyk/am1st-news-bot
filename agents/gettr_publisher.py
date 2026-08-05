from __future__ import annotations

import json
import logging
import time

import httpx

from core.config import AppConfig

logger = logging.getLogger(__name__)


class GettrPublisher:
    """Text-only post — AM1ST's original workflow never attaches media, its
    post_content is the entire payload. Same multipart/x-app-auth publish
    mechanism as the sibling bots (see russia_news/agents/gettr_publisher.py)."""

    def __init__(self, config: AppConfig, dry_run: bool = False) -> None:
        self._config = config
        self._dry_run = dry_run

    def _build_payload(self, text: str) -> dict:
        now_ms = int(time.time() * 1000)
        data = {
            "_t": "post",
            "acl": {"_t": "acl"},
            "txt": text,
            "udate": now_ms,
            "cdate": now_ms,
            "uid": self._config.gettr.user_id,
        }
        return {"data": data, "aux": None, "serial": "post"}

    async def publish(self, text: str, log_ref: str) -> str | None:
        gettr = self._config.gettr

        if self._dry_run or not gettr.user_id or not gettr.user_token:
            payload = self._build_payload(text)
            logger.info("[dry-run] would POST %s content=%s", gettr.api_url, json.dumps(payload, ensure_ascii=False))
            return "dry-run-post-id"

        payload = self._build_payload(text)
        headers = {"x-app-auth": json.dumps({"user": gettr.user_id, "token": gettr.user_token})}
        files = {"content": (None, json.dumps(payload))}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(gettr.api_url, headers=headers, files=files)
                response.raise_for_status()
                if not response.text:
                    raise ValueError("empty response body from Gettr publish endpoint")
                result = response.json()
                return result.get("result", {}).get("data", {}).get("_id")
        except Exception as e:
            logger.error("GettrPublisher: publish failed for %s: %s", log_ref, e)
            return None
