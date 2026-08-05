from __future__ import annotations

import logging
import time

import httpx

from core.config import AppConfig
from core.notion_sources import NOTION_VERSION

logger = logging.getLogger(__name__)


class AlertNotifier:
    """Per-source technical failure alerts (cookie expiry, extraction
    errors) via a Notion comment that @-mentions the user on the relevant
    source page — rides Notion's own push notifications, no new service.
    See feedback_alert_channel_preference: NOT the shared channel-monitor
    dashboard, NOT a passive table field, NOT email.

    Debounced in-memory per page_id — a still-broken source only re-alerts
    after alert_cooldown_seconds, not every single cycle. Debounce state is
    process-local and resets on restart; that's an acceptable gap for this
    scope (worst case: one extra alert after a redeploy)."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._cooldown = config.alert_cooldown_seconds
        self._last_alerted: dict[str, float] = {}

    async def alert(self, page_id: str, message: str) -> None:
        notion = self._config.notion
        if not notion.api_key or not notion.alert_user_id:
            logger.warning("AlertNotifier: NOTION_API_KEY / NOTION_ALERT_USER_ID not set — cannot send alert: %s", message)
            return
        if not page_id:
            logger.warning("AlertNotifier: no page_id to attach alert to — alert: %s", message)
            return

        last = self._last_alerted.get(page_id, 0.0)
        if time.time() - last < self._cooldown:
            logger.info("AlertNotifier: suppressing repeat alert for %s (cooldown active)", page_id)
            return

        headers = {
            "Authorization": f"Bearer {notion.api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        body = {
            "parent": {"page_id": page_id},
            "rich_text": [
                {"type": "mention", "mention": {"user": {"id": notion.alert_user_id}}},
                {"type": "text", "text": {"content": f" {message}"}},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post("https://api.notion.com/v1/comments", headers=headers, json=body)
                resp.raise_for_status()
            self._last_alerted[page_id] = time.time()
        except Exception:
            logger.exception("AlertNotifier: failed to post alert comment on %s", page_id)
