"""
OpenClaw Messaging -- Slack Bot Adapter.

Receives Slack Events API payloads and replies via the Slack Web API.

Environment variables:
    SLACK_BOT_TOKEN       -- Bot User OAuth Token (xoxb-...).
    SLACK_SIGNING_SECRET  -- Used to verify incoming requests from Slack.
"""

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Optional, Union

import httpx

from .base import BaseMessagingAdapter
from ..models import AgentResponse

logger = logging.getLogger(__name__)

SLACK_API = "https://slack.com/api"


class SlackAdapter(BaseMessagingAdapter):
    """
    Slack messaging adapter.

    Receives event payloads at ``/webhooks/slack`` and replies using
    ``chat.postMessage``.
    """

    def __init__(self, agent_client: Any) -> None:
        super().__init__(agent_client)
        self.bot_token = os.getenv("SLACK_BOT_TOKEN", "")
        self.signing_secret = os.getenv("SLACK_SIGNING_SECRET", "")
        if not self.bot_token:
            logger.warning(
                "SLACK_BOT_TOKEN is not set. "
                "Slack integration will not work until a token is provided."
            )

    # ------------------------------------------------------------------
    # BaseMessagingAdapter implementation
    # ------------------------------------------------------------------

    async def on_message(
        self,
        platform_user_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> AgentResponse:
        return await self.agent_client.run(
            user_id=f"slack_{platform_user_id}",
            prompt=text,
            return_response=True
        )

    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        channel = (metadata or {}).get("channel", platform_user_id)
        url = f"{SLACK_API}/chat.postMessage"

        response_text = response.content
        if response.action_type == "approval_required":
            response_text = f"{response_text}\n\n⚠️ *Acción requerida:* Confirma en la API o interactúa aquí."

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={"channel": channel, "text": response_text},
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok"):
                    logger.error("Slack API error: %s", data.get("error"))
                    return False
                return True
        except Exception:
            logger.exception("Failed to send Slack message to %s", channel)
            return False

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def verify_signature(self, timestamp: str, body: bytes, signature: str) -> bool:
        """Verify that the request comes from Slack."""
        if not timestamp or not signature:
            return False
        try:
            if abs(time.time() - int(timestamp)) > 300:
                return False
        except ValueError:
            return False

        base = f"v0:{timestamp}:{body.decode(errors='ignore')}".encode()
        expected = "v0=" + hmac.HMAC(
            self.signing_secret.encode(), base, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, str(signature))

    # ------------------------------------------------------------------
    # Webhook helper
    # ------------------------------------------------------------------

    async def process_event(self, payload: dict) -> Optional[AgentResponse]:
        """
        Process an incoming Slack Events API payload.

        Handles ``url_verification`` challenges and ``message`` events.
        Returns the agent response or *None* for non-message events.
        """
        # Slack URL verification challenge
        if payload.get("type") == "url_verification":
            # Note: This is an exception where we return a challenge string
            # In a real API this would be handled before reaching the adapter logic
            return None # AgentResponse(content=payload.get("challenge", ""), action_type="final_answer")

        event = payload.get("event", {})
        event_type = event.get("type", "")

        if event_type != "message" or event.get("bot_id"):
            return None  # Ignore bot messages and non-message events

        user_id = event.get("user", "unknown")
        channel = event.get("channel", "")
        text = event.get("text", "").strip()

        if not text:
            return None

        return await self.handle(
            platform_user_id=user_id,
            text=text,
            metadata={"channel": channel},
        )

