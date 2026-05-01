"""
OpenClaw Messaging -- Discord Bot Adapter.

Receives messages via Discord webhook (or gateway) and replies through
the Discord REST API.

Environment variables:
    DISCORD_BOT_TOKEN   -- Bot token from the Discord Developer Portal.
    DISCORD_APP_ID      -- Application ID (for interactions / slash commands).
"""

import logging
import os
from typing import Any, Optional, Union

import httpx

from .base import BaseMessagingAdapter
from ..models import AgentResponse

logger = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"


class DiscordAdapter(BaseMessagingAdapter):
    """
    Discord messaging adapter.

    Supports two ingestion modes:
    1. **Interaction webhook** -- Discord sends interaction payloads to
       ``/webhooks/discord``.
    2. **Gateway bot** -- for richer real-time presence (not covered here;
       use ``discord.py`` library directly and call :meth:`on_message`).
    """

    def __init__(self, agent_client: Any, bot_token: Optional[str] = None) -> None:
        super().__init__(agent_client)
        self.bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN", "")
        self.app_id = os.getenv("DISCORD_APP_ID", "")
        if not self.bot_token:
            logger.warning(
                "DISCORD_BOT_TOKEN is not set. "
                "Discord integration will not work until a token is provided."
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
        # Note: agent_client.run should return AgentResponse when return_response=True
        return await self.agent_client.run(
            user_id=f"discord_{platform_user_id}",
            prompt=text,
            return_response=True
        )

    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        channel_id = (metadata or {}).get("channel_id")
        if not channel_id:
            logger.error("Cannot send Discord message without channel_id")
            return False

        response_text = response.content
        if response.action_type == "approval_required":
            response_text = f"{response_text}\n\n⚠️ **Acción requerida:** Aprueba vía API para continuar."

        url = f"{DISCORD_API}/channels/{channel_id}/messages"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={"content": response_text[:2000]},
                    headers={"Authorization": f"Bot {self.bot_token}"},
                )
                resp.raise_for_status()
                return True
        except Exception:
            logger.exception("Failed to send Discord message to channel %s", channel_id)
            return False

    # ------------------------------------------------------------------
    # Webhook / Interaction helpers
    # ------------------------------------------------------------------

    async def process_interaction(self, payload: dict) -> Optional[AgentResponse]:
        """
        Process an incoming Discord Interaction payload.

        Handles ``type=2`` (APPLICATION_COMMAND) and ``type=3`` (MESSAGE_COMPONENT).
        Returns the agent response or *None* for pings.
        """
        interaction_type = payload.get("type", 0)

        # Type 1 = PING (health-check from Discord)
        if interaction_type == 1:
            return None

        # Extract user and message content
        user = payload.get("member", {}).get("user", {}) or payload.get("user", {})
        user_id = user.get("id", "unknown")
        channel_id = payload.get("channel_id", "")

        # Slash command data
        data = payload.get("data", {})
        options = data.get("options", [])
        text = ""
        for opt in options:
            if opt.get("name") == "prompt":
                text = opt.get("value", "")
                break
        if not text:
            text = data.get("custom_id", "no input")

        return await self.handle(
            platform_user_id=user_id,
            text=text,
            metadata={"channel_id": channel_id, "interaction_id": payload.get("id")},
        )

