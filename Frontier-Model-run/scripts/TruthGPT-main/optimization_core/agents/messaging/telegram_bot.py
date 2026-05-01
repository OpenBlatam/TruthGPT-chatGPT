"""
OpenClaw Messaging – Telegram Bot Adapter.

Provides a Telegram bot that forwards messages to the AgentClient and
replies with the agent's response.  Uses the ``python-telegram-bot`` library.

Environment variables:
    TELEGRAM_BOT_TOKEN  –  Bot token obtained from @BotFather.
"""

import logging
import os
from typing import Any, Optional

import httpx

from .base import BaseMessagingAdapter
from ..models import AgentResponse

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramAdapter(BaseMessagingAdapter):
    """
    Telegram messaging adapter.

    Two modes of operation:
    1. **Webhook mode** – FastAPI receives updates via ``/webhooks/telegram``.
    2. **Polling mode** – call :meth:`start_polling` for local development.
    """

    def __init__(self, agent_client: Any, bot_token: Optional[str] = None) -> None:
        super().__init__(agent_client)
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.bot_token:
            logger.warning(
                "TELEGRAM_BOT_TOKEN is not set. "
                "Telegram integration will not work until a token is provided."
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
        """Forward the message to the AgentClient and return its response."""
        return await self.agent_client.run(
            user_id=f"telegram_{platform_user_id}",
            prompt=text,
            return_response=True
        )

    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Send a text message back to the Telegram user."""
        chat_id = (metadata or {}).get("chat_id", platform_user_id)
        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"

        response_text = response.content
        # Special handling for HITL on Telegram
        if response.action_type == "approval_required":
            response_text = f"{response_text}\n\n⚠️ Usa la API `/approve_tool` o responde para confirmar."

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={"chat_id": chat_id, "text": response_text, "parse_mode": "Markdown"},
                )
                resp.raise_for_status()
                return True
        except Exception:
            logger.exception("Failed to send Telegram message to %s", chat_id)
            return False

    # ------------------------------------------------------------------
    # Webhook helpers
    # ------------------------------------------------------------------

    async def process_update(self, update: dict) -> Optional[AgentResponse]:
        """
        Parse a raw Telegram ``Update`` dict and process the message.

        Returns the agent response, or *None* if the update is not a text message.
        """
        message = update.get("message") or update.get("edited_message")
        if not message or "text" not in message:
            return None

        user_id = str(message["from"]["id"])
        chat_id = str(message["chat"]["id"])
        text = message["text"]

        return await self.handle(
            platform_user_id=user_id,
            text=text,
            metadata={"chat_id": chat_id},
        )

    async def set_webhook(self, webhook_url: str) -> bool:
        """Register a webhook URL with the Telegram Bot API."""
        url = f"{TELEGRAM_API}/bot{self.bot_token}/setWebhook"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={"url": webhook_url})
                resp.raise_for_status()
                logger.info("Telegram webhook set to %s", webhook_url)
                return True
        except Exception:
            logger.exception("Failed to set Telegram webhook")
            return False

    async def delete_webhook(self) -> bool:
        """Remove the current webhook."""
        url = f"{TELEGRAM_API}/bot{self.bot_token}/deleteWebhook"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url)
                resp.raise_for_status()
                return True
        except Exception:
            logger.exception("Failed to delete Telegram webhook")
            return False

