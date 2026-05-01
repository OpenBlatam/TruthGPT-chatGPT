"""
OpenClaw Messaging -- FastAPI webhook routes.

Exposes webhook endpoints for all supported messaging platforms:
Telegram, WhatsApp, Discord, Signal, Slack, and Microsoft Teams.
"""

import logging
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ..models import AgentResponse

from .telegram_bot import TelegramAdapter
from .whatsapp_webhook import WhatsAppAdapter
from .discord_bot import DiscordAdapter
from .signal_adapter import SignalAdapter
from .slack_bot import SlackAdapter
from .teams_adapter import TeamsAdapter

logger = logging.getLogger(__name__)


class WebhookStatus(BaseModel):
    """Generic webhook response."""

    ok: bool
    detail: str = ""


def create_messaging_router(agent_client: Any) -> APIRouter:
    """
    Build an APIRouter with webhook endpoints for all messaging platforms.

    Args:
        agent_client: An ``AgentClient`` instance shared by all adapters.

    Returns:
        A configured ``APIRouter``.
    """

    router = APIRouter(
        prefix="/webhooks",
        tags=["Messaging Webhooks"],
    )

    telegram = TelegramAdapter(agent_client)
    whatsapp = WhatsAppAdapter(agent_client)
    discord = DiscordAdapter(agent_client)
    signal = SignalAdapter(agent_client)
    slack = SlackAdapter(agent_client)
    teams = TeamsAdapter(agent_client)

    # ==================================================================
    # Telegram
    # ==================================================================

    @router.post("/telegram", response_model=WebhookStatus)
    async def telegram_webhook(request: Request) -> WebhookStatus:
        """Receive a Telegram Bot API update."""
        try:
            update = await request.json()
            response_text = await telegram.process_update(update)
            if response_text is None:
                return WebhookStatus(ok=True, detail="Update ignored (no text message).")
            return WebhookStatus(ok=True, detail="Message processed.")
        except Exception:
            logger.exception("Telegram webhook error")
            return WebhookStatus(ok=False, detail="Internal server error")

    @router.post("/telegram/setup", response_model=WebhookStatus)
    async def telegram_setup_webhook(request: Request) -> WebhookStatus:
        """Register the Telegram webhook URL."""
        body = await request.json()
        url = body.get("webhook_url", "")
        if not url:
            return WebhookStatus(ok=False, detail="Missing 'webhook_url' in body.")
        success = await telegram.set_webhook(url)
        return WebhookStatus(ok=success, detail="Webhook registered." if success else "Failed.")

    # ==================================================================
    # WhatsApp (Twilio)
    # ==================================================================

    @router.post("/whatsapp", response_model=WebhookStatus)
    async def whatsapp_webhook(request: Request) -> WebhookStatus:
        """Receive a Twilio WhatsApp webhook (form-encoded)."""
        try:
            form = await request.form()
            form_data = dict(form)
            response_text = await whatsapp.process_webhook(form_data)
            if response_text is None:
                return WebhookStatus(ok=True, detail="Empty body, ignored.")
            return WebhookStatus(ok=True, detail="Message processed.")
        except Exception:
            logger.exception("WhatsApp webhook error")
            return WebhookStatus(ok=False, detail="Internal server error")

    # ==================================================================
    # Discord
    # ==================================================================

    @router.post("/discord")
    async def discord_webhook(request: Request):
        """Receive a Discord interaction payload."""
        try:
            payload = await request.json()

            # Discord PING verification
            if payload.get("type") == 1:
                return JSONResponse({"type": 1})

            response = await discord.process_interaction(payload)
            if response is None:
                return WebhookStatus(ok=True, detail="Interaction ignored.")

            if isinstance(response, AgentResponse):
                content = response.content
            else:
                content = str(response)

            # Respond with an interaction response (type 4 = CHANNEL_MESSAGE_WITH_SOURCE)
            return JSONResponse({
                "type": 4,
                "data": {"content": content[:2000]},
            })
        except Exception:
            logger.exception("Discord webhook error")
            return WebhookStatus(ok=False, detail="Internal server error")

    # ==================================================================
    # Signal (signal-cli REST API)
    # ==================================================================

    @router.post("/signal", response_model=WebhookStatus)
    async def signal_webhook(request: Request) -> WebhookStatus:
        """Receive a signal-cli webhook payload."""
        try:
            payload = await request.json()
            response_text = await signal.process_webhook(payload)
            if response_text is None:
                return WebhookStatus(ok=True, detail="No text message, ignored.")
            return WebhookStatus(ok=True, detail="Message processed.")
        except Exception:
            logger.exception("Signal webhook error")
            return WebhookStatus(ok=False, detail="Internal server error")

    # ==================================================================
    # Slack (Events API)
    # ==================================================================

    @router.post("/slack")
    async def slack_webhook(request: Request):
        """Receive a Slack Events API payload."""
        try:
            payload = await request.json()

            # Slack URL verification challenge
            if payload.get("type") == "url_verification":
                return JSONResponse({"challenge": payload.get("challenge", "")})

            response_text = await slack.process_event(payload)
            if response_text is None:
                return WebhookStatus(ok=True, detail="Event ignored.")
            return WebhookStatus(ok=True, detail="Message processed.")
        except Exception:
            logger.exception("Slack webhook error")
            return WebhookStatus(ok=False, detail="Internal server error")

    # ==================================================================
    # Microsoft Teams (Bot Framework)
    # ==================================================================

    @router.post("/teams", response_model=WebhookStatus)
    async def teams_webhook(request: Request) -> WebhookStatus:
        """Receive a Bot Framework activity from Microsoft Teams."""
        try:
            activity = await request.json()
            response_text = await teams.process_activity(activity)
            if response_text is None:
                return WebhookStatus(ok=True, detail="Activity ignored.")
            return WebhookStatus(ok=True, detail="Message processed.")
        except Exception:
            logger.exception("Teams webhook error")
            return WebhookStatus(ok=False, detail="Internal server error")

    return router

