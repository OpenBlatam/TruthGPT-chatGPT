"""
OpenClaw Messaging — Unified Adapter Layer.

Provides platform-specific messaging adapters (Telegram, WhatsApp,
Discord, Signal, Slack, Microsoft Teams, Email) and a FastAPI router
that wires all webhook endpoints together.
"""

from .base import BaseMessagingAdapter
from .telegram_bot import TelegramAdapter
from .whatsapp_webhook import WhatsAppAdapter
from .discord_bot import DiscordAdapter
from .signal_adapter import SignalAdapter
from .slack_bot import SlackAdapter
from .teams_adapter import TeamsAdapter
from .email_adapter import EmailAdapter
from .api_routes import create_messaging_router

__all__ = [
    "BaseMessagingAdapter",
    "TelegramAdapter",
    "WhatsAppAdapter",
    "DiscordAdapter",
    "SignalAdapter",
    "SlackAdapter",
    "TeamsAdapter",
    "EmailAdapter",
    "create_messaging_router",
]
