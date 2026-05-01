"""
OpenClaw Messaging – WhatsApp Adapter (Twilio).

Receives incoming WhatsApp messages via a Twilio webhook and responds
through the Twilio REST API.

Environment variables:
    TWILIO_ACCOUNT_SID     –  Twilio Account SID.
    TWILIO_AUTH_TOKEN       –  Twilio Auth Token.
    TWILIO_WHATSAPP_FROM   –  Sender number, e.g. ``whatsapp:+14155238886``.
"""

import logging
import os
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from .base import BaseMessagingAdapter
from ..models import AgentResponse

logger = logging.getLogger(__name__)

TWILIO_API = "https://api.twilio.com/2010-04-01"


class WhatsAppAdapter(BaseMessagingAdapter):
    """
    WhatsApp messaging adapter powered by Twilio.

    Operation:
    1. Twilio forwards incoming WhatsApp messages to ``/webhooks/whatsapp``.
    2. The adapter processes the message through the AgentClient.
    3. The reply is sent back via the Twilio Messages API.
    """

    def __init__(self, agent_client: Any) -> None:
        super().__init__(agent_client)
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

        if not self.account_sid or not self.auth_token:
            logger.warning(
                "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set. "
                "WhatsApp integration will not work until credentials are provided."
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
        """Forward the message to the AgentClient."""
        return await self.agent_client.run(
            user_id=f"whatsapp_{platform_user_id}",
            prompt=text,
            return_response=True
        )

    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Send a WhatsApp message back via Twilio."""
        to_number = (metadata or {}).get("from_number", platform_user_id)
        url = f"{TWILIO_API}/Accounts/{self.account_sid}/Messages.json"

        response_text = response.content
        if response.action_type == "approval_required":
            response_text = f"{response_text}\n\n*HITL Required:* Aprueba en la API para continuar."

        payload = urlencode({
            "From": self.from_number,
            "To": to_number,
            "Body": response_text[:1600],  # WhatsApp limit
        })

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    url,
                    content=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    auth=(self.account_sid, self.auth_token),
                )
                resp.raise_for_status()
                logger.info("WhatsApp message sent to %s", to_number)
                return True
        except Exception:
            logger.exception("Failed to send WhatsApp message to %s", to_number)
            return False

    # ------------------------------------------------------------------
    # Webhook helpers
    # ------------------------------------------------------------------

    async def process_webhook(self, form_data: dict) -> Optional[AgentResponse]:
        """
        Parse an incoming Twilio webhook form payload.

        Twilio sends ``From``, ``Body``, ``To``, etc. as form fields.
        Returns the agent response, or *None* if the payload has no body.
        """
        body = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "")

        if not body:
            return None

        # Normalise the user ID (remove "whatsapp:" prefix for storage)
        user_id = from_number.replace("whatsapp:", "").replace("+", "")

        return await self.handle(
            platform_user_id=user_id,
            text=body,
            metadata={"from_number": from_number},
        )

