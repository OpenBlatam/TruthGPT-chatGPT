"""
OpenClaw Messaging -- Signal Adapter.

Communicates with Signal via the ``signal-cli`` REST API
(https://github.com/bbernhard/signal-cli-rest-api).

Environment variables:
    SIGNAL_CLI_API_URL   -- Base URL of the signal-cli REST API (default http://localhost:8079).
    SIGNAL_SENDER_NUMBER -- The phone number registered in signal-cli, e.g. +1234567890.
"""

import logging
import os
from typing import Any, Optional

import httpx

from .base import BaseMessagingAdapter
from ..models import AgentResponse

logger = logging.getLogger(__name__)


class SignalAdapter(BaseMessagingAdapter):
    """
    Signal messaging adapter via ``signal-cli-rest-api``.

    Deploy ``signal-cli-rest-api`` as a Docker container alongside your
    OpenClaw server, then point ``SIGNAL_CLI_API_URL`` to it.
    """

    def __init__(self, agent_client: Any) -> None:
        super().__init__(agent_client)
        self.api_url = os.getenv("SIGNAL_CLI_API_URL", "http://localhost:8079")
        self.sender = os.getenv("SIGNAL_SENDER_NUMBER", "")
        if not self.sender:
            logger.warning(
                "SIGNAL_SENDER_NUMBER is not set. "
                "Signal integration will not work until a sender number is provided."
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
            user_id=f"signal_{platform_user_id}",
            prompt=text,
            return_response=True
        )

    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        url = f"{self.api_url}/v2/send"

        response_text = response.content
        if response.action_type == "approval_required":
            response_text = f"{response_text}\n\n[HITL] Aprobación requerida en la API."
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    url,
                    json={
                        "message": response_text,
                        "number": self.sender,
                        "recipients": [platform_user_id],
                    },
                )
                resp.raise_for_status()
                return True
        except Exception:
            logger.exception("Failed to send Signal message to %s", platform_user_id)
            return False

    # ------------------------------------------------------------------
    # Webhook helper
    # ------------------------------------------------------------------

    async def process_webhook(self, payload: dict) -> Optional[AgentResponse]:
        """
        Process an incoming signal-cli webhook payload.

        The ``signal-cli-rest-api`` can POST received messages to a URL.
        Expected shape::

            {
                "envelope": {
                    "source": "+1234567890",
                    "dataMessage": {"message": "Hello agent"}
                }
            }
        """
        envelope = payload.get("envelope", {})
        source = envelope.get("source", "")
        data_msg = envelope.get("dataMessage", {})
        text = data_msg.get("message", "").strip()

        if not text or not source:
            return None

        user_id = source.replace("+", "")
        return await self.handle(
            platform_user_id=user_id,
            text=text,
            metadata={"source_number": source},
        )

