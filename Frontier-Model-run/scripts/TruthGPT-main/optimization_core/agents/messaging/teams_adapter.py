"""
OpenClaw Messaging -- Microsoft Teams Adapter.

Receives Bot Framework activity payloads and replies through the
Bot Framework REST API.

Environment variables:
    TEAMS_APP_ID        -- Microsoft App ID (from Azure Bot registration).
    TEAMS_APP_PASSWORD  -- Microsoft App Password / Client Secret.
"""

import logging
import os
from typing import Any, Optional

import httpx

from .base import BaseMessagingAdapter
from ..models import AgentResponse

logger = logging.getLogger(__name__)

BF_AUTH_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"


class TeamsAdapter(BaseMessagingAdapter):
    """
    Microsoft Teams messaging adapter via the Bot Framework.

    Receives activity payloads at ``/webhooks/teams`` and replies using
    the Bot Framework ``/v3/conversations`` API.
    """

    def __init__(self, agent_client: Any) -> None:
        super().__init__(agent_client)
        self.app_id = os.getenv("TEAMS_APP_ID", "")
        self.app_password = os.getenv("TEAMS_APP_PASSWORD", "")
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        if not self.app_id or not self.app_password:
            logger.warning(
                "TEAMS_APP_ID / TEAMS_APP_PASSWORD not set. "
                "Teams integration will not work until credentials are provided."
            )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _get_token(self, force_refresh: bool = False) -> str:
        """Obtain a Bot Framework access token. Refreshes if expired or forced."""
        import time

        now = time.time()
        # If not forcing, and we have a token that hasn't expired (give a 5-min buffer)
        if not force_refresh and self._access_token and self._token_expires_at > (now + 300):
            return self._access_token

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    BF_AUTH_URL,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.app_id,
                        "client_secret": self.app_password,
                        "scope": "https://api.botframework.com/.default",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                self._access_token = data.get("access_token", "")
                expires_in = data.get("expires_in", 3600)
                self._token_expires_at = now + float(expires_in)
                return self._access_token
        except Exception:
            logger.exception("Failed to obtain Bot Framework token")
            return ""

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
            user_id=f"teams_{platform_user_id}",
            prompt=text,
            return_response=True
        )

    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        meta = metadata or {}
        service_url = meta.get("service_url", "")
        conversation_id = meta.get("conversation_id", "")

        if not service_url or not conversation_id:
            logger.error("Cannot reply to Teams without service_url and conversation_id")
            return False

        response_text = response.content
        if response.action_type == "approval_required":
            response_text = f"{response_text}\n\n[HITL] Aprobación requerida para continuar."

        url = f"{service_url}v3/conversations/{conversation_id}/activities"

        return await self._send_with_retry(url, response_text)

    async def _send_with_retry(self, url: str, response_text: str, _is_retry: bool = False) -> bool:
        """Attempt to send the message. Automatically retry once if 401 Unauthorized."""
        token = await self._get_token(force_refresh=_is_retry)
        if not token:
            return False

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={
                        "type": "message",
                        "text": response_text,
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code == 401 and not _is_retry:
                    logger.warning("Received 401 from Teams. Forcing token refresh and retrying.")
                    return await self._send_with_retry(url, response_text, _is_retry=True)
                
                resp.raise_for_status()
                return True
        except Exception:
            logger.exception("Failed to send Teams reply")
            return False

    # ------------------------------------------------------------------
    # Webhook helper
    # ------------------------------------------------------------------

    async def process_activity(self, activity: dict) -> Optional[AgentResponse]:
        """
        Process an incoming Bot Framework Activity.

        Handles ``type=message`` activities.
        """
        if activity.get("type") != "message":
            return None

        user_id = activity.get("from", {}).get("id", "unknown")
        text = activity.get("text", "").strip()
        service_url = activity.get("serviceUrl", "")
        conversation_id = activity.get("conversation", {}).get("id", "")

        if not text:
            return None

        return await self.handle(
            platform_user_id=user_id,
            text=text,
            metadata={
                "service_url": service_url,
                "conversation_id": conversation_id,
            },
        )

