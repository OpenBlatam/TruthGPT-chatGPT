"""
OpenClaw Messaging -- Base Adapter.

Abstract base class providing the unified messaging adapter interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from ..models import AgentResponse

logger = logging.getLogger(__name__)


class BaseMessagingAdapter(ABC):
    """
    Abstract base for messaging platform adapters (Telegram, WhatsApp, etc.).
    Unified under the Pydantic-first AgentResponse model.
    """

    def __init__(self, agent_client: Any) -> None:
        self.agent_client = agent_client

    @abstractmethod
    async def on_message(
        self,
        platform_user_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> AgentResponse:
        """
        Process an incoming message and return the agent's response.
        MUST return an AgentResponse object.
        """
        pass

    @abstractmethod
    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Send a response back to the user on the platform.
        """
        pass

    async def handle(
        self,
        platform_user_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> AgentResponse:
        """
        Full round-trip: receive a message, run the agent, send the reply.
        """
        logger.info(
            "[%s] Message from %s: %s",
            self.__class__.__name__,
            platform_user_id,
            text[:80],
        )
        
        response = await self.on_message(platform_user_id, text, metadata)
        await self.send_response(platform_user_id, response, metadata)
        return response

