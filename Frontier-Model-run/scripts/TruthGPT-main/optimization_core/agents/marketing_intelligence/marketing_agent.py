"""
OpenClaw Marketing Intelligence Agent — Pydantic-First Architecture.

Specialised in creating marketing content and competitive analysis
by leveraging web-search tools through the ReAct architecture.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..arquitecturas_fundamentales.base_agent import BaseAgent
from ..razonamiento_planificacion.orchestrator import MultiUserReActAgent
from ..models import AgentResponse, AgentConfig
# from ..registry import get_tool (Moved to local import to avoid circularity)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class MarketingConfig(BaseModel):
    """Configuration for the Marketing Intelligence Agent."""
    system_extension: str = Field(
        default=(
            "\n\n"
            "SPECIFIC ROLE: You are an autonomous Digital Marketing Intelligence, "
            "SEO, and Growth Hacking agent.\n"
            "You MUST use your web tools (web_search, web_reader) to research "
            "competitors when solving problems. Search for current trends and "
            "provide data-driven content strategies."
        ),
        description="System prompt extension appended to the base ReAct instructions",
    )
    tools: List[str] = Field(
        default=["web_search", "web_reader"],
        description="Tools to register on the internal ReAct agent",
    )


class MarketingResult(BaseModel):
    """Structured metadata returned alongside the AgentResponse."""
    user_id: str
    tools_registered: List[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Internal ReAct Subclass
# ---------------------------------------------------------------------------

class _MarketingReActAgent(MultiUserReActAgent):
    """ReAct agent with marketing-specific system instructions."""

    def __init__(self, *args: Any, system_extension: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._system_extension = system_extension

    def _get_system_instructions(self) -> str:
        base = super()._get_system_instructions()
        return f"{base}{self._system_extension}"


# ---------------------------------------------------------------------------
# Marketing Agent
# ---------------------------------------------------------------------------

class MarketingAgent(BaseAgent):
    """
    OpenClaw Marketing Intelligence Agent.

    Delegates query processing to an internal ReAct agent that has
    web-search and web-reader tools, with a marketing-focused system prompt.
    Configuration is driven by ``MarketingConfig``.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_engine: Optional[Any] = None,
        marketing_config: Optional[MarketingConfig] = None,
    ) -> None:
        super().__init__(
            name="MarketingAgent",
            role="Marketing and SEO Specialist",
        )
        self.marketing_config = marketing_config or MarketingConfig()

        self._react = _MarketingReActAgent(
            config=config,
            llm_engine=llm_engine,
            system_extension=self.marketing_config.system_extension,
        )

        self._registered_tools: List[str] = []
        for tool_name in self.marketing_config.tools:
            from ..registry import get_tool
            tool_cls = get_tool(tool_name)
            if tool_cls:
                self._react.register_tool(tool_cls())
                self._registered_tools.append(tool_name)

    async def process(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process a marketing query using the internal ReAct agent."""
        logger.info("%s processing: %s", self.name, query)
        start = time.monotonic()

        user_id = (
            context.get("user_id", "default_marketing_user")
            if context
            else "default_marketing_user"
        )

        try:
            response = await self._react.process_message(user_id, query)
            self.add_to_memory("user", query)
            
            # Robust content access
            content = response.content if hasattr(response, 'content') else str(response)
            self.add_to_memory("assistant", content)

            latency_ms = (time.monotonic() - start) * 1000
            response.metadata.update(
                MarketingResult(
                    user_id=user_id,
                    tools_registered=self._registered_tools,
                    latency_ms=round(latency_ms, 2),
                ).model_dump()
            )
            return response
        except Exception:
            logger.exception("Error processing in %s", self.name)
            return AgentResponse(
                content=f"Error in {self.name}: processing failed.",
                action_type="final_answer",
            )

