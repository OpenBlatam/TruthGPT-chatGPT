"""
OpenClaw Swarm Orchestrator — Pydantic-First Architecture.

Routes user queries to the most appropriate agent within a swarm,
using either LLM-based intelligent routing or keyword-based fallback.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from ..arquitecturas_fundamentales.base_agent import BaseAgent
from ..models import AgentResponse
from ..exceptions import RoutingError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SwarmConfig(BaseModel):
    """Configuration for the Swarm Orchestrator."""
    max_routing_retries: int = Field(default=3, description="Max LLM routing attempts before keyword fallback")
    default_agent_name: Optional[str] = Field(default=None, description="Fallback agent name if routing fails")


class RouterDecision(BaseModel):
    """Pydantic model for the LLM router's structured JSON output."""
    reasoning: str = Field(..., description="Chain-of-thought explanation for the routing decision")
    target_agent: str = Field(..., description="Exact registered name of the selected agent")


class RoutingResult(BaseModel):
    """Metadata attached to the AgentResponse after routing."""
    routed_to: str
    routing_method: str  # "llm" or "keyword"
    routing_latency_ms: float


# ---------------------------------------------------------------------------
# Swarm Orchestrator
# ---------------------------------------------------------------------------

class SwarmOrchestrator:
    """
    Routes queries to the best-fit agent in a multi-agent swarm.

    Routing strategy:
    1. If an LLM engine is provided, ask the LLM to pick the target agent.
    2. If the LLM returns an invalid name, fall back to keyword matching.
    3. If no LLM is available, use keyword matching directly.
    4. If nothing matches, delegate to the first registered agent.
    """

    def __init__(
        self,
        llm_engine: Optional[Any] = None,
        default_agent_name: Optional[str] = None,
        swarm_config: Optional[SwarmConfig] = None,
    ) -> None:
        self.agents: Dict[str, BaseAgent] = {}
        self.llm = llm_engine
        self.swarm_config = swarm_config or SwarmConfig(default_agent_name=default_agent_name)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent in the swarm."""
        self.agents[agent.name] = agent
        logger.info("Agent registered in swarm: %s (%s)", agent.name, agent.role)

    async def route_and_process(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Route *query* to the most suitable agent and return its response."""
        if not self.agents:
            raise RoutingError("No agents registered in the swarm.")

        start = time.monotonic()
        target_agent, method = await self._route_query(query)

        if target_agent is None:
            if self.swarm_config.default_agent_name and self.swarm_config.default_agent_name in self.agents:
                target_agent = self.agents[self.swarm_config.default_agent_name]
                method = "default_config"
            else:
                target_agent = next(iter(self.agents.values()))
                method = "first_registered"
            logger.info("Routing fallback: using %s (%s)", target_agent.name, method)

        logger.info("Routing query to agent: %s", target_agent.name)
        response = await target_agent.process(query, context)

        # Inject routing telemetry
        routing_latency_ms = (time.monotonic() - start) * 1000
        response.metadata.update(
            RoutingResult(
                routed_to=target_agent.name,
                routing_method=method,
                routing_latency_ms=round(routing_latency_ms, 2),
            ).model_dump()
        )
        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scrub_json(self, text: str) -> str:
        """Clean up LLM response to get a raw JSON string."""
        clean = text.strip()
        if "```" in clean:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
            if match:
                clean = match.group(1).strip()
        if not (clean.startswith('{') and clean.endswith('}')):
            start = clean.find('{')
            end = clean.rfind('}')
            if start != -1 and end != -1:
                clean = clean[start:end+1]
        return clean

    async def _route_query(self, query: str) -> tuple[Optional[BaseAgent], str]:
        """Determine which agent should handle *query*."""
        if self.llm:
            agent = await self._route_via_llm(query)
            if agent:
                return agent, "llm"
        kw_agent = self._route_via_keywords(query)
        return kw_agent, "keyword"

    async def _route_via_llm(self, query: str) -> Optional[BaseAgent]:
        """Use the LLM to decide the target agent using Pydantic schema."""
        agents_info = "\n".join(
            f"- {a.name}: {a.role}" for a in self.agents.values()
        )

        # Use Pydantic's own schema generator instead of hand-rolled dicts
        schema_str = json.dumps(RouterDecision.model_json_schema(), indent=2)

        prompt = (
            "### MASTER ROUTER ROLE\n"
            "You are the Master Router of a SOTA 2025 Multi-Agent Swarm.\n"
            "Your task is to route the user query to the most capable specialized agent.\n\n"
            "### AVAILABLE AGENTS\n"
            f"{agents_info}\n\n"
            f"### USER QUERY\n"
            f"'{query}'\n\n"
            "### INSTRUCTIONS\n"
            "1. Analyze the query semantic intent.\n"
            "2. Select the agent whose role best matches the query.\n"
            "3. Return a JSON response matching the following schema:\n"
            f"{schema_str}\n\n"
            "### CRITICAL RULE\n"
            "Respond ONLY with raw JSON. No conversational text, no markdown wrappers."
        )

        for attempt in range(self.swarm_config.max_routing_retries):
            try:
                decision_raw = await self.llm(prompt)
                clean_resp = self._scrub_json(decision_raw)
                decision = RouterDecision.model_validate_json(clean_resp)

                logger.debug("[Router CoT] %s", decision.reasoning)

                if decision.target_agent in self.agents:
                    logger.info("LLM Semantic Router selected: %s", decision.target_agent)
                    return self.agents[decision.target_agent]

                logger.warning(
                    "Swarm LLM returned invalid name: '%s'. Attempt %d/%d",
                    decision.target_agent, attempt + 1, self.swarm_config.max_routing_retries,
                )
            except json.JSONDecodeError as e:
                logger.warning("Router LLM JSON Parse Error: %s. Retry...", e)
                prompt += f"\nERROR: Your response was not valid JSON ({e}). Respond ONLY with JSON."
            except Exception:
                logger.exception("LLM routing internal error.")
                break

        logger.warning("All LLM routing attempts failed. Falling back to keyword routing.")
        return None

    def _route_via_keywords(self, query: str) -> BaseAgent:
        """Match agents by checking whether their name or role appears in the query."""
        query_lower = query.lower()
        for agent in self.agents.values():
            if agent.name.lower() in query_lower or agent.role.lower() in query_lower:
                return agent
        return next(iter(self.agents.values()))

