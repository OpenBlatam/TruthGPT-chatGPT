"""
OpenClaw Agent Client (SDK) — Pydantic-First Architecture.

Provides a high-level interface for initialising and using autonomous agents,
supporting both single-agent ReAct mode and multi-agent swarm mode.
"""

import json
import logging
import time
import inspect
from typing import Any, AsyncIterator, Dict, Optional, Union

from .multi_agentes.swarm_orchestrator import SwarmOrchestrator
from .razonamiento_planificacion.orchestrator import MultiUserReActAgent
from .models import AgentResponse, AgentConfig
from .registry import registry
from .engines import DummyAsyncLLM
from .exceptions import HandoffError

logger = logging.getLogger(__name__)


# Opentelemetry support (optional)
try:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
except ImportError:
    tracer = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AgentClient:
    """
    High-level client for OpenClaw autonomous agents.

    Args:
        config: ``AgentConfig`` Pydantic model with all settings.
        llm_engine: An async-callable LLM engine (``await engine(prompt)``).
                    Falls back to :class:`DummyAsyncLLM` when *None*.
    """

    AVAILABLE_TOOLS = registry.get_all_tools()

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        # Legacy positional args for backward compatibility
        llm_engine: Optional[Any] = None,
        memory_db_path: str = "openclaw_memory.db",
        use_swarm: bool = False,
        use_vector_memory: bool = False,
        use_reflexion: bool = False,
    ) -> None:
        if config is None:
            config = AgentConfig(
                llm_engine=llm_engine,
                memory_db_path=memory_db_path,
                use_swarm=use_swarm,
                use_vector_memory=use_vector_memory,
                use_reflexion=use_reflexion,
            )

        self.config = config
        self.llm_engine = config.llm_engine or DummyAsyncLLM()

        # Lazy import to avoid circular dependency issues
        from .memoria_aprendizaje.sqlite_memory import SQLiteMemory
        self.memory = SQLiteMemory(db_path=config.memory_db_path)

        self.use_swarm = config.use_swarm
        self.use_reflexion = config.use_reflexion

        # Init Vector Memory if requested
        self.vector_memory = None
        if config.use_vector_memory:
            try:
                from .memoria_aprendizaje.vector_memory import VectorMemory
                self.vector_memory = VectorMemory()
            except ImportError:
                logger.warning("VectorMemory not available (chromadb missing).")

        # Swarm or single-agent ReAct
        self.swarm: Optional[SwarmOrchestrator] = None
        self.agent: Optional[MultiUserReActAgent] = None

        if self.use_swarm:
            self.swarm = SwarmOrchestrator(
                llm_engine=self.llm_engine,
                default_agent_name=config.default_agent_name,
            )
            self._init_default_swarm()
        else:
            self.agent = MultiUserReActAgent(
                config=config,
                llm_engine=self.llm_engine,
                vector_memory=self.vector_memory,
            )
            self._register_default_tools()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_default_swarm(self) -> None:
        """Register the default set of agents in the swarm using the ComponentRegistry."""
        assert self.swarm is not None

        # These names must match the registered names in the registry
        default_agent_classes = [
            "marketing_agent",
            "rl_agent",
            "code_interpreter",
            "data_analysis",
            "system_agent",
        ]

        # Lazy-import and register if not already there (bootstrap)
        self._ensure_default_agents_registered()

        for agent_name in default_agent_classes:
            agent_cls = registry.get_agent(agent_name)
            if agent_cls:
                try:
                    agent_instance = agent_cls(config=self.config, llm_engine=self.llm_engine)
                    self.swarm.register_agent(agent_instance)
                except Exception as e:
                    logger.warning("Could not instantiate default agent %s: %s", agent_name, e)
            else:
                logger.warning("Default agent %s not found in registry.", agent_name)

        logger.info(
            "Swarm initialised with agents: %s",
            [a.name for a in self.swarm.agents.values()],
        )

    def _ensure_default_agents_registered(self) -> None:
        """Helper to ensure built-in agents are in the registry."""
        # Check if already registered to avoid redundant imports
        if registry.get_agent("marketing_agent"):
            return

        try:
            from .marketing_intelligence.marketing_agent import MarketingAgent
            registry.register_agent("marketing_agent", MarketingAgent)
            
            from .embodied_rl.rl_agent import RLAgent
            registry.register_agent("rl_agent", RLAgent)
            
            from .code_interpreter import CodeInterpreterAgent
            registry.register_agent("code_interpreter", CodeInterpreterAgent)
            
            from .data_analysis import DataAnalysisAgent
            registry.register_agent("data_analysis", DataAnalysisAgent)

            from .system_intelligence.system_agent import SystemAgent
            registry.register_agent("system_agent", SystemAgent)
        except ImportError as e:
            logger.error("Failed to import built-in agents during registration: %s", e)

    def _register_default_tools(self) -> None:
        """Register all built-in tools on the single-agent ReAct instance."""
        assert self.agent is not None

        for tool_name, tool_cls in registry.get_all_tools().items():
            try:
                # Check if tool_cls requires arguments in __init__
                sig = inspect.signature(tool_cls.__init__)
                # Exclude 'self' and count required args
                required_args = [
                    p for p in sig.parameters.values() 
                    if p.name != 'self' and p.default is p.empty and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
                ]
                
                if len(required_args) > 0:
                    logger.debug("Skipping auto-registration for tool %s: requires arguments", tool_name)
                    continue

                tool_instance = tool_cls()
                if hasattr(tool_instance, "agent_client"):
                    tool_instance.agent_client = self
                self.agent.register_tool(tool_instance)
            except Exception as e:
                logger.warning("Could not register tool %s: %s", tool_name, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_tool(self, tool_name: str) -> bool:
        """
        Enable a specific tool on the single-agent instance.

        Returns *True* if the tool was registered, *False* otherwise.
        """
        if self.use_swarm or self.agent is None:
            return False

        tool_cls = registry.get_tool(tool_name)
        if tool_cls is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            return False

        tool_instance = tool_cls()
        if hasattr(tool_instance, "agent_client"):
            tool_instance.agent_client = self

        self.agent.register_tool(tool_instance)
        return True

    async def run(
        self,
        user_id: str,
        prompt: str,
        depth: int = 0,
        return_response: bool = False,
    ) -> Union[str, AgentResponse]:
        """
        Execute the agent (or swarm) to process *prompt*.

        If ``return_response`` is True, returns the full ``AgentResponse``.
        Otherwise, returns just the string content (default for backward compat).
        """
        if depth > self.config.max_handoff_depth:
            err_msg = f"Error: Maximum swarm handoff depth ({self.config.max_handoff_depth}) exceeded."
            if return_response:
                return AgentResponse(content=err_msg, action_type="error")
            return err_msg

        final_resp: AgentResponse

        if self.use_swarm and self.swarm is not None:
            final_resp = await self.swarm.route_and_process(
                prompt, context={"user_id": user_id}
            )

            # Handle recursive handoff
            if final_resp.action_type == "handoff" and final_resp.handoff_target:
                handoff_resp = await self._handle_handoff(
                    user_id, prompt, final_resp.handoff_target, depth + 1
                )
                final_resp = handoff_resp

        elif self.agent is not None:
            final_resp = await self.agent.process_message(user_id, prompt)

            if final_resp.action_type == "handoff" and final_resp.handoff_target:
                final_resp = await self._handle_handoff(
                    user_id, prompt, final_resp.handoff_target, depth + 1
                )
        else:
            raise RuntimeError("Neither swarm nor single agent is initialised.")

        if return_response:
            return final_resp if not isinstance(final_resp, str) else AgentResponse(content=final_resp, action_type="final_answer")
        else:
            return final_resp.content if hasattr(final_resp, "content") else str(final_resp)

    async def _handle_handoff(
        self, user_id: str, prompt: str, target: str, depth: int
    ) -> AgentResponse:
        """Transfer control to a named agent in the swarm."""
        logger.info("AgentClient detected Handoff to %s. Transferring control...", target)

        if self.use_swarm and self.swarm and target in self.swarm.agents:
            target_agent = self.swarm.agents[target]
            handoff_prompt = (
                f"[SYSTEM: CONTEXT HANDOFF]\nUser request: {prompt}\nRespond as {target}."
            )
            return await target_agent.process(handoff_prompt, context={"user_id": user_id})

        raise HandoffError(
            f"Cannot handoff to '{target}' (Not found or Swarm mode disabled).",
            metadata={"target": target, "user_id": user_id},
        )

    async def astream_run(self, user_id: str, prompt: str) -> AsyncIterator[str]:
        """Execute the agent and stream the response via SSE."""
        if self.agent is not None:
            async for chunk in self.agent.astream_process_message(user_id, prompt):
                yield chunk
        elif self.use_swarm and self.swarm is not None:
            yield json.dumps({"event": "thinking", "content": "Swarm orchestrator is routing your request..."}) + "\n"
            resp = await self.run(user_id, prompt, return_response=True)
            yield json.dumps({"event": "final_answer", "content": resp.content}) + "\n"
        else:
            yield json.dumps({"event": "error", "message": "Agent client not properly initialised."}) + "\n"

    async def clear_memory(self, user_id: str) -> bool:
        """Clear episodic memory for a given user."""
        await self.memory.clear_memory(user_id)
        return True

