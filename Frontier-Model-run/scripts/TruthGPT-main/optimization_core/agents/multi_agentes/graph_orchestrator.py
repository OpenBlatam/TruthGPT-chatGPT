"""
OpenClaw Multi-Agent — Graph-Based Orchestrator (LangGraph) — Pydantic-First.

Implements a Directed Acyclic Graph (DAG) state machine for agents using
LangGraph.  Sequential and conditional workflows where one agent's output
becomes the next agent's input, with persistence and cyclic support.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

try:
    from agents.arquitecturas_fundamentales.base_agent import BaseAgent
    from agents.models import AgentResponse
except ImportError:
    from ..arquitecturas_fundamentales.base_agent import BaseAgent
    from ..models import AgentResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class GraphConfig(BaseModel):
    """Configuration for the Graph Orchestrator pipeline."""
    max_nodes: int = Field(default=20, description="Max nodes allowed in the graph")
    enable_persistence: bool = Field(default=False, description="Enable LangGraph checkpoint persistence")


class GraphState(BaseModel):
    """Shared state passed between agents in the LangGraph pipeline."""
    user_id: str
    initial_input: str
    current_input: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphExecutionResult(BaseModel):
    """Structured result of a graph execution."""
    final_output: str
    steps_executed: int
    node_history: List[Dict[str, str]]


# ---------------------------------------------------------------------------
# Graph Orchestrator
# ---------------------------------------------------------------------------

class GraphOrchestrator:
    """
    LangGraph-based orchestrator for sequential and conditional workflows.

    Usage::

        orchestrator = GraphOrchestrator()
        orchestrator.add_node("data_agent", data_agent)
        orchestrator.add_node("writer_agent", writer_agent)
        orchestrator.add_edge("data_agent", "writer_agent")
        orchestrator.set_entry_point("data_agent")
        orchestrator.compile()
        result = await orchestrator.run("user_1", "Analyze sales.csv")
    """

    def __init__(self, graph_config: Optional[GraphConfig] = None) -> None:
        self.graph_config = graph_config or GraphConfig()
        self.nodes: Dict[str, BaseAgent] = {}

        if not HAS_LANGGRAPH:
            logger.warning(
                "LangGraph not installed. Install via 'pip install langgraph'."
            )

        # LangGraph still needs TypedDict internally for its state schema
        from typing import TypedDict, Annotated, Sequence
        import operator

        class _LangGraphState(TypedDict):
            user_id: str
            initial_input: str
            current_input: str
            history: Annotated[list[dict], operator.add]
            metadata: dict

        self._state_cls = _LangGraphState
        self.workflow = StateGraph(_LangGraphState) if HAS_LANGGRAPH else None
        self.app = None
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, agent: BaseAgent) -> None:
        """Register an agent as a node in the graph."""
        if len(self.nodes) >= self.graph_config.max_nodes:
            raise ValueError(f"Max nodes ({self.graph_config.max_nodes}) exceeded.")

        self.nodes[name] = agent

        async def _node_executor(state: dict) -> dict:
            logger.info("LangGraph executing node: %s", name)
            prompt = (
                f"[SYSTEM INSTRUCTION: You are part of a pipeline. "
                f"Previous output: {state['current_input']}]\n\n"
                f"Next task: Please continue the workflow."
            )
            if len(state["history"]) == 0:
                prompt = state["initial_input"]

            result = await agent.process(prompt, context={"user_id": state["user_id"]})
            result_text = result.content if isinstance(result, AgentResponse) else str(result)

            return {
                "current_input": result_text,
                "history": [{"agent": name, "result": result_text}],
            }

        if self.workflow:
            self.workflow.add_node(name, _node_executor)
        logger.info("Added LangGraph node: %s", name)

    def set_entry_point(self, name: str) -> None:
        """Set the starting node of the graph."""
        self.entry_point = name
        if self.workflow:
            if name not in self.nodes:
                raise ValueError(f"Entry point {name} is not a valid node.")
            self.workflow.set_entry_point(name)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a static sequence from one node to another."""
        if self.workflow:
            target = END if to_node == "__end__" else to_node
            self.workflow.add_edge(from_node, target)

    def add_conditional_edge(
        self, from_node: str, condition_func: Callable
    ) -> None:
        """Add a dynamic edge based on the state."""
        if self.workflow:
            self.workflow.add_conditional_edges(from_node, condition_func)

    def compile(self) -> None:
        """Compile the LangGraph for execution."""
        if self.workflow:
            self.app = self.workflow.compile()
            logger.info("LangGraph compiled successfully.")

    async def run(self, user_id: str, initial_input: str) -> AgentResponse:
        """Execute the compiled graph and return a structured AgentResponse."""
        if not HAS_LANGGRAPH or not self.app:
            raise RuntimeError(
                "LangGraph not installed or graph not compiled. "
                "Run 'pip install langgraph' and call compile()."
            )

        initial_state = {
            "user_id": user_id,
            "initial_input": initial_input,
            "current_input": initial_input,
            "history": [],
            "metadata": {},
        }

        logger.info("Starting LangGraph Orchestrator for %s", user_id)
        final_state = await self.app.ainvoke(initial_state)

        exec_result = GraphExecutionResult(
            final_output=final_state["current_input"],
            steps_executed=len(final_state["history"]),
            node_history=final_state["history"],
        )

        return AgentResponse(
            content=exec_result.final_output,
            action_type="final_answer",
            metadata=exec_result.model_dump(),
        )

