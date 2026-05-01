"""
OpenClaw Agent — Code Interpreter — Pydantic-First Architecture.

A specialised autonomous agent that writes, executes, and iterates on
Python code to solve tasks.  Uses the LLM as the reasoning engine and
the ``PythonExecutionTool`` + ``FileWriteTool`` combo for execution.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..arquitecturas_fundamentales.base_agent import BaseAgent
from ..razonamiento_planificacion.orchestrator import MultiUserReActAgent
from ..models import AgentResponse, AgentConfig
# from ..registry import get_tool (Lazy loaded in __init__)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CodeInterpreterConfig(BaseModel):
    """Configuration for the Code Interpreter Agent."""
    custom_instructions: str = Field(
        default=(
            "You are an autonomous Code Interpreter agent.\n"
            "Your workflow:\n"
            "1. Analyse the user request and plan a Python solution.\n"
            "2. Write the code to a temporary file using file_write.\n"
            "3. Execute it with python_execute and inspect the output.\n"
            "4. If there are errors, fix the code and re-execute.\n"
            "5. Return the final output to the user.\n"
            "Always show the code you wrote AND its output in the final answer."
        ),
        description="System prompt for the internal ReAct agent",
    )
    tools: List[str] = Field(
        default=["python_execute", "file_write", "file_read"],
        description="Tools to register on the internal ReAct agent",
    )


class CodeInterpreterResult(BaseModel):
    """Structured metadata for code interpreter responses."""
    user_id: str
    tools_registered: List[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Code Interpreter Agent
# ---------------------------------------------------------------------------

class CodeInterpreterAgent(BaseAgent):
    """
    Autonomous code-writing and execution agent.

    Given a task, the agent will:
    1. Write Python code to solve it.
    2. Execute the code and inspect the output.
    3. Fix errors or refine the solution iteratively (ReAct loop).

    Configuration is driven by ``CodeInterpreterConfig``.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_engine: Optional[Any] = None,
        interpreter_config: Optional[CodeInterpreterConfig] = None,
    ) -> None:
        super().__init__(
            name="CodeInterpreterAgent",
            role="Interprete de Codigo y Ejecucion Autonoma de Python",
        )
        self.interpreter_config = interpreter_config or CodeInterpreterConfig()

        self.react_agent = MultiUserReActAgent(
            config=config,
            llm_engine=llm_engine,
            custom_system_instructions=self.interpreter_config.custom_instructions,
        )

        self._registered_tools: List[str] = []
        for tool_name in self.interpreter_config.tools:
            from ..registry import get_tool
            tool_cls = get_tool(tool_name)
            if tool_cls:
                self.react_agent.register_tool(tool_cls())
                self._registered_tools.append(tool_name)

    async def process(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute the code interpreter ReAct loop."""
        user_id = (context or {}).get("user_id", "code_interpreter_default")
        logger.info("[%s] Processing: %s", self.name, query[:80])
        start = time.monotonic()

        try:
            response = await self.react_agent.process_message(user_id, query)
            self.add_to_memory("user", query)
            
            # Robust content access
            content = response.content if hasattr(response, 'content') else str(response)
            self.add_to_memory("assistant", content)

            latency_ms = (time.monotonic() - start) * 1000
            response.metadata.update(
                CodeInterpreterResult(
                    user_id=user_id,
                    tools_registered=self._registered_tools,
                    latency_ms=round(latency_ms, 2),
                ).model_dump()
            )
            return response
        except Exception as e:
            logger.error("[%s] Error: %s", self.name, e)
            return AgentResponse(
                content=f"Error in {self.name}: {e}",
                action_type="final_answer",
            )

