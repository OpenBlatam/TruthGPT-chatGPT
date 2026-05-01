"""
OpenClaw Agent — Data Analysis — Pydantic-First Architecture.

Specialised agent for reading, parsing and analysing structured data
(CSV, JSON, Excel).  Uses PythonExecutionTool to run pandas/matplotlib
scripts and returns insights plus generated charts.
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

class DataAnalysisConfig(BaseModel):
    """Configuration for the Data Analysis Agent."""
    custom_instructions: str = Field(
        default=(
            "You are an autonomous Data Analysis agent.\n"
            "Your workflow:\n"
            "1. Read the data file the user provides (use file_read).\n"
            "2. Write a Python script that uses pandas to clean and analyse the data.\n"
            "3. Execute the script (python_execute) and inspect the output.\n"
            "4. If the user asks for a chart, generate it with matplotlib/seaborn "
            "and save it as a PNG file (use file_write for any helper code).\n"
            "5. Summarise your findings in a clear, structured report.\n"
            "Always include numeric insights (mean, median, std, top values)."
        ),
        description="System prompt for the internal ReAct agent",
    )
    tools: List[str] = Field(
        default=["python_execute", "file_read", "file_write"],
        description="Tools to register on the internal ReAct agent",
    )


class DataAnalysisResult(BaseModel):
    """Structured metadata for data analysis responses."""
    user_id: str
    tools_registered: List[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Data Analysis Agent
# ---------------------------------------------------------------------------

class DataAnalysisAgent(BaseAgent):
    """
    Autonomous data analysis agent.

    Capabilities:
    - Read CSV / JSON / Excel files.
    - Generate summary statistics, correlations, and distributions.
    - Create matplotlib / seaborn charts and save them to disk.
    - Answer natural-language questions about the data.

    Configuration is driven by ``DataAnalysisConfig``.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_engine: Optional[Any] = None,
        analysis_config: Optional[DataAnalysisConfig] = None,
    ) -> None:
        super().__init__(
            name="DataAnalysisAgent",
            role="Analista de Datos Autonomo con Python y Pandas",
        )
        self.analysis_config = analysis_config or DataAnalysisConfig()

        self.react_agent = MultiUserReActAgent(
            config=config,
            llm_engine=llm_engine,
            custom_system_instructions=self.analysis_config.custom_instructions,
        )

        self._registered_tools: List[str] = []
        for tool_name in self.analysis_config.tools:
            from ..registry import get_tool
            tool_cls = get_tool(tool_name)
            if tool_cls:
                self.react_agent.register_tool(tool_cls())
                self._registered_tools.append(tool_name)

    async def process(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute the data analysis ReAct loop."""
        user_id = (context or {}).get("user_id", "data_analysis_default")
        logger.info("[%s] Processing: %s", self.name, query[:80])
        start = time.monotonic()

        try:
            response: AgentResponse = await self.react_agent.process_message(user_id, query)
            self.add_to_memory("user", query)
            self.add_to_memory("assistant", response.content)

            latency_ms = (time.monotonic() - start) * 1000
            response.metadata.update(
                DataAnalysisResult(
                    user_id=user_id,
                    tools_registered=self._registered_tools,
                    latency_ms=round(latency_ms, 2),
                ).model_dump()
            )
            return response
        except Exception as e:
            logger.exception("[%s] Error processing query: %s", self.name, e)
            return AgentResponse(
                content=f"Error in {self.name}: {str(e)}",
                action_type="final_answer",
            )

