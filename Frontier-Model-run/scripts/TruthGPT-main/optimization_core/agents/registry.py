"""
Centralized Component Registry for TruthGPT — Pydantic-First.

Provides a singleton registry for discovering, registering, and
introspecting both tools and agents available to the ecosystem.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from .razonamiento_planificacion.tools import (
    BaseTool,
)
from .arquitecturas_fundamentales.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ToolInfo(BaseModel):
    """Structured introspection data for a registered tool."""
    name: str
    class_name: str
    module: str = ""
    has_run: bool = True


class AgentInfo(BaseModel):
    """Structured introspection data for a registered agent."""
    name: str
    role: str
    class_name: str
    module: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ComponentRegistry:
    """Dynamic singleton registry for TruthGPT components (tools and agents)."""

    _instance = None
    _tools: Dict[str, Type[BaseTool]] = {}
    _agents: Dict[str, Type[BaseAgent]] = {}

    def __new__(cls) -> "ComponentRegistry":
        if cls._instance is None:
            cls._instance = super(ComponentRegistry, cls).__new__(cls)
            cls._instance._init_builtins()
            cls._instance.discover_plugins()
        return cls._instance

    def _init_builtins(self) -> None:
        """Register core tools."""
        from .razonamiento_planificacion.tools import (
            FileReadTool, FileWriteTool, PythonExecutionTool, SystemBashTool,
            WebReaderTool, WebSearchTool, DelegateTaskTool
        )
        from .system_intelligence.system_tools import (
            ListPapersTool, PaperInfoTool, SystemHealthTool, RunOptimizationTool,
            ModelInferenceTool, ModelTrainTool, ArXivSearchTool, PaperSynthesisTool
        )
        
        self._tools = {
            "system_bash": SystemBashTool,
            "web_search": WebSearchTool,
            "web_reader": WebReaderTool,
            "file_read": FileReadTool,
            "file_write": FileWriteTool,
            "python_execute": PythonExecutionTool,
            "delegate_task": DelegateTaskTool,
            "system_papers_list": ListPapersTool,
            "system_papers_info": PaperInfoTool,
            "system_health": SystemHealthTool,
            "system_run_optimization": RunOptimizationTool,
            "system_model_inference": ModelInferenceTool,
            "system_model_train": ModelTrainTool,
            "arxiv_search": ArXivSearchTool,
            "paper_synthesis": PaperSynthesisTool,
        }
        from .marketing_intelligence.marketing_agent import MarketingAgent
        from .embodied_rl.rl_agent import RLAgent
        from .system_intelligence.system_agent import SystemAgent
        from .system_intelligence.research_agent import ResearchAgent
        
        # Standardize on snake_case for all agents to avoid duplicates with client.py
        self._agents = {
            "research_agent": ResearchAgent,
            "marketing_agent": MarketingAgent,
            "rl_agent": RLAgent,
            "system_agent": SystemAgent,
        }

    # --- Tool Management ---

    def register_tool(self, name: str, tool_cls: Type[BaseTool]) -> None:
        """Manually register a tool."""
        self._tools[name] = tool_cls
        logger.info("Tool registered: %s -> %s", name, tool_cls.__name__)

    def get_tool(self, name: str) -> Optional[Type[BaseTool]]:
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Type[BaseTool]]:
        """Return all valid registered tools."""
        return {
            k: v for k, v in self._tools.items()
            if isinstance(k, str) and not k.startswith("__")
        }

    def list_tools(self) -> List[ToolInfo]:
        """Return structured Pydantic introspection of all registered tools."""
        return [
            ToolInfo(
                name=name,
                class_name=cls.__name__,
                module=cls.__module__ if hasattr(cls, "__module__") else "",
                has_run=hasattr(cls, "run") or hasattr(cls, "process"),
            )
            for name, cls in self.get_all_tools().items()
        ]

    # --- Agent Management ---

    def register_agent(self, name: str, agent_cls: Type[BaseAgent]) -> None:
        """Manually register an agent."""
        self._agents[name] = agent_cls
        logger.info("Agent registered: %s -> %s", name, agent_cls.__name__)

    def get_agent(self, name: str) -> Optional[Type[BaseAgent]]:
        return self._agents.get(name)

    def get_all_agents(self) -> Dict[str, Type[BaseAgent]]:
        return dict(self._agents)

    def list_agents(self) -> List[AgentInfo]:
        """Return structured Pydantic introspection of all registered agents."""
        return [
            AgentInfo(
                name=name,
                role=getattr(cls, "role", "Unknown Role"),
                class_name=cls.__name__,
                module=cls.__module__,
            )
            for name, cls in self._agents.items()
        ]

    def register(self, name: str, cls: Type[Any]) -> None:
        """
        Generic registration method.

        Routes to register_tool or register_agent based on the class type.
        """
        if issubclass(cls, BaseTool):
            self.register_tool(name, cls)
        elif issubclass(cls, BaseAgent):
            self.register_agent(name, cls)
        else:
            logger.warning("ComponentRegistry: Unknown component type for %s (%s)", name, cls)
            # Fallback to tool if it's not an agent but we want to try anyway
            self._tools[name] = cls

    # --- Discovery ---

    def discover_plugins(self, plugins_dir: str = "plugins") -> None:
        """Dynamically load tools and agents from a directory."""
        path = Path(plugins_dir)
        if not path.exists():
            return

        for file in path.glob("*.py"):
            if file.name == "__init__.py":
                continue

            module_name = f"{plugins_dir}.{file.stem}"
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        # Discover Tools
                        if issubclass(obj, BaseTool) and obj is not BaseTool:
                            tool_name = getattr(obj, "name", obj.__name__.lower())
                            self.register_tool(tool_name, obj)
                            logger.info("Tool plugin discovered: %s", tool_name)
                        
                        # Discover Agents
                        elif issubclass(obj, BaseAgent) and obj is not BaseAgent:
                            agent_name = getattr(obj, "name", obj.__name__)
                            self.register_agent(agent_name, obj)
                            logger.info("Agent plugin discovered: %s", agent_name)

            except Exception as e:
                logger.warning("Error loading plugin %s: %s", module_name, e)


# Global singleton
registry = ComponentRegistry()

# Backward-compatible aliases
ToolRegistry = ComponentRegistry
register = registry.register_tool
get_tool = registry.get_tool
get_all_tools = registry.get_all_tools
list_tools = registry.list_tools

# New Agent API
register_agent = registry.register_agent
get_agent = registry.get_agent
get_all_agents = registry.get_all_agents
list_agents = registry.list_agents

