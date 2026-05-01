"""
OpenClaw Base Agent Interface — Pydantic-First Architecture.

Defines the foundational abstract class for ALL agents in the OpenClaw
ecosystem. Every specialised agent (RL, Marketing, CodeInterpreter, etc.)
MUST inherit from this class.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models import AgentResponse


# ---------------------------------------------------------------------------
# Pydantic Value Objects
# ---------------------------------------------------------------------------

class MemoryEntry(BaseModel):
    """A single typed entry in the agent's episodic memory."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message content")
    timestamp: float = Field(default_factory=time.time)


class AgentStatus(BaseModel):
    """Structured snapshot of an agent's current state."""
    name: str
    role: str
    memory_size: int = 0
    is_active: bool = True


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    OpenClaw Base Agent Interface.

    Provides:
    - Typed episodic memory via ``MemoryEntry``
    - Structured status reporting via ``AgentStatus``
    - Abstract ``process()`` contract for all subclasses
    """

    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role
        self.memory: List[MemoryEntry] = []

    @abstractmethod
    async def process(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process a query and return a structured AgentResponse."""
        pass

    def add_to_memory(self, role: str, content: str) -> None:
        """Append a typed MemoryEntry to the episodic buffer."""
        self.memory.append(MemoryEntry(role=role, content=content))

    def get_memory(self) -> List[Dict[str, Any]]:
        """Return memory as a list of dicts (backward-compatible)."""
        return [entry.model_dump() for entry in self.memory]

    def get_memory_entries(self) -> List[MemoryEntry]:
        """Return raw typed MemoryEntry list."""
        return list(self.memory)

    def clear_memory(self) -> None:
        self.memory.clear()

    def get_status(self) -> AgentStatus:
        """Return a Pydantic-validated status snapshot."""
        return AgentStatus(
            name=self.name,
            role=self.role,
            memory_size=len(self.memory),
        )

