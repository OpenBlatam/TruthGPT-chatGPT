"""
Unified Exception Hierarchy for TruthGPT / OpenClaw — Pydantic-First.

Provides a complete, structured exception tree that covers all failure
domains: inference, tools, routing, handoffs, memory, and timeouts.
"""

from typing import Any, Dict, Optional


class TruthGPTError(Exception):
    """Base exception for all TruthGPT / OpenClaw errors."""

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class InferenceError(TruthGPTError):
    """Raised when the LLM engine fails or returns invalid output."""
    pass


class ToolExecutionError(TruthGPTError):
    """Raised when a registered tool fails during execution."""
    pass


class RegistryError(TruthGPTError):
    """Raised when there is an issue with Tool or Engine registration."""
    pass


class ConfigurationError(TruthGPTError):
    """Raised when the AgentConfig is invalid or missing required parameters."""
    pass


class AgentMemoryError(TruthGPTError):
    """Raised when episodic or vector memory operations fail."""
    pass


class HandoffError(TruthGPTError):
    """Raised when an agent-to-agent handoff fails (target not found, depth exceeded)."""
    pass


class RoutingError(TruthGPTError):
    """Raised when the Swarm Router cannot determine a valid target agent."""
    pass


class AgentTimeoutError(TruthGPTError):
    """Raised when an agent operation exceeds the configured time limit."""
    pass

