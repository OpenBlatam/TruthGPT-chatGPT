"""
Agent Communication Models.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

class AgentAction(BaseModel):
    """Universal model for an LLM reasoning step or action."""
    thought: Optional[str] = Field(None, description="Internal reasoning or thought process.")
    tool: Optional[str] = Field(None, description="Name of the tool to call. Null if providing a final answer.")
    tool_input: Optional[Any] = Field(None, description="Arguments for the tool call (usually a string or JSON).")
    final_answer: Optional[str] = Field(None, description="Final message to the user.")
    handoff: Optional[str] = Field(None, description="Target agent name for a handoff transfer.")

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        """Override to ensure LLM-friendly descriptions."""
        schema = super().model_json_schema(*args, **kwargs)
        return schema

class AgentResponse(BaseModel):
    """Response from the agent orchestrator to the client."""
    content: str
    action_type: str  # 'final_answer', 'handoff', 'approval_required'
    metadata: Dict[str, Any] = Field(default_factory=dict)
    handoff_target: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

class InferenceResult(BaseModel):
    """Unified model for LLM inference outputs."""
    text: str = Field(..., description="The generated text content.")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens produced.")
    latency_ms: Optional[float] = Field(None, description="Time taken for inference in milliseconds.")
    model_name: Optional[str] = Field(None, description="Name of the model that generated this.")
    finish_reason: Optional[str] = Field(None, description="Why the generation stopped.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentConfig(BaseModel):
    """Configuration for the AgentClient."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_engine: Optional[Any] = None
    memory_db_path: str = "openclaw_memory.db"
    use_swarm: bool = False
    use_vector_memory: bool = False
    use_reflexion: bool = False
    max_handoff_depth: int = 5
    default_agent_name: Optional[str] = None
    enable_telemetry: bool = True

