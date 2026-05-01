"""
OpenClaw Agent API Router.

Provides REST endpoints for running agents and managing user memory.
"""

import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .models import AgentResponse as InternalAgentResponse
import time
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    """Request body for running an agent."""

    prompt: str = Field(..., description="Instruction for the agent to execute")
    user_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="User ID for maintaining memory and context",
    )
    enable_swarm: bool = Field(
        False,
        description="Use the agent swarm instead of a single ReAct agent",
    )
    tools: Optional[List[str]] = Field(
        None,
        description="Optional list of tools to register/enable",
    )


class APIAgentResponse(BaseModel):
    """Response body returned after agent execution."""

    user_id: str
    response: str
    action_type: str
    latency_ms: float
    handoff_target: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryClearResponse(BaseModel):
    """Response body returned after clearing agent memory."""

    success: bool
    message: str


# ---------------------------------------------------------------------------
# Factory: creates the router with an injected AgentClient
# ---------------------------------------------------------------------------

def create_agent_router(agent_client: "AgentClient") -> APIRouter:  # noqa: F821
    """
    Build and return an APIRouter wired to the given *agent_client*.

    Using a factory avoids module-level global mutable state and makes
    the router testable by allowing dependency injection.

    Args:
        agent_client: An ``AgentClient`` instance (with or without swarm).

    Returns:
        A configured ``APIRouter``.
    """

    router = APIRouter(
        prefix="/v1/agent",
        tags=["Agent API"],
        responses={404: {"description": "Not found"}},
    )

    @router.post("/run", response_model=APIAgentResponse)
    async def run_agent(req: AgentRequest) -> APIAgentResponse:
        """Execute an autonomous OpenClaw agent."""
        start_time = time.monotonic()

        try:
            # Switch swarm mode if requested and different from current
            if req.enable_swarm != agent_client.use_swarm:
                agent_client.use_swarm = req.enable_swarm
                if req.enable_swarm and not getattr(agent_client, "swarm", None):
                    # Lazy import: decouple from specific orchestrator module
                    try:
                        from .multi_agentes.swarm_orchestrator import SwarmOrchestrator
                        agent_client.swarm = SwarmOrchestrator(
                            llm_engine=agent_client.llm_engine,
                        )
                        agent_client._init_default_swarm()
                    except ImportError:
                        logger.error("SwarmOrchestrator not available; falling back to single-agent mode")
                        agent_client.use_swarm = False

            # Register tools on-the-fly (single-agent mode only)
            if req.tools and not agent_client.use_swarm:
                for tool_name in req.tools:
                    agent_client.add_tool(tool_name)

            # Re-running the agent through the client
            # We use return_response=True to get the full AgentResponse object
            response_obj = await agent_client.run(
                user_id=req.user_id,
                prompt=req.prompt,
                return_response=True
            )

            latency_ms = (time.monotonic() - start_time) * 1000
            
            # Map InternalAgentResponse to APIAgentResponse
            return APIAgentResponse(
                user_id=req.user_id,
                response=response_obj.content,
                action_type=response_obj.action_type,
                latency_ms=round(latency_ms, 2),
                handoff_target=response_obj.handoff_target,
                metadata=response_obj.metadata
            )

        except Exception as e:
            logger.exception("Agent execution failed for user %s", req.user_id)
            raise HTTPException(
                status_code=500,
                detail=f"Internal agent error: {e}",
            ) from e

    from fastapi.responses import StreamingResponse
    
    @router.post("/stream")
    async def stream_agent(req: AgentRequest) -> StreamingResponse:
        """Execute an autonomous OpenClaw agent and stream the reasoning process via SSE."""
        if req.enable_swarm:
            raise HTTPException(status_code=400, detail="Streaming not supported in Swarm mode.")
            
        async def event_generator():
            try:
                async for chunk in agent_client.astream_run(req.user_id, req.prompt):
                    yield chunk
            except Exception as e:
                logger.exception("Agent streaming failed at router level")
                import json
                yield json.dumps({"event": "error", "message": f"Server Error: {str(e)}"}) + "\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        })

    class ToolApprovalRequest(BaseModel):
        user_id: str
        tool_name: str
        cmd: str
        approved: bool

    @router.post("/approve_tool")
    async def approve_tool(req: ToolApprovalRequest):
        """Resume a paused agent execution by approving or rejecting a required tool."""
        if agent_client.use_swarm or not agent_client.agent:
            raise HTTPException(status_code=400, detail="HITL tool approval is currently only supported in single-agent mode.")
            
        agent = agent_client.agent
        if req.tool_name not in agent.tools:
            raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not registered.")
            
        tool = agent.tools[req.tool_name]
        
        try:
            if req.approved:
                logger.info(f"HITL: Usuario {req.user_id} aprobó la ejecución de {req.tool_name}")
                result = await tool.run(req.cmd)
                await agent.memory.add_message(req.user_id, "user", f"[SYSTEM INTERNAL]: The user APPROVED the execution. Result: {result}")
                return {"success": True, "status": "approved", "result": result}
            else:
                logger.info(f"HITL: Usuario {req.user_id} denegó la ejecución de {req.tool_name}")
                await agent.memory.add_message(req.user_id, "user", "[SYSTEM INTERNAL]: The user REJECTED the execution of the tool. You must ask for instructions or find another way.")
                return {"success": True, "status": "rejected"}
        except Exception as e:
            logger.exception("Failed to execute approved tool")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/memory/{user_id}", response_model=MemoryClearResponse)
    async def clear_agent_memory(user_id: str) -> MemoryClearResponse:
        """Clear the agent's memory context for a given user."""
        try:
            success = await agent_client.clear_memory(user_id)
            return MemoryClearResponse(
                success=success,
                message=f"Memory for {user_id} cleared.",
            )
        except Exception as e:
            logger.exception("Failed to clear memory for user %s", user_id)
            raise HTTPException(
                status_code=500,
                detail=f"Error clearing memory: {e}",
            ) from e

    return router

