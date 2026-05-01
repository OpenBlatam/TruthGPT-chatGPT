"""
OpenClaw Scheduler -- FastAPI endpoints — Pydantic-First Architecture.

Exposes typed endpoints for managing scheduled agent tasks at runtime.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class TaskCreateRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier.")
    user_id: str = Field("default", description="User context for the agent.")
    prompt: str = Field(..., description="Prompt the agent will execute.")
    interval_seconds: float = Field(60, description="Interval between runs (seconds).", gt=0)
    repeat: bool = Field(True, description="Whether the task repeats.")
    max_runs: int = Field(0, description="Max number of executions (0 = unlimited).", ge=0)


class TaskCreateResponse(BaseModel):
    ok: bool
    task_id: str
    mode: str = Field(description="'recurring' or 'delayed'")


class TaskCancelResponse(BaseModel):
    ok: bool
    task_id: str


class SchedulerStatusResponse(BaseModel):
    ok: bool
    message: str


class TaskStatus(BaseModel):
    task_id: str
    user_id: str
    prompt: str
    interval: float
    repeat: bool
    runs: int
    cancelled: bool
    is_active: bool = True


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------

def create_scheduler_router(scheduler: "AgentScheduler") -> APIRouter:  # noqa: F821
    """Build an APIRouter with typed responses for managing scheduled tasks."""

    router = APIRouter(
        prefix="/v1/scheduler",
        tags=["Scheduler"],
    )

    @router.post("/tasks", response_model=TaskCreateResponse)
    async def create_task(req: TaskCreateRequest) -> TaskCreateResponse:
        """Register a new scheduled task."""
        try:
            if req.repeat:
                scheduler.add_recurring(
                    task_id=req.task_id,
                    user_id=req.user_id,
                    prompt=req.prompt,
                    interval_seconds=req.interval_seconds,
                    max_runs=req.max_runs,
                )
                mode = "recurring"
            else:
                scheduler.add_delayed(
                    task_id=req.task_id,
                    user_id=req.user_id,
                    prompt=req.prompt,
                    delay_seconds=req.interval_seconds,
                )
                mode = "delayed"
            return TaskCreateResponse(ok=True, task_id=req.task_id, mode=mode)
        except Exception as e:
            logger.error("Failed to create task %s: %s", req.task_id, e)
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

    @router.get("/tasks", response_model=List[TaskStatus])
    async def list_tasks() -> List[TaskStatus]:
        """List all registered tasks."""
        try:
            summaries = scheduler.list_tasks()
            return [
                TaskStatus(
                    task_id=s.task_id,
                    user_id=s.user_id,
                    prompt=s.prompt,
                    interval=s.interval,
                    repeat=s.repeat,
                    runs=s.runs,
                    cancelled=s.cancelled,
                    is_active=s.is_active,
                )
                for s in summaries
            ]
        except Exception as e:
            logger.error("Failed to list tasks: %s", e)
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Failed to list tasks.")

    @router.delete("/tasks/{task_id}", response_model=TaskCancelResponse)
    async def cancel_task(task_id: str) -> TaskCancelResponse:
        """Cancel a scheduled task."""
        try:
            success = scheduler.cancel(task_id)
            return TaskCancelResponse(ok=success, task_id=task_id)
        except Exception as e:
            logger.error("Failed to cancel task %s: %s", task_id, e)
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Failed to cancel task {task_id}.")

    @router.post("/start", response_model=SchedulerStatusResponse)
    async def start_scheduler() -> SchedulerStatusResponse:
        """Start all registered tasks."""
        await scheduler.start()
        return SchedulerStatusResponse(ok=True, message="Scheduler started.")

    @router.post("/stop", response_model=SchedulerStatusResponse)
    async def stop_scheduler() -> SchedulerStatusResponse:
        """Stop all running tasks."""
        await scheduler.stop()
        return SchedulerStatusResponse(ok=True, message="Scheduler stopped.")

    return router


