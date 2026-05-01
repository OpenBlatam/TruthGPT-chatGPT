"""
OpenClaw Observability -- FastAPI endpoints.

Exposes read-only endpoints for inspecting traces, spans, and agent stats.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from . import global_tracer

logger = logging.getLogger(__name__)


class TraceStats(BaseModel):
    total_traces: int
    total_spans: int
    error_spans: int
    error_rate: float


def create_observability_router() -> APIRouter:
    """Create an APIRouter for the observability endpoints."""

    router = APIRouter(
        prefix="/v1/traces",
        tags=["Observability"],
    )

    @router.get("/stats", response_model=TraceStats)
    async def get_stats() -> TraceStats:
        """Return aggregate stats across all stored traces."""
        return TraceStats(**global_tracer.get_stats())

    @router.get("/recent")
    async def recent_traces(limit: int = Query(20, ge=1, le=100)):
        """Return a summary of the most recent traces."""
        return global_tracer.get_recent_traces(limit=limit)

    @router.get("/{trace_id}")
    async def get_trace(trace_id: str):
        """Return all spans for a specific trace."""
        spans = global_tracer.get_trace(trace_id)
        if not spans:
            return {"detail": "Trace not found."}
        return spans

    return router

