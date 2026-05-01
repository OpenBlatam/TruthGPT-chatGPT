"""
Unit tests for agents.observability — Span & Tracer.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path


class TestSpan:
    """Test the Span Pydantic model."""

    def test_span_default_fields(self):
        from optimization_core.agents.observability import Span

        span = Span(name="test_span", agent_name="TestAgent")
        assert span.name == "test_span"
        assert span.agent_name == "TestAgent"
        assert span.kind == "internal"
        assert span.status == "ok"
        assert span.end_time == 0.0
        assert span.duration_ms == 0.0
        assert len(span.span_id) == 8

    def test_span_finish(self):
        from optimization_core.agents.observability import Span

        span = Span(name="test")
        time.sleep(0.01)
        span.finish(output="result", status="ok", metadata={"key": "val"})

        assert span.end_time > span.start_time
        assert span.output_data == "result"
        assert span.status == "ok"
        assert span.duration_ms > 0
        assert span.metadata["key"] == "val"

    def test_span_finish_error(self):
        from optimization_core.agents.observability import Span

        span = Span(name="test")
        span.finish(output="fail", status="error")
        assert span.status == "error"

    def test_span_to_dict(self):
        from optimization_core.agents.observability import Span

        span = Span(name="test", trace_id="abc123", agent_name="Agent")
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["trace_id"] == "abc123"
        assert d["agent"] == "Agent"
        assert "duration_ms" in d

    def test_span_model_dump(self):
        from optimization_core.agents.observability import Span

        span = Span(name="test")
        dumped = span.model_dump()
        assert "name" in dumped
        assert "span_id" in dumped

    def test_span_model_validate_roundtrip(self):
        from optimization_core.agents.observability import Span

        span = Span(name="roundtrip", trace_id="x")
        span.finish(output="done")
        data = span.model_dump()
        restored = Span.model_validate(data)
        assert restored.name == "roundtrip"
        assert restored.trace_id == "x"
        assert restored.output_data == "done"


class TestTracer:
    """Test the Tracer class."""

    def _make_tracer(self, tmp_path: Path):
        from optimization_core.agents.observability import Tracer
        return Tracer(max_traces=50, persistence_path=str(tmp_path / "test_traces.json"))

    def test_start_trace(self, tmp_path):
        tracer = self._make_tracer(tmp_path)
        trace_id = tracer.start_trace("test_trace", agent_name="TestAgent")
        assert trace_id
        assert len(trace_id) == 12

    def test_start_span(self, tmp_path):
        tracer = self._make_tracer(tmp_path)
        tid = tracer.start_trace("test")
        span = tracer.start_span(tid, "span_1", kind="tool_call", input_data="hello")
        assert span.name == "span_1"
        assert span.kind == "tool_call"
        assert span.trace_id == tid

    def test_finish_trace(self, tmp_path):
        tracer = self._make_tracer(tmp_path)
        tid = tracer.start_trace("test")
        tracer.finish_trace(tid)
        spans = tracer.get_trace(tid)
        assert len(spans) >= 1
        root = spans[0]
        assert root["duration_ms"] >= 0

    def test_get_trace(self, tmp_path):
        tracer = self._make_tracer(tmp_path)
        tid = tracer.start_trace("test")
        tracer.start_span(tid, "a")
        tracer.start_span(tid, "b")
        spans = tracer.get_trace(tid)
        assert len(spans) == 3  # root + 2 spans

    def test_get_recent_traces(self, tmp_path):
        tracer = self._make_tracer(tmp_path)
        for i in range(5):
            tracer.start_trace(f"trace_{i}")
        recent = tracer.get_recent_traces(limit=3)
        assert len(recent) == 3

    def test_get_stats(self, tmp_path):
        tracer = self._make_tracer(tmp_path)
        tid = tracer.start_trace("test")
        s = tracer.start_span(tid, "err_span")
        s.finish(status="error")
        stats = tracer.get_stats()
        assert stats["total_traces"] == 1
        assert stats["total_spans"] == 2  # root + err_span
        assert stats["error_spans"] == 1
        assert stats["error_rate"] == 0.5

    def test_eviction(self, tmp_path):
        from optimization_core.agents.observability import Tracer
        tracer = Tracer(max_traces=3, persistence_path=str(tmp_path / "evict.json"))
        for i in range(5):
            tracer.start_trace(f"trace_{i}")
        assert tracer.get_stats()["total_traces"] == 3

    def test_persistence_roundtrip(self, tmp_path):
        from optimization_core.agents.observability import Tracer

        path = str(tmp_path / "persist.json")

        # Write
        t1 = Tracer(max_traces=50, persistence_path=path)
        tid = t1.start_trace("persist_test", agent_name="A")
        t1.start_span(tid, "span_a", kind="llm_call")
        t1.finish_trace(tid)

        # Read back
        t2 = Tracer(max_traces=50, persistence_path=path)
        spans = t2.get_trace(tid)
        assert len(spans) == 2
        assert spans[0]["name"] == "persist_test"
