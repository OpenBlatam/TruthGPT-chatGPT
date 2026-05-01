"""
Unit tests for agents.scheduler — ScheduledTask & AgentScheduler.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


class TestScheduledTask:
    """Test the ScheduledTask Pydantic model."""

    def test_default_is_active(self):
        from optimization_core.agents.scheduler import ScheduledTask
        task = ScheduledTask(
            task_id="t1",
            user_id="u1",
            prompt="Do something",
            interval_seconds=60,
        )
        assert task.is_active is True
        assert task.run_count == 0
        assert task.cancelled is False

    def test_cancel_deactivates(self):
        from optimization_core.agents.scheduler import ScheduledTask
        task = ScheduledTask(
            task_id="t1", user_id="u1", prompt="p", interval_seconds=10,
        )
        task.cancel()
        assert task.is_active is False
        assert task.cancelled is True

    def test_max_runs_deactivates(self):
        from optimization_core.agents.scheduler import ScheduledTask
        task = ScheduledTask(
            task_id="t1", user_id="u1", prompt="p",
            interval_seconds=10, max_runs=3, run_count=3,
        )
        assert task.is_active is False

    def test_model_dump_roundtrip(self):
        from optimization_core.agents.scheduler import ScheduledTask
        task = ScheduledTask(
            task_id="t1", user_id="u1", prompt="hello",
            interval_seconds=30, repeat=True, max_runs=5,
        )
        data = task.model_dump(exclude={"is_active"})
        restored = ScheduledTask.model_validate(data)
        assert restored.task_id == "t1"
        assert restored.interval_seconds == 30


class TestTaskSummary:
    """Test the TaskSummary response model."""

    def test_creation(self):
        from optimization_core.agents.scheduler import TaskSummary
        summary = TaskSummary(
            task_id="t1", user_id="u1", prompt="hello",
            interval=60, repeat=True, runs=5,
            cancelled=False, is_active=True,
        )
        assert summary.task_id == "t1"
        assert summary.runs == 5


class TestAgentScheduler:
    """Test the AgentScheduler (asyncio fallback mode)."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.run = AsyncMock(return_value="done")
        return client

    @pytest.fixture
    def scheduler(self, mock_client, tmp_path):
        from optimization_core.agents.scheduler import AgentScheduler
        return AgentScheduler(mock_client, persistence_path=str(tmp_path / "sched.json"))

    def test_add_recurring(self, scheduler):
        task = scheduler.add_recurring("rec1", "u1", "Do it", interval_seconds=10)
        assert task.task_id == "rec1"
        assert task.repeat is True
        assert task.interval_seconds == 10

    def test_add_delayed(self, scheduler):
        task = scheduler.add_delayed("del1", "u1", "Do it once", delay_seconds=5)
        assert task.task_id == "del1"
        assert task.repeat is False
        assert task.max_runs == 1

    def test_cancel(self, scheduler):
        scheduler.add_recurring("cancel1", "u1", "prompt", interval_seconds=10)
        assert scheduler.cancel("cancel1") is True
        tasks = scheduler.list_tasks()
        cancelled = [t for t in tasks if t.task_id == "cancel1"]
        assert len(cancelled) == 1
        assert cancelled[0].cancelled is True

    def test_cancel_nonexistent(self, scheduler):
        assert scheduler.cancel("no_such_task") is False

    def test_list_tasks(self, scheduler):
        scheduler.add_recurring("a", "u1", "p1", interval_seconds=5)
        scheduler.add_delayed("b", "u1", "p2", delay_seconds=3)
        tasks = scheduler.list_tasks()
        assert len(tasks) == 2
        ids = {t.task_id for t in tasks}
        assert ids == {"a", "b"}

    def test_persistence(self, mock_client, tmp_path):
        from optimization_core.agents.scheduler import AgentScheduler
        path = str(tmp_path / "persist_sched.json")

        s1 = AgentScheduler(mock_client, persistence_path=path)
        s1.add_recurring("persist_task", "u1", "hello", interval_seconds=60)

        s2 = AgentScheduler(mock_client, persistence_path=path)
        tasks = s2.list_tasks()
        assert any(t.task_id == "persist_task" for t in tasks)

    def test_on_result_callback(self, scheduler):
        results = []
        scheduler.on_result(lambda tid, res: results.append((tid, res)))
        assert scheduler._on_result is not None
