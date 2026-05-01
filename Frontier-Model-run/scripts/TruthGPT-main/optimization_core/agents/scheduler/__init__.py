"""
OpenClaw -- Agent Task Scheduler — Pydantic-First Architecture.

Allows agents to be triggered on a schedule (cron-like) or after a delay.
Uses ``asyncio`` tasks for lightweight in-process scheduling.

For production, integrate with Celery, APScheduler, or a cloud scheduler.
"""

import asyncio
import logging
import time
import json
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, computed_field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ScheduledTask(BaseModel):
    """Represents a single scheduled agent task (Pydantic-validated)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    user_id: str
    prompt: str
    interval_seconds: float
    repeat: bool = True
    max_runs: int = Field(default=0, description="0 = unlimited")
    last_run_time: Optional[float] = None
    run_count: int = 0
    cancelled: bool = False

    @computed_field  # type: ignore[misc]
    @property
    def is_active(self) -> bool:
        """True if the task is not cancelled and hasn't exceeded max_runs."""
        if self.cancelled:
            return False
        if self.max_runs > 0 and self.run_count >= self.max_runs:
            return False
        return True

    def cancel(self) -> None:
        self.cancelled = True


class TaskSummary(BaseModel):
    """Lightweight view of a scheduled task for API responses."""
    task_id: str
    user_id: str
    prompt: str
    interval: float
    repeat: bool
    runs: int
    cancelled: bool
    is_active: bool


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class AgentScheduler:
    """
    In-process task scheduler for OpenClaw agents.

    Usage::

        from agents.client import AgentClient
        from agents.scheduler import AgentScheduler

        client = AgentClient()
        scheduler = AgentScheduler(client)

        # Run a prompt every 60 seconds
        scheduler.add_recurring("daily_report", "user1",
                                "Generate a summary of today's metrics",
                                interval_seconds=60)

        # Run a prompt once after 10 seconds
        scheduler.add_delayed("welcome", "user2",
                              "Send a welcome message",
                              delay_seconds=10)

        await scheduler.start()
    """

    def __init__(self, agent_client: Any, persistence_path: str = "scheduler_registry.json") -> None:
        self.agent_client = agent_client
        self.persistence_path = Path(persistence_path)
        self.tasks: Dict[str, ScheduledTask] = {}
        self._on_result: Optional[Callable] = None
        self._persistence_loaded = False
        self._running_tasks: Dict[str, asyncio.Task] = {}

        if agent_client is None:
            logger.warning("AgentScheduler initialized without agent_client. Task execution will fail.")

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            self.scheduler_engine = AsyncIOScheduler()
            self.has_apscheduler = True
        except ImportError:
            self.scheduler_engine = None
            self.has_apscheduler = False
            logger.warning("APScheduler not found. Using asyncio fallback. 'pip install apscheduler'")

    def _ensure_loaded(self) -> None:
        """Lazy-load persisted tasks on first access."""
        if not self._persistence_loaded:
            self._load_tasks()
            self._persistence_loaded = True

    def on_result(self, callback: Callable) -> None:
        """Register a callback ``(task_id, result) -> None`` for task results."""
        self._on_result = callback

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def add_recurring(
        self,
        task_id: str,
        user_id: str,
        prompt: str,
        interval_seconds: float = 60,
        max_runs: int = 0,
    ) -> ScheduledTask:
        """Add a recurring task that fires every ``interval_seconds``."""
        if interval_seconds < 0.1:
            logger.warning("Interval %ss is too short. Minimum is 0.1s. Adjusting to 0.1s.", interval_seconds)
            interval_seconds = 0.1

        self._ensure_loaded()
        task = ScheduledTask(
            task_id=task_id,
            user_id=user_id,
            prompt=prompt,
            interval_seconds=interval_seconds,
            repeat=True,
            max_runs=max_runs,
        )
        self.tasks[task_id] = task
        self._save_tasks()
        logger.info("Recurring task registered: %s (every %ss)", task_id, interval_seconds)

        if self.scheduler_engine and self.scheduler_engine.state == 1:
            self._schedule_job(task)

        return task

    def add_delayed(
        self,
        task_id: str,
        user_id: str,
        prompt: str,
        delay_seconds: float = 10,
    ) -> ScheduledTask:
        """Add a one-shot task that fires after ``delay_seconds``."""
        if delay_seconds < 0.1:
            logger.warning("Delay %ss is too short. Minimum is 0.1s. Adjusting to 0.1s.", delay_seconds)
            delay_seconds = 0.1

        self._ensure_loaded()
        task = ScheduledTask(
            task_id=task_id,
            user_id=user_id,
            prompt=prompt,
            interval_seconds=delay_seconds,
            repeat=False,
            max_runs=1,
        )
        self.tasks[task_id] = task
        self._save_tasks()
        logger.info("Delayed task registered: %s (in %ss)", task_id, delay_seconds)

        if self.scheduler_engine and self.scheduler_engine.state == 1:
            self._schedule_job(task)

        return task

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        self._ensure_loaded()
        task = self.tasks.get(task_id)
        if task:
            task.cancel()
            if self.has_apscheduler and self.scheduler_engine:
                try:
                    self.scheduler_engine.remove_job(task_id)
                except Exception:
                    pass
            else:
                asyncio_task = self._running_tasks.get(task_id)
                if asyncio_task:
                    asyncio_task.cancel()

            self._save_tasks()
            logger.info("Task cancelled: %s", task_id)
            return True
        return False

    def list_tasks(self) -> List[TaskSummary]:
        """Return a typed summary of all registered tasks."""
        self._ensure_loaded()
        return [
            TaskSummary(
                task_id=t.task_id,
                user_id=t.user_id,
                prompt=t.prompt[:50],
                interval=t.interval_seconds,
                repeat=t.repeat,
                runs=t.run_count,
                cancelled=t.cancelled,
                is_active=t.is_active,
            )
            for t in self.tasks.values()
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all registered tasks using APScheduler or asyncio fallback."""
        self._ensure_loaded()
        if self.has_apscheduler and self.scheduler_engine:
            if self.scheduler_engine.state == 0:
                self.scheduler_engine.start()
            for task in self.tasks.values():
                if task.is_active:
                    self._schedule_job(task)
            logger.info("APScheduler started with %d tasks", len(self.tasks))
        else:
            for task_id, task in self.tasks.items():
                if task_id not in self._running_tasks and task.is_active:
                    initial_delay = task.interval_seconds
                    if task.last_run_time:
                        elapsed = time.time() - task.last_run_time
                        initial_delay = max(0.1, task.interval_seconds - elapsed)

                    self._running_tasks[task_id] = asyncio.create_task(
                        self._run_loop(task, initial_delay=initial_delay)
                    )
            logger.info("Asyncio Fallback Scheduler started.")

    async def stop(self) -> None:
        """Cancel all running tasks and stop the scheduler."""
        for task in self.tasks.values():
            task.cancel()

        if self.has_apscheduler and self.scheduler_engine:
            self.scheduler_engine.shutdown(wait=False)
        else:
            for asyncio_task in self._running_tasks.values():
                asyncio_task.cancel()
            self._running_tasks.clear()

        logger.info("Scheduler stopped.")

    def _schedule_job(self, task: ScheduledTask) -> None:
        """Schedule a task via APScheduler."""
        from datetime import datetime, timedelta

        next_run = None
        if task.last_run_time:
            next_run = datetime.fromtimestamp(task.last_run_time) + timedelta(seconds=task.interval_seconds)
            if next_run < datetime.now():
                next_run = datetime.now() + timedelta(seconds=0.1)

        if task.repeat:
            self.scheduler_engine.add_job(
                self._execute_task,
                "interval",
                seconds=task.interval_seconds,
                id=task.task_id,
                args=[task],
                next_run_time=next_run,
                replace_existing=True,
            )
        else:
            run_date = next_run or (datetime.now() + timedelta(seconds=task.interval_seconds))
            self.scheduler_engine.add_job(
                self._execute_task,
                "date",
                run_date=run_date,
                id=task.task_id,
                args=[task],
                replace_existing=True,
            )

    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute and record a scheduled task run."""
        if task.cancelled:
            return

        if not self.agent_client:
            logger.error("Cannot execute task '%s': agent_client is not initialized.", task.task_id)
            return

        task.run_count += 1
        task.last_run_time = time.time()
        self._save_tasks()

        logger.info("Executing scheduled task '%s' (run #%d)", task.task_id, task.run_count)

        try:
            # Type check for agent_client.run
            if hasattr(self.agent_client, "run"):
                result = await self.agent_client.run(user_id=task.user_id, prompt=task.prompt)
                if self._on_result:
                    self._on_result(task.task_id, result)
            else:
                 logger.error("agent_client missing 'run' method for task '%s'.", task.task_id)
        except Exception as e:
            logger.exception("Task '%s' failed: %s", task.task_id, str(e))

        if task.max_runs > 0 and task.run_count >= task.max_runs:
            logger.info("Task '%s' reached max runs (%d)", task.task_id, task.max_runs)
            self.cancel(task.task_id)

    async def _run_loop(self, task: ScheduledTask, initial_delay: Optional[float] = None) -> None:
        """Fallback internal loop via asyncio."""
        try:
            delay = initial_delay if initial_delay is not None else task.interval_seconds
            await asyncio.sleep(delay)

            while not task.cancelled:
                await self._execute_task(task)

                if not task.repeat or task.cancelled:
                    break

                await asyncio.sleep(task.interval_seconds)

        except asyncio.CancelledError:
            logger.info("Task '%s' was cancelled.", task.task_id)
            raise

    # ------------------------------------------------------------------
    # Persistence (Pydantic model_dump / model_validate)
    # ------------------------------------------------------------------

    def _save_tasks(self) -> None:
        """Serialize current tasks via Pydantic model_dump."""
        try:
            data = {}
            for tid, t in self.tasks.items():
                if not t.cancelled:
                    data[tid] = t.model_dump(exclude={"is_active"})

            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save scheduler registry: %s", e)

    def _load_tasks(self) -> None:
        """Load tasks via Pydantic model_validate."""
        if not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for tid, t_data in data.items():
                    task = ScheduledTask.model_validate(t_data)
                    self.tasks[tid] = task
            logger.info("Restored %d tasks from persistence.", len(self.tasks))
        except Exception as e:
            logger.error("Failed to load scheduler registry: %s", e)


