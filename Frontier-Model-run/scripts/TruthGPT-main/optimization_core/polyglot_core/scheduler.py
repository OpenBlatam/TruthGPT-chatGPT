"""
Task scheduling utilities for polyglot_core.

Provides task scheduling and background job execution.
"""

from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
from queue import Queue, Empty


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    schedule_time: Optional[datetime] = None
    interval: Optional[timedelta] = None
    max_retries: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskScheduler:
    """
    Task scheduler for polyglot_core.
    
    Provides background task scheduling and execution.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize task scheduler.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self._tasks: Dict[str, Task] = {}
        self._queue = Queue()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
    
    def schedule(
        self,
        task_id: str,
        func: Callable,
        *args,
        schedule_time: Optional[datetime] = None,
        interval: Optional[timedelta] = None,
        max_retries: int = 0,
        **kwargs
    ) -> str:
        """
        Schedule a task.
        
        Args:
            task_id: Unique task ID
            func: Function to execute
            *args: Positional arguments
            schedule_time: When to execute (None = immediately)
            interval: Repeat interval (None = one-time)
            max_retries: Maximum retry attempts
            **kwargs: Keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            schedule_time=schedule_time,
            interval=interval,
            max_retries=max_retries
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        if schedule_time is None or schedule_time <= datetime.now():
            self._queue.put(task)
        else:
            # Schedule for later
            delay = (schedule_time - datetime.now()).total_seconds()
            threading.Timer(delay, lambda: self._queue.put(task)).start()
        
        return task_id
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False if not found
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    return True
        return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """List tasks, optionally filtered by status."""
        with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            return tasks
    
    def start(self):
        """Start scheduler workers."""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self._workers.append(worker)
    
    def stop(self, wait: bool = True):
        """
        Stop scheduler.
        
        Args:
            wait: Whether to wait for running tasks
        """
        self._running = False
        
        if wait:
            for worker in self._workers:
                worker.join()
    
    def _worker(self):
        """Worker thread function."""
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
                
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                try:
                    result = task.func(*task.args, **task.kwargs)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    
                    if task.max_retries > 0:
                        task.max_retries -= 1
                        task.status = TaskStatus.PENDING
                        self._queue.put(task)
                    else:
                        task.status = TaskStatus.FAILED
                
                task.completed_at = datetime.now()
                
                # Reschedule if interval specified
                if task.interval and task.status == TaskStatus.COMPLETED:
                    next_time = datetime.now() + task.interval
                    task.schedule_time = next_time
                    task.status = TaskStatus.PENDING
                    delay = task.interval.total_seconds()
                    threading.Timer(delay, lambda: self._queue.put(task)).start()
                
            except Empty:
                continue
            except Exception as e:
                # Log error but continue
                print(f"Worker error: {e}")


# Global scheduler
_global_scheduler = TaskScheduler()


def get_scheduler() -> TaskScheduler:
    """Get global task scheduler."""
    return _global_scheduler


def schedule_task(
    task_id: str,
    func: Callable,
    *args,
    schedule_time: Optional[datetime] = None,
    interval: Optional[timedelta] = None,
    **kwargs
) -> str:
    """Convenience function to schedule task."""
    return _global_scheduler.schedule(task_id, func, *args, schedule_time=schedule_time, interval=interval, **kwargs)













