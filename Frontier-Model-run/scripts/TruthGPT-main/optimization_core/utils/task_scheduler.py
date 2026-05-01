"""
Task scheduling utilities for optimization_core.

Provides utilities for scheduling and managing tasks.
"""
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from queue import Queue, Empty
from pydantic import Field

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


from .base import BaseOptimizationModel


class Task(BaseOptimizationModel):
    """Task data structure."""
    
    id: str
    func: Callable
    args: tuple = Field(default_factory=tuple)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration."""
        if self.completed_at is not None and self.started_at is not None:
            return self.completed_at - self.started_at
        return None


class TaskScheduler:
    """Scheduler for managing tasks."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize task scheduler.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.task_queue: Queue = Queue()
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Start scheduler."""
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Task scheduler started with {self.max_workers} workers")
    
    def stop(self):
        """Stop scheduler."""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.workers.clear()
        logger.info("Task scheduler stopped")
    
    def submit(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Task:
        """
        Submit a task.
        
        Args:
            task_id: Unique task identifier
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Task object
        """
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        with self.lock:
            self.tasks[task_id] = task
        
        self.task_queue.put(task)
        logger.debug(f"Task {task_id} submitted")
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Task object or None
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            True if cancelled, False otherwise
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Task {task_id} cancelled")
                return True
        return False
    
    def _worker(self):
        """Worker thread function."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                try:
                    result = task.func(*task.args, **task.kwargs)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    logger.debug(f"Task {task.id} completed")
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    logger.error(f"Task {task.id} failed: {e}", exc_info=True)
                finally:
                    task.completed_at = time.time()
                
                self.task_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            tasks = list(self.tasks.values())
        
        return {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
            "running": len([t for t in tasks if t.status == TaskStatus.RUNNING]),
            "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "cancelled": len([t for t in tasks if t.status == TaskStatus.CANCELLED]),
        }













