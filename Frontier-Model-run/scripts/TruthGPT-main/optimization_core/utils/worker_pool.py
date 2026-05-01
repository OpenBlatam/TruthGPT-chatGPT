"""
Worker pool utilities for optimization_core.

Provides utilities for managing worker pools.
"""
import logging
import threading
import time
from typing import Callable, Any, Optional, List, Dict
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future
from pydantic import Field

logger = logging.getLogger(__name__)


from .base import BaseOptimizationModel


class WorkerTask(BaseOptimizationModel):
    """Worker task definition."""
    
    func: Callable
    args: tuple = Field(default_factory=tuple)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    task_id: Optional[str] = None


class WorkerPool:
    """Pool of worker threads."""
    
    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 100
    ):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Number of worker threads
            max_queue_size: Maximum queue size
        """
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.task_queue: Queue = Queue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, Future] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def submit(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> Future:
        """
        Submit a task to the pool.
        
        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Optional task identifier
            **kwargs: Function keyword arguments
        
        Returns:
            Future object
        """
        if task_id is None:
            task_id = f"task_{time.time()}_{id(func)}"
        
        future = self.executor.submit(func, *args, **kwargs)
        
        with self.lock:
            self.active_tasks[task_id] = future
        
        # Add callback to track completion
        def done_callback(fut: Future):
            with self.lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                self.completed_tasks.append({
                    "task_id": task_id,
                    "completed_at": time.time(),
                    "success": not fut.exception(),
                })
        
        future.add_done_callback(done_callback)
        
        return future
    
    def submit_batch(
        self,
        tasks: List[WorkerTask]
    ) -> List[Future]:
        """
        Submit multiple tasks.
        
        Args:
            tasks: List of tasks
        
        Returns:
            List of futures
        """
        futures = []
        for task in tasks:
            future = self.submit(
                task.func,
                *task.args,
                task_id=task.task_id,
                **task.kwargs
            )
            futures.append(future)
        return futures
    
    def wait_all(self, timeout: Optional[float] = None):
        """
        Wait for all active tasks to complete.
        
        Args:
            timeout: Optional timeout in seconds
        """
        with self.lock:
            futures = list(self.active_tasks.values())
        
        for future in futures:
            future.result(timeout=timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            return {
                "num_workers": self.num_workers,
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "queue_size": self.task_queue.qsize(),
            }
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the worker pool.
        
        Args:
            wait: Whether to wait for tasks to complete
        """
        self.executor.shutdown(wait=wait)
        logger.info("Worker pool shut down")


def create_worker_pool(
    num_workers: int = 4,
    max_queue_size: int = 100
) -> WorkerPool:
    """
    Create a worker pool.
    
    Args:
        num_workers: Number of workers
        max_queue_size: Maximum queue size
    
    Returns:
        Worker pool
    """
    return WorkerPool(num_workers, max_queue_size)













