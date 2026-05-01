"""
Advanced Batch Processing for Inference
========================================

High-performance batch processing with:
- Dynamic batching
- Priority queues
- Batch optimization
- Memory management
- Parallel processing
"""

import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, Generic, TypeVar
from enum import Enum
from collections import deque
import heapq

from ..exceptions import BatchProcessingError

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchPriority(Enum):
    """Batch priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BatchItem(Generic[T]):
    """Item in a batch."""
    data: T
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by priority (higher priority first)."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


@dataclass
class Batch(Generic[T]):
    """Batch of items."""
    items: List[BatchItem[T]]
    created_at: float = field(default_factory=time.time)
    max_size: int = 32
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.items)
    
    @property
    def is_full(self) -> bool:
        """Check if batch is full."""
        return self.size >= self.max_size
    
    @property
    def age(self) -> float:
        """Get batch age in seconds."""
        return time.time() - self.created_at


class DynamicBatcher(Generic[T, R]):
    """
    Dynamic batcher with priority queue and optimization.
    
    Features:
    - Priority-based batching
    - Dynamic batch sizing
    - Time-based flushing
    - Memory-efficient
    - Thread-safe
    """
    
    def __init__(
        self,
        processor: Callable[[List[T]], List[R]],
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        max_wait_time: float = 0.1,
        max_queue_size: int = 1000,
        optimize_batches: bool = True
    ):
        """
        Initialize dynamic batcher.
        
        Args:
            processor: Function to process batches
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size before processing
            max_wait_time: Maximum time to wait before processing
            max_queue_size: Maximum queue size
            optimize_batches: Whether to optimize batch order
        """
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        self.optimize_batches = optimize_batches
        
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "total_wait_time": 0.0,
            "total_process_time": 0.0,
        }
    
    def start(self):
        """Start the batcher thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("Dynamic batcher started")
    
    def stop(self, timeout: Optional[float] = None):
        """Stop the batcher thread."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Dynamic batcher stopped")
    
    def submit(
        self,
        item: T,
        priority: BatchPriority = BatchPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Submit an item for batching.
        
        Args:
            item: Item to batch
            priority: Item priority
            metadata: Optional metadata
        """
        batch_item = BatchItem(
            data=item,
            priority=priority,
            metadata=metadata or {}
        )
        
        try:
            self._queue.put_nowait(batch_item)
        except queue.Full:
            raise BatchProcessingError("Batch queue is full")
    
    def _process_loop(self):
        """Main processing loop."""
        current_batch: List[BatchItem[T]] = []
        batch_start_time = time.time()
        
        while self._running:
            try:
                # Try to get item with timeout
                try:
                    item = self._queue.get(timeout=self.max_wait_time)
                    current_batch.append(item)
                except queue.Empty:
                    # Timeout - process batch if we have items
                    if current_batch:
                        self._process_batch(current_batch)
                        current_batch = []
                        batch_start_time = time.time()
                    continue
                
                # Check if batch should be processed
                should_process = (
                    len(current_batch) >= self.max_batch_size or
                    (len(current_batch) >= self.min_batch_size and
                     time.time() - batch_start_time >= self.max_wait_time)
                )
                
                if should_process:
                    self._process_batch(current_batch)
                    current_batch = []
                    batch_start_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}", exc_info=True)
    
    def _process_batch(self, items: List[BatchItem[T]]):
        """Process a batch of items."""
        if not items:
            return
        
        start_time = time.time()
        
        # Extract data from batch items
        batch_data = [item.data for item in items]
        
        # Optimize batch order if enabled
        if self.optimize_batches:
            batch_data = self._optimize_batch(batch_data)
        
        try:
            # Process batch
            results = self.processor(batch_data)
            
            # Update stats
            process_time = time.time() - start_time
            with self._lock:
                self._stats["batches_processed"] += 1
                self._stats["items_processed"] += len(items)
                self._stats["total_process_time"] += process_time
                self._stats["total_wait_time"] += sum(
                    time.time() - item.created_at for item in items
                )
            
            logger.debug(
                f"Processed batch of {len(items)} items in {process_time:.3f}s"
            )
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            raise BatchProcessingError(f"Batch processing failed: {e}") from e
    
    def _optimize_batch(self, batch: List[T]) -> List[T]:
        """Optimize batch order for better performance."""
        # Default: no optimization
        # Subclasses can override for custom optimization
        return batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        with self._lock:
            avg_batch_size = (
                self._stats["items_processed"] / self._stats["batches_processed"]
                if self._stats["batches_processed"] > 0 else 0
            )
            avg_process_time = (
                self._stats["total_process_time"] / self._stats["batches_processed"]
                if self._stats["batches_processed"] > 0 else 0
            )
            avg_wait_time = (
                self._stats["total_wait_time"] / self._stats["items_processed"]
                if self._stats["items_processed"] > 0 else 0
            )
            
            return {
                **self._stats,
                "queue_size": self._queue.qsize(),
                "avg_batch_size": avg_batch_size,
                "avg_process_time": avg_process_time,
                "avg_wait_time": avg_wait_time,
            }


class ContinuousBatcher(DynamicBatcher[T, R]):
    """
    Continuous batcher that processes items as they arrive.
    
    Similar to dynamic batcher but optimized for continuous streams.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_results: Dict[int, queue.Queue] = {}
        self._result_counter = 0
        self._result_lock = threading.Lock()
    
    def submit_async(
        self,
        item: T,
        priority: BatchPriority = BatchPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Submit item and return result ID for async retrieval.
        
        Returns:
            Result ID for retrieving result later
        """
        result_id = self._get_next_result_id()
        result_queue = queue.Queue()
        
        with self._result_lock:
            self._pending_results[result_id] = result_queue
        
        # Add result_id to metadata
        if metadata is None:
            metadata = {}
        metadata["result_id"] = result_id
        
        self.submit(item, priority, metadata)
        
        return result_id
    
    def get_result(self, result_id: int, timeout: Optional[float] = None) -> R:
        """Get result for a submitted item."""
        with self._result_lock:
            if result_id not in self._pending_results:
                raise ValueError(f"Result ID {result_id} not found")
            result_queue = self._pending_results[result_id]
        
        try:
            result = result_queue.get(timeout=timeout)
            with self._result_lock:
                del self._pending_results[result_id]
            return result
        except queue.Empty:
            raise TimeoutError(f"Result for {result_id} not available within timeout")
    
    def _get_next_result_id(self) -> int:
        """Get next result ID."""
        with self._result_lock:
            self._result_counter += 1
            return self._result_counter


