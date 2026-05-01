"""
Common async utilities for optimization_core.

Provides reusable async/concurrency functions for parallel processing.
"""

import asyncio
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    TypeVar,
    Union,
)
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC GATHERING WITH LIMITS
# ════════════════════════════════════════════════════════════════════════════════

async def gather_with_limit(
    tasks: List[Awaitable[T]],
    limit: int = 10,
    return_exceptions: bool = False
) -> List[T]:
    """
    Execute multiple async tasks with concurrency limit.
    
    Args:
        tasks: List of async tasks
        limit: Maximum concurrent tasks
        return_exceptions: If True, exceptions are returned instead of raised
    
    Returns:
        List of results
    
    Example:
        >>> tasks = [async_func(i) for i in range(100)]
        >>> results = await gather_with_limit(tasks, limit=10)
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_task(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *[bounded_task(task) for task in tasks],
        return_exceptions=return_exceptions
    )


async def async_map(
    func: Callable[[T], Union[R, Awaitable[R]]],
    items: List[T],
    max_concurrent: int = 10
) -> List[R]:
    """
    Apply function to items with controlled concurrency.
    
    Args:
        func: Function to apply (can be sync or async)
        items: List of items to process
        max_concurrent: Maximum concurrent executions
    
    Returns:
        List of results
    
    Example:
        >>> results = await async_map(process_item, items, max_concurrent=5)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item: T) -> R:
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=False)


async def async_filter(
    func: Callable[[T], Union[bool, Awaitable[bool]]],
    items: List[T],
    max_concurrent: int = 10
) -> List[T]:
    """
    Filter items with controlled concurrency.
    
    Args:
        func: Filter function (can be sync or async)
        items: List of items to filter
        max_concurrent: Maximum concurrent executions
    
    Returns:
        List of items where func returns True
    
    Example:
        >>> filtered = await async_filter(is_valid, items, max_concurrent=5)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def filter_item(item: T) -> Optional[T]:
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                should_include = await func(item)
            else:
                loop = asyncio.get_event_loop()
                should_include = await loop.run_in_executor(None, func, item)
            
            return item if should_include else None
    
    tasks = [filter_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    return [
        result for result in results
        if result is not None and not isinstance(result, Exception)
    ]


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC TIMEOUT
# ════════════════════════════════════════════════════════════════════════════════

async def async_timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        default: Default value on timeout
    
    Returns:
        Result or default
    
    Example:
        >>> result = await async_timeout(slow_operation(), timeout=5.0, default=None)
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC BATCH PROCESSING
# ════════════════════════════════════════════════════════════════════════════════

async def async_batch_process(
    items: List[T],
    processor: Callable[[List[T]], Union[List[R], Awaitable[List[R]]]],
    batch_size: int = 100,
    max_concurrent_batches: int = 5
) -> List[R]:
    """
    Process items in batches with controlled concurrency.
    
    Args:
        items: List of items to process
        processor: Batch processor function (can be sync or async)
        batch_size: Size of each batch
        max_concurrent_batches: Maximum concurrent batches
    
    Returns:
        List of all results
    
    Example:
        >>> results = await async_batch_process(
        ...     items,
        ...     process_batch,
        ...     batch_size=50,
        ...     max_concurrent_batches=3
        ... )
    """
    from .collection_utils import chunk_list
    
    batches = chunk_list(items, batch_size)
    
    async def process_batch(batch: List[T]) -> List[R]:
        if asyncio.iscoroutinefunction(processor):
            return await processor(batch)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, processor, batch)
    
    batch_results = await gather_with_limit(
        [process_batch(batch) for batch in batches],
        limit=max_concurrent_batches
    )
    
    # Flatten results
    from .collection_utils import flatten_list
    return flatten_list(batch_results)


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC CONTEXT MANAGERS
# ════════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def async_retry_context(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """
    Context manager for async retry logic.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
    
    Example:
        >>> async with async_retry_context(max_attempts=3, delay=1.0):
        ...     result = await risky_operation()
    """
    attempt = 0
    current_delay = delay
    
    while attempt < max_attempts:
        try:
            yield attempt
            return
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            
            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed: {e}. "
                f"Retrying in {current_delay}s..."
            )
            await asyncio.sleep(current_delay)
            current_delay *= backoff


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC TO SYNC HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def run_async(coro: Awaitable[T]) -> T:
    """
    Run async coroutine in sync context.
    
    Args:
        coro: Coroutine to run
    
    Returns:
        Result of coroutine
    
    Example:
        >>> result = run_async(async_function())
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Gathering
    "gather_with_limit",
    "async_map",
    "async_filter",
    # Timeout
    "async_timeout",
    # Batch processing
    "async_batch_process",
    # Context managers
    "async_retry_context",
    # Sync helpers
    "run_async",
]













