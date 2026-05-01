"""
Batch processing utilities for optimization_core.

Provides utilities for processing data in batches.
"""
import logging
from typing import List, Any, Callable, Optional, Iterator, TypeVar, Generic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchResult(BaseModel):
    """Result of batch processing."""
    batch_index: int
    items_processed: int
    results: List[Any]
    errors: List[str] = Field(default_factory=list)


class BatchProcessor(Generic[T, R]):
    """Processor for batch operations."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_workers: Optional[int] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
            max_workers: Maximum number of workers (None for sequential)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process(
        self,
        items: List[T],
        func: Callable[[List[T]], List[R]],
        show_progress: bool = False
    ) -> List[R]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            func: Processing function
            show_progress: Whether to show progress
        
        Returns:
            List of results
        """
        all_results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_index = i // self.batch_size + 1
            
            if show_progress:
                logger.info(f"Processing batch {batch_index}/{total_batches}")
            
            try:
                batch_results = func(batch)
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch {batch_index} failed: {e}", exc_info=True)
                # Continue with next batch
                continue
        
        return all_results
    
    def process_with_results(
        self,
        items: List[T],
        func: Callable[[List[T]], List[R]]
    ) -> List[BatchResult]:
        """
        Process items and return detailed results.
        
        Args:
            items: Items to process
            func: Processing function
        
        Returns:
            List of batch results
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_index = i // self.batch_size + 1
            
            errors = []
            batch_results = []
            
            try:
                batch_results = func(batch)
            except Exception as e:
                errors.append(str(e))
                logger.error(f"Batch {batch_index} failed: {e}", exc_info=True)
            
            results.append(BatchResult(
                batch_index=batch_index,
                items_processed=len(batch),
                results=batch_results,
                errors=errors
            ))
        
        return results
    
    def iterate_batches(
        self,
        items: List[T]
    ) -> Iterator[List[T]]:
        """
        Iterate over items in batches.
        
        Args:
            items: Items to iterate
        
        Yields:
            Batches of items
        """
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]


def create_batch_processor(
    batch_size: int = 32,
    max_workers: Optional[int] = None
) -> BatchProcessor:
    """
    Create a batch processor.
    
    Args:
        batch_size: Size of each batch
        max_workers: Maximum number of workers
    
    Returns:
        Batch processor
    """
    return BatchProcessor(batch_size, max_workers)













