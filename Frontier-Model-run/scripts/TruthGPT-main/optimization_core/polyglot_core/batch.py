"""
Batch processing utilities for polyglot_core.

Provides batching, chunking, and parallel processing capabilities.
"""

from typing import List, Iterator, Callable, Any, Optional, TypeVar
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


T = TypeVar('T')


class BatchProcessor:
    """
    Batch processor for polyglot_core operations.
    
    Provides efficient batching, chunking, and parallel processing.
    """
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size
            max_workers: Maximum number of parallel workers
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def batch(self, items: List[T], batch_size: Optional[int] = None) -> Iterator[List[T]]:
        """
        Split items into batches.
        
        Args:
            items: List of items
            batch_size: Batch size (default: self.batch_size)
            
        Yields:
            Batches of items
        """
        batch_size = batch_size or self.batch_size
        
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    def process_batches(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], Any],
        batch_size: Optional[int] = None,
        parallel: bool = False
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            process_fn: Processing function
            batch_size: Batch size
            parallel: Whether to process batches in parallel
            
        Returns:
            List of results
        """
        batches = list(self.batch(items, batch_size))
        results = []
        
        if parallel and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_fn, batch) for batch in batches]
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for batch in batches:
                results.append(process_fn(batch))
        
        return results
    
    def chunk_tensor(
        self,
        tensor: np.ndarray,
        chunk_size: int,
        axis: int = 0
    ) -> Iterator[np.ndarray]:
        """
        Chunk tensor along specified axis.
        
        Args:
            tensor: Input tensor
            chunk_size: Chunk size
            axis: Axis to chunk along
            
        Yields:
            Tensor chunks
        """
        total_size = tensor.shape[axis]
        
        for i in range(0, total_size, chunk_size):
            end = min(i + chunk_size, total_size)
            
            if axis == 0:
                yield tensor[i:end]
            elif axis == 1:
                yield tensor[:, i:end]
            else:
                # General case
                indices = [slice(None)] * len(tensor.shape)
                indices[axis] = slice(i, end)
                yield tensor[tuple(indices)]
    
    def pad_batch(self, sequences: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
        """
        Pad sequences to same length.
        
        Args:
            sequences: List of sequences
            pad_value: Padding value
            
        Returns:
            Padded batch tensor
        """
        max_len = max(seq.shape[0] for seq in sequences)
        batch_size = len(sequences)
        feature_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
        
        if len(sequences[0].shape) > 1:
            padded = np.full((batch_size, max_len, feature_dim), pad_value, dtype=sequences[0].dtype)
        else:
            padded = np.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)
        
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        
        return padded


def batch(items: List[T], batch_size: int = 32) -> Iterator[List[T]]:
    """Convenience function to batch items."""
    processor = BatchProcessor(batch_size=batch_size)
    return processor.batch(items)


def process_batches(
    items: List[T],
    process_fn: Callable[[List[T]], Any],
    batch_size: int = 32,
    parallel: bool = False,
    max_workers: int = 4
) -> List[Any]:
    """Convenience function to process batches."""
    processor = BatchProcessor(batch_size=batch_size, max_workers=max_workers)
    return processor.process_batches(items, process_fn, parallel=parallel)


def pad_batch(sequences: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """Convenience function to pad sequences."""
    processor = BatchProcessor()
    return processor.pad_batch(sequences, pad_value)













