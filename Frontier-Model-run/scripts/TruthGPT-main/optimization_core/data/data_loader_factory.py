"""
Factory for creating DataLoaders with various configurations.
"""
import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, Dataset

from factories.collate import COLLATE

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating optimized DataLoaders."""
    
    @staticmethod
    def create_loader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        collate_fn: Optional[Any] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        batch_sampler: Optional[Any] = None,
    ) -> DataLoader:
        """
        Create a DataLoader with optimized settings.
        
        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size (ignored if batch_sampler is provided)
            shuffle: Whether to shuffle
            collate_fn: Collate function
            num_workers: Number of worker processes
            prefetch_factor: Prefetch factor for workers
            persistent_workers: Keep workers alive between epochs
            pin_memory: Pin memory for faster GPU transfer
            batch_sampler: Optional batch sampler
        
        Returns:
            Configured DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=None if batch_sampler else batch_size,
            shuffle=shuffle and batch_sampler is None,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            pin_memory=pin_memory,
            batch_sampler=batch_sampler,
        )
    
    @staticmethod
    def create_train_loader(
        texts: List[str],
        tokenizer: Any,
        max_length: int,
        batch_size: int,
        collate_type: str = "lm",
        bucket_by_length: bool = False,
        bucket_bins: Optional[List[int]] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> DataLoader:
        """
        Create training DataLoader with optional length bucketing.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            batch_size: Batch size
            collate_type: Type of collator (lm|cv)
            bucket_by_length: Whether to use length bucketing
            bucket_bins: Bucket size bins for length bucketing
            num_workers: Number of workers
            prefetch_factor: Prefetch factor
            persistent_workers: Keep workers alive
        
        Returns:
            Configured training DataLoader
        """
        # Create collate function
        collate_fn = COLLATE.build(collate_type)(tokenizer, max_length)
        
        # Create batch sampler if using length bucketing
        batch_sampler = None
        if bucket_by_length and collate_type == "lm":
            batch_sampler = DataLoaderFactory._create_length_bucket_sampler(
                texts, tokenizer, batch_size, bucket_bins or [64, 128, 256, 512]
            )
        
        return DataLoaderFactory.create_loader(
            dataset=list(texts),  # Use raw texts with dynamic padding
            batch_size=batch_size,
            shuffle=not bucket_by_length,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            batch_sampler=batch_sampler,
        )
    
    @staticmethod
    def create_val_loader(
        texts: List[str],
        tokenizer: Any,
        max_length: int,
        batch_size: int,
        collate_type: str = "lm",
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> DataLoader:
        """
        Create validation DataLoader.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            batch_size: Batch size
            collate_type: Type of collator (lm|cv)
            num_workers: Number of workers
            prefetch_factor: Prefetch factor
            persistent_workers: Keep workers alive
        
        Returns:
            Configured validation DataLoader
        """
        collate_fn = COLLATE.build(collate_type)(tokenizer, max_length)
        
        return DataLoaderFactory.create_loader(
            dataset=list(texts),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    
    @staticmethod
    def _create_length_bucket_sampler(
        texts: List[str],
        tokenizer: Any,
        batch_size: int,
        bucket_bins: List[int],
    ) -> Any:
        """
        Create batch sampler that groups samples by length.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer for computing lengths
            batch_size: Batch size
            bucket_bins: Bucket size bins
        
        Returns:
            Batch sampler instance
        """
        # Precompute lengths
        lengths = [
            len(tokenizer.encode(t, add_special_tokens=False))
            for t in texts
        ]
        
        # Assign to bins
        bin_indices: Dict[int, List[int]] = {b: [] for b in bucket_bins}
        for idx, length in enumerate(lengths):
            bin_size = next((b for b in bucket_bins if length <= b), bucket_bins[-1])
            bin_indices[bin_size].append(idx)
        
        # Build batches per bin
        batches: List[List[int]] = []
        for bin_size in bucket_bins:
            indices = bin_indices[bin_size]
            for i in range(0, len(indices), batch_size):
                batches.append(indices[i:i + batch_size])
        
        # Create sampler class
        class LengthBucketSampler:
            def __init__(self, batches):
                self.batches = batches
            
            def __iter__(self):
                return iter(self.batches)
            
            def __len__(self):
                return len(self.batches)
        
        logger.debug(
            f"Created length bucket sampler with {len(batches)} batches "
            f"across {len(bucket_bins)} bins"
        )
        return LengthBucketSampler(batches)


class DataLoaderBuilder:
    """
    Builder pattern for creating DataLoaders.
    Allows fluent API for configuration.
    """
    
    def __init__(self):
        self._texts: Optional[List[str]] = None
        self._tokenizer: Optional[Any] = None
        self._max_length: int = 512
        self._batch_size: int = 8
        self._collate_type: str = "lm"
        self._bucket_by_length: bool = False
        self._bucket_bins: Optional[List[int]] = None
        self._num_workers: int = 4
        self._prefetch_factor: int = 2
        self._persistent_workers: bool = True
        self._shuffle: bool = True
    
    def with_texts(self, texts: List[str]) -> "DataLoaderBuilder":
        """Set text samples."""
        self._texts = texts
        return self
    
    def with_tokenizer(self, tokenizer: Any) -> "DataLoaderBuilder":
        """Set tokenizer."""
        self._tokenizer = tokenizer
        return self
    
    def with_max_length(self, max_length: int) -> "DataLoaderBuilder":
        """Set maximum sequence length."""
        self._max_length = max_length
        return self
    
    def with_batch_size(self, batch_size: int) -> "DataLoaderBuilder":
        """Set batch size."""
        self._batch_size = batch_size
        return self
    
    def with_collate_type(self, collate_type: str) -> "DataLoaderBuilder":
        """Set collate type."""
        self._collate_type = collate_type
        return self
    
    def with_length_bucketing(
        self,
        enabled: bool = True,
        bins: Optional[List[int]] = None
    ) -> "DataLoaderBuilder":
        """Enable/disable length bucketing."""
        self._bucket_by_length = enabled
        if bins:
            self._bucket_bins = bins
        return self
    
    def with_workers(
        self,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent: bool = True
    ) -> "DataLoaderBuilder":
        """Configure worker processes."""
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._persistent_workers = persistent
        return self
    
    def with_shuffle(self, shuffle: bool = True) -> "DataLoaderBuilder":
        """Enable/disable shuffling."""
        self._shuffle = shuffle
        return self
    
    def build_train(self) -> DataLoader:
        """Build training DataLoader."""
        if not self._texts or not self._tokenizer:
            raise ValueError("texts and tokenizer must be set")
        
        return DataLoaderFactory.create_train_loader(
            texts=self._texts,
            tokenizer=self._tokenizer,
            max_length=self._max_length,
            batch_size=self._batch_size,
            collate_type=self._collate_type,
            bucket_by_length=self._bucket_by_length,
            bucket_bins=self._bucket_bins,
            num_workers=self._num_workers,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=self._persistent_workers,
        )
    
    def build_val(self) -> DataLoader:
        """Build validation DataLoader."""
        if not self._texts or not self._tokenizer:
            raise ValueError("texts and tokenizer must be set")
        
        return DataLoaderFactory.create_val_loader(
            texts=self._texts,
            tokenizer=self._tokenizer,
            max_length=self._max_length,
            batch_size=self._batch_size,
            collate_type=self._collate_type,
            num_workers=self._num_workers,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=self._persistent_workers,
        )



