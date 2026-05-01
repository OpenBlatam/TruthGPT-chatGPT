"""
Data Manager — Pydantic-First Architecture.

Manages data loading, preprocessing, and bucketed sampling.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Callable
import torch
from torch.utils.data import DataLoader, Dataset
from pydantic import BaseModel, Field, ConfigDict

from optimization_core.trainers.config import TrainingConfig, HardwareConfig
from optimization_core.factories.collate import COLLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models for Data
# ---------------------------------------------------------------------------

class DataOptions(BaseModel):
    """Pydantic model for data loading options."""
    collate: str = Field(default="lm", description="Collate function type")
    bucket_by_length: bool = Field(default=False, description="Enable length bucketing")
    bucket_bins: List[int] = Field(
        default_factory=lambda: [64, 128, 256, 512],
        description="Bins for length bucketing"
    )


class DataLoadResult(BaseModel):
    """Result of data loading and loader creation."""
    train_samples: int = 0
    val_samples: int = 0
    train_batches: int = 0
    val_batches: int = 0
    collate_used: str = "none"
    bucketing_enabled: bool = False
    elapsed_ms: float = 0.0


class HFTextDataset(Dataset):
    """Simple dataset for HuggingFace tokenized text."""
    
    def __init__(self, tokenizer, texts: List[str], max_length: int):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attn_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


class DataManager:
    """
    Data Manager — Pydantic-First Architecture.
    
    Manages data loading, preprocessing, and bucketed sampling.
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        hardware_config: HardwareConfig,
        tokenizer,
        text_field_max_len: int = 512,
        data_options: Optional[Dict[str, Any]] = None,
    ):
        self.training_config = training_config
        self.hardware_config = hardware_config
        self.tokenizer = tokenizer
        self.text_field_max_len = text_field_max_len
        
        # Parse into Pydantic model
        self.opts = DataOptions(**(data_options or {}))
        
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
    
    def create_loaders(
        self,
        train_texts: List[str],
        val_texts: List[str],
    ) -> tuple[DataLoader, DataLoader, DataLoadResult]:
        """
        Create training and validation DataLoaders.
        """
        import time
        start = time.monotonic()
        
        # Determine collate function
        collate_name = self.opts.collate
        use_lm_collate = collate_name == "lm"
        collate_fn: Optional[Callable] = None
        
        if use_lm_collate:
            collate_fn = COLLATE.build("lm")(self.tokenizer, self.text_field_max_len)
        
        # Check for bucketing
        bucket_by_length = self.opts.bucket_by_length and use_lm_collate
        bucket_bins = self.opts.bucket_bins
        
        # Create train loader
        if collate_fn is not None:
            train_dataset = list(train_texts)
            val_dataset = list(val_texts)
            
            batch_sampler = None
            if bucket_by_length:
                batch_sampler = self._create_bucket_sampler(train_dataset, bucket_bins)
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=None if batch_sampler is not None else self.training_config.train_batch_size,
                shuffle=(batch_sampler is None),
                num_workers=self.hardware_config.num_workers,
                pin_memory=True,
                prefetch_factor=self.hardware_config.prefetch_factor if self.hardware_config.num_workers > 0 else None,
                persistent_workers=self.hardware_config.persistent_workers if self.hardware_config.num_workers > 0 else False,
                collate_fn=collate_fn,
                batch_sampler=batch_sampler,
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.eval_batch_size,
                shuffle=False,
                num_workers=self.hardware_config.num_workers,
                pin_memory=True,
                prefetch_factor=self.hardware_config.prefetch_factor if self.hardware_config.num_workers > 0 else None,
                persistent_workers=self.hardware_config.persistent_workers if self.hardware_config.num_workers > 0 else False,
                collate_fn=collate_fn,
            )
        else:
            # Fallback to static padding
            self.train_loader = DataLoader(
                HFTextDataset(self.tokenizer, train_texts, self.text_field_max_len),
                batch_size=self.training_config.train_batch_size,
                shuffle=True,
                num_workers=self.hardware_config.num_workers,
                pin_memory=True,
                prefetch_factor=self.hardware_config.prefetch_factor if self.hardware_config.num_workers > 0 else None,
                persistent_workers=self.hardware_config.persistent_workers if self.hardware_config.num_workers > 0 else False,
            )
            
            self.val_loader = DataLoader(
                HFTextDataset(self.tokenizer, val_texts, self.text_field_max_len),
                batch_size=self.training_config.eval_batch_size,
                shuffle=False,
                num_workers=self.hardware_config.num_workers,
                pin_memory=True,
                prefetch_factor=self.hardware_config.prefetch_factor if self.hardware_config.num_workers > 0 else None,
                persistent_workers=self.hardware_config.persistent_workers if self.hardware_config.num_workers > 0 else False,
            )
        
        logger.info(f"Created DataLoaders: train={len(self.train_loader)}, val={len(self.val_loader)}")
        
        elapsed_ms = (time.monotonic() - start) * 1000
        result = DataLoadResult(
            train_samples=len(train_texts),
            val_samples=len(val_texts),
            train_batches=len(self.train_loader),
            val_batches=len(self.val_loader),
            collate_used=collate_name,
            bucketing_enabled=bucket_by_length,
            elapsed_ms=round(elapsed_ms, 2)
        )
        
        return self.train_loader, self.val_loader, result
    
    def _create_bucket_sampler(self, dataset: List[str], bucket_bins: List[int]):
        """Create batch sampler with length bucketing."""
        # Precompute lengths
        lengths = [len(self.tokenizer.encode(t, add_special_tokens=False)) for t in dataset]
        
        # Assign to bins
        bin_indices: Dict[int, List[int]] = {b: [] for b in bucket_bins}
        for idx, length in enumerate(lengths):
            # Find appropriate bin
            bin_size = next((bb for bb in bucket_bins if length <= bb), bucket_bins[-1])
            bin_indices[bin_size].append(idx)
        
        # Create batches per bin
        batches: List[List[int]] = []
        batch_size = self.training_config.train_batch_size
        for bin_size in bucket_bins:
            indices = bin_indices[bin_size]
            for i in range(0, len(indices), batch_size):
                batches.append(indices[i:i + batch_size])
        
        # Create sampler
        class BucketBatchSampler:
            def __iter__(self):
                for batch in batches:
                    yield batch
            
            def __len__(self):
                return len(batches)
        
        logger.info(f"Created bucket sampler with {len(batches)} batches")
        return BucketBatchSampler()




