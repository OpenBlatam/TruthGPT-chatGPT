"""
Dataset management module for loading and preprocessing datasets.
"""
import logging
from typing import List, Optional, Tuple
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset loading from various sources."""
    
    @staticmethod
    def load_hf_dataset(
        dataset_name: str,
        subset: Optional[str] = None,
        text_field: str = "text",
        streaming: bool = False,
        split: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Load dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset
            subset: Optional subset name
            text_field: Field containing text data
            streaming: Whether to use streaming
            split: Optional split specification
        
        Returns:
            Tuple of (train_texts, val_texts)
        """
        try:
            logger.info(f"Loading HuggingFace dataset: {dataset_name} (subset: {subset})")
            
            if subset:
                ds = load_dataset(dataset_name, subset, streaming=streaming)
            else:
                ds = load_dataset(dataset_name, streaming=streaming)
            
            if "train" not in ds:
                raise ValueError(f"Dataset {dataset_name} does not contain 'train' split")
            
            # Get training data
            train_data = ds["train"][text_field] if not streaming else ds["train"]
            
            # Get validation data
            if "validation" in ds:
                val_data = ds["validation"][text_field] if not streaming else ds["validation"]
            elif "val" in ds:
                val_data = ds["val"][text_field] if not streaming else ds["val"]
            else:
                logger.warning("No validation split found, using train split for validation")
                val_data = train_data
            
            # Convert to lists if not streaming
            if streaming:
                train_list = list(train_data.take(5000))
                val_list = list(val_data.take(500))
            else:
                train_list = list(train_data) if not isinstance(train_data, list) else train_data
                val_list = list(val_data) if not isinstance(val_data, list) else val_data
            
            logger.info(
                f"Loaded {len(train_list)} training samples and "
                f"{len(val_list)} validation samples"
            )
            return train_list, val_list
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}", exc_info=True)
            raise
    
    @staticmethod
    def load_jsonl_dataset(
        path: str,
        text_field: str = "text",
        train_split: float = 0.9,
    ) -> Tuple[List[str], List[str]]:
        """
        Load dataset from JSONL file.
        
        Args:
            path: Path to JSONL file
            text_field: Field containing text data
            train_split: Ratio for train/val split
        
        Returns:
            Tuple of (train_texts, val_texts)
        """
        import json
        
        try:
            logger.info(f"Loading JSONL dataset from {path}")
            
            texts = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if text_field in data:
                        texts.append(data[text_field])
            
            # Split train/val
            split_idx = int(len(texts) * train_split)
            train_texts = texts[:split_idx]
            val_texts = texts[split_idx:]
            
            logger.info(
                f"Loaded {len(train_texts)} training samples and "
                f"{len(val_texts)} validation samples"
            )
            return train_texts, val_texts
            
        except Exception as e:
            logger.error(f"Error loading JSONL dataset {path}: {e}", exc_info=True)
            raise
    
    @staticmethod
    def load_text_file(
        path: str,
        train_split: float = 0.9,
        chunk_size: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Load dataset from text file.
        
        Args:
            path: Path to text file
            train_split: Ratio for train/val split
            chunk_size: Optional chunk size for splitting large texts
        
        Returns:
            Tuple of (train_texts, val_texts)
        """
        try:
            logger.info(f"Loading text file from {path}")
            
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if chunk_size:
                # Split into chunks
                texts = [
                    content[i:i + chunk_size]
                    for i in range(0, len(content), chunk_size)
                ]
            else:
                # Split by paragraphs
                texts = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            # Split train/val
            split_idx = int(len(texts) * train_split)
            train_texts = texts[:split_idx]
            val_texts = texts[split_idx:]
            
            logger.info(
                f"Loaded {len(train_texts)} training samples and "
                f"{len(val_texts)} validation samples"
            )
            return train_texts, val_texts
            
        except Exception as e:
            logger.error(f"Error loading text file {path}: {e}", exc_info=True)
            raise
    
    @classmethod
    def load_dataset(
        cls,
        source: str,
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        Load dataset from any supported source.
        
        Args:
            source: Source type (hf|jsonl|text)
            **kwargs: Source-specific arguments
        
        Returns:
            Tuple of (train_texts, val_texts)
        """
        if source == "hf":
            return cls.load_hf_dataset(**kwargs)
        elif source == "jsonl":
            return cls.load_jsonl_dataset(**kwargs)
        elif source == "text":
            return cls.load_text_file(**kwargs)
        else:
            raise ValueError(f"Unsupported dataset source: {source}")



