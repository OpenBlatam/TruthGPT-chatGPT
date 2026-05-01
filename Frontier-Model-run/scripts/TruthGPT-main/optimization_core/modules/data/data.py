"""
TruthGPT Data Module
Advanced data loading and preprocessing for TruthGPT models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import random
from transformers import AutoTokenizer, AutoModel
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTDataConfig:
    """Configuration for TruthGPT data loading."""
    # Data configuration
    data_path: str = ""
    tokenizer_name: str = "gpt2"
    max_sequence_length: int = 2048
    batch_size: int = 32
    
    # Data loading configuration
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Data preprocessing
    enable_tokenization: bool = True
    enable_padding: bool = True
    enable_truncation: bool = True
    padding_side: str = "right"  # left, right
    
    # Data augmentation
    enable_augmentation: bool = False
    augmentation_probability: float = 0.1
    augmentation_types: List[str] = field(default_factory=lambda: ["shuffle", "mask", "replace"])
    
    # Data splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Advanced features
    enable_caching: bool = True
    cache_dir: str = "./cache"
    enable_streaming: bool = False
    streaming_chunk_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_path': self.data_path,
            'tokenizer_name': self.tokenizer_name,
            'max_sequence_length': self.max_sequence_length,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'prefetch_factor': self.prefetch_factor,
            'enable_tokenization': self.enable_tokenization,
            'enable_padding': self.enable_padding,
            'enable_truncation': self.enable_truncation,
            'padding_side': self.padding_side,
            'enable_augmentation': self.enable_augmentation,
            'augmentation_probability': self.augmentation_probability,
            'augmentation_types': self.augmentation_types,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'enable_caching': self.enable_caching,
            'cache_dir': self.cache_dir,
            'enable_streaming': self.enable_streaming,
            'streaming_chunk_size': self.streaming_chunk_size
        }

class TruthGPTDataset(Dataset):
    """Advanced dataset for TruthGPT models."""
    
    def __init__(self, config: TruthGPTDataConfig, data: Optional[List[str]] = None):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize tokenizer
        self.tokenizer = self._setup_tokenizer()
        
        # Load data
        self.data = self._load_data(data)
        
        # Setup caching
        if config.enable_caching:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache = {}
        
        self.logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def _setup_tokenizer(self):
        """Setup tokenizer for TruthGPT."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Set padding side
            tokenizer.padding_side = self.config.padding_side
            
            self.logger.info(f"Tokenizer loaded: {self.config.tokenizer_name}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_data(self, data: Optional[List[str]] = None) -> List[str]:
        """Load data for TruthGPT."""
        if data is not None:
            return data
        
        if not self.config.data_path:
            # Create dummy data for demonstration
            dummy_data = [
                "This is a sample text for TruthGPT training.",
                "Another example of text data for the model.",
                "More training data to improve model performance."
            ] * 1000  # Repeat to create more data
            return dummy_data
        
        # Load data from file
        data_path = Path(self.config.data_path)
        
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.suffix == '.txt':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = f.readlines()
        elif data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        return data
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset."""
        # Get text
        text = self.data[idx]
        
        # Apply augmentation if enabled
        if self.config.enable_augmentation and random.random() < self.config.augmentation_probability:
            text = self._apply_augmentation(text)
        
        # Tokenize text
        if self.config.enable_tokenization:
            tokens = self._tokenize_text(text)
        else:
            # Convert to tensor directly (for demonstration)
            tokens = torch.tensor([ord(c) for c in text[:self.config.max_sequence_length]])
            tokens = F.pad(tokens, (0, self.config.max_sequence_length - len(tokens)))
        
        return {
            'input_ids': tokens,
            'labels': tokens.clone()  # For language modeling, labels are the same as input_ids
        }
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text for TruthGPT."""
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.config.max_sequence_length,
            padding='max_length' if self.config.enable_padding else False,
            truncation=self.config.enable_truncation,
            return_tensors='pt'
        )
        
        return tokens['input_ids'].squeeze(0)
    
    def _apply_augmentation(self, text: str) -> str:
        """Apply data augmentation to text."""
        augmented_text = text
        
        for aug_type in self.config.augmentation_types:
            if random.random() < self.config.augmentation_probability:
                if aug_type == "shuffle":
                    augmented_text = self._shuffle_words(augmented_text)
                elif aug_type == "mask":
                    augmented_text = self._mask_tokens(augmented_text)
                elif aug_type == "replace":
                    augmented_text = self._replace_tokens(augmented_text)
        
        return augmented_text
    
    def _shuffle_words(self, text: str) -> str:
        """Shuffle words in text."""
        words = text.split()
        random.shuffle(words)
        return ' '.join(words)
    
    def _mask_tokens(self, text: str) -> str:
        """Mask random tokens in text."""
        words = text.split()
        mask_indices = random.sample(range(len(words)), min(3, len(words)))
        for idx in mask_indices:
            words[idx] = '[MASK]'
        return ' '.join(words)
    
    def _replace_tokens(self, text: str) -> str:
        """Replace random tokens in text."""
        words = text.split()
        replace_indices = random.sample(range(len(words)), min(3, len(words)))
        for idx in replace_indices:
            words[idx] = random.choice(['[UNK]', '[REPLACE]', '[RANDOM]'])
        return ' '.join(words)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.data:
            return {}
        
        # Calculate statistics
        text_lengths = [len(text) for text in self.data]
        
        return {
            'total_samples': len(self.data),
            'avg_text_length': np.mean(text_lengths),
            'min_text_length': np.min(text_lengths),
            'max_text_length': np.max(text_lengths),
            'vocab_size': len(self.tokenizer) if self.tokenizer else 0
        }

class TruthGPTDataLoader:
    """Advanced data loader for TruthGPT models."""
    
    def __init__(self, config: TruthGPTDataConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize dataset
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.logger.info("TruthGPT DataLoader initialized")
    
    def create_dataset(self, data: Optional[List[str]] = None) -> TruthGPTDataset:
        """Create TruthGPT dataset."""
        self.dataset = TruthGPTDataset(self.config, data)
        self.logger.info(f"Dataset created with {len(self.dataset)} samples")
        return self.dataset
    
    def create_dataloaders(self, dataset: TruthGPTDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders."""
        if not dataset:
            raise ValueError("Dataset not created. Call create_dataset() first.")
        
        # Split dataset
        train_size = int(self.config.train_split * len(dataset))
        val_size = int(self.config.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
        
        self.logger.info(f"DataLoaders created - Train: {len(self.train_loader)}, Val: {len(self.val_loader)}, Test: {len(self.test_loader)}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get dataloaders."""
        if not all([self.train_loader, self.val_loader, self.test_loader]):
            raise ValueError("DataLoaders not created. Call create_dataloaders() first.")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.dataset:
            return {}
        
        return self.dataset.get_statistics()

class TruthGPTDataPreprocessor:
    """Data preprocessor for TruthGPT models."""
    
    def __init__(self, config: TruthGPTDataConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize tokenizer
        self.tokenizer = self._setup_tokenizer()
        
        self.logger.info("TruthGPT DataPreprocessor initialized")
    
    def _setup_tokenizer(self):
        """Setup tokenizer for preprocessing."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.logger.info(f"Preprocessor tokenizer loaded: {self.config.tokenizer_name}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for TruthGPT."""
        # Basic text cleaning
        text = text.strip()
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text for TruthGPT."""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.config.max_sequence_length,
            padding='max_length' if self.config.enable_padding else False,
            truncation=self.config.enable_truncation,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    
    def batch_tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize batch of texts."""
        # Preprocess texts
        texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize batch
        tokens = self.tokenizer(
            texts,
            max_length=self.config.max_sequence_length,
            padding='max_length' if self.config.enable_padding else False,
            truncation=self.config.enable_truncation,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }

# Factory functions
def create_truthgpt_dataset(config: TruthGPTDataConfig, data: Optional[List[str]] = None) -> TruthGPTDataset:
    """Create TruthGPT dataset."""
    return TruthGPTDataset(config, data)

def create_truthgpt_dataloader(config: TruthGPTDataConfig) -> TruthGPTDataLoader:
    """Create TruthGPT data loader."""
    return TruthGPTDataLoader(config)

def create_truthgpt_preprocessor(config: TruthGPTDataConfig) -> TruthGPTDataPreprocessor:
    """Create TruthGPT data preprocessor."""
    return TruthGPTDataPreprocessor(config)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT data loading
    print("🚀 TruthGPT Data Module Demo")
    print("=" * 50)
    
    # Create configuration
    config = TruthGPTDataConfig(
        tokenizer_name="gpt2",
        max_sequence_length=512,
        batch_size=16,
        enable_augmentation=True
    )
    
    # Create dataset
    dataset = create_truthgpt_dataset(config)
    
    # Create data loader
    data_loader = create_truthgpt_dataloader(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(dataset)
    
    # Get statistics
    stats = data_loader.get_dataset_statistics()
    print(f"Dataset statistics: {stats}")
    
    # Test data loading
    for batch in train_loader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break
    
    print("✅ TruthGPT data loading completed!")



