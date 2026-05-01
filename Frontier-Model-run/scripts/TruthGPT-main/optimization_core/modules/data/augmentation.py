"""
TruthGPT Advanced Data Augmentation Module
Advanced data augmentation strategies for TruthGPT models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTAugmentationConfig:
    """Configuration for TruthGPT data augmentation."""
    # Basic augmentation settings
    enable_augmentation: bool = True
    augmentation_probability: float = 0.1
    augmentation_strength: float = 0.5
    
    # Text augmentation types
    enable_token_shuffle: bool = True
    enable_token_mask: bool = True
    enable_token_replace: bool = True
    enable_token_insert: bool = True
    enable_token_delete: bool = True
    
    # Advanced augmentation types
    enable_synonym_replacement: bool = False
    enable_back_translation: bool = False
    enable_paraphrasing: bool = False
    enable_style_transfer: bool = False
    
    # Augmentation parameters
    shuffle_ratio: float = 0.1
    mask_ratio: float = 0.1
    replace_ratio: float = 0.1
    insert_ratio: float = 0.05
    delete_ratio: float = 0.05
    
    # Advanced parameters
    synonym_replacement_ratio: float = 0.1
    back_translation_ratio: float = 0.05
    paraphrasing_ratio: float = 0.05
    style_transfer_ratio: float = 0.05
    
    # Performance settings
    enable_parallel_augmentation: bool = True
    num_workers: int = 4
    enable_caching: bool = True
    cache_size: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_augmentation': self.enable_augmentation,
            'augmentation_probability': self.augmentation_probability,
            'augmentation_strength': self.augmentation_strength,
            'enable_token_shuffle': self.enable_token_shuffle,
            'enable_token_mask': self.enable_token_mask,
            'enable_token_replace': self.enable_token_replace,
            'enable_token_insert': self.enable_token_insert,
            'enable_token_delete': self.enable_token_delete,
            'enable_synonym_replacement': self.enable_synonym_replacement,
            'enable_back_translation': self.enable_back_translation,
            'enable_paraphrasing': self.enable_paraphrasing,
            'enable_style_transfer': self.enable_style_transfer,
            'shuffle_ratio': self.shuffle_ratio,
            'mask_ratio': self.mask_ratio,
            'replace_ratio': self.replace_ratio,
            'insert_ratio': self.insert_ratio,
            'delete_ratio': self.delete_ratio,
            'synonym_replacement_ratio': self.synonym_replacement_ratio,
            'back_translation_ratio': self.back_translation_ratio,
            'paraphrasing_ratio': self.paraphrasing_ratio,
            'style_transfer_ratio': self.style_transfer_ratio,
            'enable_parallel_augmentation': self.enable_parallel_augmentation,
            'num_workers': self.num_workers,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size
        }

class TruthGPTTokenAugmenter:
    """Advanced token-level augmenter for TruthGPT."""
    
    def __init__(self, config: TruthGPTAugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Augmentation state
        self.augmentation_stats = {
            'total_augmentations': 0,
            'shuffle_count': 0,
            'mask_count': 0,
            'replace_count': 0,
            'insert_count': 0,
            'delete_count': 0
        }
    
    def augment_tokens(self, tokens: List[int]) -> List[int]:
        """Apply token-level augmentation."""
        if not self.config.enable_augmentation:
            return tokens
        
        augmented_tokens = tokens.copy()
        
        # Apply different augmentation techniques
        if self.config.enable_token_shuffle and random.random() < self.config.augmentation_probability:
            augmented_tokens = self._shuffle_tokens(augmented_tokens)
            self.augmentation_stats['shuffle_count'] += 1
        
        if self.config.enable_token_mask and random.random() < self.config.augmentation_probability:
            augmented_tokens = self._mask_tokens(augmented_tokens)
            self.augmentation_stats['mask_count'] += 1
        
        if self.config.enable_token_replace and random.random() < self.config.augmentation_probability:
            augmented_tokens = self._replace_tokens(augmented_tokens)
            self.augmentation_stats['replace_count'] += 1
        
        if self.config.enable_token_insert and random.random() < self.config.augmentation_probability:
            augmented_tokens = self._insert_tokens(augmented_tokens)
            self.augmentation_stats['insert_count'] += 1
        
        if self.config.enable_token_delete and random.random() < self.config.augmentation_probability:
            augmented_tokens = self._delete_tokens(augmented_tokens)
            self.augmentation_stats['delete_count'] += 1
        
        self.augmentation_stats['total_augmentations'] += 1
        return augmented_tokens
    
    def _shuffle_tokens(self, tokens: List[int]) -> List[int]:
        """Shuffle tokens within segments."""
        if len(tokens) < 2:
            return tokens
        
        # Create segments for shuffling
        segment_size = max(1, int(len(tokens) * self.config.shuffle_ratio))
        shuffled_tokens = tokens.copy()
        
        for i in range(0, len(tokens), segment_size):
            segment = shuffled_tokens[i:i+segment_size]
            if len(segment) > 1:
                random.shuffle(segment)
                shuffled_tokens[i:i+segment_size] = segment
        
        return shuffled_tokens
    
    def _mask_tokens(self, tokens: List[int]) -> List[int]:
        """Mask random tokens."""
        if len(tokens) < 1:
            return tokens
        
        masked_tokens = tokens.copy()
        num_masks = max(1, int(len(tokens) * self.config.mask_ratio))
        
        # Select random positions to mask
        mask_positions = random.sample(range(len(tokens)), min(num_masks, len(tokens)))
        
        # Apply masking (using 0 as mask token)
        for pos in mask_positions:
            masked_tokens[pos] = 0
        
        return masked_tokens
    
    def _replace_tokens(self, tokens: List[int]) -> List[int]:
        """Replace random tokens with random values."""
        if len(tokens) < 1:
            return tokens
        
        replaced_tokens = tokens.copy()
        num_replacements = max(1, int(len(tokens) * self.config.replace_ratio))
        
        # Select random positions to replace
        replace_positions = random.sample(range(len(tokens)), min(num_replacements, len(tokens)))
        
        # Apply replacement
        for pos in replace_positions:
            replaced_tokens[pos] = random.randint(1, 1000)  # Random token ID
        
        return replaced_tokens
    
    def _insert_tokens(self, tokens: List[int]) -> List[int]:
        """Insert random tokens at random positions."""
        if len(tokens) < 1:
            return tokens
        
        inserted_tokens = tokens.copy()
        num_insertions = max(1, int(len(tokens) * self.config.insert_ratio))
        
        # Insert random tokens
        for _ in range(num_insertions):
            if len(inserted_tokens) < 1000:  # Limit length
                pos = random.randint(0, len(inserted_tokens))
                token = random.randint(1, 1000)
                inserted_tokens.insert(pos, token)
        
        return inserted_tokens
    
    def _delete_tokens(self, tokens: List[int]) -> List[int]:
        """Delete random tokens."""
        if len(tokens) < 2:
            return tokens
        
        deleted_tokens = tokens.copy()
        num_deletions = max(1, int(len(tokens) * self.config.delete_ratio))
        
        # Delete random tokens
        for _ in range(min(num_deletions, len(deleted_tokens) - 1)):
            if len(deleted_tokens) > 1:
                pos = random.randint(0, len(deleted_tokens) - 1)
                deleted_tokens.pop(pos)
        
        return deleted_tokens
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get augmentation statistics."""
        return self.augmentation_stats.copy()

class TruthGPTSemanticAugmenter:
    """Advanced semantic augmenter for TruthGPT."""
    
    def __init__(self, config: TruthGPTAugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Semantic augmentation state
        self.semantic_stats = {
            'synonym_replacements': 0,
            'back_translations': 0,
            'paraphrases': 0,
            'style_transfers': 0
        }
    
    def augment_text(self, text: str) -> str:
        """Apply semantic augmentation to text."""
        if not self.config.enable_augmentation:
            return text
        
        augmented_text = text
        
        # Apply synonym replacement
        if self.config.enable_synonym_replacement and random.random() < self.config.augmentation_probability:
            augmented_text = self._synonym_replacement(augmented_text)
            self.semantic_stats['synonym_replacements'] += 1
        
        # Apply back translation
        if self.config.enable_back_translation and random.random() < self.config.augmentation_probability:
            augmented_text = self._back_translation(augmented_text)
            self.semantic_stats['back_translations'] += 1
        
        # Apply paraphrasing
        if self.config.enable_paraphrasing and random.random() < self.config.augmentation_probability:
            augmented_text = self._paraphrasing(augmented_text)
            self.semantic_stats['paraphrases'] += 1
        
        # Apply style transfer
        if self.config.enable_style_transfer and random.random() < self.config.augmentation_probability:
            augmented_text = self._style_transfer(augmented_text)
            self.semantic_stats['style_transfers'] += 1
        
        return augmented_text
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        # Simplified synonym replacement
        # In practice, you would use a proper synonym dictionary
        words = text.split()
        augmented_words = []
        
        for word in words:
            if random.random() < self.config.synonym_replacement_ratio:
                # Simplified synonym replacement
                if word.lower() == "good":
                    augmented_words.append("excellent")
                elif word.lower() == "bad":
                    augmented_words.append("terrible")
                elif word.lower() == "big":
                    augmented_words.append("large")
                elif word.lower() == "small":
                    augmented_words.append("tiny")
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
        
        return " ".join(augmented_words)
    
    def _back_translation(self, text: str) -> str:
        """Apply back translation augmentation."""
        # Simplified back translation
        # In practice, you would use a proper translation model
        # For demo, we'll just return the original text
        return text
    
    def _paraphrasing(self, text: str) -> str:
        """Apply paraphrasing augmentation."""
        # Simplified paraphrasing
        # In practice, you would use a proper paraphrasing model
        # For demo, we'll just return the original text
        return text
    
    def _style_transfer(self, text: str) -> str:
        """Apply style transfer augmentation."""
        # Simplified style transfer
        # In practice, you would use a proper style transfer model
        # For demo, we'll just return the original text
        return text
    
    def get_semantic_stats(self) -> Dict[str, Any]:
        """Get semantic augmentation statistics."""
        return self.semantic_stats.copy()

class TruthGPTSequenceAugmenter:
    """Advanced sequence-level augmenter for TruthGPT."""
    
    def __init__(self, config: TruthGPTAugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Sequence augmentation state
        self.sequence_stats = {
            'sequence_shuffles': 0,
            'sequence_splits': 0,
            'sequence_merges': 0,
            'sequence_padding': 0
        }
    
    def augment_sequence(self, sequence: List[int]) -> List[int]:
        """Apply sequence-level augmentation."""
        if not self.config.enable_augmentation:
            return sequence
        
        augmented_sequence = sequence.copy()
        
        # Apply sequence shuffling
        if random.random() < self.config.augmentation_probability:
            augmented_sequence = self._shuffle_sequence(augmented_sequence)
            self.sequence_stats['sequence_shuffles'] += 1
        
        # Apply sequence splitting
        if random.random() < self.config.augmentation_probability:
            augmented_sequence = self._split_sequence(augmented_sequence)
            self.sequence_stats['sequence_splits'] += 1
        
        # Apply sequence merging
        if random.random() < self.config.augmentation_probability:
            augmented_sequence = self._merge_sequence(augmented_sequence)
            self.sequence_stats['sequence_merges'] += 1
        
        # Apply sequence padding
        if random.random() < self.config.augmentation_probability:
            augmented_sequence = self._pad_sequence(augmented_sequence)
            self.sequence_stats['sequence_padding'] += 1
        
        return augmented_sequence
    
    def _shuffle_sequence(self, sequence: List[int]) -> List[int]:
        """Shuffle sequence segments."""
        if len(sequence) < 2:
            return sequence
        
        # Create segments
        segment_size = max(1, len(sequence) // 4)
        segments = [sequence[i:i+segment_size] for i in range(0, len(sequence), segment_size)]
        
        # Shuffle segments
        random.shuffle(segments)
        
        # Flatten segments
        shuffled_sequence = []
        for segment in segments:
            shuffled_sequence.extend(segment)
        
        return shuffled_sequence
    
    def _split_sequence(self, sequence: List[int]) -> List[int]:
        """Split sequence into multiple parts."""
        if len(sequence) < 4:
            return sequence
        
        # Split sequence
        split_point = random.randint(1, len(sequence) - 1)
        first_part = sequence[:split_point]
        second_part = sequence[split_point:]
        
        # Return one of the parts
        return first_part if random.random() < 0.5 else second_part
    
    def _merge_sequence(self, sequence: List[int]) -> List[int]:
        """Merge sequence with random tokens."""
        if len(sequence) < 1:
            return sequence
        
        # Add random tokens
        num_random_tokens = random.randint(1, min(10, len(sequence)))
        random_tokens = [random.randint(1, 1000) for _ in range(num_random_tokens)]
        
        # Insert random tokens
        insert_position = random.randint(0, len(sequence))
        merged_sequence = sequence[:insert_position] + random_tokens + sequence[insert_position:]
        
        return merged_sequence
    
    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad sequence with random tokens."""
        if len(sequence) < 1:
            return sequence
        
        # Add padding tokens
        padding_size = random.randint(1, min(10, len(sequence)))
        padding_tokens = [0] * padding_size  # Using 0 as padding token
        
        # Add padding at random position
        if random.random() < 0.5:
            padded_sequence = sequence + padding_tokens
        else:
            padded_sequence = padding_tokens + sequence
        
        return padded_sequence
    
    def get_sequence_stats(self) -> Dict[str, Any]:
        """Get sequence augmentation statistics."""
        return self.sequence_stats.copy()

class TruthGPTAugmentationManager:
    """Advanced augmentation manager for TruthGPT."""
    
    def __init__(self, config: TruthGPTAugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Augmentation components
        self.token_augmenter = TruthGPTTokenAugmenter(config)
        self.semantic_augmenter = TruthGPTSemanticAugmenter(config)
        self.sequence_augmenter = TruthGPTSequenceAugmenter(config)
        
        # Augmentation state
        self.augmentation_history = []
        self.augmentation_stats = {}
    
    def augment_data(self, data: Union[str, List[int], List[str]]) -> Union[str, List[int], List[str]]:
        """Apply comprehensive augmentation to data."""
        if not self.config.enable_augmentation:
            return data
        
        self.logger.info("🔧 Applying TruthGPT data augmentation")
        
        if isinstance(data, str):
            # Text augmentation
            augmented_data = self.semantic_augmenter.augment_text(data)
        elif isinstance(data, list) and all(isinstance(x, int) for x in data):
            # Token sequence augmentation
            augmented_data = self.token_augmenter.augment_tokens(data)
            augmented_data = self.sequence_augmenter.augment_sequence(augmented_data)
        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            # Text list augmentation
            augmented_data = [self.semantic_augmenter.augment_text(text) for text in data]
        else:
            self.logger.warning(f"Unknown data type for augmentation: {type(data)}")
            return data
        
        # Record augmentation
        self.augmentation_history.append({
            'original_data': data,
            'augmented_data': augmented_data,
            'timestamp': time.time()
        })
        
        self.logger.info("✅ TruthGPT data augmentation completed")
        return augmented_data
    
    def batch_augment(self, data_list: List[Union[str, List[int]]]) -> List[Union[str, List[int]]]:
        """Apply augmentation to batch of data."""
        if not self.config.enable_augmentation:
            return data_list
        
        self.logger.info(f"🔧 Applying batch augmentation to {len(data_list)} samples")
        
        augmented_data_list = []
        for data in data_list:
            augmented_data = self.augment_data(data)
            augmented_data_list.append(augmented_data)
        
        self.logger.info("✅ Batch augmentation completed")
        return augmented_data_list
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get comprehensive augmentation statistics."""
        stats = {
            'total_augmentations': len(self.augmentation_history),
            'token_stats': self.token_augmenter.get_augmentation_stats(),
            'semantic_stats': self.semantic_augmenter.get_semantic_stats(),
            'sequence_stats': self.sequence_augmenter.get_sequence_stats()
        }
        
        return stats
    
    def get_augmentation_history(self) -> List[Dict[str, Any]]:
        """Get augmentation history."""
        return self.augmentation_history.copy()
    
    def clear_history(self) -> None:
        """Clear augmentation history."""
        self.augmentation_history.clear()
        self.logger.info("Augmentation history cleared")

# Factory functions
def create_truthgpt_augmentation_manager(config: TruthGPTAugmentationConfig) -> TruthGPTAugmentationManager:
    """Create TruthGPT augmentation manager."""
    return TruthGPTAugmentationManager(config)

def augment_truthgpt_data(data: Union[str, List[int]], config: TruthGPTAugmentationConfig) -> Union[str, List[int]]:
    """Quick augment TruthGPT data."""
    manager = create_truthgpt_augmentation_manager(config)
    return manager.augment_data(data)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT data augmentation
    print("🚀 TruthGPT Advanced Data Augmentation Demo")
    print("=" * 50)
    
    # Create augmentation configuration
    config = TruthGPTAugmentationConfig(
        enable_augmentation=True,
        augmentation_probability=0.3,
        enable_token_shuffle=True,
        enable_token_mask=True,
        enable_token_replace=True,
        shuffle_ratio=0.1,
        mask_ratio=0.1,
        replace_ratio=0.1
    )
    
    # Create augmentation manager
    manager = create_truthgpt_augmentation_manager(config)
    
    # Test token augmentation
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    augmented_tokens = manager.augment_data(tokens)
    print(f"Original tokens: {tokens}")
    print(f"Augmented tokens: {augmented_tokens}")
    
    # Test text augmentation
    text = "This is a sample text for augmentation"
    augmented_text = manager.augment_data(text)
    print(f"Original text: {text}")
    print(f"Augmented text: {augmented_text}")
    
    # Get augmentation stats
    stats = manager.get_augmentation_stats()
    print(f"Augmentation stats: {stats}")
    
    print("✅ TruthGPT data augmentation demo completed!")



