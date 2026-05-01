"""
Ultra-Advanced Multi-Modal Fusion Engine Module
===============================================

This module provides advanced multi-modal fusion capabilities for TruthGPT models,
including text, image, audio, video, and sensor data fusion with attention mechanisms.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pydantic import BaseModel, Field, model_validator
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of modalities for fusion."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"
    TABULAR = "tabular"
    GRAPH = "graph"
    TIME_SERIES = "time_series"

class FusionStrategy(Enum):
    """Fusion strategies for multi-modal data."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    INTERMEDIATE_FUSION = "intermediate_fusion"
    ATTENTION_FUSION = "attention_fusion"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    HIERARCHICAL_FUSION = "hierarchical_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"

class AttentionMechanism(Enum):
    """Attention mechanisms for fusion."""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    SPATIAL_ATTENTION = "spatial_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    CHANNEL_ATTENTION = "channel_attention"

class MultimodalConfig(BaseModel):
    """Configuration for multi-modal fusion."""
    modalities: List[ModalityType] = Field(default_factory=lambda: [ModalityType.TEXT, ModalityType.IMAGE])
    fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION
    attention_mechanism: AttentionMechanism = AttentionMechanism.MULTI_HEAD_ATTENTION
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_fusion_layers: int = 3
    dropout_rate: float = 0.1
    fusion_dim: int = 256
    max_sequence_length: int = 512
    image_size: Tuple[int, int] = (224, 224)
    audio_sample_rate: int = 16000
    video_fps: int = 30
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./multimodal_results"

    @model_validator(mode='after')
    def validate_config(self) -> 'MultimodalConfig':
        """Post-initialization validation."""
        if self.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("Number of attention heads must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        return self

class ModalityEncoder(nn.Module):
    """Base class for modality encoders."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for modality encoding."""
        raise NotImplementedError

class TextEncoder(ModalityEncoder):
    """Text encoder for multi-modal fusion."""
    
    def __init__(self, config: MultimodalConfig, vocab_size: int = 30000):
        super().__init__(config)
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(config.max_sequence_length, config.hidden_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dropout=config.dropout_rate,
                batch_first=True
            ),
            num_layers=config.num_fusion_layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for text encoding."""
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.shape
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        embedded += self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Transformer encoding
        encoded = self.transformer(embedded)
        
        return encoded

class ImageEncoder(ModalityEncoder):
    """Image encoder for multi-modal fusion."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(256 * 7 * 7, config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for image encoding."""
        # x shape: (batch_size, channels, height, width)
        features = self.conv_layers(x)
        flattened = self.flatten(features)
        projected = self.projection(flattened)
        
        # Reshape to sequence format for fusion
        batch_size = x.shape[0]
        encoded = projected.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        return encoded

class AudioEncoder(ModalityEncoder):
    """Audio encoder for multi-modal fusion."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            
            nn.Conv1d(64, 128, kernel_size=11, stride=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.projection = nn.Linear(256, config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for audio encoding."""
        # x shape: (batch_size, 1, audio_length)
        features = self.conv1d_layers(x)
        flattened = features.squeeze(-1)  # (batch_size, 256)
        projected = self.projection(flattened)
        
        # Reshape to sequence format for fusion
        batch_size = x.shape[0]
        encoded = projected.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        return encoded

class VideoEncoder(ModalityEncoder):
    """Video encoder for multi-modal fusion."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 7, 7))
        )
        
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(256 * 7 * 7, config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for video encoding."""
        # x shape: (batch_size, channels, frames, height, width)
        features = self.conv3d_layers(x)
        flattened = self.flatten(features)
        projected = self.projection(flattened)
        
        # Reshape to sequence format for fusion
        batch_size = x.shape[0]
        encoded = projected.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        return encoded

class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for attention fusion."""
        # modality_features: List of tensors with shape (batch_size, seq_len, hidden_dim)
        
        # Concatenate modality features
        concatenated = torch.cat(modality_features, dim=1)  # (batch_size, total_seq_len, hidden_dim)
        
        # Self-attention
        attended, attention_weights = self.attention(
            concatenated, concatenated, concatenated
        )
        
        # Residual connection and layer norm
        fused = self.layer_norm(concatenated + self.dropout(attended))
        
        return fused, attention_weights

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, query_modality: torch.Tensor, 
                key_value_modality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for cross-modal attention."""
        # Cross-attention between modalities
        attended, attention_weights = self.cross_attention(
            query_modality, key_value_modality, key_value_modality
        )
        
        # Residual connection and layer norm
        fused = self.layer_norm(query_modality + self.dropout(attended))
        
        return fused, attention_weights

class HierarchicalFusion(nn.Module):
    """Hierarchical fusion mechanism."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dropout=config.dropout_rate,
                batch_first=True
            ) for _ in range(config.num_fusion_layers)
        ])
        
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for hierarchical fusion."""
        # Concatenate modality features
        concatenated = torch.cat(modality_features, dim=1)
        
        # Apply hierarchical fusion layers
        fused = concatenated
        for layer in self.fusion_layers:
            fused = layer(fused)
        
        return fused

class AdaptiveFusion(nn.Module):
    """Adaptive fusion mechanism that learns fusion weights."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.num_modalities = len(config.modalities)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(self.num_modalities) / self.num_modalities)
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.fusion_dim) 
            for _ in range(self.num_modalities)
        ])
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(config.fusion_dim * self.num_modalities, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for adaptive fusion."""
        # Project each modality
        projected_features = []
        for i, features in enumerate(modality_features):
            projected = self.modality_projections[i](features.mean(dim=1))  # Global average pooling
            projected_features.append(projected)
        
        # Weighted combination
        weighted_features = []
        for i, features in enumerate(projected_features):
            weighted = features * self.fusion_weights[i]
            weighted_features.append(weighted)
        
        # Concatenate and fuse
        concatenated = torch.cat(weighted_features, dim=1)
        fused = self.fusion_network(concatenated)
        
        return fused

class MultimodalFusionEngine(nn.Module):
    """Main multi-modal fusion engine."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.setup_logging()
        self.setup_device()
        
        # Initialize modality encoders
        self.encoders = nn.ModuleDict()
        self._initialize_encoders()
        
        # Initialize fusion mechanism
        self.fusion_mechanism = self._initialize_fusion_mechanism()
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.fusion_dim)
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def setup_device(self):
        """Setup computation device."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")
    
    def _initialize_encoders(self):
        """Initialize modality encoders."""
        for modality in self.config.modalities:
            if modality == ModalityType.TEXT:
                self.encoders[modality.value] = TextEncoder(self.config)
            elif modality == ModalityType.IMAGE:
                self.encoders[modality.value] = ImageEncoder(self.config)
            elif modality == ModalityType.AUDIO:
                self.encoders[modality.value] = AudioEncoder(self.config)
            elif modality == ModalityType.VIDEO:
                self.encoders[modality.value] = VideoEncoder(self.config)
            else:
                logger.warning(f"Unsupported modality: {modality.value}")
    
    def _initialize_fusion_mechanism(self):
        """Initialize fusion mechanism based on strategy."""
        if self.config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            return AttentionFusion(self.config)
        elif self.config.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            return CrossModalAttention(self.config)
        elif self.config.fusion_strategy == FusionStrategy.HIERARCHICAL_FUSION:
            return HierarchicalFusion(self.config)
        elif self.config.fusion_strategy == FusionStrategy.ADAPTIVE_FUSION:
            return AdaptiveFusion(self.config)
        else:
            return AttentionFusion(self.config)  # Default
    
    def forward(self, multimodal_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-modal fusion."""
        # Encode each modality
        modality_features = []
        modality_names = []
        
        for modality_name, data in multimodal_data.items():
            if modality_name in self.encoders:
                encoded = self.encoders[modality_name](data)
                modality_features.append(encoded)
                modality_names.append(modality_name)
        
        if not modality_features:
            raise ValueError("No valid modality data provided")
        
        # Apply fusion mechanism
        if self.config.fusion_strategy == FusionStrategy.CROSS_MODAL_ATTENTION:
            # Cross-modal attention between first two modalities
            if len(modality_features) >= 2:
                fused, attention_weights = self.fusion_mechanism(
                    modality_features[0], modality_features[1]
                )
            else:
                fused = modality_features[0]
                attention_weights = None
        else:
            fused = self.fusion_mechanism(modality_features)
            attention_weights = None
        
        # Output projection
        if fused.dim() == 3:  # (batch_size, seq_len, hidden_dim)
            fused = fused.mean(dim=1)  # Global average pooling
        
        output = self.output_projection(fused)
        
        return {
            'fused_features': output,
            'modality_features': modality_features,
            'attention_weights': attention_weights,
            'modality_names': modality_names
        }
    
    def get_modality_importance(self, multimodal_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get importance scores for each modality."""
        with torch.no_grad():
            # Forward pass
            results = self.forward(multimodal_data)
            
            # Calculate importance based on attention weights or feature norms
            importance_scores = {}
            
            if results['attention_weights'] is not None:
                # Use attention weights for importance
                attention_weights = results['attention_weights']
                avg_attention = attention_weights.mean(dim=1).mean(dim=0)  # Average over heads and positions
                
                for i, modality_name in enumerate(results['modality_names']):
                    importance_scores[modality_name] = avg_attention[i].item()
            else:
                # Use feature norms for importance
                for i, modality_name in enumerate(results['modality_names']):
                    feature_norm = torch.norm(results['modality_features'][i]).item()
                    importance_scores[modality_name] = feature_norm
            
            # Normalize importance scores
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
            
            return importance_scores

class TruthGPTMultimodalManager:
    """Main manager for TruthGPT multi-modal fusion."""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.fusion_engine = MultimodalFusionEngine(config)
        self.fusion_results = {}
        self.performance_metrics = {}
        
    def fuse_modalities(self, multimodal_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Fuse multiple modalities."""
        logger.info(f"Fusing modalities: {list(multimodal_data.keys())}")
        
        start_time = time.time()
        
        # Perform fusion
        fusion_results = self.fusion_engine(multimodal_data)
        
        # Calculate fusion time
        fusion_time = time.time() - start_time
        
        # Get modality importance
        importance_scores = self.fusion_engine.get_modality_importance(multimodal_data)
        
        # Store results
        self.fusion_results = {
            'fused_features': fusion_results['fused_features'],
            'modality_features': fusion_results['modality_features'],
            'attention_weights': fusion_results['attention_weights'],
            'modality_names': fusion_results['modality_names'],
            'importance_scores': importance_scores,
            'fusion_time': fusion_time,
            'fusion_strategy': self.config.fusion_strategy.value
        }
        
        logger.info(f"Multi-modal fusion completed in {fusion_time:.4f}s")
        
        return self.fusion_results
    
    def evaluate_fusion_quality(self, multimodal_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate the quality of multi-modal fusion."""
        # Perform fusion
        fusion_results = self.fuse_modalities(multimodal_data)
        
        # Calculate quality metrics
        quality_metrics = {}
        
        # Feature diversity (higher is better)
        modality_features = fusion_results['modality_features']
        if len(modality_features) > 1:
            feature_diversity = self._calculate_feature_diversity(modality_features)
            quality_metrics['feature_diversity'] = feature_diversity
        
        # Attention consistency (lower is better)
        if fusion_results['attention_weights'] is not None:
            attention_consistency = self._calculate_attention_consistency(fusion_results['attention_weights'])
            quality_metrics['attention_consistency'] = attention_consistency
        
        # Fusion efficiency (higher is better)
        fusion_efficiency = 1.0 / fusion_results['fusion_time'] if fusion_results['fusion_time'] > 0 else 0.0
        quality_metrics['fusion_efficiency'] = fusion_efficiency
        
        # Modality balance (closer to uniform is better)
        importance_scores = fusion_results['importance_scores']
        modality_balance = self._calculate_modality_balance(importance_scores)
        quality_metrics['modality_balance'] = modality_balance
        
        self.performance_metrics = quality_metrics
        
        return quality_metrics
    
    def _calculate_feature_diversity(self, modality_features: List[torch.Tensor]) -> float:
        """Calculate diversity between modality features."""
        if len(modality_features) < 2:
            return 0.0
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(modality_features)):
            for j in range(i + 1, len(modality_features)):
                # Flatten features for similarity calculation
                feat1 = modality_features[i].mean(dim=1)  # Global average pooling
                feat2 = modality_features[j].mean(dim=1)
                
                # Cosine similarity
                similarity = F.cosine_similarity(feat1, feat2, dim=1).mean().item()
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = statistics.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_similarity
        
        return diversity
    
    def _calculate_attention_consistency(self, attention_weights: torch.Tensor) -> float:
        """Calculate consistency of attention weights."""
        # Calculate variance of attention weights across heads
        attention_variance = torch.var(attention_weights, dim=1).mean().item()
        
        # Consistency is inverse of variance
        consistency = 1.0 / (1.0 + attention_variance)
        
        return consistency
    
    def _calculate_modality_balance(self, importance_scores: Dict[str, float]) -> float:
        """Calculate balance of modality importance scores."""
        if not importance_scores:
            return 0.0
        
        scores = list(importance_scores.values())
        uniform_score = 1.0 / len(scores)
        
        # Calculate how close the scores are to uniform distribution
        balance = 1.0 - sum(abs(score - uniform_score) for score in scores) / 2.0
        
        return balance
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        return {
            'num_modalities': len(self.config.modalities),
            'fusion_strategy': self.config.fusion_strategy.value,
            'attention_mechanism': self.config.attention_mechanism.value,
            'hidden_dim': self.config.hidden_dim,
            'num_attention_heads': self.config.num_attention_heads,
            'performance_metrics': self.performance_metrics
        }

# Factory functions
def create_multimodal_config(modalities: List[ModalityType] = None,
                           fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION,
                           hidden_dim: int = 512,
                           **kwargs) -> MultimodalConfig:
    """Create multi-modal configuration."""
    if modalities is None:
        modalities = [ModalityType.TEXT, ModalityType.IMAGE]
    
    return MultimodalConfig(
        modalities=modalities,
        fusion_strategy=fusion_strategy,
        hidden_dim=hidden_dim,
        **kwargs
    )

def create_multimodal_fusion_engine(config: Optional[MultimodalConfig] = None) -> MultimodalFusionEngine:
    """Create multi-modal fusion engine."""
    if config is None:
        config = create_multimodal_config()
    return MultimodalFusionEngine(config)

def create_multimodal_manager(config: Optional[MultimodalConfig] = None) -> TruthGPTMultimodalManager:
    """Create multi-modal manager."""
    if config is None:
        config = create_multimodal_config()
    return TruthGPTMultimodalManager(config)

# Example usage
def example_multimodal_fusion():
    """Example of multi-modal fusion."""
    # Create configuration
    config = create_multimodal_config(
        modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO],
        fusion_strategy=FusionStrategy.ATTENTION_FUSION,
        hidden_dim=256,
        num_attention_heads=8
    )
    
    # Create multi-modal manager
    multimodal_manager = create_multimodal_manager(config)
    
    # Create dummy multi-modal data
    batch_size = 4
    
    multimodal_data = {
        'text': torch.randint(0, 1000, (batch_size, 50)),  # (batch_size, seq_len)
        'image': torch.randn(batch_size, 3, 224, 224),     # (batch_size, channels, height, width)
        'audio': torch.randn(batch_size, 1, 16000)         # (batch_size, channels, audio_length)
    }
    
    # Perform fusion
    fusion_results = multimodal_manager.fuse_modalities(multimodal_data)
    
    print(f"Fusion results: {fusion_results}")
    
    # Evaluate fusion quality
    quality_metrics = multimodal_manager.evaluate_fusion_quality(multimodal_data)
    print(f"Quality metrics: {quality_metrics}")
    
    # Get fusion statistics
    statistics = multimodal_manager.get_fusion_statistics()
    print(f"Fusion statistics: {statistics}")
    
    return fusion_results

if __name__ == "__main__":
    # Run example
    example_multimodal_fusion()

