"""
Momentum & Memory
=================

Momentum encoder and memory bank implementations.
"""
import torch
import torch.nn as nn
import logging
from typing import Tuple
from .config import SSLConfig

logger = logging.getLogger(__name__)

class MomentumEncoder:
    """Momentum encoder implementation"""
    
    def __init__(self, config: SSLConfig):
        self.config = config
        self.encoder = self._create_encoder()
        self.momentum_encoder = self._create_encoder()
        self._update_momentum_encoder()
        logger.info("✅ Momentum Encoder initialized")
    
    def _create_encoder(self) -> nn.Module:
        """Create encoder network"""
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.config.encoder_dim)
        )
        return encoder
    
    def _update_momentum_encoder(self):
        """Update momentum encoder"""
        for param, momentum_param in zip(self.encoder.parameters(), 
                                       self.momentum_encoder.parameters()):
            momentum_param.data = momentum_param.data * self.config.momentum + \
                                param.data * (1 - self.config.momentum)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Current encoder
        current_features = self.encoder(x)
        
        # Momentum encoder
        with torch.no_grad():
            momentum_features = self.momentum_encoder(x)
        
        return current_features, momentum_features
    
    def update_momentum(self):
        """Update momentum encoder"""
        self._update_momentum_encoder()
        
class MemoryBank:
    """Memory bank implementation"""
    
    def __init__(self, config: SSLConfig):
        self.config = config
        self.memory_bank = torch.randn(config.memory_bank_size, config.encoder_dim)
        self.memory_labels = torch.randint(0, 1000, (config.memory_bank_size,))
        self.current_index = 0
        logger.info("✅ Memory Bank initialized")
    
    def update(self, features: torch.Tensor, labels: torch.Tensor = None):
        """Update memory bank"""
        batch_size = features.shape[0]
        
        for i in range(batch_size):
            self.memory_bank[self.current_index] = features[i]
            if labels is not None:
                self.memory_labels[self.current_index] = labels[i]
            
            self.current_index = (self.current_index + 1) % self.config.memory_bank_size
    
    def get_negative_samples(self, num_samples: int) -> torch.Tensor:
        """Get negative samples from memory bank"""
        indices = torch.randint(0, self.config.memory_bank_size, (num_samples,))
        return self.memory_bank[indices]
    
    def get_positive_samples(self, labels: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Get positive samples from memory bank"""
        positive_indices = []
        for label in labels:
            label_indices = torch.where(self.memory_labels == label)[0]
            if len(label_indices) > 0:
                positive_indices.append(label_indices[0])
        
        if len(positive_indices) >= num_samples:
            selected_indices = torch.tensor(positive_indices[:num_samples])
            return self.memory_bank[selected_indices]
        else:
            return self.get_negative_samples(num_samples)

