"""
Neural Network for Optimization Strategy Learning
"""

import torch
import torch.nn as nn
from typing import Tuple

from .config import DEFAULT_DROPOUT_RATE


class NeuralOptimizationNetwork(nn.Module):
    """Neural network for learning optimization strategies."""
    
    def __init__(self, input_size: int = 1024, hidden_size: int = 512, num_strategies: int = 20):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_strategies = num_strategies
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        self.performance_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.intelligence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        strategy_probs = self.strategy_head(features)
        performance_pred = self.performance_head(features)
        intelligence_score = self.intelligence_head(features)
        return strategy_probs, performance_pred, intelligence_score


