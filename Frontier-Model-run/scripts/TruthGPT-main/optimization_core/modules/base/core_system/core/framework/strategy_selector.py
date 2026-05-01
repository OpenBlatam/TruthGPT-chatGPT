"""
Strategy Selection Module
Handles AI-powered strategy selection for optimization
"""

import torch
import numpy as np
from typing import Tuple, List
import logging

from .neural_network import NeuralOptimizationNetwork
from .learning_mechanism import LearningMechanism
from .config import DEFAULT_STRATEGY_CONFIDENCE

logger = logging.getLogger(__name__)


class StrategySelector:
    """Handles AI-powered strategy selection."""
    
    def __init__(
        self,
        neural_network: NeuralOptimizationNetwork,
        learning_mechanism: LearningMechanism,
        strategies: List[str]
    ):
        self.neural_network = neural_network
        self.learning_mechanism = learning_mechanism
        self.strategies = strategies
    
    def select_strategy(self, model_features: torch.Tensor) -> Tuple[str, float]:
        """Select optimization strategy using AI."""
        if len(self.strategies) == 0:
            logger.warning("No strategies available, using default quantization")
            return 'quantization', DEFAULT_STRATEGY_CONFIDENCE
        
        with torch.no_grad():
            strategy_probs, _, _ = self.neural_network(model_features.unsqueeze(0))
        
        if np.random.random() < self.learning_mechanism.get_exploration_rate():
            strategy_probs = torch.softmax(torch.randn_like(strategy_probs), dim=-1)
        
        strategy_idx = torch.multinomial(strategy_probs, 1).item()
        if strategy_idx >= len(self.strategies):
            strategy_idx = 0
        
        strategy = self.strategies[strategy_idx]
        confidence = float(strategy_probs[strategy_idx].item())
        
        return strategy, confidence


