"""
Library Optimizer
=================
Optimizer using the modular library system.
"""
from typing import Dict, Any, Optional
import torch.nn as nn

from optimization_core.modules.optimizers.core.base_truthgpt_optimizer import BaseTruthGPTOptimizer
from optimization_core.modules.optimizers.core.strategies.library_strategy import LibraryStrategy


class LibraryOptimizer(BaseTruthGPTOptimizer):
    """
    Optimizer that uses the modular library system.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config=config)
        self.strategy = LibraryStrategy(config=config)
        
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        return self.strategy.apply(model, **kwargs)

def create_library_optimizer(config: Dict[str, Any] = None) -> LibraryOptimizer:
    return LibraryOptimizer(config=config)

