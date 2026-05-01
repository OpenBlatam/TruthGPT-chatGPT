"""
Optimizer Factory Module
Factory and builder patterns for creating optimizers
"""

from typing import Dict, Any, Optional

from .ai_extreme_optimizer import AIExtremeOptimizer


class OptimizerBuilder:
    """Builder for creating AIExtremeOptimizer instances."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
    
    def with_level(self, level: str) -> 'OptimizerBuilder':
        """Set optimization level."""
        self.config['level'] = level
        return self
    
    def with_learning_rate(self, rate: float) -> 'OptimizerBuilder':
        """Set learning rate."""
        self.config['learning_rate'] = rate
        return self
    
    def with_exploration_rate(self, rate: float) -> 'OptimizerBuilder':
        """Set exploration rate."""
        self.config['exploration_rate'] = rate
        return self
    
    def with_custom_config(self, config: Dict[str, Any]) -> 'OptimizerBuilder':
        """Add custom configuration."""
        self.config.update(config)
        return self
    
    def build(self) -> AIExtremeOptimizer:
        """Build the optimizer instance."""
        return AIExtremeOptimizer(self.config)


def create_optimizer(
    level: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> AIExtremeOptimizer:
    """Create optimizer with specified level and configuration."""
    builder = OptimizerBuilder()
    
    if level:
        builder.with_level(level)
    
    if config:
        builder.with_custom_config(config)
    
    return builder.build()


def create_default_optimizer() -> AIExtremeOptimizer:
    """Create optimizer with default settings."""
    return AIExtremeOptimizer()


