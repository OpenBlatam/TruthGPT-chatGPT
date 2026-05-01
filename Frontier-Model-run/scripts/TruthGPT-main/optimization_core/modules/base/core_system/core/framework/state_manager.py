"""
State Manager Module
Handles state validation and restoration
"""

import logging
from typing import Dict, Any

from .state_persistence import StatePersistence
from .neural_network import NeuralOptimizationNetwork
from .strategy_selector import StrategySelector
from .insights_generator import InsightsGenerator
from .optimization_pipeline import OptimizationPipeline
from .metrics_calculator import AIOptimizationLevel
from .optimization_strategies import StrategyRegistry
from .component_factory import ComponentFactory
from .config import REQUIRED_STATE_KEYS

logger = logging.getLogger(__name__)


class StateManager:
    """Manages optimizer state validation and restoration."""
    
    def __init__(
        self,
        neural_network: NeuralOptimizationNetwork,
        optimizer,
        strategy_registry: StrategyRegistry
    ):
        self.neural_network = neural_network
        self.optimizer = optimizer
        self.strategy_registry = strategy_registry
    
    def validate_state(self, state: Dict[str, Any]) -> None:
        """Validate that state contains required keys."""
        missing_keys = [
            key for key in REQUIRED_STATE_KEYS if key not in state
        ]
        if missing_keys:
            raise ValueError(
                f"Invalid state file: missing required keys: {missing_keys}"
            )
    
    def restore_neural_network(self, state: Dict[str, Any]) -> None:
        """Restore neural network state."""
        self.neural_network.load_state_dict(state['neural_network_state'])
    
    def restore_optimizer(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state."""
        self.optimizer.load_state_dict(state['optimizer_state'])
    
    def restore_optimization_level(
        self,
        state: Dict[str, Any],
        current_level: AIOptimizationLevel
    ) -> AIOptimizationLevel:
        """Restore optimization level with fallback."""
        try:
            return AIOptimizationLevel(state['optimization_level'])
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Invalid optimization level in state: {e}, keeping current level"
            )
            return current_level
    
    def restore_learning_components(
        self,
        state: Dict[str, Any],
        optimization_level: AIOptimizationLevel
    ) -> Dict[str, Any]:
        """Restore learning mechanism and related components."""
        learning_mechanism = StatePersistence.restore_learning_mechanism(state)
        
        insights_generator = ComponentFactory.create_insights_generator(
            optimization_level, learning_mechanism
        )
        
        strategies = self.strategy_registry.get_all_strategy_names()
        strategy_selector = ComponentFactory.create_strategy_selector(
            self.neural_network, learning_mechanism, strategies
        )
        
        return {
            'learning_mechanism': learning_mechanism,
            'insights_generator': insights_generator,
            'strategy_selector': strategy_selector,
            'strategies': strategies
        }
    
    def restore_optimization_pipeline(
        self,
        components: Dict[str, Any],
        optimization_level: AIOptimizationLevel
    ) -> OptimizationPipeline:
        """Restore optimization pipeline with restored components."""
        return ComponentFactory.create_optimization_pipeline(
            feature_extractor=components['feature_extractor'],
            strategy_selector=components['strategy_selector'],
            strategy_registry=self.strategy_registry,
            learning_mechanism=components['learning_mechanism'],
            metrics_calculator=components['metrics_calculator'],
            insights_generator=components['insights_generator'],
            optimization_level=optimization_level
        )


