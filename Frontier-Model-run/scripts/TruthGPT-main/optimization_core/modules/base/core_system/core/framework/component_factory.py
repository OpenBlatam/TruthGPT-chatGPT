"""
Component Factory Module
Creates and initializes all optimizer components
"""

import torch.optim as optim
from typing import Dict, Any, Optional

from .optimization_strategies import StrategyRegistry
from .model_features import ModelFeatureExtractor
from .metrics_calculator import MetricsCalculator, AIOptimizationLevel
from .learning_mechanism import LearningMechanism
from .insights_generator import InsightsGenerator
from .neural_network import NeuralOptimizationNetwork
from .strategy_selector import StrategySelector
from .optimization_pipeline import OptimizationPipeline
from .config import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_EXPLORATION_RATE,
    DEFAULT_MEMORY_DECAY,
    DEFAULT_ADAPTATION_RATE,
    DEFAULT_TARGET_FEATURE_SIZE,
    DEFAULT_EXPERIENCE_BUFFER_SIZE,
    DEFAULT_LEARNING_HISTORY_SIZE,
)


class ComponentFactory:
    """Factory for creating optimizer components."""
    
    @staticmethod
    def create_neural_network() -> NeuralOptimizationNetwork:
        """Create neural network for optimization."""
        network = NeuralOptimizationNetwork()
        network.eval()
        return network
    
    @staticmethod
    def create_optimizer(network: NeuralOptimizationNetwork) -> optim.Adam:
        """Create optimizer for neural network."""
        return optim.Adam(network.parameters(), lr=DEFAULT_LEARNING_RATE)
    
    @staticmethod
    def create_strategy_registry() -> StrategyRegistry:
        """Create strategy registry."""
        return StrategyRegistry()
    
    @staticmethod
    def create_feature_extractor() -> ModelFeatureExtractor:
        """Create model feature extractor."""
        return ModelFeatureExtractor(target_feature_size=DEFAULT_TARGET_FEATURE_SIZE)
    
    @staticmethod
    def create_metrics_calculator(optimization_level: AIOptimizationLevel) -> MetricsCalculator:
        """Create metrics calculator."""
        return MetricsCalculator(optimization_level)
    
    @staticmethod
    def create_learning_mechanism(config: Optional[Dict[str, Any]] = None) -> LearningMechanism:
        """Create learning mechanism."""
        config = config or {}
        return LearningMechanism(
            learning_rate=config.get('learning_rate', DEFAULT_LEARNING_RATE),
            exploration_rate=config.get('exploration_rate', DEFAULT_EXPLORATION_RATE),
            memory_decay=config.get('memory_decay', DEFAULT_MEMORY_DECAY),
            adaptation_rate=config.get('adaptation_rate', DEFAULT_ADAPTATION_RATE),
            experience_buffer_size=config.get(
                'experience_buffer_size', DEFAULT_EXPERIENCE_BUFFER_SIZE
            ),
            learning_history_size=config.get(
                'learning_history_size', DEFAULT_LEARNING_HISTORY_SIZE
            )
        )
    
    @staticmethod
    def create_insights_generator(
        optimization_level: AIOptimizationLevel,
        learning_mechanism: LearningMechanism
    ) -> InsightsGenerator:
        """Create insights generator."""
        return InsightsGenerator(optimization_level, learning_mechanism)
    
    @staticmethod
    def create_strategy_selector(
        neural_network: NeuralOptimizationNetwork,
        learning_mechanism: LearningMechanism,
        strategies: list
    ) -> StrategySelector:
        """Create strategy selector."""
        return StrategySelector(neural_network, learning_mechanism, strategies)
    
    @staticmethod
    def create_optimization_pipeline(
        feature_extractor: ModelFeatureExtractor,
        strategy_selector: StrategySelector,
        strategy_registry: StrategyRegistry,
        learning_mechanism: LearningMechanism,
        metrics_calculator: MetricsCalculator,
        insights_generator: InsightsGenerator,
        optimization_level: AIOptimizationLevel
    ) -> OptimizationPipeline:
        """Create optimization pipeline."""
        return OptimizationPipeline(
            feature_extractor=feature_extractor,
            strategy_selector=strategy_selector,
            strategy_registry=strategy_registry,
            learning_mechanism=learning_mechanism,
            metrics_calculator=metrics_calculator,
            insights_generator=insights_generator,
            optimization_level=optimization_level
        )
    
    @staticmethod
    def create_all_components(
        optimization_level: AIOptimizationLevel,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create all optimizer components."""
        neural_network = ComponentFactory.create_neural_network()
        optimizer = ComponentFactory.create_optimizer(neural_network)
        strategy_registry = ComponentFactory.create_strategy_registry()
        feature_extractor = ComponentFactory.create_feature_extractor()
        learning_mechanism = ComponentFactory.create_learning_mechanism(config)
        metrics_calculator = ComponentFactory.create_metrics_calculator(optimization_level)
        insights_generator = ComponentFactory.create_insights_generator(
            optimization_level, learning_mechanism
        )
        
        strategies = strategy_registry.get_all_strategy_names()
        strategy_selector = ComponentFactory.create_strategy_selector(
            neural_network, learning_mechanism, strategies
        )
        
        optimization_pipeline = ComponentFactory.create_optimization_pipeline(
            feature_extractor,
            strategy_selector,
            strategy_registry,
            learning_mechanism,
            metrics_calculator,
            insights_generator,
            optimization_level
        )
        
        return {
            'neural_network': neural_network,
            'optimizer': optimizer,
            'strategy_registry': strategy_registry,
            'feature_extractor': feature_extractor,
            'learning_mechanism': learning_mechanism,
            'metrics_calculator': metrics_calculator,
            'insights_generator': insights_generator,
            'strategy_selector': strategy_selector,
            'optimization_pipeline': optimization_pipeline,
            'strategies': strategies
        }







