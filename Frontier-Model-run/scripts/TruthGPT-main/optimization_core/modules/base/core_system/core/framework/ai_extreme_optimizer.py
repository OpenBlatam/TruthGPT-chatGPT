"""
AI Extreme Optimizer - Next-generation AI-powered optimization
Implements the most advanced AI optimization techniques with machine learning
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import time
import logging
from collections import deque
from contextlib import contextmanager
import warnings

from .metrics_calculator import AIOptimizationLevel
from .state_persistence import StatePersistence
from .models import AIOptimizationResult
from .statistics_calculator import StatisticsCalculator
from .component_factory import ComponentFactory
from .state_manager import StateManager
from .config import DEFAULT_OPTIMIZATION_HISTORY_SIZE

warnings.filterwarnings('ignore')


class AIExtremeOptimizer:
    """AI-powered extreme optimization system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimization_level = AIOptimizationLevel(
            self.config.get('level', 'intelligent')
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing AI extreme optimization system")
        
        components = ComponentFactory.create_all_components(
            self.optimization_level, self.config
        )
        
        self.neural_network = components['neural_network']
        self.optimizer = components['optimizer']
        self.strategy_registry = components['strategy_registry']
        self.feature_extractor = components['feature_extractor']
        self.learning_mechanism = components['learning_mechanism']
        self.metrics_calculator = components['metrics_calculator']
        self.insights_generator = components['insights_generator']
        self.strategy_selector = components['strategy_selector']
        self.optimization_pipeline = components['optimization_pipeline']
        self.strategies = components['strategies']
        
        self.optimization_history = deque(maxlen=DEFAULT_OPTIMIZATION_HISTORY_SIZE)
        
        self.state_manager = StateManager(
            self.neural_network, self.optimizer, self.strategy_registry
        )
        
        self.logger.info("AI system initialized successfully")
    
    def optimize_with_ai(self, model: nn.Module) -> AIOptimizationResult:
        """Optimize model using AI techniques."""
        start_time = time.perf_counter()
        self.logger.info(f"AI optimization started (level: {self.optimization_level.value})")
        
        result = self.optimization_pipeline.execute(model, start_time)
        self.optimization_history.append(result)
        self._log_optimization_completion(result, result.optimization_time)
        
        return result
    
    def _log_optimization_completion(self, result: AIOptimizationResult, optimization_time: float) -> None:
        """Log optimization completion."""
        self.logger.info(
            f"AI optimization completed: {result.speed_improvement:.1f}x speedup "
            f"in {optimization_time:.3f}ms"
        )
    
    def get_ai_statistics(self) -> Dict[str, Any]:
        """Get AI optimization statistics."""
        return StatisticsCalculator.calculate(
            list(self.optimization_history),
            self.optimization_level,
            self.learning_mechanism
        )
    
    def save_ai_state(self, filepath: str) -> None:
        """Save AI optimization state."""
        StatePersistence.save_state(
            neural_network_state=self.neural_network.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            learning_history=self.learning_mechanism.get_learning_history(),
            experience_buffer=self.learning_mechanism.get_experience_buffer(),
            exploration_rate=self.learning_mechanism.get_exploration_rate(),
            optimization_level=self.optimization_level,
            filepath=filepath
        )
    
    def load_ai_state(self, filepath: str) -> None:
        """Load AI optimization state."""
        try:
            state = StatePersistence.load_state(filepath)
            self.state_manager.validate_state(state)
            
            self.state_manager.restore_neural_network(state)
            self.state_manager.restore_optimizer(state)
            
            self.optimization_level = self.state_manager.restore_optimization_level(
                state, self.optimization_level
            )
            
            restored_components = self.state_manager.restore_learning_components(
                state, self.optimization_level
            )
            
            self.learning_mechanism = restored_components['learning_mechanism']
            self.insights_generator = restored_components['insights_generator']
            self.strategy_selector = restored_components['strategy_selector']
            self.strategies = restored_components['strategies']
            
            components = {
                'feature_extractor': self.feature_extractor,
                'strategy_selector': self.strategy_selector,
                'learning_mechanism': self.learning_mechanism,
                'metrics_calculator': self.metrics_calculator,
                'insights_generator': self.insights_generator
            }
            
            self.optimization_pipeline = self.state_manager.restore_optimization_pipeline(
                components, self.optimization_level
            )
        except FileNotFoundError:
            self.logger.error(f"State file not found: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load AI state: {e}", exc_info=True)
            raise

# Factory functions
def create_ai_extreme_optimizer(config: Optional[Dict[str, Any]] = None) -> AIExtremeOptimizer:
    """Create AI extreme optimizer."""
    return AIExtremeOptimizer(config)

@contextmanager
def ai_extreme_optimization_context(config: Optional[Dict[str, Any]] = None):
    """Context manager for AI extreme optimization."""
    optimizer = create_ai_extreme_optimizer(config)
    try:
        yield optimizer
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

