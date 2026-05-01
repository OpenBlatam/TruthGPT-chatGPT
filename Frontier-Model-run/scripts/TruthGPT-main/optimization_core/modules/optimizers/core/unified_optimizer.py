"""
Unified Optimizer
=================
Consolidates all optimization core implementations using Strategy Pattern.
Replaces all *optimization_core.py files.
"""
import torch.nn as nn
from typing import Dict, Any, List, Optional
import time
import logging

from .base_truthgpt_optimizer import (
    BaseTruthGPTOptimizer,
    OptimizationResult,
    OptimizationLevel
)

logger = logging.getLogger(__name__)


class UnifiedOptimizer(BaseTruthGPTOptimizer):
    """
    Unified optimizer that consolidates all optimization core implementations.
    """
    
    # Mapping from level to default strategy class name (string) to avoid eager imports
    LEVEL_TO_STRATEGY_NAME = {
        OptimizationLevel.BASIC: 'BasicStrategy',
        OptimizationLevel.ADVANCED: 'EnhancedStrategy',
        OptimizationLevel.EXPERT: 'HybridStrategy',
        OptimizationLevel.MASTER: 'HybridStrategy',
        OptimizationLevel.SUPREME: 'HybridStrategy',
        OptimizationLevel.TRANSCENDENT: 'HybridStrategy',
        OptimizationLevel.ULTRA_FAST: 'HybridStrategy',
        OptimizationLevel.LIBRARY: 'LibraryStrategy',
    }
    
    def __init__(
        self,
        level: OptimizationLevel = OptimizationLevel.ADVANCED,
        strategies: Optional[List[Any]] = None, # Changed hint to avoid eager OptimizationStrategy
        config: Dict[str, Any] = None
    ):
        """
        Initialize unified optimizer.
        """
        from .strategies import (
            OptimizationStrategy,
            BasicStrategy,
            EnhancedStrategy,
            HybridStrategy,
            LibraryStrategy,
        )
        
        # Local mapping for actual classes
        level_to_class = {
            OptimizationLevel.BASIC: BasicStrategy,
            OptimizationLevel.ADVANCED: EnhancedStrategy,
            OptimizationLevel.EXPERT: HybridStrategy,
            OptimizationLevel.MASTER: HybridStrategy,
            OptimizationLevel.SUPREME: HybridStrategy,
            OptimizationLevel.TRANSCENDENT: HybridStrategy,
            OptimizationLevel.ULTRA_FAST: HybridStrategy,
            OptimizationLevel.LIBRARY: LibraryStrategy,
        }

        """
        Initialize unified optimizer.
        
        Args:
            level: Optimization level (used if strategies not provided)
            strategies: List of strategies to apply (optional)
            config: Configuration dictionary
        """
        super().__init__(config, level)
        
        # Use provided strategies or create default based on level
        if strategies:
            self.strategies = strategies
            # Update level to match first strategy if provided
            if strategies:
                self.level = strategies[0].level
        else:
            strategy_class = level_to_class.get(
                level, 
                EnhancedStrategy  # Default fallback
            )
            self.strategies = [strategy_class(config)]
    
    def optimize(self, model: nn.Module, **kwargs) -> OptimizationResult:
        """
        Apply optimization using configured strategies.
        
        Args:
            model: Model to optimize
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with optimized model and metrics
        """
        start_time = time.perf_counter()
        
        self.logger.info(
            f"🚀 Unified optimization started (level: {self.level.value}, "
            f"strategies: {len(self.strategies)})"
        )
        
        # Apply strategies in sequence
        optimized_model = model
        all_techniques = []
        
        for i, strategy in enumerate(self.strategies):
            self.logger.debug(f"Applying strategy {i+1}/{len(self.strategies)}: {strategy.__class__.__name__}")
            
            try:
                optimized_model = strategy.apply(optimized_model, **kwargs)
                all_techniques.extend(strategy.get_techniques())
            except Exception as e:
                self.logger.error(f"Error applying strategy {strategy.__class__.__name__}: {e}")
                # Continue with next strategy even if one fails
                continue
        
        optimization_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, optimized_model)
        
        # Store for history
        self._last_applied_techniques = all_techniques
        self._last_optimization_time = optimization_time
        
        result = OptimizationResult(
            optimized_model=optimized_model,
            speed_improvement=metrics.get("speed_improvement", 1.0),
            memory_reduction=metrics.get("memory_reduction", 0.0),
            accuracy_preservation=metrics.get("accuracy_preservation", 1.0),
            energy_efficiency=metrics.get("energy_efficiency", 1.0),
            optimization_time=optimization_time,
            level=self.level,
            techniques_applied=all_techniques,
            performance_metrics=metrics
        )
        
        self.optimization_history.append(result)
        self.logger.info(
            f"✅ Optimization completed: {len(all_techniques)} techniques applied "
            f"in {optimization_time:.2f}ms"
        )
        
        return result
    
    def _calculate_metrics(
        self, 
        original_model: nn.Module, 
        optimized_model: nn.Module
    ) -> Dict[str, float]:
        """
        Calculate performance metrics by aggregating from all strategies.
        
        Args:
            original_model: Original model
            optimized_model: Optimized model
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Aggregate metrics from all strategies
        all_metrics = {}
        
        for strategy in self.strategies:
            strategy_metrics = strategy.calculate_metrics(original_model, optimized_model)
            # Merge metrics (could use more sophisticated aggregation)
            for key, value in strategy_metrics.items():
                if key in all_metrics:
                    # Average for now, could use other aggregation methods
                    all_metrics[key] = (all_metrics[key] + value) / 2
                else:
                    all_metrics[key] = value
        
        # Fallback to base implementation if no strategies provided metrics
        if not all_metrics:
            all_metrics = super()._calculate_metrics(original_model, optimized_model)
        
        return all_metrics


