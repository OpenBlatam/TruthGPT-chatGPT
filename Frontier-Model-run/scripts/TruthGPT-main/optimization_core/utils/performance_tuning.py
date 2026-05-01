"""
Performance tuning utilities for optimization_core.

Provides utilities for automatic performance tuning.
"""
import logging
import time
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)


from .base import BaseOptimizationModel


class TuningResult(BaseOptimizationModel):
    """Result of performance tuning."""
    best_config: Dict[str, Any]
    best_performance: float
    iterations: int
    history: List[Dict[str, Any]]


class PerformanceTuner:
    """Tuner for automatic performance optimization."""
    
    def __init__(
        self,
        objective_func: Callable,
        param_space: Dict[str, List[Any]],
        max_iterations: int = 100
    ):
        """
        Initialize performance tuner.
        
        Args:
            objective_func: Function to optimize (returns performance metric)
            param_space: Parameter space to search
            max_iterations: Maximum iterations
        """
        self.objective_func = objective_func
        self.param_space = param_space
        self.max_iterations = max_iterations
    
    def tune(
        self,
        method: str = "random",
        **kwargs
    ) -> TuningResult:
        """
        Tune performance.
        
        Args:
            method: Tuning method (random, grid, bayesian)
            **kwargs: Additional arguments
        
        Returns:
            Tuning result
        """
        logger.info(f"Starting performance tuning with method: {method}")
        
        best_config = None
        best_performance = float('-inf')
        history = []
        
        if method == "random":
            configs = self._random_search()
        elif method == "grid":
            configs = self._grid_search()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for i, config in enumerate(configs[:self.max_iterations]):
            try:
                performance = self.objective_func(**config)
                
                if performance > best_performance:
                    best_performance = performance
                    best_config = config
                
                history.append({
                    "iteration": i + 1,
                    "config": config,
                    "performance": performance,
                })
                
                logger.debug(f"Iteration {i + 1}: performance={performance:.4f}")
            except Exception as e:
                logger.warning(f"Iteration {i + 1} failed: {e}")
                continue
        
        return TuningResult(
            best_config=best_config or {},
            best_performance=best_performance,
            iterations=len(history),
            history=history
        )
    
    def _random_search(self) -> List[Dict[str, Any]]:
        """Generate random configurations."""
        import random
        
        configs = []
        for _ in range(self.max_iterations):
            config = {}
            for param, values in self.param_space.items():
                config[param] = random.choice(values)
            configs.append(config)
        
        return configs
    
    def _grid_search(self) -> List[Dict[str, Any]]:
        """Generate grid configurations."""
        from itertools import product
        
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        
        configs = []
        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            configs.append(config)
        
        return configs


def auto_tune_performance(
    func: Callable,
    param_space: Dict[str, List[Any]],
    max_iterations: int = 100,
    method: str = "random"
) -> TuningResult:
    """
    Automatically tune performance.
    
    Args:
        func: Function to optimize
        param_space: Parameter space
        max_iterations: Maximum iterations
        method: Tuning method
    
    Returns:
        Tuning result
    """
    tuner = PerformanceTuner(func, param_space, max_iterations)
    return tuner.tune(method=method)













