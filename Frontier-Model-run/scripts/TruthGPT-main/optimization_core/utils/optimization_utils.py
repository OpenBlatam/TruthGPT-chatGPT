"""
Optimization utilities for optimization_core.

Provides utilities for automatic optimization and tuning.
"""
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationResult(BaseModel):
    """Result of optimization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    best_params: Dict[str, Any]
    best_score: float
    iterations: int
    history: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class HyperparameterOptimizer:
    """Optimizer for hyperparameters."""
    
    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float]],
        objective_func: Callable[[Dict[str, float]], float],
        method: str = "random"
    ):
        """
        Initialize optimizer.
        
        Args:
            param_space: Parameter space (name -> (min, max))
            objective_func: Objective function to minimize
            method: Optimization method (random, grid, bayesian)
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.method = method
        self.history: List[float] = []
    
    def optimize(
        self,
        n_iterations: int = 100,
        random_state: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize hyperparameters.
        
        Args:
            n_iterations: Number of iterations
            random_state: Random seed
        
        Returns:
            OptimizationResult
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        best_score = float('inf')
        best_params = {}
        
        for i in range(n_iterations):
            # Sample parameters
            params = self._sample_params()
            
            # Evaluate
            try:
                score = self.objective_func(params)
                self.history.append(score)
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"Iteration {i+1}: New best score {best_score:.4f}")
            except Exception as e:
                logger.warning(f"Iteration {i+1} failed: {e}")
                self.history.append(float('inf'))
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            iterations=n_iterations,
            history=self.history
        )
    
    def _sample_params(self) -> Dict[str, float]:
        """Sample parameters from space."""
        params = {}
        for name, (min_val, max_val) in self.param_space.items():
            if self.method == "random":
                params[name] = np.random.uniform(min_val, max_val)
            elif self.method == "grid":
                # Simple grid search (would need more sophisticated implementation)
                params[name] = np.random.uniform(min_val, max_val)
            else:
                params[name] = np.random.uniform(min_val, max_val)
        return params


def optimize_batch_size(
    model_func: Callable,
    input_size: int,
    min_batch: int = 1,
    max_batch: int = 128,
    target_memory: Optional[float] = None
) -> int:
    """
    Optimize batch size for a model.
    
    Args:
        model_func: Function that takes batch_size and returns memory usage
        input_size: Input size
        min_batch: Minimum batch size
        max_batch: Maximum batch size
        target_memory: Target memory usage (MB)
    
    Returns:
        Optimal batch size
    """
    best_batch = min_batch
    best_score = float('inf')
    
    for batch_size in range(min_batch, max_batch + 1):
        try:
            memory = model_func(batch_size)
            
            if target_memory:
                score = abs(memory - target_memory)
            else:
                score = memory
            
            if score < best_score:
                best_score = score
                best_batch = batch_size
        except Exception as e:
            logger.warning(f"Batch size {batch_size} failed: {e}")
            break
    
    return best_batch













