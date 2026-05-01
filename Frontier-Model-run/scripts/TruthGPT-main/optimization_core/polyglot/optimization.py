"""
Unified Optimization Interface

Provides Python interface to Julia optimization backends.
"""
from typing import Optional, Dict, Any, Callable, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from julia import TruthGPTCore
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False

class HyperparameterOptimizer:
    """
    Unified hyperparameter optimization interface.
    
    Uses Julia backend for mathematical optimization when available.
    Falls back to Python scipy.optimize otherwise.
    """
    
    def __init__(self, use_julia: bool = True):
        self.use_julia = use_julia and JULIA_AVAILABLE
        if self.use_julia:
            logger.info("Using Julia backend for optimization")
        else:
            logger.info("Using Python backend for optimization")
    
    def optimize(
        self,
        loss_fn: Callable[[Dict[str, Any]], float],
        bounds: Dict[str, Tuple[float, float]],
        method: str = "bayesian",
        max_iters: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            loss_fn: Function that takes params dict and returns loss
            bounds: Dictionary of parameter bounds {param: (min, max)}
            method: Optimization method (bayesian, random, grid)
            max_iters: Maximum iterations
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with best parameters and metrics
        """
        if self.use_julia:
            return self._optimize_julia(loss_fn, bounds, method, max_iters, **kwargs)
        else:
            return self._optimize_python(loss_fn, bounds, method, max_iters, **kwargs)
    
    def _optimize_julia(
        self,
        loss_fn: Callable,
        bounds: Dict[str, Tuple[float, float]],
        method: str,
        max_iters: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using Julia backend."""
        try:
            # Convert bounds to Julia format
            lr_range = bounds.get("lr", (1e-6, 1e-2))
            batch_range = bounds.get("batch_size", (8, 128))
            dropout_range = bounds.get("dropout", (0.0, 0.5))
            warmup_range = bounds.get("warmup_steps", (100, 2000))
            
            julia_bounds = TruthGPTCore.HyperparamBounds(
                lr_range=lr_range,
                batch_range=batch_range,
                dropout_range=dropout_range,
                warmup_range=warmup_range
            )
            
            # Wrap loss function for Julia
            def julia_loss_fn(params):
                return loss_fn({
                    "lr": params[:lr],
                    "batch_size": int(params[:batch_size]),
                    "dropout": params[:dropout],
                    "warmup_steps": int(params[:warmup_steps]),
                })
            
            # Run optimization
            result = TruthGPTCore.optimize_hyperparams(
                julia_loss_fn,
                julia_bounds,
                method=method,
                max_iters=max_iters
            )
            
            return {
                "best_params": {
                    "lr": result.best_params[:lr],
                    "batch_size": int(result.best_params[:batch_size]),
                    "dropout": result.best_params[:dropout],
                    "warmup_steps": int(result.best_params[:warmup_steps]),
                },
                "best_loss": result.best_loss,
                "iterations": result.iterations,
                "history": result.history,
            }
        except Exception as e:
            logger.warning(f"Julia optimization failed: {e}, falling back to Python")
            return self._optimize_python(loss_fn, bounds, method, max_iters, **kwargs)
    
    def _optimize_python(
        self,
        loss_fn: Callable,
        bounds: Dict[str, Tuple[float, float]],
        method: str,
        max_iters: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using Python backend."""
        import random
        import numpy as np
        
        best_loss = float('inf')
        best_params = {}
        history = []
        
        random.seed(kwargs.get("seed", 42))
        np.random.seed(kwargs.get("seed", 42))
        
        for i in range(max_iters):
            # Sample parameters
            params = {}
            for param, (min_val, max_val) in bounds.items():
                if method == "random":
                    if isinstance(min_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
                elif method == "grid":
                    t = i / max_iters
                    if isinstance(min_val, int):
                        params[param] = int(min_val + t * (max_val - min_val))
                    else:
                        params[param] = min_val + t * (max_val - min_val)
                else:  # bayesian (simplified)
                    if isinstance(min_val, int):
                        params[param] = random.randint(min_val, max_val)
                    else:
                        params[param] = random.uniform(min_val, max_val)
            
            try:
                loss = loss_fn(params)
                history.append(loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                    logger.info(f"Iter {i}: New best loss = {loss:.4f}")
            except Exception as e:
                logger.warning(f"Iter {i} failed: {e}")
                history.append(float('inf'))
        
        return {
            "best_params": best_params,
            "best_loss": best_loss,
            "iterations": max_iters,
            "history": history,
        }

def create_optimizer(use_julia: bool = True) -> HyperparameterOptimizer:
    """Factory function to create optimizer."""
    return HyperparameterOptimizer(use_julia=use_julia)

__all__ = ["HyperparameterOptimizer", "create_optimizer"]













