"""
Multi-Objective Optimizer
=========================

Multi-objective hyperparameter optimization implementation.
"""
import numpy as np
import logging
from typing import Dict, Any, List, Callable
from .config import HpoConfig

logger = logging.getLogger(__name__)

class MultiObjectiveOptimizer:
    """Multi-objective optimization implementation"""
    
    def __init__(self, config: HpoConfig):
        self.config = config
        self.pareto_front = []
        self.training_history = []
        logger.info("✅ Multi-Objective Optimizer initialized")
    
    def optimize(self, objective_function: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using multi-objective optimization"""
        logger.info("🎯 Optimizing hyperparameters using multi-objective optimization")
        
        # Multi-objective optimization loop
        for trial in range(self.config.n_trials):
            # Sample parameters
            params = self._sample_params(search_space)
            
            # Evaluate objectives
            objectives = objective_function(params)
            if not isinstance(objectives, (list, tuple)):
                objectives = [objectives]
            
            # Update Pareto front
            self._update_pareto_front(params, objectives)
            
            # Store training history
            self.training_history.append({
                'trial': trial,
                'params': params,
                'objectives': objectives,
                'pareto_front_size': len(self.pareto_front)
            })
            
            if trial % 10 == 0:
                logger.info(f"   Trial {trial}: Pareto front size = {len(self.pareto_front)}")
        
        optimization_result = {
            'algorithm': 'multi_objective',
            'n_trials': self.config.n_trials,
            'pareto_front': self.pareto_front,
            'training_history': self.training_history,
            'status': 'success'
        }
        
        return optimization_result
    
    def _sample_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from search space"""
        params = {}
        for param_name, param_range in search_space.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = np.random.choice(param_range)
            else:
                params[param_name] = param_range
        
        return params
    
    def _update_pareto_front(self, params: Dict[str, Any], objectives: List[float]):
        """Update Pareto front"""
        # Check if current solution is dominated
        is_dominated = False
        dominated_indices = []
        
        for i, (front_params, front_objectives) in enumerate(self.pareto_front):
            if self._dominates(front_objectives, objectives):
                is_dominated = True
                break
            elif self._dominates(objectives, front_objectives):
                dominated_indices.append(i)
        
        # Remove dominated solutions
        for i in reversed(dominated_indices):
            self.pareto_front.pop(i)
        
        # Add current solution if not dominated
        if not is_dominated:
            self.pareto_front.append((params, objectives))
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2"""
        return all(o1 >= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 > o2 for o1, o2 in zip(obj1, obj2))

