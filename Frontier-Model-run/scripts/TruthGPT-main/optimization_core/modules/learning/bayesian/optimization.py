"""
Advanced Optimization Strategies
===============================

Multi-objective and constrained optimization strategies.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from .config import BayesianOptimizationConfig

logger = logging.getLogger(__name__)

class MultiObjectiveOptimizer:
    """Multi-objective optimization"""
    
    def __init__(self, config: BayesianOptimizationConfig):
        self.config = config
        self.pareto_front = []
        self.optimization_history = []
        logger.info("✅ Multi-Objective Optimizer initialized")
    
    def optimize_multi_objective(self, objective_function: Callable, 
                                bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        logger.info("🎯 Optimizing multiple objectives")
        self.pareto_front = []
        for iteration in range(self.config.n_iterations):
            candidates = self._generate_candidates(bounds, self.config.n_candidates)
            objectives = np.array([objective_function(c) for c in candidates])
            self._update_pareto_front(candidates, objectives)
            
            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}: Pareto front size = {len(self.pareto_front)}")
                
        return {'pareto_front': self.pareto_front, 'status': 'success'}
    
    def _generate_candidates(self, bounds: List[Tuple[float, float]], n_candidates: int) -> np.ndarray:
        n_dims = len(bounds)
        return np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_candidates, n_dims)
        )
    
    def _update_pareto_front(self, candidates: np.ndarray, objectives: np.ndarray):
        for candidate, obj_values in zip(candidates, objectives):
            is_dominated = False
            dominated_indices = []
            for j, (_, front_objectives) in enumerate(self.pareto_front):
                if self._dominates(front_objectives, obj_values):
                    is_dominated = True
                    break
                elif self._dominates(obj_values, front_objectives):
                    dominated_indices.append(j)
            
            for idx in reversed(dominated_indices):
                self.pareto_front.pop(idx)
            if not is_dominated:
                self.pareto_front.append((candidate, obj_values))
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

class ConstrainedOptimizer:
    """Constrained optimization"""
    
    def __init__(self, config: BayesianOptimizationConfig):
        self.config = config
        logger.info("✅ Constrained Optimizer initialized")
    
    def optimize_constrained(self, objective_function: Callable, 
                           constraint_functions: List[Callable],
                           bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        logger.info("🎯 Optimizing with constraints")
        feasible_points = []
        feasible_objectives = []
        for iteration in range(self.config.n_iterations):
            candidates = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(self.config.n_candidates, len(bounds))
            )
            for candidate in candidates:
                if all(cf(candidate) <= 0 for cf in constraint_functions):
                    feasible_points.append(candidate)
                    feasible_objectives.append(objective_function(candidate))
                    
        return {
            'feasible_points': np.array(feasible_points),
            'feasible_objectives': np.array(feasible_objectives),
            'status': 'success'
        }

