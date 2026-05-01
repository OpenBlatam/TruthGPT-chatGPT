"""
Bayesian Optimizer System
=========================

Main orchestrator for Bayesian optimization processes.
"""
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Callable

from .config import BayesianOptimizationConfig
from .models import GaussianProcessModel
from .acquisition import AcquisitionFunctionOptimizer
from .optimization import MultiObjectiveOptimizer, ConstrainedOptimizer

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """Main Bayesian optimizer"""
    
    def __init__(self, config: BayesianOptimizationConfig):
        self.config = config
        self.gp_model = GaussianProcessModel(config)
        self.acquisition_optimizer = AcquisitionFunctionOptimizer(config)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(config)
        self.constrained_optimizer = ConstrainedOptimizer(config)
        self.optimization_history = []
        self.X_observed = []
        self.y_observed = []
        
        logger.info("✅ Bayesian Optimizer initialized")
    
    def optimize(self, objective_function: Callable, 
                bounds: List[Tuple[float, float]],
                constraint_functions: List[Callable] = None) -> Dict[str, Any]:
        logger.info(f"🚀 Optimizing using Bayesian optimization")
        start_time = time.time()
        results = {'stages': {}}
        
        if self.config.enable_multi_objective:
            results['stages']['multi_objective'] = self.multi_objective_optimizer.optimize_multi_objective(objective_function, bounds)
        elif self.config.enable_constraints and constraint_functions:
            results['stages']['constrained'] = self.constrained_optimizer.optimize_constrained(objective_function, constraint_functions, bounds)
        else:
            results['stages']['standard'] = self._standard_bayesian_optimization(objective_function, bounds)
            
        results['duration'] = time.time() - start_time
        return results
    
    def _standard_bayesian_optimization(self, objective_function: Callable, 
                                      bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        # Initial points
        for _ in range(self.config.n_initial_points):
            x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            self.X_observed.append(x)
            self.y_observed.append(objective_function(x))
            
        for i in range(self.config.n_initial_points, self.config.n_iterations):
            self.gp_model.fit(np.array(self.X_observed), np.array(self.y_observed))
            next_point = self.acquisition_optimizer.optimize_acquisition(self.gp_model, bounds)
            self.X_observed.append(next_point)
            self.y_observed.append(objective_function(next_point))
            
        best_idx = np.argmax(self.y_observed)
        return {
            'best_point': self.X_observed[best_idx],
            'best_value': self.y_observed[best_idx],
            'status': 'success'
        }

    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        return f"Bayesian Optimization completed in {results.get('duration', 0):.2f}s"

