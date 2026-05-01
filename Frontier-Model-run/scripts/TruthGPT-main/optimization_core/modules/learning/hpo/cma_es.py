"""
CMA-ES Optimizer
================

Covariance Matrix Adaptation Evolution Strategy implementation.
"""
import numpy as np
import logging
from typing import Dict, Any, Callable, List
from .config import HpoConfig, HpoAlgorithm

logger = logging.getLogger(__name__)

class CMAESOptimizer:
    """CMA-ES implementation"""
    
    def __init__(self, config: HpoConfig):
        self.config = config
        self.mean = None
        self.covariance = None
        self.best_params = None
        self.best_score = -np.inf
        self.training_history = []
        logger.info("✅ CMA-ES Optimizer initialized")
    
    def optimize(self, objective_function: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using CMA-ES"""
        logger.info("🎯 Optimizing hyperparameters using CMA-ES")
        
        # Initialize CMA-ES
        param_names = list(search_space.keys())
        n_params = len(param_names)
        
        # Initialize mean and covariance
        self.mean = np.random.random(n_params)
        self.covariance = np.eye(n_params)
        
        # CMA-ES parameters
        sigma = 0.3
        mu = self.config.population_size // 2
        lambda_pop = self.config.population_size
        
        # Evolution loop
        for generation in range(self.config.n_generations):
            # Generate population
            population = []
            for _ in range(lambda_pop):
                individual = np.random.multivariate_normal(self.mean, sigma**2 * self.covariance)
                params = {param_names[i]: individual[i] for i in range(n_params)}
                population.append(params)
            
            # Evaluate population
            fitness_scores = []
            for params in population:
                score = objective_function(params)
                fitness_scores.append(score)
                
                # Update best
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
            
            # Select best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            selected_individuals = [population[i] for i in sorted_indices[:mu]]
            selected_scores = [fitness_scores[i] for i in sorted_indices[:mu]]
            
            # Update mean and covariance
            self._update_cma_es(selected_individuals, selected_scores, sigma)
            
            # Store generation history
            self.training_history.append({
                'generation': generation,
                'best_score': self.best_score,
                'avg_score': np.mean(fitness_scores),
                'best_params': self.best_params
            })
            
            if generation % 5 == 0:
                logger.info(f"   Generation {generation}: Best score = {self.best_score:.4f}")
        
        optimization_result = {
            'algorithm': HpoAlgorithm.CMA_ES.value,
            'n_generations': self.config.n_generations,
            'population_size': self.config.population_size,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'training_history': self.training_history,
            'status': 'success'
        }
        
        return optimization_result
    
    def _update_cma_es(self, selected_individuals: List[Dict[str, Any]], 
                      selected_scores: List[float], sigma: float):
        """Update CMA-ES parameters"""
        param_names = list(selected_individuals[0].keys())
        # n_params = len(param_names) # Removing unused variable
        
        # Convert to arrays
        selected_array = np.array([[ind[param] for param in param_names] 
                                  for ind in selected_individuals])
        
        # Update mean
        weights = np.array(selected_scores)
        weights = weights / np.sum(weights)  # Normalize weights
        self.mean = np.average(selected_array, axis=0, weights=weights)
        
        # Update covariance (simplified)
        centered = selected_array - self.mean
        self.covariance = np.cov(centered.T)

