"""
Acquisition Function Optimization
================================

Optimization of acquisition functions for Bayesian optimization.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from .config import BayesianOptimizationConfig
from .enums import AcquisitionFunction
from .models import GaussianProcessModel

logger = logging.getLogger(__name__)

class AcquisitionFunctionOptimizer:
    """Acquisition function optimizer"""
    
    def __init__(self, config: BayesianOptimizationConfig):
        self.config = config
        self.acquisition_history = []
        logger.info("✅ Acquisition Function Optimizer initialized")
    
    def optimize_acquisition(self, gp_model: GaussianProcessModel, 
                           bounds: List[Tuple[float, float]], 
                           n_candidates: int = None) -> np.ndarray:
        """Optimize acquisition function"""
        logger.info(f"🎯 Optimizing acquisition function: {self.config.acquisition_function.value}")
        
        if n_candidates is None:
            n_candidates = self.config.n_candidates
        
        # Generate candidate points
        candidates = self._generate_candidates(bounds, n_candidates)
        
        # Calculate acquisition function values
        acq_values = self._calculate_acquisition_function(gp_model, candidates)
        
        # Find best candidate
        best_idx = np.argmax(acq_values)
        best_candidate = candidates[best_idx]
        
        # Store acquisition history
        self.acquisition_history.append({
            'best_candidate': best_candidate,
            'best_value': acq_values[best_idx]
        })
        
        return best_candidate
    
    def _generate_candidates(self, bounds: List[Tuple[float, float]], 
                           n_candidates: int) -> np.ndarray:
        """Generate candidate points"""
        n_dims = len(bounds)
        candidates = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_candidates, n_dims)
        )
        return candidates
    
    def _calculate_acquisition_function(self, gp_model: GaussianProcessModel, 
                                      candidates: np.ndarray) -> np.ndarray:
        """Calculate acquisition function values"""
        if self.config.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(gp_model, candidates)
        elif self.config.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(gp_model, candidates)
        elif self.config.acquisition_function == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT:
            return self._probability_of_improvement(gp_model, candidates)
        elif self.config.acquisition_function == AcquisitionFunction.ENTROPY_SEARCH:
            return self._entropy_search(gp_model, candidates)
        elif self.config.acquisition_function == AcquisitionFunction.KNOWLEDGE_GRADIENT:
            return self._knowledge_gradient(gp_model, candidates)
        elif self.config.acquisition_function == AcquisitionFunction.MUTUAL_INFORMATION:
            return self._mutual_information(gp_model, candidates)
        elif self.config.acquisition_function == AcquisitionFunction.THOMPSON_SAMPLING:
            return self._thompson_sampling(gp_model, candidates)
        else:
            return self._expected_improvement(gp_model, candidates)
    
    def _expected_improvement(self, gp_model: GaussianProcessModel, 
                            candidates: np.ndarray) -> np.ndarray:
        mean, std = gp_model.predict(candidates, return_std=True)
        best_value = np.max(gp_model.y_train)
        improvement = mean - best_value - self.config.acquisition_xi
        z = improvement / (std + 1e-9)
        ei = improvement * self._normal_cdf(z) + std * self._normal_pdf(z)
        return ei
    
    def _upper_confidence_bound(self, gp_model: GaussianProcessModel, 
                               candidates: np.ndarray) -> np.ndarray:
        mean, std = gp_model.predict(candidates, return_std=True)
        return mean + self.config.acquisition_kappa * std
    
    def _probability_of_improvement(self, gp_model: GaussianProcessModel, 
                                   candidates: np.ndarray) -> np.ndarray:
        mean, std = gp_model.predict(candidates, return_std=True)
        best_value = np.max(gp_model.y_train)
        improvement = mean - best_value - self.config.acquisition_xi
        z = improvement / (std + 1e-9)
        return self._normal_cdf(z)
    
    def _entropy_search(self, gp_model: GaussianProcessModel, 
                       candidates: np.ndarray) -> np.ndarray:
        mean, std = gp_model.predict(candidates, return_std=True)
        return 0.5 * np.log(2 * np.pi * np.e * std**2 + 1e-9)
    
    def _knowledge_gradient(self, gp_model: GaussianProcessModel, 
                          candidates: np.ndarray) -> np.ndarray:
        mean, std = gp_model.predict(candidates, return_std=True)
        return mean + self.config.acquisition_beta * std
    
    def _mutual_information(self, gp_model: GaussianProcessModel, 
                          candidates: np.ndarray) -> np.ndarray:
        mean, std = gp_model.predict(candidates, return_std=True)
        return 0.5 * np.log(1 + std**2)
    
    def _thompson_sampling(self, gp_model: GaussianProcessModel, 
                          candidates: np.ndarray) -> np.ndarray:
        samples = gp_model.sample_y(candidates, n_samples=1)
        return samples.flatten()
    
    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

