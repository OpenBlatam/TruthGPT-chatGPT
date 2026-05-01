"""
Bayesian Optimizer
==================

Bayesian optimization implementation using Gaussian Processes.
"""
import numpy as np
import logging
from typing import Dict, Any, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from .config import HpoConfig, HpoAlgorithm

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """Bayesian optimization implementation"""
    
    def __init__(self, config: HpoConfig):
        self.config = config
        self.gp_model = None
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        self.training_history = []
        logger.info("✅ Bayesian Optimizer initialized")
    
    def create_gp_model(self):
        """Create Gaussian Process model"""
        if self.config.kernel_type == "rbf":
            kernel = RBF(length_scale=1.0)
        elif self.config.kernel_type == "matern":
            kernel = Matern(length_scale=1.0, nu=2.5)
        else:
            kernel = RBF(length_scale=1.0)
        
        kernel += WhiteKernel(noise_level=self.config.alpha)
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.alpha,
            n_restarts_optimizer=10,
            random_state=42
        )
    
    def acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate acquisition function"""
        if self.gp_model is None:
            return np.random.random(X.shape[0])
        
        # Get GP predictions
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        if self.config.acquisition_function == "expected_improvement":
            return self._expected_improvement(mu, sigma)
        elif self.config.acquisition_function == "upper_confidence_bound":
            return self._upper_confidence_bound(mu, sigma)
        elif self.config.acquisition_function == "probability_of_improvement":
            return self._probability_of_improvement(mu, sigma)
        else:
            return self._expected_improvement(mu, sigma)
    
    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function"""
        improvement = mu - self.best_score
        z = improvement / (sigma + 1e-9)
        
        ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)
        return ei
    
    def _upper_confidence_bound(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function"""
        beta = 2.0  # Exploration parameter
        return mu + beta * sigma
    
    def _probability_of_improvement(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Probability of Improvement acquisition function"""
        improvement = mu - self.best_score
        z = improvement / (sigma + 1e-9)
        return self._normal_cdf(z)
    
    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Normal CDF approximation"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def optimize(self, objective_function: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using Bayesian optimization"""
        logger.info("🔍 Optimizing hyperparameters using Bayesian optimization")
        
        # Create GP model
        self.create_gp_model()
        
        # Initialize with random samples
        n_init = min(5, self.config.n_trials // 4)
        for i in range(n_init):
            params = self._sample_params(search_space)
            score = objective_function(params)
            self._update_observations(params, score)
        
        # Bayesian optimization loop
        for trial in range(n_init, self.config.n_trials):
            # Fit GP model
            if len(self.X_observed) > 0:
                X_array = np.array(self.X_observed)
                y_array = np.array(self.y_observed)
                self.gp_model.fit(X_array, y_array)
            
            # Find next point to evaluate
            next_params = self._find_next_point(search_space)
            score = objective_function(next_params)
            
            # Update observations
            self._update_observations(next_params, score)
            
            if trial % 10 == 0:
                logger.info(f"   Trial {trial}: Best score = {self.best_score:.4f}")
        
        optimization_result = {
            'algorithm': HpoAlgorithm.BAYESIAN_OPTIMIZATION.value,
            'n_trials': self.config.n_trials,
            'best_params': self.best_params,
            'best_score': self.best_score,
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
    
    def _find_next_point(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Find next point to evaluate using acquisition function"""
        # Generate candidate points
        n_candidates = 1000
        candidates = []
        
        for _ in range(n_candidates):
            candidate = self._sample_params(search_space)
            candidates.append(list(candidate.values()))
        
        candidates = np.array(candidates)
        
        # Calculate acquisition function values
        acq_values = self.acquisition_function(candidates)
        
        # Select best candidate
        best_idx = np.argmax(acq_values)
        best_candidate = candidates[best_idx]
        
        # Convert back to parameter dictionary
        param_names = list(search_space.keys())
        next_params = {name: best_candidate[i] for i, name in enumerate(param_names)}
        
        return next_params
    
    def _update_observations(self, params: Dict[str, Any], score: float):
        """Update observations"""
        # Convert params to array
        param_array = list(params.values())
        self.X_observed.append(param_array)
        self.y_observed.append(score)
        
        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
        
        # Store training history
        self.training_history.append({
            'trial': len(self.training_history),
            'params': params,
            'score': score,
            'best_score': self.best_score
        })

