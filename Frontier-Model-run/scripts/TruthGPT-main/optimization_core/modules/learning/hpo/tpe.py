"""
TPE Optimizer
=============

Tree-structured Parzen Estimator implementation.
"""
import numpy as np
import logging
from typing import Dict, Any, Callable
from .config import HpoConfig, HpoAlgorithm

logger = logging.getLogger(__name__)

class TPEOptimizer:
    """Tree-structured Parzen Estimator implementation"""
    
    def __init__(self, config: HpoConfig):
        self.config = config
        self.trials = []
        self.best_params = None
        self.best_score = -np.inf
        self.training_history = []
        logger.info("✅ TPE Optimizer initialized")
    
    def optimize(self, objective_function: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using TPE"""
        logger.info("🌳 Optimizing hyperparameters using TPE")
        
        # TPE optimization loop
        for trial in range(self.config.n_trials):
            # Sample parameters
            if trial < self.config.n_startup_trials:
                # Random sampling for startup trials
                params = self._random_sample(search_space)
            else:
                # TPE sampling
                params = self._tpe_sample(search_space)
            
            # Evaluate objective
            score = objective_function(params)
            
            # Store trial
            self.trials.append({
                'trial': trial,
                'params': params,
                'score': score
            })
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            
            # Store training history
            self.training_history.append({
                'trial': trial,
                'params': params,
                'score': score,
                'best_score': self.best_score
            })
            
            if trial % 10 == 0:
                logger.info(f"   Trial {trial}: Best score = {self.best_score:.4f}")
        
        optimization_result = {
            'algorithm': HpoAlgorithm.TPE.value,
            'n_trials': self.config.n_trials,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'training_history': self.training_history,
            'status': 'success'
        }
        
        return optimization_result
    
    def _random_sample(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Random sampling"""
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
    
    def _tpe_sample(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """TPE sampling"""
        # Split trials into good and bad
        scores = [trial['score'] for trial in self.trials]
        threshold = np.percentile(scores, self.config.pruning_percentile)
        
        good_trials = [trial for trial in self.trials if trial['score'] >= threshold]
        bad_trials = [trial for trial in self.trials if trial['score'] < threshold]
        
        # Sample from good trials
        if len(good_trials) > 0:
            good_trial = np.random.choice(good_trials)
            params = good_trial['params'].copy()
            
            # Add some noise
            for param_name, param_range in search_space.items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        noise = np.random.randint(-1, 2)
                        params[param_name] = max(param_range[0], 
                                               min(param_range[1], params[param_name] + noise))
                    else:
                        noise = np.random.normal(0, 0.1)
                        params[param_name] = max(param_range[0], 
                                               min(param_range[1], params[param_name] + noise))
        else:
            params = self._random_sample(search_space)
        
        return params

