"""
Hyperparameter Optimization Module
Advanced hyperparameter optimization capabilities for TruthGPT
"""

import torch
import torch.nn as nn
import numpy as np
import random
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from abc import ABC, abstractmethod
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class OptimizationAlgorithm(Enum):
    """Hyperparameter optimization algorithms."""
    BAYESIAN = "bayesian"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    OPTUNA = "optuna"
    HYPEROPT = "hyperopt"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"

class SearchSpace(BaseModel):
    """Hyperparameter search space definition."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    learning_rate: Tuple[float, float] = (1e-5, 1e-1)
    batch_size: List[int] = Field(default_factory=lambda: [16, 32, 64, 128, 256])
    weight_decay: Tuple[float, float] = (1e-6, 1e-2)
    dropout_rate: Tuple[float, float] = (0.0, 0.5)
    hidden_dim: List[int] = Field(default_factory=lambda: [64, 128, 256, 512, 1024])
    num_layers: Tuple[int, int] = (2, 12)
    activation: List[str] = Field(default_factory=lambda: ['relu', 'gelu', 'swish', 'mish'])
    optimizer: List[str] = Field(default_factory=lambda: ['adam', 'adamw', 'sgd', 'rmsprop'])
    scheduler: List[str] = Field(default_factory=lambda: ['cosine', 'step', 'exponential', 'plateau'])
    warmup_steps: Tuple[int, int] = (0, 1000)
    gradient_clip: Tuple[float, float] = (0.0, 5.0)

class HyperparameterConfig(BaseModel):
    """Configuration for hyperparameter optimization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.BAYESIAN
    search_space: SearchSpace = Field(default_factory=SearchSpace)
    max_trials: int = 100
    max_evaluations: int = 50
    timeout: float = 3600.0  # 1 hour
    n_jobs: int = 1
    random_seed: int = 42
    early_stopping_patience: int = 5
    performance_threshold: float = 0.8
    enable_pruning: bool = True
    enable_parallel_evaluation: bool = False

class HyperparameterTrial(BaseModel):
    """Hyperparameter trial result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    trial_id: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    training_time: float = 0.0
    evaluation_time: float = 0.0
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None

class BaseHyperparameterOptimizer(ABC):
    """Base class for hyperparameter optimization algorithms."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.trials: List[HyperparameterTrial] = []
        self.best_trial: Optional[HyperparameterTrial] = None
        self.optimization_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    @abstractmethod
    def optimize(self, objective_function: Callable) -> HyperparameterTrial:
        """Optimize hyperparameters."""
        pass
    
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest next hyperparameters to try."""
        pass
    
    def update_trial(self, trial: HyperparameterTrial, metrics: Dict[str, float]):
        """Update trial with evaluation results."""
        trial.performance_metrics.update(metrics)
        trial.status = "completed"
        trial.completed_at = time.time()
        
        # Update best trial
        if self.best_trial is None or self._is_better_trial(trial, self.best_trial):
            self.best_trial = trial
        
        self.trials.append(trial)
        self._update_optimization_history(trial)
    
    def _is_better_trial(self, trial1: HyperparameterTrial, trial2: HyperparameterTrial) -> bool:
        """Check if trial1 is better than trial2."""
        # Use accuracy as primary metric
        acc1 = trial1.performance_metrics.get('accuracy', 0.0)
        acc2 = trial2.performance_metrics.get('accuracy', 0.0)
        return acc1 > acc2
    
    def _update_optimization_history(self, trial: HyperparameterTrial):
        """Update optimization history."""
        history_entry = {
            'trial_id': trial.trial_id,
            'hyperparameters': trial.hyperparameters,
            'performance': trial.performance_metrics,
            'timestamp': trial.completed_at
        }
        self.optimization_history.append(history_entry)
    
    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters found."""
        if self.best_trial:
            return self.best_trial.hyperparameters
        return None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.trials:
            return {}
        
        completed_trials = [t for t in self.trials if t.status == "completed"]
        if not completed_trials:
            return {}
        
        accuracies = [t.performance_metrics.get('accuracy', 0.0) for t in completed_trials]
        training_times = [t.training_time for t in completed_trials]
        
        return {
            'total_trials': len(self.trials),
            'completed_trials': len(completed_trials),
            'best_accuracy': max(accuracies),
            'average_accuracy': sum(accuracies) / len(accuracies),
            'average_training_time': sum(training_times) / len(training_times),
            'optimization_time': time.time() - self.start_time,
            'best_hyperparameters': self.get_best_hyperparameters()
        }

class BayesianOptimizer(BaseHyperparameterOptimizer):
    """Bayesian optimization for hyperparameters."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
        self.acquisition_function = "expected_improvement"
        self.gaussian_process = None
    
    def optimize(self, objective_function: Callable) -> HyperparameterTrial:
        """Bayesian optimization."""
        self.logger.info("Starting Bayesian hyperparameter optimization")
        
        for trial_num in range(self.config.max_trials):
            # Suggest hyperparameters
            hyperparameters = self._suggest_hyperparameters_bayesian()
            
            # Create trial
            trial = HyperparameterTrial(
                trial_id=f"bayesian_{trial_num}",
                hyperparameters=hyperparameters,
                status="running"
            )
            
            # Evaluate objective function
            try:
                start_time = time.time()
                metrics = objective_function(hyperparameters)
                trial.training_time = time.time() - start_time
                
                # Update trial
                self.update_trial(trial, metrics)
                
                self.logger.info(f"Trial {trial_num + 1}: Accuracy = {metrics.get('accuracy', 0.0):.4f}")
                
                # Check early stopping
                if self._should_stop_early():
                    break
                    
            except Exception as e:
                self.logger.error(f"Trial {trial_num + 1} failed: {e}")
                trial.status = "failed"
                trial.completed_at = time.time()
                self.trials.append(trial)
        
        return self.best_trial or self.trials[0]
    
    def _suggest_hyperparameters_bayesian(self) -> Dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization."""
        if len(self.trials) < 5:
            # Random exploration for first few trials
            return self._random_sample()
        
        # Use acquisition function to suggest next point
        # Simplified implementation
        return self._acquisition_sample()
    
    def _random_sample(self) -> Dict[str, Any]:
        """Random hyperparameter sampling."""
        return {
            'learning_rate': random.uniform(*self.config.search_space.learning_rate),
            'batch_size': random.choice(self.config.search_space.batch_size),
            'weight_decay': random.uniform(*self.config.search_space.weight_decay),
            'dropout_rate': random.uniform(*self.config.search_space.dropout_rate),
            'hidden_dim': random.choice(self.config.search_space.hidden_dim),
            'num_layers': random.randint(*self.config.search_space.num_layers),
            'activation': random.choice(self.config.search_space.activation),
            'optimizer': random.choice(self.config.search_space.optimizer),
            'scheduler': random.choice(self.config.search_space.scheduler),
            'warmup_steps': random.randint(*self.config.search_space.warmup_steps),
            'gradient_clip': random.uniform(*self.config.search_space.gradient_clip)
        }
    
    def _acquisition_sample(self) -> Dict[str, Any]:
        """Sample using acquisition function."""
        # Simplified acquisition function implementation
        # In practice, this would use Gaussian Process and acquisition function
        
        # Find promising regions based on previous trials
        if self.trials:
            best_trial = max(self.trials, key=lambda t: t.performance_metrics.get('accuracy', 0.0))
            
            # Perturb best hyperparameters
            hyperparameters = best_trial.hyperparameters.copy()
            
            # Add some noise to explore around best point
            hyperparameters['learning_rate'] *= random.uniform(0.8, 1.2)
            hyperparameters['weight_decay'] *= random.uniform(0.8, 1.2)
            hyperparameters['dropout_rate'] = max(0.0, min(0.5, 
                hyperparameters['dropout_rate'] + random.uniform(-0.1, 0.1)))
            
            return hyperparameters
        
        return self._random_sample()
    
    def _should_stop_early(self) -> bool:
        """Check if optimization should stop early."""
        if len(self.trials) < 10:
            return False
        
        # Check if best accuracy hasn't improved recently
        recent_trials = self.trials[-10:]
        recent_accuracies = [t.performance_metrics.get('accuracy', 0.0) for t in recent_trials if t.status == "completed"]
        
        if len(recent_accuracies) >= 5:
            return max(recent_accuracies) - min(recent_accuracies) < 0.01
        
        return False

class RandomSearchOptimizer(BaseHyperparameterOptimizer):
    """Random search hyperparameter optimization."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
    
    def optimize(self, objective_function: Callable) -> HyperparameterTrial:
        """Random search optimization."""
        self.logger.info("Starting random search hyperparameter optimization")
        
        for trial_num in range(self.config.max_trials):
            # Random hyperparameters
            hyperparameters = self._random_sample()
            
            # Create trial
            trial = HyperparameterTrial(
                trial_id=f"random_{trial_num}",
                hyperparameters=hyperparameters,
                status="running"
            )
            
            # Evaluate objective function
            try:
                start_time = time.time()
                metrics = objective_function(hyperparameters)
                trial.training_time = time.time() - start_time
                
                # Update trial
                self.update_trial(trial, metrics)
                
                self.logger.info(f"Trial {trial_num + 1}: Accuracy = {metrics.get('accuracy', 0.0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Trial {trial_num + 1} failed: {e}")
                trial.status = "failed"
                trial.completed_at = time.time()
                self.trials.append(trial)
        
        return self.best_trial or self.trials[0]
    
    def _random_sample(self) -> Dict[str, Any]:
        """Random hyperparameter sampling."""
        return {
            'learning_rate': random.uniform(*self.config.search_space.learning_rate),
            'batch_size': random.choice(self.config.search_space.batch_size),
            'weight_decay': random.uniform(*self.config.search_space.weight_decay),
            'dropout_rate': random.uniform(*self.config.search_space.dropout_rate),
            'hidden_dim': random.choice(self.config.search_space.hidden_dim),
            'num_layers': random.randint(*self.config.search_space.num_layers),
            'activation': random.choice(self.config.search_space.activation),
            'optimizer': random.choice(self.config.search_space.optimizer),
            'scheduler': random.choice(self.config.search_space.scheduler),
            'warmup_steps': random.randint(*self.config.search_space.warmup_steps),
            'gradient_clip': random.uniform(*self.config.search_space.gradient_clip)
        }

class GridSearchOptimizer(BaseHyperparameterOptimizer):
    """Grid search hyperparameter optimization."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
        self.grid_points = self._generate_grid_points()
        self.current_index = 0
    
    def optimize(self, objective_function: Callable) -> HyperparameterTrial:
        """Grid search optimization."""
        self.logger.info("Starting grid search hyperparameter optimization")
        
        for trial_num, hyperparameters in enumerate(self.grid_points):
            if trial_num >= self.config.max_trials:
                break
            
            # Create trial
            trial = HyperparameterTrial(
                trial_id=f"grid_{trial_num}",
                hyperparameters=hyperparameters,
                status="running"
            )
            
            # Evaluate objective function
            try:
                start_time = time.time()
                metrics = objective_function(hyperparameters)
                trial.training_time = time.time() - start_time
                
                # Update trial
                self.update_trial(trial, metrics)
                
                self.logger.info(f"Trial {trial_num + 1}: Accuracy = {metrics.get('accuracy', 0.0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Trial {trial_num + 1} failed: {e}")
                trial.status = "failed"
                trial.completed_at = time.time()
                self.trials.append(trial)
        
        return self.best_trial or self.trials[0]
    
    def _generate_grid_points(self) -> List[Dict[str, Any]]:
        """Generate grid search points."""
        grid_points = []
        
        # Create grid for key hyperparameters
        learning_rates = np.logspace(
            np.log10(self.config.search_space.learning_rate[0]),
            np.log10(self.config.search_space.learning_rate[1]),
            num=5
        )
        
        batch_sizes = self.config.search_space.batch_size[:3]  # Limit for grid search
        hidden_dims = self.config.search_space.hidden_dim[:3]
        activations = self.config.search_space.activation[:2]
        optimizers = self.config.search_space.optimizer[:2]
        
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for hidden_dim in hidden_dims:
                    for activation in activations:
                        for optimizer in optimizers:
                            grid_points.append({
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'weight_decay': 1e-4,
                                'dropout_rate': 0.1,
                                'hidden_dim': hidden_dim,
                                'num_layers': 4,
                                'activation': activation,
                                'optimizer': optimizer,
                                'scheduler': 'cosine',
                                'warmup_steps': 100,
                                'gradient_clip': 1.0
                            })
        
        return grid_points

class OptunaOptimizer(BaseHyperparameterOptimizer):
    """Optuna-based hyperparameter optimization."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
        self.study = None
    
    def optimize(self, objective_function: Callable) -> HyperparameterTrial:
        """Optuna optimization."""
        self.logger.info("Starting Optuna hyperparameter optimization")
        
        # Simplified Optuna implementation
        for trial_num in range(self.config.max_trials):
            # Suggest hyperparameters using Optuna-like approach
            hyperparameters = self._suggest_hyperparameters_optuna()
            
            # Create trial
            trial = HyperparameterTrial(
                trial_id=f"optuna_{trial_num}",
                hyperparameters=hyperparameters,
                status="running"
            )
            
            # Evaluate objective function
            try:
                start_time = time.time()
                metrics = objective_function(hyperparameters)
                trial.training_time = time.time() - start_time
                
                # Update trial
                self.update_trial(trial, metrics)
                
                self.logger.info(f"Trial {trial_num + 1}: Accuracy = {metrics.get('accuracy', 0.0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Trial {trial_num + 1} failed: {e}")
                trial.status = "failed"
                trial.completed_at = time.time()
                self.trials.append(trial)
        
        return self.best_trial or self.trials[0]
    
    def _suggest_hyperparameters_optuna(self) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna-like approach."""
        # Simplified Optuna suggestion
        return {
            'learning_rate': random.uniform(*self.config.search_space.learning_rate),
            'batch_size': random.choice(self.config.search_space.batch_size),
            'weight_decay': random.uniform(*self.config.search_space.weight_decay),
            'dropout_rate': random.uniform(*self.config.search_space.dropout_rate),
            'hidden_dim': random.choice(self.config.search_space.hidden_dim),
            'num_layers': random.randint(*self.config.search_space.num_layers),
            'activation': random.choice(self.config.search_space.activation),
            'optimizer': random.choice(self.config.search_space.optimizer),
            'scheduler': random.choice(self.config.search_space.scheduler),
            'warmup_steps': random.randint(*self.config.search_space.warmup_steps),
            'gradient_clip': random.uniform(*self.config.search_space.gradient_clip)
        }

class HyperoptOptimizer(BaseHyperparameterOptimizer):
    """Hyperopt-based hyperparameter optimization."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
        self.tpe = None  # Tree-structured Parzen Estimator
    
    def optimize(self, objective_function: Callable) -> HyperparameterTrial:
        """Hyperopt optimization."""
        self.logger.info("Starting Hyperopt hyperparameter optimization")
        
        # Simplified Hyperopt implementation
        for trial_num in range(self.config.max_trials):
            # Suggest hyperparameters using Hyperopt-like approach
            hyperparameters = self._suggest_hyperparameters_hyperopt()
            
            # Create trial
            trial = HyperparameterTrial(
                trial_id=f"hyperopt_{trial_num}",
                hyperparameters=hyperparameters,
                status="running"
            )
            
            # Evaluate objective function
            try:
                start_time = time.time()
                metrics = objective_function(hyperparameters)
                trial.training_time = time.time() - start_time
                
                # Update trial
                self.update_trial(trial, metrics)
                
                self.logger.info(f"Trial {trial_num + 1}: Accuracy = {metrics.get('accuracy', 0.0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Trial {trial_num + 1} failed: {e}")
                trial.status = "failed"
                trial.completed_at = time.time()
                self.trials.append(trial)
        
        return self.best_trial or self.trials[0]
    
    def _suggest_hyperparameters_hyperopt(self) -> Dict[str, Any]:
        """Suggest hyperparameters using Hyperopt-like approach."""
        # Simplified Hyperopt suggestion
        return {
            'learning_rate': random.uniform(*self.config.search_space.learning_rate),
            'batch_size': random.choice(self.config.search_space.batch_size),
            'weight_decay': random.uniform(*self.config.search_space.weight_decay),
            'dropout_rate': random.uniform(*self.config.search_space.dropout_rate),
            'hidden_dim': random.choice(self.config.search_space.hidden_dim),
            'num_layers': random.randint(*self.config.search_space.num_layers),
            'activation': random.choice(self.config.search_space.activation),
            'optimizer': random.choice(self.config.search_space.optimizer),
            'scheduler': random.choice(self.config.search_space.scheduler),
            'warmup_steps': random.randint(*self.config.search_space.warmup_steps),
            'gradient_clip': random.uniform(*self.config.search_space.gradient_clip)
        }

class TruthGPTHyperparameterManager:
    """TruthGPT Hyperparameter Optimization Manager."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.optimizer = self._create_optimizer()
        self.optimization_results: List[HyperparameterTrial] = []
    
    def _create_optimizer(self) -> BaseHyperparameterOptimizer:
        """Create optimizer based on algorithm."""
        if self.config.algorithm == OptimizationAlgorithm.BAYESIAN:
            return BayesianOptimizer(self.config)
        elif self.config.algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
            return RandomSearchOptimizer(self.config)
        elif self.config.algorithm == OptimizationAlgorithm.GRID_SEARCH:
            return GridSearchOptimizer(self.config)
        elif self.config.algorithm == OptimizationAlgorithm.OPTUNA:
            return OptunaOptimizer(self.config)
        elif self.config.algorithm == OptimizationAlgorithm.HYPEROPT:
            return HyperoptOptimizer(self.config)
        else:
            return BayesianOptimizer(self.config)  # Default
    
    def optimize_hyperparameters(
        self,
        objective_function: Callable,
        task_name: str = "default"
    ) -> HyperparameterTrial:
        """Optimize hyperparameters for a task."""
        self.logger.info(f"Starting hyperparameter optimization for task: {task_name}")
        
        start_time = time.time()
        best_trial = self.optimizer.optimize(objective_function)
        optimization_time = time.time() - start_time
        
        # Add metadata
        best_trial.performance_metrics['optimization_time'] = optimization_time
        best_trial.performance_metrics['task_name'] = task_name
        
        self.optimization_results.append(best_trial)
        
        self.logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f}s")
        self.logger.info(f"Best accuracy: {best_trial.performance_metrics.get('accuracy', 0.0):.4f}")
        
        return best_trial
    
    def get_optimization_results(self) -> List[HyperparameterTrial]:
        """Get all optimization results."""
        return self.optimization_results.copy()
    
    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """Get best hyperparameters from all optimizations."""
        if not self.optimization_results:
            return None
        
        best_trial = max(self.optimization_results, 
                        key=lambda t: t.performance_metrics.get('accuracy', 0.0))
        return best_trial.hyperparameters
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_results:
            return {}
        
        accuracies = [r.performance_metrics.get('accuracy', 0.0) for r in self.optimization_results]
        training_times = [r.training_time for r in self.optimization_results]
        
        return {
            'total_optimizations': len(self.optimization_results),
            'best_accuracy': max(accuracies),
            'average_accuracy': sum(accuracies) / len(accuracies),
            'average_training_time': sum(training_times) / len(training_times),
            'optimizer_summary': self.optimizer.get_optimization_summary()
        }

# Factory functions
def create_hyperparameter_manager(config: HyperparameterConfig) -> TruthGPTHyperparameterManager:
    """Create hyperparameter manager."""
    return TruthGPTHyperparameterManager(config)

def create_bayesian_optimizer(config: HyperparameterConfig) -> BayesianOptimizer:
    """Create Bayesian optimizer."""
    config.algorithm = OptimizationAlgorithm.BAYESIAN
    return BayesianOptimizer(config)

def create_optuna_optimizer(config: HyperparameterConfig) -> OptunaOptimizer:
    """Create Optuna optimizer."""
    config.algorithm = OptimizationAlgorithm.OPTUNA
    return OptunaOptimizer(config)

def create_hyperopt_optimizer(config: HyperparameterConfig) -> HyperoptOptimizer:
    """Create Hyperopt optimizer."""
    config.algorithm = OptimizationAlgorithm.HYPEROPT
    return HyperoptOptimizer(config)


