"""
Meta-Learner System
===================

Integrated system for meta-learning management and rapid adaptation.
"""
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

from .config import MetaLearningConfig
from .enums import MetaLearningAlgorithm
from .task_gen import TaskGenerator
from .algorithms.maml import MAML
from .algorithms.reptile import Reptile

logger = logging.getLogger(__name__)

class MetaLearner:
    """Main meta-learning orchestrator."""
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.task_generator = TaskGenerator(config)
        
        # Select meta-algo
        if config.algorithm == MetaLearningAlgorithm.MAML:
            self.algorithm = MAML(model, config)
        elif config.algorithm == MetaLearningAlgorithm.REPTILE:
            self.algorithm = Reptile(model, config)
        else:
            self.algorithm = MAML(model, config)
            
        self.training_history = []
        self.validation_history = []
        self.best_performance = 0.0
        
        logger.info(f"✅ MetaLearner initialized with {config.algorithm.value}")
        
    def train(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """Execute meta-training cycle."""
        logger.info(f"🚀 Meta-training for {num_iterations} iterations")
        start_time = time.time()
        
        for i in range(num_iterations):
            # Batch of Tasks
            tasks = self.task_generator.generate_task_batch(self.config.meta_batch_size)
            
            # Update meta-weights
            stats = self.algorithm.meta_update(tasks)
            
            self.training_history.append({
                'iteration': i,
                'meta_loss': stats['meta_loss'],
                'avg_task_loss': np.mean(stats['task_losses'])
            })
            
            # Validation cycle
            if self.config.enable_meta_validation and (i % self.config.validation_frequency == 0):
                perf = self.validate()
                self.validation_history.append({'iteration': i, 'perf': perf})
                self.best_performance = max(self.best_performance, perf)
                logger.info(f"Iteration {i}: Loss={stats['meta_loss']:.4f}, Val Acc={perf:.4f}")
                
            if self._check_early_stopping():
                logger.info("✅ Early stopping triggered.")
                break
                
        return {
            'iterations': len(self.training_history),
            'time': time.time() - start_time,
            'best_val': self.best_performance
        }

    def validate(self, n_tasks: int = 10) -> float:
        """Evaluate meta-initialization Rapid Adaptation on new tasks."""
        v_tasks = self.task_generator.generate_task_batch(n_tasks)
        scores = []
        
        for t in v_tasks:
            # Adapt copy of model to this task
            adapted_m = self.algorithm.adapt_to_task(t)
            
            # Eval on query set
            q_data, q_labels = t['query_data'], t['query_labels']
            with torch.no_grad():
                out = adapted_m(q_data)
                _, pred = torch.max(out, 1)
                acc = (pred == q_labels).float().mean().item()
                scores.append(acc)
                
        return np.mean(scores)

    def few_shot_learn(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                       query_data: torch.Tensor, query_labels: torch.Tensor) -> Tuple[nn.Module, float]:
        """Practical interface for few-shot adaptation on a single task."""
        t = {
            'support_data': support_data,
            'support_labels': support_labels,
            'query_data': query_data,
            'query_labels': query_labels
        }
        adapted = self.algorithm.adapt_to_task(t)
        
        with torch.no_grad():
            out = adapted(query_data)
            _, pred = torch.max(out, 1)
            acc = (pred == query_labels).float().mean().item()
            
        return adapted, acc

    def _check_early_stopping(self) -> bool:
        """Heuristic for stopping training if no progress."""
        patience = self.config.early_stopping_patience
        if len(self.validation_history) < patience:
            return False
            
        recent = [h['perf'] for h in self.validation_history[-patience:]]
        if max(recent) - min(recent) < 1e-4:
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate stats."""
        return {
            'algo': self.config.algorithm.value,
            'best_acc': self.best_performance,
            'iterations': len(self.training_history)
        }

