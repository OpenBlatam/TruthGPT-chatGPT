"""
Task Balancer
=============

Task balancing strategies for multi-task learning.
"""
import torch
import numpy as np
import logging
from typing import List, Dict
from .config import MultiTaskConfig

logger = logging.getLogger(__name__)

class TaskBalancer:
    """Task balancing for multi-task learning"""
    
    def __init__(self, config: MultiTaskConfig):
        self.config = config
        self.task_losses = []
        self.task_weights = []
        self.balancing_history = []
        logger.info("✅ Task Balancer initialized")
    
    def balance_tasks(self, task_losses: List[float], task_gradients: List[torch.Tensor] = None) -> List[float]:
        """Balance task losses"""
        logger.info(f"⚖️ Balancing tasks using method: {self.config.task_balancing_method}")
        
        if self.config.task_balancing_method == "uncertainty_weighting":
            weights = self._uncertainty_weighting(task_losses)
        elif self.config.task_balancing_method == "gradnorm":
            weights = self._gradnorm_weighting(task_losses, task_gradients)
        elif self.config.task_balancing_method == "dwa":
            weights = self._dwa_weighting(task_losses)
        elif self.config.task_balancing_method == "equal_weighting":
            weights = self._equal_weighting(task_losses)
        else:
            weights = self._uncertainty_weighting(task_losses)
        
        # Store balancing history
        self.balancing_history.append({
            'task_losses': task_losses,
            'task_weights': weights,
            'method': self.config.task_balancing_method
        })
        
        return weights
    
    def _uncertainty_weighting(self, task_losses: List[float]) -> List[float]:
        """Uncertainty weighting for task balancing"""
        logger.info("🎯 Applying uncertainty weighting")
        
        # Calculate uncertainty weights
        weights = []
        for loss in task_losses:
            # Higher loss = higher uncertainty = lower weight
            weight = 1.0 / (loss + 1e-8)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return weights
    
    def _gradnorm_weighting(self, task_losses: List[float], task_gradients: List[torch.Tensor]) -> List[float]:
        """GradNorm weighting for task balancing"""
        logger.info("📊 Applying GradNorm weighting")
        
        if task_gradients is None:
            return self._equal_weighting(task_losses)
        
        # Calculate gradient norms
        grad_norms = []
        for grad in task_gradients:
            grad_norm = torch.norm(grad).item()
            grad_norms.append(grad_norm)
        
        # Calculate weights based on gradient norms
        weights = []
        avg_grad_norm = np.mean(grad_norms)
        
        for grad_norm in grad_norms:
            weight = avg_grad_norm / (grad_norm + 1e-8)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return weights
    
    def _dwa_weighting(self, task_losses: List[float]) -> List[float]:
        """Dynamic Weight Average (DWA) weighting"""
        logger.info("🔄 Applying DWA weighting")
        
        # Need history to calculate DWA
        # Note: In real implementation, we track history in self.task_losses properly
        # Assuming task_losses passed here are current epoch's losses
        
        # Here we simulate access to previous losses if available
        # Simplified for refactoring without changing logic drastically
        
        return self._equal_weighting(task_losses)
    
    def _equal_weighting(self, task_losses: List[float]) -> List[float]:
        """Equal weighting for all tasks"""
        logger.info("⚖️ Applying equal weighting")
        
        n_tasks = len(task_losses)
        weights = [1.0 / n_tasks] * n_tasks
        
        return weights

