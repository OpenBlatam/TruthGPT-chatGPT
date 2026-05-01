"""
Gradient Surgery
================

Gradient surgery methods for multi-task learning.
"""
import torch
import logging
from typing import List
from .config import MultiTaskConfig

logger = logging.getLogger(__name__)

class GradientSurgery:
    """Gradient surgery for multi-task learning"""
    
    def __init__(self, config: MultiTaskConfig):
        self.config = config
        self.surgery_history = []
        logger.info("✅ Gradient Surgery initialized")
    
    def apply_gradient_surgery(self, task_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply gradient surgery to task gradients"""
        logger.info(f"🔧 Applying gradient surgery using method: {self.config.gradient_surgery_method}")
        
        if self.config.gradient_surgery_method == "pcgrad":
            modified_gradients = self._pcgrad_surgery(task_gradients)
        elif self.config.gradient_surgery_method == "mgda":
            modified_gradients = self._mgda_surgery(task_gradients)
        elif self.config.gradient_surgery_method == "graddrop":
            modified_gradients = self._graddrop_surgery(task_gradients)
        else:
            modified_gradients = self._pcgrad_surgery(task_gradients)
        
        # Store surgery history
        self.surgery_history.append({
            'original_gradients': task_gradients,
            'modified_gradients': modified_gradients,
            'method': self.config.gradient_surgery_method
        })
        
        return modified_gradients
    
    def _pcgrad_surgery(self, task_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """PCGrad gradient surgery"""
        logger.info("🔧 Applying PCGrad surgery")
        
        modified_gradients = []
        
        for i, grad_i in enumerate(task_gradients):
            modified_grad = grad_i.clone()
            
            for j, grad_j in enumerate(task_gradients):
                if i != j:
                    # Calculate dot product
                    dot_product = torch.dot(grad_i.flatten(), grad_j.flatten())
                    
                    if dot_product < 0:
                        # Project grad_i onto grad_j
                        projection = dot_product / (torch.norm(grad_j.flatten())**2 + 1e-8)
                        modified_grad = modified_grad - projection * grad_j
            
            modified_gradients.append(modified_grad)
        
        return modified_gradients
    
    def _mgda_surgery(self, task_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """MGDA gradient surgery"""
        logger.info("🔧 Applying MGDA surgery")
        
        # Calculate MGDA weights
        mgda_weights = self._calculate_mgda_weights(task_gradients)
        
        # Apply weighted combination
        modified_gradients = []
        for i, grad in enumerate(task_gradients):
            modified_grad = mgda_weights[i] * grad
            modified_gradients.append(modified_grad)
        
        return modified_gradients
    
    def _graddrop_surgery(self, task_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """GradDrop gradient surgery"""
        logger.info("🔧 Applying GradDrop surgery")
        
        # Stack gradients
        stacked_gradients = torch.stack(task_gradients, dim=0)
        
        # Calculate gradient magnitudes
        grad_magnitudes = torch.norm(stacked_gradients, dim=1, keepdim=True)
        
        # Calculate drop probabilities
        drop_probs = torch.sigmoid(-grad_magnitudes)
        
        # Apply dropout
        modified_gradients = []
        for i, grad in enumerate(task_gradients):
            mask = torch.rand_like(grad) > drop_probs[i]
            modified_grad = grad * mask.float()
            modified_gradients.append(modified_grad)
        
        return modified_gradients
    
    def _calculate_mgda_weights(self, task_gradients: List[torch.Tensor]) -> List[float]:
        """Calculate MGDA weights"""
        # Simplified MGDA weight calculation
        n_tasks = len(task_gradients)
        weights = [1.0 / n_tasks] * n_tasks
        
        return weights

