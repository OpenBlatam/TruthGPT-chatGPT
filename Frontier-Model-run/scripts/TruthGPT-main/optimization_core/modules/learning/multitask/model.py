"""
Multi-Task Network
==================

Multi-task neural network architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict
from .config import MultiTaskConfig
from .enums import TaskType
from .layers import SharedRepresentation, MultiTaskHead
from .balancer import TaskBalancer
from .surgery import GradientSurgery

logger = logging.getLogger(__name__)

class MultiTaskNetwork(nn.Module):
    """Multi-task neural network"""
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.task_balancer = TaskBalancer(config)
        self.gradient_surgery = GradientSurgery(config)
        self.shared_representation = SharedRepresentation(config)
        
        # Network components
        self.shared_layers = None
        self.task_specific_layers = nn.ModuleDict()
        
        logger.info("✅ Multi-Task Network initialized")
    
    def build_network(self, input_dim: int, task_output_dims: Dict[TaskType, int]) -> nn.Module:
        """Build multi-task network"""
        logger.info("🏗️ Building multi-task network")
        
        # Create shared representation
        self.shared_layers = self.shared_representation.create_shared_representation(input_dim)
        
        # Create task-specific heads
        for task_type, output_dim in task_output_dims.items():
            task_head = MultiTaskHead(self.config, task_type)
            self.task_specific_layers[task_type.value] = task_head.create_task_head(
                self.config.shared_hidden_dim, output_dim
            )
        
        return self
    
    def forward(self, x: torch.Tensor, task_type: TaskType) -> torch.Tensor:
        """Forward pass for specific task"""
        # Shared representation
        shared_features = self.shared_layers(x)
        
        # Task-specific head
        task_output = self.task_specific_layers[task_type.value](shared_features)
        
        return task_output
    
    def compute_task_losses(self, outputs: Dict[TaskType, torch.Tensor], 
                           targets: Dict[TaskType, torch.Tensor]) -> Dict[TaskType, torch.Tensor]:
        """Compute losses for all tasks"""
        task_losses = {}
        
        for task_type in outputs.keys():
            if task_type == TaskType.CLASSIFICATION:
                loss = F.cross_entropy(outputs[task_type], targets[task_type])
            elif task_type == TaskType.REGRESSION:
                loss = F.mse_loss(outputs[task_type], targets[task_type])
            else:
                loss = F.mse_loss(outputs[task_type], targets[task_type])
            
            task_losses[task_type] = loss
        
        return task_losses
    
    def compute_weighted_loss(self, task_losses: Dict[TaskType, torch.Tensor]) -> torch.Tensor:
        """Compute weighted loss across tasks"""
        if self.config.enable_task_balancing:
            # Get task weights
            loss_values = [loss.item() for loss in task_losses.values()]
            # Pass task_gradients as None for now, as we don't have them easily accessible here
            # Ideally they should be passed from the trainer or computed here if we had retain_graph=True
            task_weights = self.task_balancer.balance_tasks(loss_values)
            
            # Apply weights
            weighted_loss = torch.tensor(0.0)
            for i, (task_type, loss) in enumerate(task_losses.items()):
                weighted_loss += task_weights[i] * loss
        else:
            # Equal weighting
            weighted_loss = sum(task_losses.values()) / len(task_losses)
        
        return weighted_loss

