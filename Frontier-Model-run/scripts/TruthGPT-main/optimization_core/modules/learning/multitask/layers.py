"""
Multi-Task Layers
=================

Shared representation and task-specific head implementations.
"""
import torch
import torch.nn as nn
import logging
from typing import List
from .config import MultiTaskConfig
from .enums import TaskType

logger = logging.getLogger(__name__)

class SharedRepresentation:
    """Shared representation learning"""
    
    def __init__(self, config: MultiTaskConfig):
        self.config = config
        self.shared_layers = []
        self.representation_history = []
        logger.info("✅ Shared Representation initialized")
    
    def create_shared_representation(self, input_dim: int) -> nn.Module:
        """Create shared representation layers"""
        logger.info(f"🏗️ Creating shared representation with {self.config.num_shared_layers} layers")
        
        layers = []
        current_dim = input_dim
        
        for i in range(self.config.num_shared_layers):
            layers.append(nn.Linear(current_dim, self.config.shared_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = self.config.shared_hidden_dim
        
        shared_representation = nn.Sequential(*layers)
        
        # Store shared layers
        self.shared_layers = layers
        
        return shared_representation
    
    def update_shared_representation(self, shared_repr: nn.Module, task_gradients: List[torch.Tensor]):
        """Update shared representation based on task gradients"""
        logger.info("🔄 Updating shared representation")
        
        # Calculate average gradient for shared layers
        avg_gradient = torch.zeros_like(list(shared_repr.parameters())[0])
        
        for grad in task_gradients:
            avg_gradient += grad
        
        avg_gradient /= len(task_gradients)
        
        # Update shared representation
        with torch.no_grad():
            for param in shared_repr.parameters():
                param.data -= self.config.learning_rate * avg_gradient
        
        # Store representation history
        self.representation_history.append({
            'avg_gradient': avg_gradient,
            'task_gradients': task_gradients
        })

class MultiTaskHead:
    """Multi-task head for specific tasks"""
    
    def __init__(self, config: MultiTaskConfig, task_type: TaskType):
        self.config = config
        self.task_type = task_type
        self.task_head = None
        logger.info(f"✅ Multi-Task Head for {task_type.value} initialized")
    
    def create_task_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create task-specific head"""
        logger.info(f"🎯 Creating task head for {self.task_type.value}")
        
        layers = []
        current_dim = input_dim
        
        for i in range(self.config.num_task_specific_layers):
            if i == self.config.num_task_specific_layers - 1:
                # Output layer
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                # Hidden layers
                layers.append(nn.Linear(current_dim, self.config.task_specific_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                current_dim = self.config.task_specific_dim
        
        task_head = nn.Sequential(*layers)
        self.task_head = task_head
        
        return task_head

