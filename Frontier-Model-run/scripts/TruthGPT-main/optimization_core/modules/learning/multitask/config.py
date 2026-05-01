"""
Multi-Task Learning Configuration
=================================

Configuration for multi-task learning systems.
"""
from dataclasses import dataclass, field
from typing import List
from .enums import TaskType, TaskRelationship, SharingStrategy

@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning system"""
    # Basic settings
    task_types: List[TaskType] = field(default_factory=lambda: [TaskType.CLASSIFICATION, TaskType.REGRESSION])
    task_relationships: List[TaskRelationship] = field(default_factory=lambda: [TaskRelationship.RELATED])
    sharing_strategy: SharingStrategy = SharingStrategy.HARD_SHARING
    
    # Model architecture
    shared_hidden_dim: int = 512
    task_specific_dim: int = 256
    num_shared_layers: int = 3
    num_task_specific_layers: int = 2
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    
    # Task balancing
    enable_task_balancing: bool = True
    task_balancing_method: str = "uncertainty_weighting"
    task_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Gradient surgery
    enable_gradient_surgery: bool = True
    gradient_surgery_method: str = "pcgrad"
    gradient_surgery_lambda: float = 0.1
    
    # Advanced features
    enable_meta_learning: bool = False
    enable_transfer_learning: bool = True
    enable_continual_learning: bool = False
    enable_adaptive_sharing: bool = True
    
    def __post_init__(self):
        """Validate multi-task configuration"""
        if self.shared_hidden_dim <= 0:
            raise ValueError("Shared hidden dimension must be positive")
        if self.task_specific_dim <= 0:
            raise ValueError("Task-specific dimension must be positive")
        if self.num_shared_layers <= 0:
            raise ValueError("Number of shared layers must be positive")
        if self.num_task_specific_layers <= 0:
            raise ValueError("Number of task-specific layers must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if not (0 <= self.gradient_surgery_lambda <= 1):
            raise ValueError("Gradient surgery lambda must be between 0 and 1")

