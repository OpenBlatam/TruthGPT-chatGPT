"""
Meta-Learning Configuration
===========================

Configuration class for the meta-learning system.
"""
from dataclasses import dataclass, field
from .enums import MetaLearningAlgorithm, TaskDistribution

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning"""
    # Algorithm settings
    algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML
    task_distribution: TaskDistribution = TaskDistribution.UNIFORM
    
    # Training parameters
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    inner_steps: int = 5
    meta_batch_size: int = 16
    num_tasks: int = 1000
    
    # Task parameters (defaults for generator)
    support_size: int = 5
    query_size: int = 15
    num_ways: int = 5
    
    # Advanced features
    enable_second_order: bool = True
    enable_learned_initialization: bool = True
    enable_task_embedding: bool = True
    enable_meta_regularization: bool = True
    
    # Performance
    enable_meta_validation: bool = True
    validation_frequency: int = 100
    early_stopping_patience: int = 50
    
    def __post_init__(self):
        """Validate meta-learning configuration"""
        if self.inner_steps < 1:
            raise ValueError("Inner steps must be at least 1")
        if self.support_size < 1 or self.query_size < 1:
            raise ValueError("Support and query sizes must be at least 1")

