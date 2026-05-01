"""
Adaptive Learning Configuration
==============================

Configuration dataclasses for adaptive learning systems.
"""
from dataclasses import dataclass, field

@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning systems"""
    # Learning parameters
    learning_rate: float = 0.001
    adaptation_rate: float = 0.01
    exploration_rate: float = 0.1
    exploitation_rate: float = 0.9
    
    # Meta-learning settings
    enable_meta_learning: bool = True
    meta_learning_steps: int = 100
    meta_batch_size: int = 32
    meta_learning_rate: float = 0.0001
    
    # Self-improvement settings
    enable_self_improvement: bool = True
    improvement_threshold: float = 0.05
    improvement_patience: int = 10
    improvement_memory_size: int = 1000
    
    # Adaptive mechanisms
    enable_adaptive_lr: bool = True
    enable_adaptive_architecture: bool = True
    enable_adaptive_optimization: bool = True
    
    # Monitoring
    enable_performance_tracking: bool = True
    enable_learning_curves: bool = True
    enable_adaptation_logging: bool = True
    
    def __post_init__(self):
        """Validate adaptive learning configuration"""
        if not (0.0 <= self.exploration_rate <= 1.0):
            raise ValueError("Exploration rate must be between 0.0 and 1.0")
        if not (0.0 <= self.exploitation_rate <= 1.0):
            raise ValueError("Exploitation rate must be between 0.0 and 1.0")

