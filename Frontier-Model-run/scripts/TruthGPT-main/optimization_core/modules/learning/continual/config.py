"""
Continual Learning Configuration
================================

Configuration dataclasses for continual learning systems.
"""
from dataclasses import dataclass, field
from .enums import CLStrategy, ReplayStrategy, MemoryType

@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning system"""
    # Basic settings
    cl_strategy: CLStrategy = CLStrategy.EWC
    replay_strategy: ReplayStrategy = ReplayStrategy.RANDOM_REPLAY
    memory_type: MemoryType = MemoryType.EPISODIC_MEMORY
    
    # Model settings
    model_dim: int = 512
    hidden_dim: int = 256
    num_tasks: int = 5
    
    # EWC settings
    ewc_lambda: float = 1000.0
    ewc_importance: float = 1.0
    
    # Replay settings
    replay_buffer_size: int = 1000
    replay_batch_size: int = 32
    replay_frequency: int = 10
    
    # Progressive networks
    enable_progressive_networks: bool = True
    progressive_expansion_factor: float = 1.2
    
    # Multi-task learning
    enable_multi_task_learning: bool = True
    task_balancing_weight: float = 0.5
    
    # Lifelong learning
    enable_lifelong_learning: bool = True
    knowledge_retention_rate: float = 0.8
    
    # Advanced features
    enable_catastrophic_forgetting_prevention: bool = True
    enable_knowledge_distillation: bool = True
    enable_meta_learning: bool = False
    
    def __post_init__(self):
        """Validate continual learning configuration"""
        if self.model_dim <= 0:
            raise ValueError("Model dimension must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")
        if self.num_tasks <= 0:
            raise ValueError("Number of tasks must be positive")
        if self.ewc_lambda <= 0:
            raise ValueError("EWC lambda must be positive")
        if self.ewc_importance <= 0:
            raise ValueError("EWC importance must be positive")
        if self.replay_buffer_size <= 0:
            raise ValueError("Replay buffer size must be positive")
        if self.replay_batch_size <= 0:
            raise ValueError("Replay batch size must be positive")
        if self.replay_frequency <= 0:
            raise ValueError("Replay frequency must be positive")
        if self.progressive_expansion_factor <= 0:
            raise ValueError("Progressive expansion factor must be positive")
        if not (0 <= self.task_balancing_weight <= 1):
            raise ValueError("Task balancing weight must be between 0 and 1")
        if not (0 <= self.knowledge_retention_rate <= 1):
            raise ValueError("Knowledge retention rate must be between 0 and 1")

