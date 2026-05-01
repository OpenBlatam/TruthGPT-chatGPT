"""
Reinforcement Learning Configuration
====================================

Configuration class for RL agents and training.
"""
from dataclasses import dataclass, field
from .enums import RLAlgorithm, EnvironmentType

@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning"""
    # Algorithm settings
    algorithm: RLAlgorithm = RLAlgorithm.DQN
    environment_type: EnvironmentType = EnvironmentType.DISCRETE
    
    # Training parameters
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    
    # Experience replay
    buffer_size: int = 10000
    min_buffer_size: int = 1000
    update_frequency: int = 4
    
    # Target network
    target_update_frequency: int = 100
    soft_update_tau: float = 0.001
    
    # Advanced features
    enable_double_dqn: bool = True
    enable_dueling: bool = True
    enable_prioritized_replay: bool = True
    enable_noisy_networks: bool = False
    
    # Multi-agent settings
    num_agents: int = 1
    enable_centralized_training: bool = True
    enable_decentralized_execution: bool = True
    
    # Dimensions (often set at runtime)
    state_dim: int = 10
    action_dim: int = 4
    
    def __post_init__(self):
        """Validate RL configuration"""
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("Gamma must be between 0.0 and 1.0")
        if self.epsilon_start < 0.0 or self.epsilon_start > 1.0:
            raise ValueError("Epsilon start must be between 0.0 and 1.0")

