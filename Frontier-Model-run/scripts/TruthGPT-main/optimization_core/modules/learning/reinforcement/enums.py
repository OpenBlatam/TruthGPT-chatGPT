"""
Reinforcement Learning Enums
============================

Algorithms and environment types for RL.
"""
from enum import Enum

class RLAlgorithm(Enum):
    """Reinforcement Learning algorithms"""
    DQN = "dqn"  # Deep Q-Network
    DDQN = "ddqn"  # Double DQN
    D3QN = "d3qn"  # Dueling Double DQN
    A3C = "a3c"  # Asynchronous Advantage Actor-Critic
    PPO = "ppo"  # Proximal Policy Optimization
    SAC = "sac"  # Soft Actor-Critic
    TD3 = "td3"  # Twin Delayed Deep Deterministic Policy Gradient
    MADDPG = "maddpg"  # Multi-Agent Deep Deterministic Policy Gradient

class EnvironmentType(Enum):
    """Environment types"""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_AGENT = "multi_agent"
    PARTIALLY_OBSERVABLE = "partially_observable"
    HIERARCHICAL = "hierarchical"

