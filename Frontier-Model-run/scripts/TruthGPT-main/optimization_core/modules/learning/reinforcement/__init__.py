"""
Reinforcement Learning Module
=============================

Package for RL agents, environments, and training systems.
"""
from .enums import RLAlgorithm, EnvironmentType
from .config import RLConfig
from .buffer import ExperienceReplay
from .networks import DQNNetwork, DuelingDQNNetwork
from .agents import DQNAgent, PPOAgent
from .environment import MultiAgentEnvironment
from .system import RLTrainingManager

# Compatibility aliases
RLSystem = RLTrainingManager
RLBuffer = ExperienceReplay

# Factory functions for convenient instantiation
def create_rl_config(**kwargs) -> RLConfig:
    return RLConfig(**kwargs)

def create_dqn_agent(state_dim: int, action_dim: int, config: RLConfig) -> DQNAgent:
    return DQNAgent(state_dim, action_dim, config)

def create_ppo_agent(state_dim: int, action_dim: int, config: RLConfig) -> PPOAgent:
    return PPOAgent(state_dim, action_dim, config)

def create_rl_training_manager(config: RLConfig) -> RLTrainingManager:
    return RLTrainingManager(config)

__all__ = [
    'RLAlgorithm',
    'EnvironmentType',
    'RLConfig',
    'ExperienceReplay',
    'DQNNetwork',
    'DuelingDQNNetwork',
    'DQNAgent',
    'PPOAgent',
    'MultiAgentEnvironment',
    'RLTrainingManager',
    'create_rl_config',
    'create_dqn_agent',
    'create_ppo_agent',
    'create_rl_training_manager'
]

