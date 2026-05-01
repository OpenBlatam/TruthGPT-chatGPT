"""
RL Training Manager
===================

System for managing RL training loops and agent creation.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Union

from .config import RLConfig
from .enums import RLAlgorithm
from .agents.dqn import DQNAgent
from .agents.ppo import PPOAgent

logger = logging.getLogger(__name__)

class RLTrainingManager:
    """Manages RL agent creation and training sessions."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.training_history: List[Dict[str, Any]] = []
        logger.info("✅ RL Training Manager initialized")
    
    def create_agent(self, state_dim: int, action_dim: int) -> Union[DQNAgent, PPOAgent]:
        """Factory method for creating agents."""
        if self.config.algorithm == RLAlgorithm.DQN:
            return DQNAgent(state_dim, action_dim, self.config)
        elif self.config.algorithm == RLAlgorithm.PPO:
            return PPOAgent(state_dim, action_dim, self.config)
        return DQNAgent(state_dim, action_dim, self.config)
    
    def train_agent(self, agent: Union[DQNAgent, PPOAgent], num_episodes: int = 1000) -> Dict[str, Any]:
        """General training loop for RL agents."""
        logger.info(f"🚀 Training {agent.__class__.__name__} for {num_episodes} episodes")
        
        stats = {
            'rewards': [],
            'lengths': [],
            'losses': []
        }
        
        # State dim heuristic from config or default
        s_dim = self.config.state_dim
        
        for ep in range(num_episodes):
            state = np.random.random(s_dim)
            ep_reward = 0.0
            ep_length = 0
            
            while True:
                if isinstance(agent, DQNAgent):
                    action = agent.select_action(state, training=True)
                    next_state = np.random.random(s_dim)
                    reward = np.random.normal(0.0, 1.0)
                    done = (np.random.random() < 0.1)
                    
                    agent.store_experience(state, action, reward, next_state, done)
                    
                    if len(agent.replay_buffer) >= self.config.min_buffer_size:
                        loss_info = agent.train()
                        stats['losses'].append(loss_info['loss'])
                        
                    state = next_state
                    ep_reward += reward
                    ep_length += 1
                    if done: break
                
                elif isinstance(agent, PPOAgent):
                    # Simplified placeholder for PPO episode loop
                    action, log_prob, value = agent.select_action(state)
                    state = np.random.random(s_dim)
                    reward = np.random.normal(0.0, 1.0)
                    done = (np.random.random() < 0.1)
                    ep_reward += reward
                    ep_length += 1
                    if done: break
            
            stats['rewards'].append(ep_reward)
            stats['lengths'].append(ep_length)
            
            if ep % 100 == 0:
                logger.debug(f"Episode {ep}: Avg Reward (recent) = {np.mean(stats['rewards'][-100:]):.2f}")
        
        summary = {
            'total_episodes': num_episodes,
            'avg_reward': np.mean(stats['rewards']),
            'max_reward': np.max(stats['rewards']),
            'min_reward': np.min(stats['rewards']),
            'avg_length': np.mean(stats['lengths']),
            'avg_loss': np.mean(stats['losses']) if stats['losses'] else 0.0,
            'history': stats
        }
        
        self.training_history.append(summary)
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieve aggregated training stats."""
        if not self.training_history:
            return {}
        return {
            'sessions': len(self.training_history),
            'latest': self.training_history[-1],
            'algorithm': self.config.algorithm.value
        }

