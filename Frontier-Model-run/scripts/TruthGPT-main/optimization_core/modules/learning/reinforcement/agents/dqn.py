"""
DQN Agent
=========

Deep Q-Network agent implementation.
"""
import torch
import torch.optim as optim
import numpy as np
import random
import logging
from typing import Dict, Any, List

from ..config import RLConfig
from ..buffer import ExperienceReplay
from ..networks import DQNNetwork, DuelingDQNNetwork

logger = logging.getLogger(__name__)

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Networks
        if config.enable_dueling:
            self.q_network = DuelingDQNNetwork(state_dim, action_dim)
            self.target_network = DuelingDQNNetwork(state_dim, action_dim)
        else:
            self.q_network = DQNNetwork(state_dim, action_dim)
            self.target_network = DQNNetwork(state_dim, action_dim)
        
        # Initial weight synchronization
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(
            config.buffer_size, state_dim, action_dim
        )
        
        # Agent state
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        self.training_history = []
        
        logger.info(f"✅ DQNAgent initialized (state: {state_dim}, action: {action_dim})")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in the replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> Dict[str, float]:
        """Train agent using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        experiences, indices, weights = self.replay_buffer.sample(self.config.batch_size)
        if experiences is None:
            return {'loss': 0.0, 'q_value': 0.0}
        
        # Prepare tensors
        states = torch.FloatTensor([exp[0] for exp in experiences])
        actions = torch.LongTensor([exp[1] for exp in experiences])
        rewards = torch.FloatTensor([exp[2] for exp in experiences])
        next_states = torch.FloatTensor([exp[3] for exp in experiences])
        dones = torch.BoolTensor([exp[4] for exp in experiences])
        weights_tensor = torch.FloatTensor(weights)
        
        # Get current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q-values
        with torch.no_grad():
            if self.config.enable_double_dqn:
                # Double DQN logic: use policy network for action selection
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN logic: use target network for greedy selection
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Loss calculation (Huber loss vs MSE depends on needs, sticking to MSE matching original)
        td_errors = current_q_values - target_q_values
        loss = (weights_tensor.unsqueeze(1) * td_errors.pow(2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities in buffer
        self.replay_buffer.update_priorities(indices, td_errors.detach().numpy().flatten())
        
        # Target network update check
        if self.step_count % self.config.target_update_frequency == 0:
            self._soft_update_target_network()
        
        # Epsilon decay
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        self.step_count += 1
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }
    
    def _soft_update_target_network(self):
        """Gradually update target network weights."""
        tau = self.config.soft_update_tau
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, path)
        logger.info(f"💾 DQNAgent model saved to {path}")
    
    def load_model(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"📂 DQNAgent model loaded from {path}")

