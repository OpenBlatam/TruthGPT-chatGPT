"""
PPO Agent
=========

Proximal Policy Optimization agent implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import logging
from typing import Dict, Any, Tuple

from ..config import RLConfig

logger = logging.getLogger(__name__)

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Actor-Critic Networks
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # PPO Hyperparameters (hardcoded in original, could move to config)
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        
        logger.info(f"✅ PPOAgent initialized")
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy and get value estimate."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            action_dist = distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
               old_log_probs: torch.Tensor, returns: torch.Tensor,
               advantages: torch.Tensor) -> Dict[str, float]:
        """Update actor and critic networks using PPO objective."""
        # Policy gradient update (surrogate objective)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        curr_action_probs = self.actor(states)
        curr_dist = distributions.Categorical(curr_action_probs)
        curr_log_probs = curr_dist.log_prob(actions)
        entropy = curr_dist.entropy().mean()
        
        # Ratios: exp(curr_log_prob - old_log_prob)
        ratios = torch.exp(curr_log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (critic)
        values = self.critic(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)
        
        # Total loss
        total_loss = actor_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy)
        
        # Optimization
        self._perform_update(self.actor_optimizer, actor_loss, self.actor.parameters(), retain_graph=True)
        self._perform_update(self.critic_optimizer, value_loss, self.critic.parameters())
        
        self.step_count += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }

    def _perform_update(self, optimizer: optim.Optimizer, loss: torch.Tensor, 
                        params: Any, retain_graph: bool = False):
        """Helper for optimization steps."""
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()

