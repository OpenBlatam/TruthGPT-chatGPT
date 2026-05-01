"""
Reinforcement Learning Networks
===============================

PyTorch network architectures for RL agents.
"""
import torch
import torch.nn as nn
import logging
from typing import List

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """Standard Deep Q-Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
        logger.info(f"✅ DQN Network initialized (state_dim: {state_dim}, action_dim: {action_dim})")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)

class DuelingDQNNetwork(nn.Module):
    """Dueling Deep Q-Network Architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        shared_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Value stream (Scalar V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream (Advantage vector A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        logger.info(f"✅ Dueling DQN Network initialized")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: combination of Value and Advantage."""
        shared_features = self.shared_network(state)
        
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        # Combine value and advantage using the aggregation formula:
        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

