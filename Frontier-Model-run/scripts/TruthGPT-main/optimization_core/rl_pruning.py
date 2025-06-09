"""
RL-based pruning optimization for TruthGPT.
Integrates main.py reinforcement learning pruning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
from collections import deque

class RLPruningAgent:
    """Reinforcement Learning agent for neural network pruning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_network()

    def _build_network(self):
        """Build Q-network for RL agent."""
        try:
            from optimization_core.enhanced_mlp import EnhancedLinear
            return nn.Sequential(
                EnhancedLinear(self.state_dim, 128),
                nn.ReLU(),
                EnhancedLinear(128, 128),
                nn.ReLU(),
                EnhancedLinear(128, self.action_dim)
            )
        except ImportError:
            return nn.Sequential(
                nn.Linear(self.state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_dim)
            )

    def get_state(self, layer: nn.Module) -> torch.Tensor:
        """Extract state features from a neural network layer."""
        features = []
        
        if hasattr(layer, 'weight'):
            weight = layer.weight.data
            features.extend([
                weight.mean().item(),
                weight.std().item(),
                weight.abs().mean().item(),
                (weight == 0).float().mean().item(),
                weight.numel()
            ])
        
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias = layer.bias.data
            features.extend([
                bias.mean().item(),
                bias.std().item(),
                bias.abs().mean().item()
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.state_dim], dtype=torch.float32)

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select pruning action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 32):
        """Perform one training step."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

class RLPruning:
    """RL-based pruning system for neural networks."""
    
    def __init__(
        self,
        target_sparsity: float = 0.5,
        pruning_steps: int = 10,
        reward_metric: str = "accuracy",
        state_dim: int = 8,
        action_dim: int = 5
    ):
        self.target_sparsity = target_sparsity
        self.pruning_steps = pruning_steps
        self.reward_metric = reward_metric
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.agent = RLPruningAgent(state_dim, action_dim)
        self.pruning_history = []

    def get_pruning_actions(self) -> List[str]:
        """Get available pruning actions."""
        return [
            "magnitude_prune_10",
            "magnitude_prune_20",
            "structured_prune_10",
            "structured_prune_20",
            "no_prune"
        ]

    def apply_pruning_action(self, layer: nn.Module, action: int) -> float:
        """Apply pruning action to a layer."""
        actions = self.get_pruning_actions()
        action_name = actions[action]
        
        if "no_prune" in action_name:
            return 0.0
        
        if "magnitude_prune" in action_name:
            prune_ratio = 0.1 if "10" in action_name else 0.2
            return self._magnitude_prune(layer, prune_ratio)
        
        elif "structured_prune" in action_name:
            prune_ratio = 0.1 if "10" in action_name else 0.2
            return self._structured_prune(layer, prune_ratio)
        
        return 0.0

    def _magnitude_prune(self, layer: nn.Module, prune_ratio: float) -> float:
        """Apply magnitude-based pruning to a layer."""
        if not hasattr(layer, 'weight'):
            return 0.0
        
        weight = layer.weight.data
        threshold = torch.quantile(weight.abs(), prune_ratio)
        mask = weight.abs() > threshold
        
        pruned_weights = (mask == 0).sum().item()
        total_weights = weight.numel()
        
        weight.data = weight * mask.float()
        
        return pruned_weights / total_weights

    def _structured_prune(self, layer: nn.Module, prune_ratio: float) -> float:
        """Apply structured pruning to a layer."""
        if not hasattr(layer, 'weight'):
            return 0.0
        
        weight = layer.weight.data
        
        if len(weight.shape) == 2:
            channel_importance = weight.abs().sum(dim=1)
            num_channels_to_prune = int(prune_ratio * weight.shape[0])
            
            if num_channels_to_prune > 0:
                _, indices_to_prune = torch.topk(
                    channel_importance, num_channels_to_prune, largest=False
                )
                weight[indices_to_prune] = 0
                
                return num_channels_to_prune / weight.shape[0]
        
        return 0.0

    def calculate_reward(
        self,
        model: nn.Module,
        original_performance: float,
        current_performance: float,
        sparsity_increase: float
    ) -> float:
        """Calculate reward for pruning action."""
        performance_loss = max(0, original_performance - current_performance)
        
        sparsity_reward = sparsity_increase * 10
        performance_penalty = performance_loss * 100
        
        reward = sparsity_reward - performance_penalty
        
        if current_performance > original_performance * 0.95:
            reward += 5
        
        return reward

    def prune_model(
        self,
        model: nn.Module,
        validation_fn: callable,
        training_steps: int = 100
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Prune model using RL agent."""
        original_performance = validation_fn(model)
        pruning_log = {
            'original_performance': original_performance,
            'pruning_steps': [],
            'final_sparsity': 0.0,
            'final_performance': 0.0
        }
        
        pruneable_layers = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight')
        ]
        
        for step in range(training_steps):
            for layer_name, layer in pruneable_layers:
                state = self.agent.get_state(layer)
                action = self.agent.select_action(state, training=True)
                
                sparsity_before = self._get_layer_sparsity(layer)
                sparsity_increase = self.apply_pruning_action(layer, action)
                sparsity_after = self._get_layer_sparsity(layer)
                
                current_performance = validation_fn(model)
                reward = self.calculate_reward(
                    model, original_performance, current_performance, sparsity_increase
                )
                
                next_state = self.agent.get_state(layer)
                done = sparsity_after >= self.target_sparsity
                
                self.agent.store_experience(state, action, reward, next_state, done)
                
                if step % 10 == 0:
                    self.agent.train_step()
                
                pruning_log['pruning_steps'].append({
                    'step': step,
                    'layer': layer_name,
                    'action': action,
                    'reward': reward,
                    'sparsity_before': sparsity_before,
                    'sparsity_after': sparsity_after,
                    'performance': current_performance
                })
                
                if done:
                    break
            
            if step % 50 == 0:
                self.agent.update_target_network()
        
        final_sparsity = self._get_model_sparsity(model)
        final_performance = validation_fn(model)
        
        pruning_log['final_sparsity'] = final_sparsity
        pruning_log['final_performance'] = final_performance
        
        return model, pruning_log

    def _get_layer_sparsity(self, layer: nn.Module) -> float:
        """Calculate sparsity of a layer."""
        if not hasattr(layer, 'weight'):
            return 0.0
        
        weight = layer.weight.data
        total_params = weight.numel()
        zero_params = (weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0

    def _get_model_sparsity(self, model: nn.Module) -> float:
        """Calculate overall model sparsity."""
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0

class RLPruningOptimizations:
    """Utility class for applying RL pruning optimizations."""
    
    @staticmethod
    def apply_rl_pruning(
        model: nn.Module,
        validation_fn: callable,
        target_sparsity: float = 0.5,
        training_steps: int = 100
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply RL-based pruning to a model."""
        pruner = RLPruning(target_sparsity=target_sparsity)
        return pruner.prune_model(model, validation_fn, training_steps)
    
    @staticmethod
    def get_pruning_report(model: nn.Module) -> Dict[str, Any]:
        """Get a report of model pruning status."""
        total_params = 0
        zero_params = 0
        layer_sparsities = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                layer_total = weight.numel()
                layer_zeros = (weight == 0).sum().item()
                
                total_params += layer_total
                zero_params += layer_zeros
                
                layer_sparsities[name] = {
                    'sparsity': layer_zeros / layer_total if layer_total > 0 else 0.0,
                    'total_params': layer_total,
                    'zero_params': layer_zeros
                }
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            'overall_sparsity': overall_sparsity,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'layer_sparsities': layer_sparsities,
            'compression_ratio': 1 / (1 - overall_sparsity) if overall_sparsity < 1 else float('inf')
        }

def create_rl_pruning(target_sparsity: float = 0.5, **kwargs) -> RLPruning:
    """Factory function to create RLPruning."""
    return RLPruning(target_sparsity=target_sparsity, **kwargs)

def create_rl_pruning_agent(state_dim: int = 8, action_dim: int = 5, **kwargs) -> RLPruningAgent:
    """Factory function to create RLPruningAgent."""
    return RLPruningAgent(state_dim, action_dim, **kwargs)
