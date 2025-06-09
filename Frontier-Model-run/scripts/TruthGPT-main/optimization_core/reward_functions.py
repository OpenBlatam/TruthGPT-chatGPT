"""
Advanced reward functions for enhanced training.
Integrated from grpo-reward.py optimization file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import math

@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    reward_type: str = "grpo"
    kl_penalty_weight: float = 0.1
    entropy_bonus_weight: float = 0.01
    value_loss_weight: float = 0.5
    advantage_normalization: bool = True
    clip_rewards: bool = True
    reward_clip_range: Tuple[float, float] = (-10.0, 10.0)

class BaseRewardFunction(nn.Module):
    """Base class for reward functions."""
    
    def __init__(self, config: RewardConfig):
        super().__init__()
        self.config = config
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class GRPORewardFunction(BaseRewardFunction):
    """GRPO-specific reward function with KL penalty and entropy bonus."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.kl_penalty_weight = config.kl_penalty_weight
        self.entropy_bonus_weight = config.entropy_bonus_weight
    
    def compute_kl_penalty(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor,
                          action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute KL divergence penalty."""
        kl_div = ref_log_probs - log_probs
        if action_mask is not None:
            kl_div = kl_div * action_mask
        return kl_div.sum(dim=-1)
    
    def compute_entropy_bonus(self, logits: torch.Tensor,
                            action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy bonus for exploration."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        if action_mask is not None:
            entropy = entropy * action_mask.squeeze(-1)
        
        return entropy.sum(dim=-1)
    
    def forward(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor,
                logits: torch.Tensor, base_rewards: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute enhanced GRPO rewards."""
        kl_penalty = self.compute_kl_penalty(log_probs, ref_log_probs, action_mask)
        entropy_bonus = self.compute_entropy_bonus(logits, action_mask)
        
        total_rewards = (base_rewards - 
                        self.kl_penalty_weight * kl_penalty + 
                        self.entropy_bonus_weight * entropy_bonus)
        
        if self.config.clip_rewards:
            total_rewards = torch.clamp(
                total_rewards, 
                self.config.reward_clip_range[0], 
                self.config.reward_clip_range[1]
            )
        
        return total_rewards

class AdaptiveRewardFunction(BaseRewardFunction):
    """Adaptive reward function that adjusts weights based on training progress."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.step_count = 0
        self.performance_history = []
        self.adaptive_kl_weight = config.kl_penalty_weight
        self.adaptive_entropy_weight = config.entropy_bonus_weight
    
    def update_weights(self, performance_metric: float):
        """Update reward weights based on performance."""
        self.step_count += 1
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) >= 100:
            recent_performance = sum(self.performance_history[-50:]) / 50
            older_performance = sum(self.performance_history[-100:-50]) / 50
            
            improvement = recent_performance - older_performance
            
            if improvement > 0.01:
                self.adaptive_kl_weight *= 0.95
                self.adaptive_entropy_weight *= 1.05
            elif improvement < -0.01:
                self.adaptive_kl_weight *= 1.05
                self.adaptive_entropy_weight *= 0.95
            
            self.adaptive_kl_weight = max(0.01, min(1.0, self.adaptive_kl_weight))
            self.adaptive_entropy_weight = max(0.001, min(0.1, self.adaptive_entropy_weight))
    
    def forward(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor,
                logits: torch.Tensor, base_rewards: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adaptive rewards with dynamic weighting."""
        kl_div = ref_log_probs - log_probs
        if action_mask is not None:
            kl_div = kl_div * action_mask
        kl_penalty = kl_div.sum(dim=-1)
        
        probs = F.softmax(logits, dim=-1)
        log_probs_entropy = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs_entropy).sum(dim=-1)
        if action_mask is not None:
            entropy = entropy * action_mask.squeeze(-1)
        entropy_bonus = entropy.sum(dim=-1)
        
        total_rewards = (base_rewards - 
                        self.adaptive_kl_weight * kl_penalty + 
                        self.adaptive_entropy_weight * entropy_bonus)
        
        if self.config.clip_rewards:
            total_rewards = torch.clamp(
                total_rewards, 
                self.config.reward_clip_range[0], 
                self.config.reward_clip_range[1]
            )
        
        return total_rewards

class MultiObjectiveRewardFunction(BaseRewardFunction):
    """Multi-objective reward function for complex training scenarios."""
    
    def __init__(self, config: RewardConfig, objective_weights: Optional[Dict[str, float]] = None):
        super().__init__(config)
        self.objective_weights = objective_weights or {
            'quality': 0.4,
            'diversity': 0.3,
            'coherence': 0.2,
            'efficiency': 0.1
        }
    
    def compute_quality_reward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute quality-based reward."""
        similarity = F.cosine_similarity(outputs, targets, dim=-1)
        return similarity
    
    def compute_diversity_reward(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute diversity reward to encourage exploration."""
        batch_size = outputs.size(0)
        if batch_size < 2:
            return torch.zeros(batch_size, device=outputs.device)
        
        pairwise_distances = torch.cdist(outputs, outputs, p=2)
        mask = torch.eye(batch_size, device=outputs.device).bool()
        pairwise_distances = pairwise_distances.masked_fill(mask, float('inf'))
        
        min_distances = pairwise_distances.min(dim=-1)[0]
        diversity_reward = torch.tanh(min_distances)
        
        return diversity_reward
    
    def compute_coherence_reward(self, sequence_outputs: torch.Tensor) -> torch.Tensor:
        """Compute coherence reward for sequence consistency."""
        if sequence_outputs.size(1) < 2:
            return torch.zeros(sequence_outputs.size(0), device=sequence_outputs.device)
        
        consecutive_similarities = F.cosine_similarity(
            sequence_outputs[:, :-1], 
            sequence_outputs[:, 1:], 
            dim=-1
        )
        coherence_reward = consecutive_similarities.mean(dim=-1)
        
        return coherence_reward
    
    def compute_efficiency_reward(self, computation_time: torch.Tensor, 
                                 target_time: float = 1.0) -> torch.Tensor:
        """Compute efficiency reward based on computation time."""
        efficiency = torch.exp(-torch.abs(computation_time - target_time))
        return efficiency
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor,
                sequence_outputs: Optional[torch.Tensor] = None,
                computation_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-objective rewards."""
        batch_size = outputs.size(0)
        total_reward = torch.zeros(batch_size, device=outputs.device)
        
        if 'quality' in self.objective_weights:
            quality_reward = self.compute_quality_reward(outputs, targets)
            total_reward += self.objective_weights['quality'] * quality_reward
        
        if 'diversity' in self.objective_weights:
            diversity_reward = self.compute_diversity_reward(outputs)
            total_reward += self.objective_weights['diversity'] * diversity_reward
        
        if 'coherence' in self.objective_weights and sequence_outputs is not None:
            coherence_reward = self.compute_coherence_reward(sequence_outputs)
            total_reward += self.objective_weights['coherence'] * coherence_reward
        
        if 'efficiency' in self.objective_weights and computation_time is not None:
            efficiency_reward = self.compute_efficiency_reward(computation_time)
            total_reward += self.objective_weights['efficiency'] * efficiency_reward
        
        if self.config.clip_rewards:
            total_reward = torch.clamp(
                total_reward, 
                self.config.reward_clip_range[0], 
                self.config.reward_clip_range[1]
            )
        
        return total_reward

class CurriculumRewardFunction(BaseRewardFunction):
    """Curriculum learning reward function with progressive difficulty."""
    
    def __init__(self, config: RewardConfig, initial_difficulty: float = 0.1):
        super().__init__(config)
        self.difficulty = initial_difficulty
        self.max_difficulty = 1.0
        self.difficulty_increment = 0.01
        self.performance_threshold = 0.8
        self.step_count = 0
    
    def update_difficulty(self, success_rate: float):
        """Update curriculum difficulty based on success rate."""
        self.step_count += 1
        
        if success_rate > self.performance_threshold and self.step_count % 100 == 0:
            self.difficulty = min(self.max_difficulty, 
                                self.difficulty + self.difficulty_increment)
        elif success_rate < 0.5 and self.step_count % 100 == 0:
            self.difficulty = max(0.1, self.difficulty - self.difficulty_increment)
    
    def forward(self, base_rewards: torch.Tensor, task_difficulty: torch.Tensor) -> torch.Tensor:
        """Compute curriculum-adjusted rewards."""
        difficulty_mask = (task_difficulty <= self.difficulty).float()
        adjusted_rewards = base_rewards * difficulty_mask
        
        bonus_for_harder_tasks = torch.where(
            task_difficulty > self.difficulty,
            base_rewards * (task_difficulty - self.difficulty) * 2.0,
            torch.zeros_like(base_rewards)
        )
        
        total_rewards = adjusted_rewards + bonus_for_harder_tasks
        
        if self.config.clip_rewards:
            total_rewards = torch.clamp(
                total_rewards, 
                self.config.reward_clip_range[0], 
                self.config.reward_clip_range[1]
            )
        
        return total_rewards

def create_reward_function(reward_type: str = "grpo", config: Optional[RewardConfig] = None, **kwargs):
    """Factory function to create reward functions."""
    if config is None:
        config = RewardConfig(reward_type=reward_type, **kwargs)
    
    if reward_type == "grpo":
        return GRPORewardFunction(config)
    elif reward_type == "adaptive":
        return AdaptiveRewardFunction(config)
    elif reward_type == "multi_objective":
        objective_weights = kwargs.get('objective_weights')
        return MultiObjectiveRewardFunction(config, objective_weights)
    elif reward_type == "curriculum":
        initial_difficulty = kwargs.get('initial_difficulty', 0.1)
        return CurriculumRewardFunction(config, initial_difficulty)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

def compute_advantage_estimates(rewards: torch.Tensor, values: torch.Tensor,
                              next_values: torch.Tensor, gamma: float = 0.99,
                              lam: float = 0.95) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE)."""
    deltas = rewards + gamma * next_values - values
    advantages = torch.zeros_like(rewards)
    
    advantage = 0
    for t in reversed(range(len(rewards))):
        advantage = deltas[t] + gamma * lam * advantage
        advantages[t] = advantage
    
    return advantages

def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages for stable training."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)
