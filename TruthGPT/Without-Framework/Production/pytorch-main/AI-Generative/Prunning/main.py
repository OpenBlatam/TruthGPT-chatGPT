import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

class RLPruning:
    def __init__(
        self,
        model: nn.Module,
        sparsity_target: float = 0.5,
        pruning_steps: int = 100,
        reward_threshold: float = 0.0,
    ):
        """
        Reinforcement Learning based pruning.
        
        Args:
            model: The neural network model to prune
            sparsity_target: Target sparsity ratio (0.0 to 1.0)
            pruning_steps: Number of pruning iterations
            reward_threshold: Minimum reward threshold to keep weights
        """
        self.model = model
        self.sparsity_target = sparsity_target
        self.pruning_steps = pruning_steps
        self.reward_threshold = reward_threshold
        self.masks = {}
        
        # Initialize masks for each parameter
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.masks[name] = torch.ones_like(param)

    def compute_weight_importance(
        self,
        rewards: torch.Tensor,
        param_gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for weights based on rewards and gradients
        """
        importance_scores = {}
        for name, grad in param_gradients.items():
            if 'weight' in name:
                # Compute importance as gradient * reward
                importance = torch.abs(grad * rewards.reshape(-1, 1, 1))
                importance_scores[name] = importance.mean(dim=0)
        return importance_scores

    def update_masks(
        self,
        importance_scores: Dict[str, torch.Tensor]
    ):
        """
        Update binary masks based on importance scores
        """
        for name, score in importance_scores.items():
            if 'weight' in name:
                threshold = torch.quantile(
                    score.flatten(),
                    q=self.sparsity_target
                )
                self.masks[name] = (score > threshold).float()

    def apply_masks(self):
        """
        Apply masks to model weights
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name]

    def prune_step(
        self,
        rewards: torch.Tensor,
        param_gradients: Dict[str, torch.Tensor]
    ):
        """
        Perform one step of RL-based pruning
        """
        # Only prune if mean reward exceeds threshold
        if rewards.mean() > self.reward_threshold:
            importance_scores = self.compute_weight_importance(
                rewards,
                param_gradients
            )
            self.update_masks(importance_scores)
            self.apply_masks()

    def get_sparsity(self) -> float:
        """
        Calculate current sparsity ratio
        """
        total_params = 0
        zero_params = 0
        for mask in self.masks.values():
            total_params += mask.numel()
            zero_params += (mask == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0
