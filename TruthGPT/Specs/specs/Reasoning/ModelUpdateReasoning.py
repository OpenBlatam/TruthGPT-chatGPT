import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple

class ModelUpdateReasoning:
    def __init__(self, 
                 input_dim: int,
                 scaling_factor: float = 1.0,
                 epsilon: float = 1e-8,
                 preference_margin: float = 0.0):
        """
        Initialize the ModelUpdateReasoning module with reward modeling capabilities.
        
        Args:
            input_dim (int): The dimensionality of the input features
            scaling_factor (float): Scaling factor for normalized output
            epsilon (float): Small constant for numerical stability
            preference_margin (float): Margin for preference-based training
        """
        self.input_dim = input_dim
        self.scaling_factor = scaling_factor
        self.epsilon = epsilon
        self.preference_margin = preference_margin
        
        # Initialize reward model head
        self.reward_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def compute_reward_loss(self,
                          chosen_rewards: torch.Tensor,
                          rejected_rewards: torch.Tensor,
                          preference_margin: Optional[float] = None) -> torch.Tensor:
        """
        Compute the reward model loss using preference-based training.
        
        Args:
            chosen_rewards: Rewards for preferred completions
            rejected_rewards: Rewards for rejected completions
            preference_margin: Optional margin for preference (overrides default)
            
        Returns:
            Loss tensor
        """
        margin = preference_margin if preference_margin is not None else self.preference_margin
        return -nn.functional.logsigmoid(chosen_rewards - rejected_rewards - margin).mean()
    
    def train_reward_model(self,
                         chosen_features: torch.Tensor,
                         rejected_features: torch.Tensor,
                         learning_rate: float = 1e-4,
                         num_epochs: int = 1) -> Dict[str, float]:
        """
        Train the reward model using preference data.
        
        Args:
            chosen_features: Features from preferred completions
            rejected_features: Features from rejected completions
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            
        Returns:
            Training metrics
        """
        optimizer = torch.optim.Adam(self.reward_head.parameters(), lr=learning_rate)
        losses = []
        
        for _ in range(num_epochs):
            # Forward pass
            chosen_rewards = self.reward_head(chosen_features)
            rejected_rewards = self.reward_head(rejected_features)
            
            # Compute loss
            loss = self.compute_reward_loss(chosen_rewards, rejected_rewards)
            losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return {
            "final_loss": float(losses[-1]),
            "mean_loss": float(np.mean(losses))
        }
    
    def evaluate_reward_model(self,
                            test_features: torch.Tensor,
                            test_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the reward model on test data.
        
        Args:
            test_features: Test features
            test_labels: Binary labels (1 for preferred, 0 for rejected)
            
        Returns:
            Evaluation metrics
        """
        with torch.no_grad():
            rewards = self.reward_head(test_features)
            predictions = (rewards > 0).float()
            
            accuracy = (predictions == test_labels).float().mean()
            auc = torch.tensor([
                (rewards[i] > rewards[j]).float()
                for i in range(len(test_labels))
                for j in range(len(test_labels))
                if test_labels[i] > test_labels[j]
            ]).mean()
            
        return {
            "accuracy": float(accuracy),
            "auc": float(auc)
        }
    
    def check_compatibility(self, 
                          old_model_features: np.ndarray,
                          new_features: np.ndarray) -> Dict[str, float]:
        """
        Check compatibility between old and new model features using reward model.
        
        Args:
            old_model_features: Features from the old model
            new_features: Features from the new implementation
            
        Returns:
            Dict containing compatibility metrics
        """
        # Convert to torch tensors
        old_tensor = torch.from_numpy(old_model_features).float()
        new_tensor = torch.from_numpy(new_features).float()
        
        # Get reward predictions
        with torch.no_grad():
            old_rewards = self.reward_head(old_tensor)
            new_rewards = self.reward_head(new_tensor)
        
        # Compute compatibility metrics
        reward_diff = torch.abs(old_rewards - new_rewards).mean()
        reward_correlation = torch.corrcoef(
            torch.stack([old_rewards.squeeze(), new_rewards.squeeze()])
        )[0, 1]
        
        return {
            "reward_difference": float(reward_diff),
            "reward_correlation": float(reward_correlation),
            "compatibility_score": float(1.0 - reward_diff)
        }
    
    def normalize_features(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RMS normalization to input features.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Compute RMS normalization
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.epsilon)
        x_normalized = x / rms
        
        return self.scaling_factor * x_normalized
    
    def assess_update_necessity(self,
                              model_age: float,
                              performance_metrics: Dict[str, float],
                              bug_reports: List[str],
                              reward_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Union[float, bool]]:
        """
        Assess whether a model update is necessary based on various factors.
        
        Args:
            model_age: Time since last update (in days)
            performance_metrics: Dictionary of current performance metrics
            bug_reports: List of reported issues
            reward_metrics: Optional reward model evaluation metrics
            
        Returns:
            Assessment results including update recommendation
        """
        # Age factor (normalized to [0,1])
        age_factor = min(model_age / 365.0, 1.0)
        
        # Performance degradation factor
        perf_threshold = 0.8
        perf_factor = 1.0 - (
            sum(performance_metrics.values()) / len(performance_metrics)
        ) / perf_threshold
        
        # Bug severity factor
        bug_factor = min(len(bug_reports) / 10.0, 1.0)
        
        # Reward model factor (if provided)
        reward_factor = 0.0
        if reward_metrics:
            reward_factor = (
                reward_metrics.get("accuracy", 0.0) +
                reward_metrics.get("auc", 0.0)
            ) / 2.0
        
        # Combined update score with reward model consideration
        update_score = (
            0.3 * age_factor +
            0.3 * perf_factor +
            0.2 * bug_factor +
            0.2 * reward_factor
        )
        
        return {
            "update_score": float(update_score),
            "update_recommended": update_score > 0.5,
            "age_factor": float(age_factor),
            "performance_factor": float(perf_factor),
            "bug_factor": float(bug_factor),
            "reward_factor": float(reward_factor)
        }
    
    def validate_update(self,
                       old_model_output: np.ndarray,
                       new_model_output: np.ndarray,
                       threshold: float = 0.1) -> Dict[str, Union[float, bool]]:
        """
        Validate the model update by comparing outputs using reward model.
        
        Args:
            old_model_output: Output from the old model
            new_model_output: Output from the updated model
            threshold: Maximum acceptable difference
            
        Returns:
            Validation results
        """
        # Convert to torch tensors
        old_tensor = torch.from_numpy(old_model_output).float()
        new_tensor = torch.from_numpy(new_model_output).float()
        
        # Get reward predictions
        with torch.no_grad():
            old_rewards = self.reward_head(old_tensor)
            new_rewards = self.reward_head(new_tensor)
        
        # Compute difference metrics
        reward_diff = torch.abs(old_rewards - new_rewards).mean()
        reward_improvement = (new_rewards > old_rewards).float().mean()
        
        return {
            "reward_difference": float(reward_diff),
            "reward_improvement": float(reward_improvement),
            "update_valid": float(reward_diff) <= threshold,
            "consistency_score": float(1.0 - reward_diff)
        } 