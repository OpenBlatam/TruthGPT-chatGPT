import os
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict, IterableDataset
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser
from rich.logging import RichHandler
import logging

@dataclass
class KFGRPOScriptArguments(ScriptArguments):
    """Script arguments for the KF-GRPO training script."""
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    # Kalman Filter parameters
    process_noise: float = field(default=0.01, metadata={"help": "Process noise covariance (Q)"})
    measurement_noise: float = field(default=0.1, metadata={"help": "Measurement noise covariance (R)"})
    
    # CPPO parameters
    pruning_threshold: float = field(default=0.1, metadata={"help": "Threshold for sample pruning"})
    pruning_alpha: float = field(default=0.5, metadata={"help": "Alpha for dynamic K adjustment"})
    k_min: int = field(default=1, metadata={"help": "Minimum K value"})
    k_max: int = field(default=10, metadata={"help": "Maximum K value"})
    
    # AGPO parameters
    policy_clip_delta: float = field(default=0.2, metadata={"help": "Policy clipping delta"})
    length_penalty_lambda: float = field(default=0.1, metadata={"help": "Length penalty coefficient"})
    max_length: int = field(default=1000, metadata={"help": "Maximum sequence length for normalization"})

class KalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float):
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.mu = 0.0  # State estimate
        self.P = 1.0  # Error covariance
        
    def update(self, measurement: float) -> float:
        # Prediction
        mu_pred = self.mu
        P_pred = self.P + self.Q
        
        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.mu = mu_pred + K * (measurement - mu_pred)
        self.P = (1 - K) * P_pred + self.Q
        
        return self.mu

class KFGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[Any, List[Any]],
        args: Any = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, DatasetDict, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[Any] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # Initialize Kalman filter for reward estimation
        self.kf = KalmanFilter(
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise
        )
        
        # Initialize metrics
        self._metrics = {
            "kalman_reward": [],
            "pruned_samples": [],
            "length_penalty": []
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        # Get base GRPO loss
        loss = super().compute_loss(model, inputs, return_outputs)
        
        if return_outputs:
            return loss
            
        # Apply Kalman filtering to rewards
        rewards = self._get_rewards(inputs)
        filtered_rewards = torch.tensor([
            self.kf.update(r.item()) for r in rewards
        ], device=rewards.device)
        
        # Apply CPPO pruning
        advantages = self._compute_advantages(filtered_rewards)
        pruned_mask = torch.abs(advantages) > self.args.pruning_threshold
        pruned_advantages = advantages[pruned_mask]
        
        # Dynamic K adjustment
        pruning_ratio = pruned_mask.float().mean()
        k_next = torch.clamp(
            self.args.pruning_alpha * pruning_ratio,
            self.args.k_min,
            self.args.k_max
        )
        
        # Apply AGPO length penalty
        sequence_lengths = self._get_sequence_lengths(inputs)
        length_penalties = self.args.length_penalty_lambda * (sequence_lengths / self.args.max_length)
        penalized_rewards = filtered_rewards - length_penalties
        
        # Update metrics
        self._metrics["kalman_reward"].append(filtered_rewards.mean().item())
        self._metrics["pruned_samples"].append(pruning_ratio.item())
        self._metrics["length_penalty"].append(length_penalties.mean().item())
        
        # Combine losses
        final_loss = loss + self._compute_additional_losses(
            penalized_rewards,
            pruned_advantages,
            k_next
        )
        
        return final_loss

    def _compute_additional_losses(self, rewards, advantages, k):
        # Policy clipping loss
        policy_ratio = self._compute_policy_ratio()
        clipped_ratio = torch.clamp(
            policy_ratio,
            1 - self.args.policy_clip_delta,
            1 + self.args.policy_clip_delta
        )
        policy_loss = -torch.min(
            policy_ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss

    def _compute_policy_ratio(self):
        # Compute policy ratio between current and old policy
        current_logits = self._get_current_logits()
        old_logits = self._get_old_logits()
        return torch.exp(current_logits - old_logits)

    def _get_sequence_lengths(self, inputs):
        # Get sequence lengths from inputs
        return torch.tensor([
            len(input["input_ids"]) for input in inputs
        ], device=self.model.device)

    def _compute_advantages(self, rewards):
        # Compute advantages using filtered rewards
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        return (rewards - mean_reward) / std_reward

def main(script_args: KFGRPOScriptArguments, training_args: Any, model_args: Any) -> None:
    logger = setup_logging(training_args.get_process_log_level())
    set_seed(training_args.seed)
    
    # Initialize trainer
    trainer = KFGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=script_args.reward_funcs,
        args=script_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_processing_classes=reward_processing_classes,
        callbacks=callbacks,
        optimizers=optimizers,
        peft_config=peft_config,
    )
    
    # Train and evaluate
    trainer.train()
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((KFGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args) 