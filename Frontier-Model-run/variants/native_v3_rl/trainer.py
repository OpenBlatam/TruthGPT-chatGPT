"""
Advanced Trainer for Native DeepSeek-V3 with Reinforcement Learning
Specialized training loop with RL capabilities, MoE optimization, and advanced techniques.
"""

import os
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
import wandb

from .model import NativeV3RLForCausalLM, NativeV3RLConfig

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingState:
    """State for RL training components."""
    step: int = 0
    episode: int = 0
    total_reward: float = 0.0
    episode_rewards: Dict[str, float] = None
    replay_buffer: List[Dict[str, Any]] = None
    value_losses: Dict[str, float] = None
    policy_losses: List[float] = None
    curiosity_losses: List[float] = None
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = defaultdict(float)
        if self.replay_buffer is None:
            self.replay_buffer = deque(maxlen=10000)
        if self.value_losses is None:
            self.value_losses = defaultdict(float)
        if self.policy_losses is None:
            self.policy_losses = []
        if self.curiosity_losses is None:
            self.curiosity_losses = []


class RewardComputer:
    """Computes various reward signals for RL training."""
    
    def __init__(self, config: NativeV3RLConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_types = config.reward_types
        
    def compute_rewards(
        self, 
        input_ids: torch.Tensor, 
        generated_ids: torch.Tensor,
        logits: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute multiple reward signals."""
        rewards = {}
        
        for reward_type in self.reward_types:
            if reward_type == "accuracy":
                rewards[reward_type] = self._compute_accuracy_reward(input_ids, generated_ids, logits)
            elif reward_type == "fluency":
                rewards[reward_type] = self._compute_fluency_reward(generated_ids, logits)
            elif reward_type == "helpfulness":
                rewards[reward_type] = self._compute_helpfulness_reward(input_ids, generated_ids)
            elif reward_type == "safety":
                rewards[reward_type] = self._compute_safety_reward(generated_ids)
            else:
                # Default reward (can be customized)
                rewards[reward_type] = torch.zeros_like(generated_ids, dtype=torch.float)
        
        return rewards
    
    def _compute_accuracy_reward(self, input_ids: torch.Tensor, generated_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute accuracy-based reward (simplified)."""
        # For language modeling, accuracy can be based on perplexity
        with torch.no_grad():
            # Compute perplexity-based reward
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = generated_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses = losses.view(generated_ids.size(0), -1)
            
            # Convert loss to reward (lower loss = higher reward)
            perplexity = torch.exp(losses)
            reward = 1.0 / (1.0 + perplexity)  # Normalize to [0, 1]
            
            return reward
    
    def _compute_fluency_reward(self, generated_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute fluency reward based on confidence and smoothness."""
        with torch.no_grad():
            # Compute confidence (entropy-based)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            
            # Lower entropy = higher confidence = higher fluency
            max_entropy = math.log(self.config.vocab_size)
            confidence = 1.0 - (entropy / max_entropy)
            
            # Smoothness: penalize large changes in probability
            if generated_ids.size(1) > 1:
                prob_diffs = torch.diff(probs.max(dim=-1)[0], dim=1)
                smoothness = 1.0 - torch.abs(prob_diffs)
                
                # Combine confidence and smoothness
                fluency = 0.7 * confidence[:, :-1] + 0.3 * smoothness
                # Pad to match sequence length
                fluency = F.pad(fluency, (0, 1), value=confidence[:, -1])
            else:
                fluency = confidence
            
            return fluency
    
    def _compute_helpfulness_reward(self, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
        """Compute helpfulness reward (simplified heuristic)."""
        # This is a simplified implementation
        # In practice, you might use a trained reward model
        
        batch_size, seq_len = generated_ids.shape
        
        # Decode text for analysis
        rewards = []
        for i in range(batch_size):
            input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            
            # Simple heuristics for helpfulness
            helpfulness_score = 0.0
            
            # Length bonus (not too short, not too long)
            length_ratio = len(generated_text.split()) / max(len(input_text.split()), 1)
            if 0.5 <= length_ratio <= 3.0:
                helpfulness_score += 0.3
            
            # Question answering bonus
            if "?" in input_text and len(generated_text) > 10:
                helpfulness_score += 0.4
            
            # Coherence bonus (simple check for repeated words)
            words = generated_text.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                helpfulness_score += 0.3 * unique_ratio
            
            # Create reward tensor for this sequence
            seq_reward = torch.full((seq_len,), helpfulness_score, dtype=torch.float)
            rewards.append(seq_reward)
        
        return torch.stack(rewards)
    
    def _compute_safety_reward(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Compute safety reward (simplified)."""
        # This is a simplified implementation
        # In practice, you might use a safety classifier
        
        batch_size, seq_len = generated_ids.shape
        
        # Simple safety heuristics
        unsafe_tokens = set()  # Add token IDs for unsafe content
        
        rewards = []
        for i in range(batch_size):
            generated_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True).lower()
            
            # Check for unsafe patterns (simplified)
            safety_score = 1.0  # Start with full safety
            
            # Penalize certain patterns
            unsafe_patterns = ["hate", "violence", "harmful", "dangerous"]
            for pattern in unsafe_patterns:
                if pattern in generated_text:
                    safety_score -= 0.2
            
            safety_score = max(0.0, safety_score)  # Ensure non-negative
            
            # Create reward tensor for this sequence
            seq_reward = torch.full((seq_len,), safety_score, dtype=torch.float)
            rewards.append(seq_reward)
        
        return torch.stack(rewards)


class NativeV3RLTrainer:
    """Advanced trainer for Native DeepSeek-V3 with RL."""
    
    def __init__(
        self,
        model: NativeV3RLForCausalLM,
        config: NativeV3RLConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None,
        accelerator: Optional[Accelerator] = None,
        **kwargs
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        
        # Setup accelerator
        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),
            mixed_precision="bf16" if kwargs.get('bf16', False) else "fp16" if kwargs.get('fp16', False) else "no",
            log_with="wandb" if kwargs.get('use_wandb', False) else None,
        )
        
        # Training parameters
        self.num_epochs = kwargs.get('num_train_epochs', 3)
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.warmup_ratio = kwargs.get('warmup_ratio', 0.1)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.logging_steps = kwargs.get('logging_steps', 100)
        self.eval_steps = kwargs.get('eval_steps', 1000)
        self.save_steps = kwargs.get('save_steps', 2000)
        
        # RL components
        self.reward_computer = RewardComputer(config, self.tokenizer)
        self.rl_state = RLTrainingState()
        
        # Setup optimizers
        self.optimizer = None
        self.scheduler = None
        self.value_optimizers = {}
        self.curiosity_optimizer = None
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        self.best_eval_loss = float('inf')
        
    def setup_optimizers(self, num_training_steps: int):
        """Setup optimizers for different components."""
        # Main optimizer for language modeling and policy
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Scheduler
        num_warmup_steps = int(self.warmup_ratio * num_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Value function optimizers (separate for each reward type)
        if hasattr(self.model.model, 'value_heads'):
            for reward_type in self.config.reward_types:
                if reward_type in self.model.model.value_heads:
                    self.value_optimizers[reward_type] = AdamW(
                        self.model.model.value_heads[reward_type].parameters(),
                        lr=self.learning_rate * 0.5,  # Lower learning rate for value functions
                        weight_decay=self.weight_decay
                    )
        
        # Curiosity optimizer
        if hasattr(self.model.model, 'curiosity_module'):
            self.curiosity_optimizer = AdamW(
                self.model.model.curiosity_module.parameters(),
                lr=self.learning_rate * 0.1,  # Much lower learning rate for curiosity
                weight_decay=self.weight_decay
            )
    
    def train(self):
        """Main training loop."""
        logger.info("Starting Native V3 RL training...")
        
        # Setup data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Very small batch size for large model
            shuffle=True,
            collate_fn=self._data_collator,
            num_workers=2,
            pin_memory=True
        )
        
        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=self._data_collator,
                num_workers=2,
                pin_memory=True
            )
        
        # Calculate training steps
        num_training_steps = len(train_loader) * self.num_epochs
        self.setup_optimizers(num_training_steps)
        
        # Prepare with accelerator
        self.model, self.optimizer, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader
        )
        
        if eval_loader:
            eval_loader = self.accelerator.prepare(eval_loader)
        
        # Training loop
        global_step = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            self.model.train()
            epoch_metrics = defaultdict(list)
            
            for step, batch in enumerate(train_loader):
                # Standard language modeling step
                lm_loss, lm_metrics = self._language_modeling_step(batch)
                epoch_metrics['lm_loss'].append(lm_loss.item())
                
                # RL training step
                if self.config.use_reinforcement_learning and global_step % 100 == 0:
                    rl_loss, rl_metrics = self._rl_training_step(batch)
                    if rl_loss is not None:
                        epoch_metrics['rl_loss'].append(rl_loss.item())
                        for key, value in rl_metrics.items():
                            epoch_metrics[f'rl_{key}'].append(value)
                
                # Combine losses
                total_loss = lm_loss
                if 'rl_loss' in epoch_metrics and epoch_metrics['rl_loss']:
                    total_loss = total_loss + 0.1 * rl_loss  # Small RL coefficient
                
                # Backward pass
                self.accelerator.backward(total_loss)
                
                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                self.rl_state.step = global_step
                
                # Logging
                if global_step % self.logging_steps == 0:
                    self._log_metrics(epoch_metrics, global_step, epoch)
                
                # Evaluation
                if eval_loader and global_step % self.eval_steps == 0:
                    eval_metrics = self._evaluate(eval_loader)
                    self._log_eval_metrics(eval_metrics, global_step)
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval_loss']
                        self._save_model("best")
                
                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self._save_model(f"checkpoint-{global_step}")
        
        # Final save
        self._save_model("final")
        logger.info("Training completed!")
    
    def _language_modeling_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Standard language modeling training step."""
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['input_ids'],  # Causal LM
            return_dict=True
        )
        
        loss = outputs['loss']
        
        metrics = {
            'perplexity': torch.exp(loss).item(),
            'num_tokens': batch['input_ids'].numel()
        }
        
        # Add MoE auxiliary losses
        if 'aux_losses' in outputs and outputs['aux_losses']:
            aux_loss = sum(
                aux_loss_dict.get('load_balancing_loss', 0.0)
                for aux_loss_dict in outputs['aux_losses']
            ) / len(outputs['aux_losses'])
            loss = loss + 0.01 * aux_loss
            metrics['aux_loss'] = aux_loss.item()
        
        return loss, metrics
    
    def _rl_training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Reinforcement learning training step."""
        if not self.config.use_reinforcement_learning:
            return None, {}
        
        # Generate sequences for RL training
        with torch.no_grad():
            generated_outputs = self.model.generate_with_rl(
                input_ids=batch['input_ids'][:, :10],  # Use first 10 tokens as prompt
                max_length=50,
                temperature=0.8,
                do_sample=True
            )
        
        generated_ids = generated_outputs['generated_ids']
        old_log_probs = generated_outputs['log_probs']
        
        # Compute rewards
        rewards = self.reward_computer.compute_rewards(
            input_ids=batch['input_ids'],
            generated_ids=generated_ids,
            logits=None  # Would need to recompute if needed
        )
        
        # Forward pass with RL
        outputs = self.model(
            input_ids=generated_ids,
            labels=generated_ids,
            rewards=rewards,
            old_log_probs=old_log_probs,
            compute_values=True,
            return_dict=True
        )
        
        rl_info = outputs.get('rl_info', {})
        
        # Collect RL losses
        total_rl_loss = 0.0
        metrics = {}
        
        if 'policy_loss' in rl_info:
            total_rl_loss += rl_info['policy_loss']
            metrics['policy_loss'] = rl_info['policy_loss'].item()
        
        if 'value_loss' in rl_info:
            total_rl_loss += self.config.value_loss_coef * rl_info['value_loss']
            metrics['value_loss'] = rl_info['value_loss'].item()
        
        if 'entropy' in rl_info:
            metrics['entropy'] = rl_info['entropy'].item()
        
        if 'curiosity_loss' in rl_info:
            total_rl_loss += self.config.curiosity_coef * rl_info['curiosity_loss']
            metrics['curiosity_loss'] = rl_info['curiosity_loss'].item()
        
        # Update RL state
        for reward_type, reward_tensor in rewards.items():
            self.rl_state.episode_rewards[reward_type] += reward_tensor.mean().item()
        
        return total_rl_loss if total_rl_loss > 0 else None, metrics
    
    def _evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluation loop."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['input_ids'],
                    return_dict=True
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                total_tokens += batch['input_ids'].numel()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_tokens': total_tokens
        }
    
    def _data_collator(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Data collator for batching."""
        # Simple collator - in practice, you might want more sophisticated padding
        if isinstance(examples[0], dict):
            batch = {}
            for key in examples[0].keys():
                batch[key] = torch.stack([torch.tensor(ex[key]) for ex in examples])
            return batch
        else:
            # Assume examples are token sequences
            max_length = max(len(ex) for ex in examples)
            padded = []
            attention_masks = []
            
            for ex in examples:
                padded_ex = ex + [self.tokenizer.pad_token_id] * (max_length - len(ex))
                attention_mask = [1] * len(ex) + [0] * (max_length - len(ex))
                
                padded.append(padded_ex)
                attention_masks.append(attention_mask)
            
            return {
                'input_ids': torch.tensor(padded),
                'attention_mask': torch.tensor(attention_masks)
            }
    
    def _log_metrics(self, metrics: Dict[str, List[float]], step: int, epoch: int):
        """Log training metrics."""
        if self.accelerator.is_main_process:
            avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items() if values}
            
            log_dict = {
                'train/step': step,
                'train/epoch': epoch,
                'train/learning_rate': self.scheduler.get_last_lr()[0],
            }
            log_dict.update({f'train/{key}': value for key, value in avg_metrics.items()})
            
            # Add RL state metrics
            if self.rl_state.episode_rewards:
                for reward_type, reward_value in self.rl_state.episode_rewards.items():
                    log_dict[f'rl/reward_{reward_type}'] = reward_value
            
            # Log to wandb if available
            if hasattr(self.accelerator, 'log'):
                self.accelerator.log(log_dict, step=step)
            
            # Console logging
            logger.info(f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in avg_metrics.items()]))
    
    def _log_eval_metrics(self, metrics: Dict[str, float], step: int):
        """Log evaluation metrics."""
        if self.accelerator.is_main_process:
            log_dict = {f'eval/{key}': value for key, value in metrics.items()}
            
            if hasattr(self.accelerator, 'log'):
                self.accelerator.log(log_dict, step=step)
            
            logger.info(f"Eval at step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
    
    def _save_model(self, name: str):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join("./output", name)
            os.makedirs(save_path, exist_ok=True)
            
            # Save model state
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "model.pt"))
            
            # Save config
            torch.save(self.config, os.path.join(save_path, "config.pt"))
            
            # Save tokenizer
            if hasattr(self.tokenizer, 'save_pretrained'):
                self.tokenizer.save_pretrained(save_path)
            
            # Save training state
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'rl_state': self.rl_state,
                'best_eval_loss': self.best_eval_loss,
            }, os.path.join(save_path, "training_state.pt"))
            
            logger.info(f"Model saved to {save_path}")


# Export the trainer
__all__ = ['NativeV3RLTrainer', 'RewardComputer', 'RLTrainingState']