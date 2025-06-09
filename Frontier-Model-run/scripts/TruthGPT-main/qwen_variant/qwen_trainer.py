"""
Qwen model trainer with GRPO integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
import math
import time

@dataclass
class QwenTrainingConfig:
    """Configuration for Qwen training."""
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    use_grpo: bool = True
    grpo_beta: float = 0.1
    grpo_gamma: float = 0.99
    grpo_eps: float = 1e-8
    
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    output_dir: str = "./qwen_checkpoints"

class QwenGRPOLoss(nn.Module):
    """GRPO (Generalized Reward-based Policy Optimization) loss for Qwen."""
    
    def __init__(self, beta=0.1, gamma=0.99, eps=1e-8):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, logits, labels, rewards=None, old_logprobs=None):
        """Compute GRPO loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        token_losses = loss_fct(shift_logits, shift_labels)
        token_losses = token_losses.view(labels.size(0), -1)
        
        if rewards is not None and old_logprobs is not None:
            log_probs = -token_losses
            
            ratio = torch.exp(log_probs - old_logprobs)
            
            advantages = rewards - rewards.mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
            
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.beta, 1 + self.beta) * advantages
            ).mean()
            
            value_loss = F.mse_loss(rewards, torch.zeros_like(rewards))
            
            total_loss = policy_loss + 0.5 * value_loss
        else:
            total_loss = token_losses.mean()
        
        return total_loss

class QwenRewardModel(nn.Module):
    """Reward model for GRPO training."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states):
        """Compute reward scores."""
        pooled_output = hidden_states.mean(dim=1)
        rewards = self.reward_head(pooled_output)
        return rewards.squeeze(-1)

class QwenTrainer:
    """Trainer for Qwen models with GRPO support."""
    
    def __init__(self, model, config: QwenTrainingConfig, tokenizer=None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        if config.use_grpo:
            self.grpo_loss = QwenGRPOLoss(config.grpo_beta, config.grpo_gamma, config.grpo_eps)
            self.reward_model = QwenRewardModel(model.config)
        else:
            self.grpo_loss = None
            self.reward_model = None
        
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.global_step = 0
        self.epoch = 0
        
    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return max(
                0.0, float(self.config.max_steps - current_step) / float(max(1, self.config.max_steps - self.config.warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Perform a single training step."""
        self.model.train()
        
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch.get('labels', input_ids)
        
        if self.config.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                if self.config.use_grpo and self.reward_model is not None:
                    hidden_states = outputs.get('hidden_states', None)
                    if hidden_states is not None:
                        rewards = self.reward_model(hidden_states[-1])
                        old_logprobs = batch.get('old_logprobs', None)
                        
                        loss = self.grpo_loss(
                            outputs['logits'], 
                            labels, 
                            rewards, 
                            old_logprobs
                        )
                    else:
                        loss = outputs['loss']
                else:
                    loss = outputs['loss']
                
                if hasattr(outputs, 'aux_loss') and outputs['aux_loss'] is not None:
                    loss = loss + 0.01 * outputs['aux_loss']
                
                loss = loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
        
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Train the model."""
        print(f"🚀 Starting Qwen training for {self.config.max_steps} steps")
        print(f"📊 Batch size: {self.config.batch_size}")
        print(f"🔄 Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"📈 Learning rate: {self.config.learning_rate}")
        
        if self.config.use_grpo:
            print("🎯 Using GRPO optimization")
        
        total_loss = 0.0
        logging_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            if self.global_step >= self.config.max_steps:
                break
            
            batch = {k: v.to(self.model.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            loss = self.train_step(batch)
            total_loss += loss
            logging_loss += loss
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = logging_loss / self.config.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    print(f"Step {self.global_step}: Loss = {avg_loss:.4f}, LR = {lr:.2e}")
                    logging_loss = 0.0
                
                if eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    print(f"Step {self.global_step}: Eval Loss = {eval_loss:.4f}")
                
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
        
        print(f"✅ Training completed! Total steps: {self.global_step}")
        return total_loss / max(1, self.global_step)
    
    def evaluate(self, eval_dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_eval_loss = 0.0
        num_eval_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.model.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                labels = batch.get('labels', input_ids)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_eval_loss += loss.item()
                num_eval_steps += 1
        
        return total_eval_loss / max(1, num_eval_steps)
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint_path = f"{self.config.output_dir}/checkpoint-{self.global_step}"
        os.makedirs(checkpoint_path, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }, f"{checkpoint_path}/pytorch_model.bin")
        
        print(f"💾 Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(f"{checkpoint_path}/pytorch_model.bin")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        
        print(f"📂 Checkpoint loaded from {checkpoint_path}")

def create_qwen_trainer(model, config: Dict[str, Any], tokenizer=None):
    """Create Qwen trainer from configuration."""
    training_config = QwenTrainingConfig(
        learning_rate=config.get('learning_rate', 1e-4),
        batch_size=config.get('batch_size', 8),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        max_steps=config.get('max_steps', 10000),
        warmup_steps=config.get('warmup_steps', 1000),
        weight_decay=config.get('weight_decay', 0.01),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        use_grpo=config.get('use_grpo', True),
        grpo_beta=config.get('grpo_beta', 0.1),
        grpo_gamma=config.get('grpo_gamma', 0.99),
        grpo_eps=config.get('grpo_eps', 1e-8),
        use_mixed_precision=config.get('use_mixed_precision', True),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', True),
        eval_steps=config.get('eval_steps', 500),
        save_steps=config.get('save_steps', 1000),
        logging_steps=config.get('logging_steps', 100),
        output_dir=config.get('output_dir', './qwen_checkpoints')
    )
    
    return QwenTrainer(model, training_config, tokenizer)
