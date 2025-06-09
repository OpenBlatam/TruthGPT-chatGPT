"""
Generative AI Training Integration for TruthGPT
GRPO-enhanced training for generative models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging

@dataclass
class GenerativeTrainingConfig:
    """Configuration for generative model training."""
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
    
    use_adversarial_training: bool = True
    adversarial_weight: float = 0.1
    quality_weight: float = 1.0
    diversity_weight: float = 0.5
    
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_curriculum_learning: bool = True
    
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    output_dir: str = "./ia_generative_checkpoints"

class GenerativeGRPOLoss(nn.Module):
    """GRPO loss function for generative models."""
    
    def __init__(self, config: GenerativeTrainingConfig):
        super().__init__()
        self.config = config
        self.beta = config.grpo_beta
        self.gamma = config.grpo_gamma
        self.eps = config.grpo_eps
        
    def forward(self, 
                generated_outputs: torch.Tensor,
                target_outputs: torch.Tensor,
                quality_scores: torch.Tensor,
                diversity_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        generation_loss = nn.functional.mse_loss(generated_outputs, target_outputs)
        
        quality_reward = quality_scores.mean()
        
        diversity_reward = diversity_scores.mean()
        
        total_reward = (self.config.quality_weight * quality_reward + 
                       self.config.diversity_weight * diversity_reward)
        
        policy_loss = -torch.log(total_reward + self.eps) * self.beta
        
        total_loss = generation_loss + policy_loss
        
        return {
            'total_loss': total_loss,
            'generation_loss': generation_loss,
            'policy_loss': policy_loss,
            'quality_reward': quality_reward,
            'diversity_reward': diversity_reward
        }

class GenerativeRewardModel(nn.Module):
    """Reward model for evaluating generated content quality."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.quality_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.diversity_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quality_scores = self.quality_head(features)
        diversity_scores = self.diversity_head(features)
        return quality_scores, diversity_scores

class GenerativeTrainer:
    """Enhanced trainer for generative models with GRPO integration."""
    
    def __init__(self, 
                 model: nn.Module,
                 config: GenerativeTrainingConfig,
                 tokenizer: Optional[Any] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = GenerativeGRPOLoss(config)
        self.reward_model = GenerativeRewardModel(
            input_dim=getattr(model.config, 'hidden_size', 512)
        )
        
        if config.use_mixed_precision:
            self.scaler = GradScaler()
        
        if config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        self.global_step = 0
        self.epoch = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_optimizer(self) -> optim.Optimizer:
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
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with GRPO."""
        self.model.train()
        
        if self.config.use_mixed_precision:
            with autocast():
                outputs = self._forward_step(batch)
                loss_dict = self._compute_loss(outputs, batch)
        else:
            outputs = self._forward_step(batch)
            loss_dict = self._compute_loss(outputs, batch)
        
        total_loss = loss_dict['total_loss']
        
        if self.config.use_mixed_precision:
            self.scaler.scale(total_loss).backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        if self.global_step < self.config.warmup_steps:
            self.scheduler.step()
        
        self.global_step += 1
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        return {
            'logits': outputs.logits if hasattr(outputs, 'logits') else outputs,
            'hidden_states': getattr(outputs, 'hidden_states', None)
        }
    
    def _compute_loss(self, 
                     outputs: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute GRPO loss."""
        generated_outputs = outputs['logits']
        target_outputs = batch.get('labels', batch.get('input_ids'))
        
        if outputs['hidden_states'] is not None:
            features = outputs['hidden_states'][-1].mean(dim=1)  # Pool over sequence
        else:
            features = generated_outputs.mean(dim=1)
        
        quality_scores, diversity_scores = self.reward_model(features)
        
        return self.loss_fn(
            generated_outputs=generated_outputs,
            target_outputs=target_outputs,
            quality_scores=quality_scores,
            diversity_scores=diversity_scores
        )
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self._forward_step(batch)
                loss_dict = self._compute_loss(outputs, batch)
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
        
        return {'eval_loss': total_loss / num_batches}
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save model checkpoint."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")

class AdversarialTrainer:
    """Adversarial training for generative models."""
    
    def __init__(self, generator, discriminator, lr=1e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
        self.disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
        
    def train_step(self, real_data, fake_data):
        """Single training step for adversarial training."""
        self.disc_optimizer.zero_grad()
        real_loss = nn.BCELoss()(self.discriminator(real_data), torch.ones_like(real_data[:, 0]))
        fake_loss = nn.BCELoss()(self.discriminator(fake_data.detach()), torch.zeros_like(fake_data[:, 0]))
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        self.gen_optimizer.zero_grad()
        gen_loss = nn.BCELoss()(self.discriminator(fake_data), torch.ones_like(fake_data[:, 0]))
        gen_loss.backward()
        self.gen_optimizer.step()
        
        return {'disc_loss': disc_loss.item(), 'gen_loss': gen_loss.item()}
class CurriculumLearning:
    """Curriculum learning for progressive training difficulty."""
    
    def __init__(self, initial_difficulty=0.1, max_difficulty=1.0, step_size=0.1):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.step_size = step_size
        
    def update_difficulty(self, performance_metric):
        """Update curriculum difficulty based on performance."""
        if performance_metric > 0.8:  # Good performance, increase difficulty
            self.current_difficulty = min(self.max_difficulty, 
                                        self.current_difficulty + self.step_size)
        elif performance_metric < 0.5:  # Poor performance, decrease difficulty
            self.current_difficulty = max(0.1, 
                                        self.current_difficulty - self.step_size)
        return self.current_difficulty
    
    def get_current_difficulty(self):
        """Get current curriculum difficulty level."""
        return self.current_difficulty



def create_generative_trainer(model: nn.Module, 
                            config: Dict[str, Any],
                            tokenizer: Optional[Any] = None) -> GenerativeTrainer:
    """Factory function to create a generative trainer."""
    training_config = GenerativeTrainingConfig(**config)
    return GenerativeTrainer(model, training_config, tokenizer)
