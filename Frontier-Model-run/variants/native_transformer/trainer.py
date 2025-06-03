#!/usr/bin/env python3
"""
Native Transformer Trainer with Advanced Optimization Techniques
"""

import os
import sys
import time
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Union, Tuple
import yaml
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset, DatasetDict
from accelerate import Accelerator
import wandb

from model import NativeTransformerForCausalLM, NativeTransformerConfig
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from loguru import logger


# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler('native_transformer_training.log')
    ]
)


@dataclass
class NativeTransformerTrainingArguments:
    """Training arguments for Native Transformer."""
    
    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Dataset configuration
    dataset_name: str = "your_dataset"
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    max_length: int = 2048
    
    # Training parameters
    output_dir: str = "./native_transformer_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw"
    lr_scheduler_type: str = "cosine"
    num_cycles: int = 1
    use_amp: bool = True
    fp16: bool = True
    bf16: bool = False
    
    # Advanced features
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_compile: bool = True
    
    # Evaluation and logging
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Distributed training
    local_rank: int = -1
    
    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "native-transformer"
    wandb_run_name: Optional[str] = None
    
    # Advanced training techniques
    use_curriculum_learning: bool = True
    use_label_smoothing: bool = True
    label_smoothing_factor: float = 0.1
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # Memory optimization
    use_cpu_offload: bool = False
    use_activation_checkpointing: bool = True
    
    # Model specific
    use_adaptive_attention: bool = True
    use_sparse_attention: bool = True
    use_rotary_embeddings: bool = True
    attention_window_size: int = 512
    sparse_attention_ratio: float = 0.1


class CurriculumDataset(Dataset):
    """Dataset with curriculum learning support."""
    
    def __init__(self, dataset, tokenizer, max_length=2048, curriculum_stage=0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.curriculum_stage = curriculum_stage
        
        # Sort by difficulty (length as proxy)
        self.sorted_indices = sorted(
            range(len(dataset)), 
            key=lambda i: len(self.tokenizer.encode(dataset[i]['text'] if 'text' in dataset[i] else str(dataset[i])))
        )
        
    def __len__(self):
        # Gradually increase dataset size
        stage_size = len(self.dataset) // 4
        return min(len(self.dataset), stage_size * (self.curriculum_stage + 1))
    
    def __getitem__(self, idx):
        # Use sorted indices for curriculum learning
        actual_idx = self.sorted_indices[idx]
        item = self.dataset[actual_idx]
        
        # Tokenize
        if 'text' in item:
            text = item['text']
        else:
            text = str(item)
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class AdaptiveLossFunction(nn.Module):
    """Adaptive loss function with multiple components."""
    
    def __init__(self, vocab_size, label_smoothing=0.1, use_focal_loss=False, focal_alpha=1.0, focal_gamma=2.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, logits, labels):
        # Standard cross-entropy with label smoothing
        if self.label_smoothing > 0:
            # Label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            smooth_labels = torch.zeros_like(log_probs).fill_(self.label_smoothing / (self.vocab_size - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
            loss = -torch.sum(smooth_labels * log_probs, dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, labels)
        
        # Focal loss for hard examples
        if self.use_focal_loss:
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
            loss = focal_loss.mean()
        
        return loss


class NativeTransformerTrainer:
    """Advanced trainer for Native Transformer."""
    
    def __init__(self, args: NativeTransformerTrainingArguments):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16" if args.fp16 else "bf16" if args.bf16 else "no",
            log_with="wandb" if args.use_wandb else None,
            project_dir=args.output_dir
        )
        
        # Setup logging
        if self.accelerator.is_main_process and args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=args.__dict__
            )
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if args.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.curriculum_stage = 0
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model config
        config = NativeTransformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=self.args.max_length,
            use_adaptive_attention=self.args.use_adaptive_attention,
            use_sparse_attention=self.args.use_sparse_attention,
            use_rotary_embeddings=self.args.use_rotary_embeddings,
            attention_window_size=self.args.attention_window_size,
            sparse_attention_ratio=self.args.sparse_attention_ratio,
            **self.args.model_config
        )
        
        # Create model
        self.model = NativeTransformerForCausalLM(config)
        
        # Enable gradient checkpointing
        if self.args.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Compile model
        if self.args.use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def setup_optimizer_and_scheduler(self, num_training_steps):
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer and scheduler...")
        
        # Optimizer
        if self.args.optimizer_type == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.args.optimizer_type}")
        
        # Scheduler
        num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)
        
        if self.args.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.args.num_cycles
            )
        elif self.args.lr_scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.args.lr_scheduler_type}")
    
    def setup_datasets(self):
        """Setup training and evaluation datasets."""
        logger.info("Loading datasets...")
        
        # Load dataset
        if self.args.dataset_config:
            dataset = load_dataset(self.args.dataset_name, self.args.dataset_config)
        else:
            dataset = load_dataset(self.args.dataset_name)
        
        # Create curriculum datasets
        train_dataset = CurriculumDataset(
            dataset[self.args.dataset_train_split],
            self.tokenizer,
            max_length=self.args.max_length,
            curriculum_stage=self.curriculum_stage
        )
        
        eval_dataset = CurriculumDataset(
            dataset[self.args.dataset_test_split],
            self.tokenizer,
            max_length=self.args.max_length,
            curriculum_stage=3  # Use full dataset for evaluation
        )
        
        return train_dataset, eval_dataset
    
    def create_dataloaders(self, train_dataset, eval_dataset):
        """Create data loaders."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, eval_loader
    
    def train_step(self, batch, loss_fn):
        """Single training step."""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        with autocast(enabled=self.args.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Use adaptive loss
            logits = outputs['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        if self.args.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item()
    
    def evaluate(self, eval_loader, loss_fn):
        """Evaluation loop."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast(enabled=self.args.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    logits = outputs['logits']
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup
        self.setup_model_and_tokenizer()
        train_dataset, eval_dataset = self.setup_datasets()
        train_loader, eval_loader = self.create_dataloaders(train_dataset, eval_dataset)
        
        # Calculate training steps
        num_training_steps = len(train_loader) * self.args.num_train_epochs // self.args.gradient_accumulation_steps
        self.setup_optimizer_and_scheduler(num_training_steps)
        
        # Prepare with accelerator
        self.model, self.optimizer, train_loader, eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, eval_loader
        )
        
        # Loss function
        loss_fn = AdaptiveLossFunction(
            vocab_size=self.tokenizer.vocab_size,
            label_smoothing=self.args.label_smoothing_factor if self.args.use_label_smoothing else 0.0
        )
        
        # Training loop
        best_eval_loss = float('inf')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            
            task = progress.add_task("Training", total=num_training_steps)
            
            for epoch in range(self.args.num_train_epochs):
                self.epoch = epoch
                
                # Update curriculum stage
                if self.args.use_curriculum_learning:
                    self.curriculum_stage = min(3, epoch // 2)
                    train_dataset.curriculum_stage = self.curriculum_stage
                
                epoch_loss = 0
                num_batches = 0
                
                for step, batch in enumerate(train_loader):
                    # Training step
                    loss = self.train_step(batch, loss_fn)
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Gradient accumulation
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if self.args.use_amp:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                            self.optimizer.step()
                        
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        
                        progress.update(task, advance=1)
                        
                        # Logging
                        if self.global_step % self.args.logging_steps == 0:
                            avg_loss = epoch_loss / num_batches
                            lr = self.scheduler.get_last_lr()[0]
                            
                            logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                            
                            if self.accelerator.is_main_process and self.args.use_wandb:
                                wandb.log({
                                    "train/loss": avg_loss,
                                    "train/learning_rate": lr,
                                    "train/epoch": epoch,
                                    "train/curriculum_stage": self.curriculum_stage
                                }, step=self.global_step)
                        
                        # Evaluation
                        if self.global_step % self.args.eval_steps == 0:
                            eval_loss = self.evaluate(eval_loader, loss_fn)
                            logger.info(f"Evaluation loss: {eval_loss:.4f}")
                            
                            if self.accelerator.is_main_process and self.args.use_wandb:
                                wandb.log({"eval/loss": eval_loss}, step=self.global_step)
                            
                            # Save best model
                            if eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                self.save_model("best")
                        
                        # Save checkpoint
                        if self.global_step % self.args.save_steps == 0:
                            self.save_model(f"checkpoint-{self.global_step}")
        
        # Final save
        self.save_model("final")
        logger.info("Training completed!")
    
    def save_model(self, name):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.args.output_dir, name)
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "model.pt"))
            
            # Save config
            with open(os.path.join(save_path, "config.yaml"), "w") as f:
                yaml.dump(self.args.__dict__, f)
            
            logger.info(f"Model saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Native Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create training arguments
    training_args = NativeTransformerTrainingArguments(**config)
    
    # Set seed
    set_seed(42)
    
    # Create trainer and start training
    trainer = NativeTransformerTrainer(training_args)
    trainer.train()


if __name__ == "__main__":
    main()