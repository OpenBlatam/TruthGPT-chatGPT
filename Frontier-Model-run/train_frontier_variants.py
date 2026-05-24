#!/usr/bin/env python3
"""
Unified Training Script for Frontier Model Variants
Supports all implemented variants without using DeepSeek.
"""

import os
import sys
import argparse
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, set_seed
from accelerate import Accelerator
import wandb

# Add variants to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'variants'))

from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from loguru import logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler('frontier_training.log')
    ]
)

console = Console()


class FrontierModelFactory:
    """Factory for creating different frontier model variants."""
    
    @staticmethod
    def create_model(variant: str, config: Dict[str, Any]):
        """Create model based on variant type."""
        
        if variant == "native_transformer":
            from native_transformer.model import NativeTransformerForCausalLM, NativeTransformerConfig
            model_config = NativeTransformerConfig(**config.get('model_config', {}))
            return NativeTransformerForCausalLM(model_config)
            
        elif variant == "mixture_of_experts":
            from mixture_of_experts.model import MoETransformerForCausalLM, MoETransformerConfig
            model_config = MoETransformerConfig(**config.get('model_config', {}))
            return MoETransformerForCausalLM(model_config)
            
        elif variant == "retrieval_augmented":
            from retrieval_augmented.model import RAGTransformerForCausalLM, RAGTransformerConfig
            model_config = RAGTransformerConfig(**config.get('model_config', {}))
            return RAGTransformerForCausalLM(model_config)
            
        elif variant == "multi_modal":
            from multi_modal.model import MultiModalTransformerForCausalLM, MultiModalTransformerConfig
            model_config = MultiModalTransformerConfig(**config.get('model_config', {}))
            return MultiModalTransformerForCausalLM(model_config)
            
        elif variant == "reinforcement_learning":
            from reinforcement_learning.model import RLTransformerForCausalLM, RLTransformerConfig
            model_config = RLTransformerConfig(**config.get('model_config', {}))
            return RLTransformerForCausalLM(model_config)
            
        else:
            raise ValueError(f"Unknown variant: {variant}")

    @staticmethod
    def get_available_variants():
        """Get list of available variants."""
        variants_dir = Path(__file__).parent / "variants"
        variants = []
        
        for variant_dir in variants_dir.iterdir():
            if variant_dir.is_dir() and (variant_dir / "model.py").exists():
                variants.append(variant_dir.name)
        
        return variants


class UnifiedTrainer:
    """Unified trainer for all frontier model variants."""
    
    def __init__(self, variant: str, config_path: str):
        self.variant = variant
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            mixed_precision="fp16" if self.config.get('fp16', False) else "bf16" if self.config.get('bf16', False) else "no",
            log_with="wandb" if self.config.get('use_wandb', False) else None,
            project_dir=self.config.get('output_dir', './output')
        )
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Setup logging
        if self.accelerator.is_main_process and self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'frontier-models'),
                name=self.config.get('wandb_run_name', f'{variant}-experiment'),
                config=self.config
            )

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        logger.info(f"Setting up {self.variant} model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update vocab size in config
        if 'model_config' not in self.config:
            self.config['model_config'] = {}
        self.config['model_config']['vocab_size'] = self.tokenizer.vocab_size
        
        # Create model
        self.model = FrontierModelFactory.create_model(self.variant, self.config)
        
        # Enable gradient checkpointing
        if self.config.get('use_gradient_checkpointing', False):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        
        # Compile model
        if self.config.get('use_compile', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created with {num_params:,} parameters")

    def setup_datasets(self):
        """Setup training and evaluation datasets."""
        logger.info("Loading datasets...")
        
        from datasets import load_dataset
        
        # Load dataset
        dataset_name = self.config.get('dataset_name', 'wikitext')
        dataset_config = self.config.get('dataset_config')
        
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config)
        else:
            dataset = load_dataset(dataset_name)
        
        # Get splits
        train_split = self.config.get('dataset_train_split', 'train')
        test_split = self.config.get('dataset_test_split', 'validation')
        
        train_dataset = dataset[train_split]
        eval_dataset = dataset[test_split] if test_split in dataset else None
        
        return train_dataset, eval_dataset

    def create_data_collator(self):
        """Create data collator for the specific variant."""
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
            pad_to_multiple_of=8 if self.config.get('fp16', False) else None,
        )

    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer and scheduler...")
        
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 5e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Scheduler
        num_warmup_steps = int(self.config.get('warmup_ratio', 0.1) * num_training_steps)
        
        scheduler_type = self.config.get('lr_scheduler_type', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.config.get('num_cycles', 1)
            )
        elif scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.variant} variant...")
        
        # Setup components
        self.setup_model_and_tokenizer()
        train_dataset, eval_dataset = self.setup_datasets()
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
        data_collator = self.create_data_collator()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('per_device_train_batch_size', 4),
            shuffle=True,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.get('per_device_eval_batch_size', 4),
                shuffle=False,
                collate_fn=data_collator,
                num_workers=4,
                pin_memory=True
            )
        
        # Calculate training steps
        num_epochs = self.config.get('num_train_epochs', 3)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(num_training_steps)
        
        # Prepare with accelerator
        self.model, self.optimizer, train_loader, eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, eval_loader
        )
        
        # Training loop
        global_step = 0
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for step, batch in enumerate(train_loader):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.get('max_grad_norm', 1.0)
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    if self.accelerator.sync_gradients:
                        global_step += 1
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Logging
                if global_step % self.config.get('logging_steps', 100) == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    
                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                    
                    if self.accelerator.is_main_process and self.config.get('use_wandb', False):
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step
                        })
                
                # Evaluation
                if eval_loader and global_step % self.config.get('eval_steps', 500) == 0:
                    eval_loss = self.evaluate(eval_loader)
                    logger.info(f"Evaluation loss: {eval_loss:.4f}")
                    
                    if self.accelerator.is_main_process and self.config.get('use_wandb', False):
                        wandb.log({"eval/loss": eval_loss}, step=global_step)
                    
                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model("best")
                
                # Save checkpoint
                if global_step % self.config.get('save_steps', 1000) == 0:
                    self.save_model(f"checkpoint-{global_step}")
        
        # Final save
        self.save_model("final")
        logger.info("Training completed!")

    def evaluate(self, eval_loader):
        """Evaluation loop."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                outputs = self.model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0

    def save_model(self, name: str):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.config.get('output_dir', './output'), name)
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "model.pt"))
            
            # Save config
            with open(os.path.join(save_path, "config.yaml"), "w") as f:
                yaml.dump(self.config, f)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"Model saved to {save_path}")


def display_variants_table():
    """Display available variants in a nice table."""
    variants = FrontierModelFactory.get_available_variants()
    
    table = Table(title="Available Frontier Model Variants")
    table.add_column("Variant", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Key Features", style="green")
    
    variant_info = {
        "native_transformer": {
            "description": "Pure transformer with advanced attention",
            "features": "Adaptive attention, sparse patterns, RoPE"
        },
        "mixture_of_experts": {
            "description": "Sparse expert routing for scaling",
            "features": "Dynamic experts, load balancing, hierarchical routing"
        },
        "retrieval_augmented": {
            "description": "RAG with dynamic knowledge retrieval",
            "features": "Dense retrieval, cross-attention fusion, relevance scoring"
        },
        "multi_modal": {
            "description": "Cross-modal understanding",
            "features": "Vision+Audio+Text, cross-modal attention, unified embeddings"
        },
        "reinforcement_learning": {
            "description": "RL training with multiple rewards",
            "features": "PPO, multi-objective, curiosity, experience replay"
        }
    }
    
    for variant in variants:
        info = variant_info.get(variant, {"description": "Custom variant", "features": "Unknown"})
        table.add_row(variant, info["description"], info["features"])
    
    console.print(table)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Frontier Model Variants")
    parser.add_argument("--variant", type=str, required=True, 
                       help="Model variant to train")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--list-variants", action="store_true",
                       help="List available variants")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if args.list_variants:
        display_variants_table()
        return
    
    # Validate variant
    available_variants = FrontierModelFactory.get_available_variants()
    if args.variant not in available_variants:
        console.print(f"[red]Error: Unknown variant '{args.variant}'[/red]")
        console.print(f"Available variants: {', '.join(available_variants)}")
        return
    
    # Validate config file
    if not os.path.exists(args.config):
        console.print(f"[red]Error: Config file '{args.config}' not found[/red]")
        return
    
    # Set seed
    set_seed(args.seed)
    
    # Create trainer and start training
    console.print(f"[green]Starting training for {args.variant} variant[/green]")
    trainer = UnifiedTrainer(args.variant, args.config)
    trainer.train()


if __name__ == "__main__":
    main()