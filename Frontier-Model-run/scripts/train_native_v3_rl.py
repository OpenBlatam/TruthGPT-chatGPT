#!/usr/bin/env python3
"""
Training script for Native DeepSeek-V3 with Reinforcement Learning
Specialized training pipeline for the Native V3 RL variant.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator

# Add the variants directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'variants'))

from native_v3_rl.model import NativeV3RLForCausalLM, NativeV3RLConfig
from native_v3_rl.trainer import NativeV3RLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, texts, tokenizer, max_length=2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(config: Dict[str, Any], tokenizer) -> tuple:
    """Prepare training and evaluation datasets."""
    dataset_name = config.get('dataset_name', 'wikitext')
    dataset_config = config.get('dataset_config', 'wikitext-103-raw-v1')
    
    logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset(dataset_name, dataset_config)
        train_texts = [item['text'] for item in dataset['train'] if len(item['text'].strip()) > 50]
        eval_texts = [item['text'] for item in dataset['validation'] if len(item['text'].strip()) > 50]
    else:
        # Add support for other datasets
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Limit dataset size for testing
    max_train_samples = config.get('max_train_samples', 10000)
    max_eval_samples = config.get('max_eval_samples', 1000)
    
    train_texts = train_texts[:max_train_samples]
    eval_texts = eval_texts[:max_eval_samples]
    
    logger.info(f"Train samples: {len(train_texts)}, Eval samples: {len(eval_texts)}")
    
    # Create datasets
    max_length = config.get('max_length', 2048)
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length)
    
    return train_dataset, eval_dataset


def create_model(config: Dict[str, Any]) -> NativeV3RLForCausalLM:
    """Create the Native V3 RL model."""
    model_config_dict = config.get('model_config', {})
    
    # Create model configuration
    model_config = NativeV3RLConfig(**model_config_dict)
    
    logger.info(f"Creating model with config: {model_config}")
    
    # Create model
    model = NativeV3RLForCausalLM(model_config)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def setup_tokenizer(config: Dict[str, Any]):
    """Setup tokenizer."""
    tokenizer_name = config.get('tokenizer_name', 'gpt2')
    
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Native V3 RL model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../variants/native_v3_rl/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./native_v3_rl_output",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    config = load_config(config_path)
    
    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 32),
        mixed_precision="bf16" if config.get('bf16', True) else "fp16" if config.get('fp16', False) else "no",
        log_with="wandb" if config.get('use_wandb', False) else None,
        project_dir=config.get('output_dir', './output'),
    )
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False) and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.get('wandb_project', 'native-v3-rl'),
            config=config,
            init_kwargs={
                "wandb": {
                    "name": config.get('wandb_run_name', 'native-v3-rl-experiment'),
                    "tags": ["native-v3", "reinforcement-learning", "deepseek-v3"]
                }
            }
        )
    
    # Setup logging
    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
    logger.info(f"Starting Native V3 RL training with config: {config_path}")
    logger.info(f"Output directory: {config.get('output_dir', './output')}")
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = NativeV3RLTrainer(
        model=model,
        config=model.config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        accelerator=accelerator,
        **config  # Pass all config parameters
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Add checkpoint loading logic here
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if accelerator.is_main_process and config.get('use_wandb', False):
            accelerator.end_training()


if __name__ == "__main__":
    main()