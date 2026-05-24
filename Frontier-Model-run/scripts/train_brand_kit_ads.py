#!/usr/bin/env python3
"""
Training Script for Brand Kit Ads Model

This script provides a complete training pipeline for the Brand Kit Ads model,
including data preparation, model initialization, training, and evaluation.

Usage:
    python train_brand_kit_ads.py --config config.yaml --output_dir ./output
    
    # With custom parameters
    python train_brand_kit_ads.py \
        --config config.yaml \
        --output_dir ./output/brand_kit_ads \
        --model_size medium \
        --num_epochs 10 \
        --batch_size 4 \
        --learning_rate 5e-5
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
from accelerate import Accelerator
import deepspeed

# Add the variant directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'variants', 'brand_kit_ads'))

from model import BrandKitAdsModel, BrandKitAdsConfig
from trainer import BrandKitAdsTrainer, BrandKitTrainingArguments, WebsiteDataset, BrandKitEvaluator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce noise from other libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Train Brand Kit Ads Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/brand_kit_ads",
        help="Output directory for model and logs"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size variant"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (overrides config)"
    )
    
    # Data configuration
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to training data (overrides config)"
    )
    
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data (overrides config)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples"
    )
    
    # Training options
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="brand-kit-ads-model",
        help="Weights & Biases project name"
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for logging"
    )
    
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed configuration file"
    )
    
    # Evaluation options
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation frequency (overrides config)"
    )
    
    # Hardware options
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision"
    )
    
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU-only training"
    )
    
    # Debugging options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps"
    )
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configuration"""
    
    # Load base configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.model_size:
        # Apply model size variant
        if args.model_size in config.get('model_variants', {}):
            variant_config = config['model_variants'][args.model_size]
            config['model'].update(variant_config)
    
    # Training parameter overrides
    training_overrides = {
        'num_train_epochs': args.num_epochs,
        'per_device_train_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'output_dir': args.output_dir,
        'fp16': args.fp16,
        'bf16': args.bf16,
        'use_deepspeed': args.use_deepspeed,
        'report_to': "wandb" if args.use_wandb else None,
        'project_name': args.wandb_project,
        'run_name': args.run_name
    }
    
    # Apply non-None overrides
    for key, value in training_overrides.items():
        if value is not None:
            config['training'][key] = value
    
    # Data path overrides
    if args.train_data:
        config['data']['train_data_path'] = args.train_data
    if args.eval_data:
        config['data']['eval_data_path'] = args.eval_data
    
    return config


def create_dummy_tokenizer():
    """Create a dummy tokenizer for demo purposes"""
    
    class DummyTokenizer:
        def __init__(self):
            self.vocab_size = 50257
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 2
            
        def encode(self, text, max_length=512, padding=True, truncation=True, return_tensors="pt"):
            # Simple word-based tokenization
            words = text.lower().split()
            token_ids = [hash(word) % self.vocab_size for word in words]
            
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            if padding:
                while len(token_ids) < max_length:
                    token_ids.append(self.pad_token_id)
            
            if return_tensors == "pt":
                return {
                    'input_ids': torch.tensor([token_ids]),
                    'attention_mask': torch.tensor([[1 if t != self.pad_token_id else 0 for t in token_ids]])
                }
            return token_ids
        
        def decode(self, token_ids, skip_special_tokens=True):
            return f"Generated content with {len(token_ids)} tokens"
        
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "tokenizer_config.json"), 'w') as f:
                json.dump({
                    "vocab_size": self.vocab_size,
                    "pad_token_id": self.pad_token_id,
                    "eos_token_id": self.eos_token_id
                }, f)
    
    return DummyTokenizer()


def prepare_datasets(config: Dict[str, Any], tokenizer, args: argparse.Namespace):
    """Prepare training and evaluation datasets"""
    
    data_config = config['data']
    training_config = config['training']
    
    # Training dataset
    train_data_path = data_config.get('train_data_path', 'data/train_websites.json')
    
    # Create dummy data if file doesn't exist
    if not os.path.exists(train_data_path):
        logging.warning(f"Training data not found at {train_data_path}, using synthetic data")
        train_data_path = None
    
    train_dataset = WebsiteDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=training_config.get('max_sequence_length', 512),
        image_size=training_config.get('image_size', 224),
        include_synthetic=True
    )
    
    # Limit dataset size if specified
    if args.max_samples and len(train_dataset) > args.max_samples:
        train_dataset.data = train_dataset.data[:args.max_samples]
    
    # Evaluation dataset
    eval_dataset = None
    eval_data_path = data_config.get('eval_data_path')
    
    if eval_data_path and os.path.exists(eval_data_path):
        eval_dataset = WebsiteDataset(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            max_length=training_config.get('max_sequence_length', 512),
            image_size=training_config.get('image_size', 224),
            include_synthetic=False
        )
    else:
        # Create small eval dataset from training data
        eval_size = min(100, len(train_dataset) // 10)
        eval_data = train_dataset.data[:eval_size]
        
        eval_dataset = WebsiteDataset(
            data_path=None,
            tokenizer=tokenizer,
            max_length=training_config.get('max_sequence_length', 512),
            image_size=training_config.get('image_size', 224),
            include_synthetic=False
        )
        eval_dataset.data = eval_data
    
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
    return train_dataset, eval_dataset


def initialize_model(config: Dict[str, Any], args: argparse.Namespace):
    """Initialize model and configuration"""
    
    # Create model configuration
    model_config = BrandKitAdsConfig(**config['model'])
    
    # Initialize model
    if args.resume_from_checkpoint:
        logging.info(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        model = BrandKitAdsModel.from_pretrained(args.resume_from_checkpoint)
    else:
        logging.info("Initializing new model")
        model = BrandKitAdsModel(model_config)
    
    # Model info
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model initialized with {param_count:,} total parameters")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    return model, model_config


def setup_training_arguments(config: Dict[str, Any], args: argparse.Namespace):
    """Setup training arguments"""
    
    training_config = config['training']
    
    # Create training arguments
    training_args = BrandKitTrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config.get('num_train_epochs', 10),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 5e-5),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 1000),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        
        # Multi-modal learning rates
        vision_learning_rate=training_config.get('vision_learning_rate', 1e-5),
        language_learning_rate=training_config.get('language_learning_rate', 5e-5),
        brand_learning_rate=training_config.get('brand_learning_rate', 2e-5),
        
        # Loss weights
        language_loss_weight=training_config.get('language_loss_weight', 1.0),
        brand_consistency_weight=training_config.get('brand_consistency_weight', 0.3),
        visual_alignment_weight=training_config.get('visual_alignment_weight', 0.2),
        content_quality_weight=training_config.get('content_quality_weight', 0.15),
        adversarial_weight=training_config.get('adversarial_weight', 0.1),
        
        # Evaluation and logging
        eval_steps=training_config.get('eval_steps', 500),
        save_steps=training_config.get('save_steps', 1000),
        logging_steps=training_config.get('logging_steps', 100),
        
        # Advanced features
        use_adversarial_training=training_config.get('use_adversarial_training', True),
        use_curriculum_learning=training_config.get('use_curriculum_learning', True),
        use_brand_consistency_loss=training_config.get('use_brand_consistency_loss', True),
        use_visual_alignment_loss=training_config.get('use_visual_alignment_loss', True),
        use_content_quality_loss=training_config.get('use_content_quality_loss', True),
        
        # Optimization
        optimizer_type=training_config.get('optimizer_type', 'adamw'),
        scheduler_type=training_config.get('scheduler_type', 'cosine'),
        use_deepspeed=training_config.get('use_deepspeed', False),
        fp16=training_config.get('fp16', True),
        bf16=training_config.get('bf16', False),
        
        # Monitoring
        report_to=training_config.get('report_to', 'wandb'),
        run_name=training_config.get('run_name'),
        project_name=training_config.get('project_name', 'brand-kit-ads-model'),
        
        # Checkpointing
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'eval_brand_alignment'),
        greater_is_better=training_config.get('greater_is_better', True)
    )
    
    return training_args


def run_training(
    model: BrandKitAdsModel,
    training_args: BrandKitTrainingArguments,
    train_dataset: WebsiteDataset,
    eval_dataset: WebsiteDataset,
    tokenizer,
    args: argparse.Namespace
):
    """Run the training process"""
    
    # Initialize accelerator
    accelerator = None
    if not args.cpu_only:
        accelerator = Accelerator(
            mixed_precision="fp16" if training_args.fp16 else "no",
            gradient_accumulation_steps=training_args.gradient_accumulation_steps
        )
    
    # Initialize trainer
    trainer = BrandKitAdsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        accelerator=accelerator
    )
    
    # Start training
    logging.info("Starting training...")
    trainer.train()
    
    # Final evaluation
    if eval_dataset:
        logging.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logging.info(f"Final evaluation results: {eval_results}")
    
    # Save final model
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")
    
    return trainer


def run_evaluation(
    model: BrandKitAdsModel,
    eval_dataset: WebsiteDataset,
    tokenizer,
    args: argparse.Namespace
):
    """Run evaluation only"""
    
    logging.info("Running evaluation...")
    
    # Initialize evaluator
    evaluator = BrandKitEvaluator(model, tokenizer)
    
    # Test websites for brand extraction evaluation
    test_websites = [
        "https://apple.com",
        "https://google.com",
        "https://microsoft.com",
        "https://stripe.com",
        "https://airbnb.com"
    ]
    
    # Run comprehensive evaluation
    report = evaluator.generate_evaluation_report(
        test_websites=test_websites,
        output_path=os.path.join(args.output_dir, "evaluation_report.json")
    )
    
    logging.info("Evaluation completed!")
    logging.info(f"Overall Score: {report['overall_score']:.3f}")
    logging.info(f"Brand Extraction Accuracy: {report['brand_extraction_metrics']['brand_extraction_accuracy']:.3f}")
    logging.info(f"Ad Generation Quality: {report['ad_generation_metrics']['ad_generation_quality']:.3f}")
    
    return report


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training.log")
    setup_logging(args.log_level, log_file)
    
    logging.info("Starting Brand Kit Ads Model training")
    logging.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    if not os.path.exists(args.config):
        logging.error(f"Configuration file not found: {args.config}")
        return 1
    
    config = load_config(args.config, args)
    logging.info("Configuration loaded successfully")
    
    # Save configuration
    config_save_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize tokenizer
    tokenizer = create_dummy_tokenizer()
    logging.info("Tokenizer initialized")
    
    # Initialize model
    model, model_config = initialize_model(config, args)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer, args)
    
    # Setup training arguments
    training_args = setup_training_arguments(config, args)
    
    try:
        if args.eval_only:
            # Run evaluation only
            if args.resume_from_checkpoint:
                model = BrandKitAdsModel.from_pretrained(args.resume_from_checkpoint)
            
            eval_report = run_evaluation(model, eval_dataset, tokenizer, args)
            
        else:
            # Run training
            trainer = run_training(
                model, training_args, train_dataset, eval_dataset, tokenizer, args
            )
            
            # Run final evaluation if requested
            if eval_dataset:
                eval_report = run_evaluation(trainer.model, eval_dataset, tokenizer, args)
        
        logging.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)