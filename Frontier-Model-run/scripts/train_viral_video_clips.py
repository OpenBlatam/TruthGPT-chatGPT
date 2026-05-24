#!/usr/bin/env python3
"""
Training Script for Viral Video Clips Model

This script provides a complete training pipeline for the Viral Video Clips model,
including data preparation, model initialization, multi-task training, and evaluation.

Usage:
    python train_viral_video_clips.py --config config.yaml --output_dir ./output
    
    # With custom parameters
    python train_viral_video_clips.py \
        --config config.yaml \
        --output_dir ./output/viral_video_clips \
        --model_size medium \
        --num_epochs 15 \
        --batch_size 2 \
        --learning_rate 3e-5
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
import wandb
from accelerate import Accelerator
import deepspeed

# Add the variant directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'variants', 'viral_video_clips'))

from model import ViralVideoClipsModel, ViralVideoClipsConfig
from trainer import ViralVideoTrainer, ViralVideoTrainingArguments, ViralVideoDataset, ViralVideoEvaluator


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
    logging.getLogger("moviepy").setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Train Viral Video Clips Model",
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
        default="./output/viral_video_clips",
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
    
    # Multi-modal learning rates
    parser.add_argument(
        "--video_learning_rate",
        type=float,
        default=None,
        help="Video component learning rate"
    )
    
    parser.add_argument(
        "--audio_learning_rate",
        type=float,
        default=None,
        help="Audio component learning rate"
    )
    
    parser.add_argument(
        "--caption_learning_rate",
        type=float,
        default=None,
        help="Caption generation learning rate"
    )
    
    # Loss weights
    parser.add_argument(
        "--viral_prediction_weight",
        type=float,
        default=None,
        help="Viral prediction loss weight"
    )
    
    parser.add_argument(
        "--highlight_detection_weight",
        type=float,
        default=None,
        help="Highlight detection loss weight"
    )
    
    parser.add_argument(
        "--caption_generation_weight",
        type=float,
        default=None,
        help="Caption generation loss weight"
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
        default="viral-video-clips-model",
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
    
    # Advanced training features
    parser.add_argument(
        "--use_curriculum_learning",
        action="store_true",
        default=True,
        help="Use curriculum learning"
    )
    
    parser.add_argument(
        "--use_adversarial_training",
        action="store_true",
        default=True,
        help="Use adversarial training"
    )
    
    parser.add_argument(
        "--use_contrastive_learning",
        action="store_true",
        default=True,
        help="Use contrastive learning"
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
    
    # Video processing options
    parser.add_argument(
        "--video_resolution",
        type=str,
        default=None,
        help="Video resolution (e.g., '1080,1920')"
    )
    
    parser.add_argument(
        "--num_clips_to_generate",
        type=int,
        default=None,
        help="Number of clips to generate per video"
    )
    
    parser.add_argument(
        "--clip_duration_range",
        type=str,
        default=None,
        help="Clip duration range (e.g., '15,60')"
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
        'video_learning_rate': args.video_learning_rate,
        'audio_learning_rate': args.audio_learning_rate,
        'caption_learning_rate': args.caption_learning_rate,
        'viral_prediction_weight': args.viral_prediction_weight,
        'highlight_detection_weight': args.highlight_detection_weight,
        'caption_generation_weight': args.caption_generation_weight,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'output_dir': args.output_dir,
        'use_mixed_precision': args.fp16 or args.bf16,
        'use_deepspeed': args.use_deepspeed,
        'use_curriculum_learning': args.use_curriculum_learning,
        'use_adversarial_training': args.use_adversarial_training,
        'use_contrastive_learning': args.use_contrastive_learning,
        'report_to': "wandb" if args.use_wandb else None,
        'project_name': args.wandb_project,
        'run_name': args.run_name
    }
    
    # Apply non-None overrides
    for key, value in training_overrides.items():
        if value is not None:
            config['training'][key] = value
    
    # Model parameter overrides
    model_overrides = {
        'num_clips_to_generate': args.num_clips_to_generate,
    }
    
    if args.video_resolution:
        width, height = map(int, args.video_resolution.split(','))
        model_overrides['video_resolution'] = [width, height]
    
    if args.clip_duration_range:
        min_dur, max_dur = map(int, args.clip_duration_range.split(','))
        model_overrides['clip_duration_range'] = [min_dur, max_dur]
    
    # Apply model overrides
    for key, value in model_overrides.items():
        if value is not None:
            config['model'][key] = value
    
    # Data path overrides
    if args.train_data:
        config['data']['train_data_path'] = args.train_data
    if args.eval_data:
        config['data']['eval_data_path'] = args.eval_data
    
    return config


def prepare_datasets(config: Dict[str, Any], args: argparse.Namespace):
    """Prepare training and evaluation datasets"""
    
    data_config = config['data']
    model_config = ViralVideoClipsConfig(**config['model'])
    
    # Training dataset
    train_data_path = data_config.get('train_data_path', 'data/viral_videos_train.json')
    
    # Create dummy data if file doesn't exist
    if not os.path.exists(train_data_path):
        logging.warning(f"Training data not found at {train_data_path}, using synthetic data")
        train_data_path = None
    
    train_dataset = ViralVideoDataset(
        data_path=train_data_path,
        config=model_config,
        split="train",
        max_samples=args.max_samples,
        include_synthetic=True
    )
    
    # Evaluation dataset
    eval_dataset = None
    eval_data_path = data_config.get('eval_data_path')
    
    if eval_data_path and os.path.exists(eval_data_path):
        eval_dataset = ViralVideoDataset(
            data_path=eval_data_path,
            config=model_config,
            split="eval",
            include_synthetic=False
        )
    else:
        # Create small eval dataset from training data
        eval_size = min(200, len(train_dataset) // 10)
        eval_dataset = ViralVideoDataset(
            data_path=None,
            config=model_config,
            split="eval",
            max_samples=eval_size,
            include_synthetic=True
        )
    
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
    return train_dataset, eval_dataset


def initialize_model(config: Dict[str, Any], args: argparse.Namespace):
    """Initialize model and configuration"""
    
    # Create model configuration
    model_config = ViralVideoClipsConfig(**config['model'])
    
    # Initialize model
    if args.resume_from_checkpoint:
        logging.info(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        # In practice, would implement proper checkpoint loading
        model = ViralVideoClipsModel(model_config)
    else:
        logging.info("Initializing new model")
        model = ViralVideoClipsModel(model_config)
    
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
    training_args = ViralVideoTrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config.get('num_train_epochs', 15),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
        learning_rate=training_config.get('learning_rate', 3e-5),
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 1000),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        
        # Multi-modal learning rates
        video_learning_rate=training_config.get('video_learning_rate', 1e-5),
        audio_learning_rate=training_config.get('audio_learning_rate', 2e-5),
        caption_learning_rate=training_config.get('caption_learning_rate', 5e-5),
        effects_learning_rate=training_config.get('effects_learning_rate', 3e-5),
        
        # Loss weights
        video_understanding_weight=training_config.get('video_understanding_weight', 1.0),
        highlight_detection_weight=training_config.get('highlight_detection_weight', 0.8),
        caption_generation_weight=training_config.get('caption_generation_weight', 0.6),
        viral_prediction_weight=training_config.get('viral_prediction_weight', 0.4),
        engagement_prediction_weight=training_config.get('engagement_prediction_weight', 0.3),
        contrastive_loss_weight=training_config.get('contrastive_loss_weight', 0.2),
        adversarial_loss_weight=training_config.get('adversarial_loss_weight', 0.1),
        
        # Advanced training features
        use_curriculum_learning=training_config.get('use_curriculum_learning', True),
        use_adversarial_training=training_config.get('use_adversarial_training', True),
        use_contrastive_learning=training_config.get('use_contrastive_learning', True),
        use_multi_task_learning=training_config.get('use_multi_task_learning', True),
        
        # Optimization settings
        optimizer_type=training_config.get('optimizer_type', 'adamw'),
        scheduler_type=training_config.get('scheduler_type', 'cosine_with_restarts'),
        use_mixed_precision=training_config.get('use_mixed_precision', True),
        use_gradient_checkpointing=training_config.get('use_gradient_checkpointing', True),
        use_deepspeed=training_config.get('use_deepspeed', False),
        
        # Evaluation and logging
        eval_steps=training_config.get('eval_steps', 500),
        save_steps=training_config.get('save_steps', 1000),
        logging_steps=training_config.get('logging_steps', 100),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'viral_prediction_accuracy'),
        greater_is_better=training_config.get('greater_is_better', True),
        
        # Monitoring
        report_to=training_config.get('report_to', 'wandb'),
        project_name=training_config.get('project_name', 'viral-video-clips-model'),
        run_name=training_config.get('run_name'),
        
        # Data augmentation
        use_video_augmentation=training_config.get('use_video_augmentation', True),
        use_audio_augmentation=training_config.get('use_audio_augmentation', True),
        augmentation_probability=training_config.get('augmentation_probability', 0.3)
    )
    
    return training_args


def run_training(
    model: ViralVideoClipsModel,
    training_args: ViralVideoTrainingArguments,
    train_dataset: ViralVideoDataset,
    eval_dataset: ViralVideoDataset,
    args: argparse.Namespace
):
    """Run the training process"""
    
    # Initialize accelerator
    accelerator = None
    if not args.cpu_only:
        accelerator = Accelerator(
            mixed_precision="fp16" if training_args.use_mixed_precision else "no",
            gradient_accumulation_steps=training_args.gradient_accumulation_steps
        )
    
    # Initialize trainer
    trainer = ViralVideoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
    model: ViralVideoClipsModel,
    eval_dataset: ViralVideoDataset,
    args: argparse.Namespace
):
    """Run evaluation only"""
    
    logging.info("Running evaluation...")
    
    # Initialize evaluator
    evaluator = ViralVideoEvaluator(model, model.config)
    
    # Create data loader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    model.to(device)
    
    # Run comprehensive evaluation
    report = evaluator.generate_evaluation_report(
        eval_dataloader=eval_dataloader,
        device=device,
        output_path=os.path.join(args.output_dir, "evaluation_report.json")
    )
    
    logging.info("Evaluation completed!")
    logging.info(f"Overall Score: {report['overall_score']:.3f}")
    logging.info(f"Viral Prediction Accuracy: {report['viral_prediction_metrics']['viral_prediction_accuracy']:.3f}")
    logging.info(f"Highlight Detection F1: {report['highlight_detection_metrics']['highlight_detection_f1']:.3f}")
    logging.info(f"Engagement Prediction Correlation: {report['engagement_prediction_metrics']['engagement_prediction_correlation']:.3f}")
    
    return report


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training.log")
    setup_logging(args.log_level, log_file)
    
    logging.info("Starting Viral Video Clips Model training")
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
    
    # Initialize model
    model, model_config = initialize_model(config, args)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, args)
    
    # Setup training arguments
    training_args = setup_training_arguments(config, args)
    
    try:
        if args.eval_only:
            # Run evaluation only
            if args.resume_from_checkpoint:
                # Load model from checkpoint
                logging.info(f"Loading model from {args.resume_from_checkpoint}")
                # In practice, would implement proper checkpoint loading
            
            eval_report = run_evaluation(model, eval_dataset, args)
            
        else:
            # Run training
            trainer = run_training(
                model, training_args, train_dataset, eval_dataset, args
            )
            
            # Run final evaluation if requested
            if eval_dataset:
                eval_report = run_evaluation(trainer.model, eval_dataset, args)
        
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