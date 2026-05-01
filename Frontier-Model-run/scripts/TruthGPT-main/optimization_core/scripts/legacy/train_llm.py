import argparse
import os
from typing import List, Optional, Tuple
import logging

import torch
import yaml
from datasets import load_dataset

from optimization_core.trainers.config import TrainerConfig
from .build_trainer import build_trainer
from optimization_core.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def read_yaml(path: str) -> dict:
    """Read and parse YAML configuration file with error handling."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Empty or invalid YAML file: {path}")
        logger.info(f"Successfully loaded config from {path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error reading config file {path}: {e}", exc_info=True)
        raise


def load_text_splits(
    dataset: str,
    subset: Optional[str],
    text_field: str,
    limit_per_split: int = 5000
) -> Tuple[List[str], List[str]]:
    """
    Load and split dataset into training and validation sets.
    
    Args:
        dataset: Dataset name from HuggingFace
        subset: Optional subset name
        text_field: Field name containing text data
        limit_per_split: Maximum samples per split
    
    Returns:
        Tuple of (train_texts, val_texts)
    """
    try:
        logger.info(f"Loading dataset: {dataset} (subset: {subset})")
        if subset:
            ds = load_dataset(dataset, subset)
        else:
            ds = load_dataset(dataset)
        
        if "train" not in ds:
            raise ValueError(f"Dataset {dataset} does not contain 'train' split")
        
        train = ds["train"][text_field][:limit_per_split]
        val_limit = max(256, limit_per_split // 10)
        
        if "validation" in ds:
            val = ds["validation"][text_field][:val_limit]
        elif "val" in ds:
            val = ds["val"][text_field][:val_limit]
        else:
            logger.warning(f"No validation split found, using train split for validation")
            val = ds["train"][text_field][:val_limit]
        
        train_list = list(train) if not isinstance(train, list) else train
        val_list = list(val) if not isinstance(val, list) else val
        
        logger.info(f"Loaded {len(train_list)} training samples and {len(val_list)} validation samples")
        return train_list, val_list
    except Exception as e:
        logger.error(f"Error loading dataset {dataset}: {e}", exc_info=True)
        raise


def main() -> None:
    """Main training function with improved error handling and logging."""
    parser = argparse.ArgumentParser(
        description="Train LLM with TruthGPT optimization core",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="optimization_core/configs/llm_default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--data_limit",
        type=int,
        default=5000,
        help="Limit number of samples per split"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to log file"
    )
    args = parser.parse_args()

    # Setup file logging if requested
    if args.log_file:
        setup_logger(__name__, log_file=args.log_file)

    try:
        logger.info("=" * 80)
        logger.info("Starting LLM Training")
        logger.info("=" * 80)

        # Load configuration
        cfg_dict = read_yaml(args.config)
        cfg = TrainerConfig.from_dict(cfg_dict)

        # Extract data configuration
        data_cfg = cfg_dict.get("data", {})
        dataset = str(data_cfg.get("dataset", "wikitext"))
        subset = data_cfg.get("subset", "wikitext-2-raw-v1")
        text_field = str(data_cfg.get("text_field", "text"))

        # Load datasets
        train_texts, val_texts = load_text_splits(
            dataset, subset, text_field, args.data_limit
        )

        # Log configuration summary
        logger.info("Training Configuration:")
        logger.info(f"  Device: {cfg.hardware.device}")
        logger.info(f"  Model: {cfg.model.name_or_path}")
        logger.info(f"  Output Directory: {cfg.output_dir}")
        logger.info(f"  Epochs: {cfg.training.epochs}")
        logger.info(f"  Batch Size: {cfg.training.train_batch_size}")
        logger.info(f"  Learning Rate: {cfg.training.learning_rate}")
        logger.info(f"  Mixed Precision: {cfg.training.mixed_precision}")
        logger.info(f"  Gradient Accumulation Steps: {cfg.training.grad_accum_steps}")

        # Build and train
        trainer = build_trainer(
            raw_cfg=cfg_dict,
            train_texts=train_texts,
            val_texts=val_texts,
            max_seq_len=int(data_cfg.get("max_seq_len", args.max_seq_len)),
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")

        # Generate sample
        logger.info("Generating sample output...")
        sample = trainer.generate("The theory of transformers is", max_new_tokens=64)
        logger.info("Sample Output:")
        logger.info(sample)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

