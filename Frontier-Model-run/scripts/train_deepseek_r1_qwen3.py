#!/usr/bin/env python3
"""
Training Script for DeepSeek-R1-Qwen3 Frontier Model

Advanced training pipeline for the DeepSeek-R1-Qwen3 reasoning model with
multi-objective optimization, chain-of-thought training, and curriculum learning.

Usage:
    python train_deepseek_r1_qwen3.py --config variants/deepseek_r1_qwen3/config.yaml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from transformers import set_seed
import yaml

# Add the variants directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "variants", "deepseek_r1_qwen3"))

from model import DeepSeekR1Qwen3ForCausalLM, DeepSeekR1Qwen3Config
from trainer import (
    ReasoningTrainer,
    ReasoningTrainingArguments,
    ReasoningDataset,
    create_reasoning_data_collator,
    compute_reasoning_metrics,
    setup_reasoning_training,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DeepSeekR1Qwen3Trainer:
    """Main trainer class for DeepSeek-R1-Qwen3 model."""
    
    def __init__(self, config_path: str, args: argparse.Namespace):
        self.config_path = config_path
        self.args = args
        self.config = self._load_config()
        
        # Setup distributed training
        self._setup_distributed()
        
        # Set random seed
        set_seed(self.args.seed)
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._initialize_model()
        
        # Setup training components
        self.trainer, self.training_args = self._setup_training()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        logger.info(f"Loading configuration from {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with command line arguments
        if self.args.model_size:
            if self.args.model_size in config.get("model_variants", {}):
                variant_config = config["model_variants"][self.args.model_size]
                config["model_config"].update(variant_config)
                logger.info(f"Using {self.args.model_size} model variant")
        
        if self.args.output_dir:
            config["training_config"]["output_dir"] = self.args.output_dir
        
        if self.args.learning_rate:
            config["training_config"]["learning_rate"] = self.args.learning_rate
        
        if self.args.batch_size:
            config["training_config"]["per_device_train_batch_size"] = self.args.batch_size
        
        if self.args.epochs:
            config["training_config"]["num_train_epochs"] = self.args.epochs
        
        return config
    
    def _setup_distributed(self):
        """Setup distributed training if available."""
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            logger.info(f"Initialized distributed training on local rank {local_rank}")
    
    def _initialize_model(self):
        """Initialize model and tokenizer."""
        logger.info("Initializing DeepSeek-R1-Qwen3 model...")
        
        # Create model configuration
        model_config = DeepSeekR1Qwen3Config(**self.config["model_config"])
        
        # Initialize model
        if self.args.resume_from_checkpoint and os.path.exists(self.args.resume_from_checkpoint):
            logger.info(f"Loading model from checkpoint: {self.args.resume_from_checkpoint}")
            model = DeepSeekR1Qwen3ForCausalLM.from_pretrained(self.args.resume_from_checkpoint)
        else:
            logger.info("Initializing new model from config")
            model = DeepSeekR1Qwen3ForCausalLM(model_config)
        
        # Create tokenizer (in practice, load from HuggingFace)
        tokenizer = self._create_tokenizer()
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: {total_params * 4 / 1e9:.2f}GB (FP32)")
        
        return model, tokenizer
    
    def _create_tokenizer(self):
        """Create tokenizer for training."""
        # In practice, you would load from HuggingFace:
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        
        # For demo purposes, create a simple tokenizer
        class TrainingTokenizer:
            def __init__(self):
                self.vocab_size = 151936
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2
                
                # Extended vocabulary for training
                self.vocab = self._create_vocab()
                self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            def _create_vocab(self):
                """Create extended vocabulary for training."""
                vocab = {
                    "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
                    "Problem": 4, "Reasoning": 5, "Step": 6, "Answer": 7,
                    "Think": 8, "Therefore": 9, "Because": 10, "So": 11,
                    "First": 12, "Second": 13, "Third": 14, "Finally": 15,
                    "Let": 16, "me": 17, "think": 18, "about": 19, "this": 20,
                    "I": 21, "need": 22, "to": 23, "find": 24, "solve": 25,
                    "calculate": 26, "determine": 27, "analyze": 28,
                    "What": 29, "is": 30, "the": 31, "of": 32, "and": 33,
                    "a": 34, "an": 35, "in": 36, "for": 37, "with": 38,
                    "If": 39, "Then": 40, "When": 41, "Where": 42, "Why": 43,
                    "How": 44, "Which": 45, "Who": 46, "Whom": 47,
                }
                
                # Add numbers
                for i in range(1000):
                    vocab[str(i)] = len(vocab)
                
                # Add common mathematical symbols
                math_symbols = ["+", "-", "*", "/", "=", "(", ")", "[", "]", 
                               "{", "}", "^", "√", "π", "∞", "≤", "≥", "≠"]
                for symbol in math_symbols:
                    vocab[symbol] = len(vocab)
                
                # Add punctuation
                punctuation = [".", ",", "?", "!", ":", ";", "'", '"', "\n", " "]
                for punct in punctuation:
                    vocab[punct] = len(vocab)
                
                # Fill remaining vocabulary with dummy tokens
                while len(vocab) < self.vocab_size:
                    vocab[f"<token_{len(vocab)}>"] = len(vocab)
                
                return vocab
            
            def encode(self, text: str, add_special_tokens: bool = True) -> list:
                """Encode text to token IDs."""
                tokens = []
                if add_special_tokens:
                    tokens.append(self.bos_token_id)
                
                # Simple word-level tokenization
                words = text.split()
                for word in words:
                    if word in self.vocab:
                        tokens.append(self.vocab[word])
                    else:
                        # Use hash for unknown words
                        token_id = (hash(word) % (self.vocab_size - 1000)) + 1000
                        tokens.append(token_id)
                
                if add_special_tokens:
                    tokens.append(self.eos_token_id)
                
                return tokens
            
            def decode(self, token_ids: list, skip_special_tokens: bool = True) -> str:
                """Decode token IDs to text."""
                words = []
                for token_id in token_ids:
                    if skip_special_tokens and token_id in [self.pad_token_id, self.eos_token_id, self.bos_token_id]:
                        continue
                    if token_id in self.reverse_vocab:
                        words.append(self.reverse_vocab[token_id])
                    else:
                        words.append(f"<unk_{token_id}>")
                
                return " ".join(words)
            
            def __call__(self, text, return_tensors=None, padding=False, 
                        truncation=False, max_length=None, **kwargs):
                """Tokenizer call interface."""
                if isinstance(text, str):
                    token_ids = self.encode(text)
                elif isinstance(text, list):
                    token_ids = [self.encode(t) for t in text]
                else:
                    raise ValueError("Text must be string or list of strings")
                
                if isinstance(token_ids[0], list):
                    # Batch processing
                    if max_length:
                        token_ids = [ids[:max_length] for ids in token_ids]
                    
                    if padding and max_length:
                        max_len = max_length
                        token_ids = [ids + [self.pad_token_id] * (max_len - len(ids)) 
                                   for ids in token_ids]
                    
                    attention_mask = [[1 if tid != self.pad_token_id else 0 for tid in ids] 
                                    for ids in token_ids]
                else:
                    # Single sequence
                    if max_length and len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]
                    
                    if padding and max_length:
                        while len(token_ids) < max_length:
                            token_ids.append(self.pad_token_id)
                    
                    attention_mask = [1 if tid != self.pad_token_id else 0 for tid in token_ids]
                
                result = {
                    "input_ids": token_ids,
                    "attention_mask": attention_mask
                }
                
                if return_tensors == "pt":
                    result = {k: torch.tensor(v) for k, v in result.items()}
                
                return result
        
        return TrainingTokenizer()
    
    def _setup_training(self):
        """Setup training components."""
        logger.info("Setting up training components...")
        
        # Use the setup function from trainer.py
        trainer, training_args = setup_reasoning_training(
            model=self.model,
            tokenizer=self.tokenizer,
            config_path=self.config_path,
            train_data_path=self.args.train_data or "data/reasoning_train.jsonl",
            eval_data_path=self.args.eval_data or "data/reasoning_eval.jsonl",
            output_dir=self.args.output_dir,
        )
        
        return trainer, training_args
    
    def train(self):
        """Run the training process."""
        logger.info("Starting training...")
        
        try:
            # Train the model
            train_result = self.trainer.train(
                resume_from_checkpoint=self.args.resume_from_checkpoint
            )
            
            # Save the final model
            self.trainer.save_model()
            
            # Log training results
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            if hasattr(train_result, 'metrics'):
                for key, value in train_result.metrics.items():
                    logger.info(f"{key}: {value}")
            
            # Save training metrics
            metrics_path = os.path.join(self.args.output_dir, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(self):
        """Run evaluation on the trained model."""
        logger.info("Running evaluation...")
        
        try:
            eval_result = self.trainer.evaluate()
            
            logger.info("Evaluation completed!")
            for key, value in eval_result.items():
                logger.info(f"{key}: {value}")
            
            # Save evaluation results
            eval_path = os.path.join(self.args.output_dir, "eval_results.json")
            with open(eval_path, 'w') as f:
                json.dump(eval_result, f, indent=2)
            
            return eval_result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DeepSeek-R1-Qwen3 Reasoning Model")
    
    # Required arguments
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", required=True, help="Output directory for model and logs")
    
    # Model arguments
    parser.add_argument("--model_size", choices=["small", "medium", "large"], 
                       help="Model size variant")
    parser.add_argument("--resume_from_checkpoint", help="Path to checkpoint to resume from")
    
    # Data arguments
    parser.add_argument("--train_data", help="Path to training data")
    parser.add_argument("--eval_data", help="Path to evaluation data")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size per device")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Action arguments
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    
    # Logging arguments
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not (args.do_train or args.do_eval or args.eval_only):
        args.do_train = True  # Default to training
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize trainer
        logger.info("Initializing DeepSeek-R1-Qwen3 trainer...")
        trainer = DeepSeekR1Qwen3Trainer(args.config, args)
        
        # Run training and/or evaluation
        if args.eval_only:
            trainer.evaluate()
        else:
            if args.do_train:
                trainer.train()
            
            if args.do_eval:
                trainer.evaluate()
        
        logger.info("Training pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())