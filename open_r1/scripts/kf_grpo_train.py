#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import logging
import gc
import math
import platform
import warnings
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Union, Tuple
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.cpp_extension import load_inline
from torch.nn import Parameter

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from accelerate import Accelerator
from accelerate.utils import DummyScheduler, DummyOptimizer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.deepspeed import HfDeepSpeedConfig

import wandb
import mlflow
import tyro
import yaml
import argparse
from rich.logging import RichHandler
from loguru import logger
import psutil
import sentry_sdk
import torch.profiler
from torch.utils.tensorboard import SummaryWriter

from abc import ABC, abstractmethod

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler('training.log')
    ]
)

@dataclass
class PrecisionMode(Enum):
    """Precision modes for layer normalization."""
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    MIXED = auto()

@dataclass
class LayerNormConfig:
    """Configuration for layer normalization."""
    normalized_shape: Union[int, Tuple[int, ...]]
    eps: float = 1e-5
    elementwise_affine: bool = True
    device: Optional[str] = None
    precision: PrecisionMode = PrecisionMode.FP32
    use_tensor_cores: bool = True
    use_fast_math: bool = True
    use_cooperative_groups: bool = True
    use_prefetching: bool = True
    use_vectorization: bool = True
    block_size: Optional[int] = None
    shared_memory_size: Optional[int] = None

@dataclass
class DeepSeekConfig:
    """Configuration specific to DeepSeek R1 model."""
    model_type: str = "deepseek"
    model_name: str = "deepseek-ai/deepseek-r1"
    max_position_embeddings: int = 8192
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    use_rotary_embeddings: bool = True
    use_alibi: bool = False
    use_flash_attention_2: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 4096
    use_parallel_attention: bool = True
    use_parallel_mlp: bool = True
    use_parallel_layernorm: bool = True
    use_parallel_embedding: bool = True
    use_parallel_output: bool = True
    use_parallel_residual: bool = True
    use_parallel_ffn: bool = True
    use_parallel_attention_output: bool = True
    use_parallel_mlp_output: bool = True
    use_parallel_layernorm_output: bool = True
    use_parallel_embedding_output: bool = True
    use_parallel_residual_output: bool = True
    use_parallel_ffn_output: bool = True
    use_parallel_attention_input: bool = True
    use_parallel_mlp_input: bool = True
    use_parallel_layernorm_input: bool = True
    use_parallel_embedding_input: bool = True
    use_parallel_residual_input: bool = True
    use_parallel_ffn_input: bool = True

@dataclass
class KFGRPOScriptArguments(ScriptArguments):
    """Script arguments for the KF-GRPO training script with DeepSeek R1 optimizations."""
    # DeepSeek specific parameters
    deepseek_config: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    use_deepseek_optimizations: bool = field(default=True, metadata={"help": "Use DeepSeek-specific optimizations"})
    use_deepseek_attention: bool = field(default=True, metadata={"help": "Use DeepSeek attention optimizations"})
    use_deepseek_mlp: bool = field(default=True, metadata={"help": "Use DeepSeek MLP optimizations"})
    use_deepseek_layernorm: bool = field(default=True, metadata={"help": "Use DeepSeek LayerNorm optimizations"})
    use_deepseek_embedding: bool = field(default=True, metadata={"help": "Use DeepSeek embedding optimizations"})
    use_deepseek_residual: bool = field(default=True, metadata={"help": "Use DeepSeek residual optimizations"})
    use_deepseek_ffn: bool = field(default=True, metadata={"help": "Use DeepSeek FFN optimizations"})
    
    # Training parameters
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    process_noise: float = field(default=0.01, metadata={"help": "Process noise covariance (Q)"})
    measurement_noise: float = field(default=0.1, metadata={"help": "Measurement noise covariance (R)"})
    kalman_memory_size: int = field(default=1000, metadata={"help": "Size of Kalman filter memory buffer"})
    
    # Optimization parameters
    use_amp: bool = field(default=True, metadata={"help": "Use automatic mixed precision"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Ratio of warmup steps"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for optimizer"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type"})
    num_cycles: int = field(default=1, metadata={"help": "Number of cycles for cosine scheduler"})
    
    # Memory optimization
    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Use gradient checkpointing"})
    use_flash_attention: bool = field(default=True, metadata={"help": "Use flash attention"})
    use_8bit_optimizer: bool = field(default=False, metadata={"help": "Use 8-bit optimizer"})
    
    # Distributed training
    distributed_backend: str = field(default="nccl", metadata={"help": "Distributed backend"})
    distributed_world_size: int = field(default=-1, metadata={"help": "Number of distributed processes"})
    distributed_rank: int = field(default=-1, metadata={"help": "Process rank"})
    distributed_master_addr: str = field(default="localhost", metadata={"help": "Master address"})
    distributed_master_port: str = field(default="29500", metadata={"help": "Master port"})
    
    # Advanced memory management
    use_cpu_offload: bool = field(default=False, metadata={"help": "Use CPU offloading"})
    use_activation_checkpointing: bool = field(default=True, metadata={"help": "Use activation checkpointing"})
    use_attention_slicing: bool = field(default=True, metadata={"help": "Use attention slicing"})
    use_sequence_parallelism: bool = field(default=False, metadata={"help": "Use sequence parallelism"})
    
    # Performance optimization
    use_cudnn_benchmark: bool = field(default=True, metadata={"help": "Use cuDNN benchmark"})
    use_tf32: bool = field(default=True, metadata={"help": "Use TF32 precision"})
    use_channels_last: bool = field(default=True, metadata={"help": "Use channels last memory format"})
    use_compile: bool = field(default=True, metadata={"help": "Use torch.compile"})
    
    # DeepSpeed configuration
    use_deepspeed: bool = field(default=False, metadata={"help": "Use DeepSpeed for training"})
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "Path to DeepSpeed config file"})
    zero_stage: int = field(default=2, metadata={"help": "DeepSpeed ZeRO stage (0, 1, 2, 3)"})
    offload_optimizer: bool = field(default=True, metadata={"help": "Offload optimizer states to CPU"})
    offload_param: bool = field(default=False, metadata={"help": "Offload parameters to CPU"})
    gradient_clipping: float = field(default=1.0, metadata={"help": "Gradient clipping value"})
    train_batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    fp16: bool = field(default=True, metadata={"help": "Use FP16 precision"})
    bf16: bool = field(default=False, metadata={"help": "Use BF16 precision"})

class KalmanFilter:
    def __init__(self, process_noise: float, measurement_noise: float, memory_size: int = 1000):
        self.Q = process_noise
        self.R = measurement_noise
        self.mu = 0.0
        self.P = 1.0
        self.memory = []
        self.memory_size = memory_size
        self.momentum = 0.9
        self.velocity = 0.0
        
    def update(self, measurement: float) -> float:
        # Prediction with momentum
        mu_pred = self.mu + self.momentum * self.velocity
        P_pred = self.P + self.Q
        
        # Update with adaptive gain
        K = P_pred / (P_pred + self.R)
        innovation = measurement - mu_pred
        self.mu = mu_pred + K * innovation
        self.P = (1 - K) * P_pred + self.Q
        
        # Update velocity with momentum
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * innovation
        
        # Update memory with exponential moving average
        self.memory.append(measurement)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
            
        return self.mu
    
    def get_statistics(self) -> Tuple[float, float]:
        """Get mean and standard deviation of recent measurements with exponential weighting."""
        if not self.memory:
            return 0.0, 1.0
        
        weights = np.exp(np.linspace(-1, 0, len(self.memory)))
        weights /= weights.sum()
        
        weighted_mean = np.average(self.memory, weights=weights)
        weighted_std = np.sqrt(np.average((np.array(self.memory) - weighted_mean) ** 2, weights=weights))
        
        return weighted_mean, weighted_std

def setup_environment():
    """Setup the training environment with necessary configurations."""
    # Setup CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Setup memory management
    if hasattr(torch.cuda, 'memory_summary'):
        logger.info("CUDA Memory Summary:")
        logger.info(torch.cuda.memory_summary())
    
    # Setup performance optimizations
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU by default
        torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler('training.log')
        ]
    )
    
    # Setup error tracking
    try:
        sentry_sdk.init(
            "YOUR_SENTRY_DSN",  # Replace with your Sentry DSN
            traces_sample_rate=1.0,
            environment="development"
        )
        logger.info("Error tracking initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize error tracking: {e}")

def setup_deepseek_optimizations(model: PreTrainedModel, config: DeepSeekConfig):
    """Setup DeepSeek-specific optimizations for the model."""
    if not isinstance(model, PreTrainedModel):
        logger.warning("Model is not a PreTrainedModel, skipping DeepSeek optimizations")
        return
    
    # Apply DeepSeek-specific optimizations
    if config.use_flash_attention_2:
        model.config.use_flash_attention_2 = True
        logger.info("Enabled Flash Attention 2")
    
    if config.use_sliding_window:
        model.config.use_sliding_window = True
        model.config.sliding_window_size = config.sliding_window_size
        logger.info(f"Enabled sliding window attention with size {config.sliding_window_size}")
    
    # Apply parallel optimizations
    parallel_configs = {
        'attention': config.use_parallel_attention,
        'mlp': config.use_parallel_mlp,
        'layernorm': config.use_parallel_layernorm,
        'embedding': config.use_parallel_embedding,
        'residual': config.use_parallel_residual,
        'ffn': config.use_parallel_ffn
    }
    
    for name, enabled in parallel_configs.items():
        if enabled:
            setattr(model.config, f'use_parallel_{name}', True)
            logger.info(f"Enabled parallel {name}")
    
    # Apply input/output optimizations
    io_configs = {
        'attention': config.use_parallel_attention_output,
        'mlp': config.use_parallel_mlp_output,
        'layernorm': config.use_parallel_layernorm_output,
        'embedding': config.use_parallel_embedding_output,
        'residual': config.use_parallel_residual_output,
        'ffn': config.use_parallel_ffn_output
    }
    
    for name, enabled in io_configs.items():
        if enabled:
            setattr(model.config, f'use_parallel_{name}_output', True)
            logger.info(f"Enabled parallel {name} output")
    
    # Apply input optimizations
    input_configs = {
        'attention': config.use_parallel_attention_input,
        'mlp': config.use_parallel_mlp_input,
        'layernorm': config.use_parallel_layernorm_input,
        'embedding': config.use_parallel_embedding_input,
        'residual': config.use_parallel_residual_input,
        'ffn': config.use_parallel_ffn_input
    }
    
    for name, enabled in input_configs.items():
        if enabled:
            setattr(model.config, f'use_parallel_{name}_input', True)
            logger.info(f"Enabled parallel {name} input")

def setup_training_config(args: KFGRPOScriptArguments) -> Dict[str, Any]:
    """Setup training configuration from arguments."""
    config = {
        'training': {
            'batch_size': args.train_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_grad_norm': args.max_grad_norm,
            'warmup_ratio': args.warmup_ratio,
            'weight_decay': args.weight_decay,
            'lr_scheduler_type': args.lr_scheduler_type,
            'num_cycles': args.num_cycles
        },
        'optimization': {
            'use_amp': args.use_amp,
            'use_gradient_checkpointing': args.use_gradient_checkpointing,
            'use_flash_attention': args.use_flash_attention,
            'use_8bit_optimizer': args.use_8bit_optimizer,
            'use_cpu_offload': args.use_cpu_offload,
            'use_activation_checkpointing': args.use_activation_checkpointing,
            'use_attention_slicing': args.use_attention_slicing,
            'use_sequence_parallelism': args.use_sequence_parallelism
        },
        'performance': {
            'use_cudnn_benchmark': args.use_cudnn_benchmark,
            'use_tf32': args.use_tf32,
            'use_channels_last': args.use_channels_last,
            'use_compile': args.use_compile
        },
        'deepspeed': {
            'use_deepspeed': args.use_deepspeed,
            'deepspeed_config': args.deepspeed_config,
            'zero_stage': args.zero_stage,
            'offload_optimizer': args.offload_optimizer,
            'offload_param': args.offload_param,
            'gradient_clipping': args.gradient_clipping,
            'fp16': args.fp16,
            'bf16': args.bf16
        },
        'deepseek': {
            'model_type': args.deepseek_config.model_type,
            'model_name': args.deepseek_config.model_name,
            'max_position_embeddings': args.deepseek_config.max_position_embeddings,
            'hidden_size': args.deepseek_config.hidden_size,
            'num_hidden_layers': args.deepseek_config.num_hidden_layers,
            'num_attention_heads': args.deepseek_config.num_attention_heads,
            'intermediate_size': args.deepseek_config.intermediate_size,
            'hidden_dropout_prob': args.deepseek_config.hidden_dropout_prob,
            'attention_dropout_prob': args.deepseek_config.attention_dropout_prob,
            'layer_norm_eps': args.deepseek_config.layer_norm_eps,
            'use_rotary_embeddings': args.deepseek_config.use_rotary_embeddings,
            'use_alibi': args.deepseek_config.use_alibi,
            'use_flash_attention_2': args.deepseek_config.use_flash_attention_2,
            'use_sliding_window': args.deepseek_config.use_sliding_window,
            'sliding_window_size': args.deepseek_config.sliding_window_size
        }
    }
    
    return config

def setup_experiment_tracking(args: KFGRPOScriptArguments, config: Dict[str, Any]):
    """Setup experiment tracking with wandb and MLflow."""
    # Setup wandb
    if "wandb" in args.report_to:
        wandb.init(
            project="kf-grpo",
            config=config,
            settings=wandb.Settings(
                code_dir=".",
                disable_git=True,
                start_method="thread"
            )
        )
        logger.info("Wandb initialized successfully")
    
    # Setup MLflow
    mlflow.start_run()
    mlflow.log_params(config)
    logger.info("MLflow initialized successfully")

def setup_distributed_training(args: KFGRPOScriptArguments):
    """Setup distributed training environment."""
    if args.distributed_world_size > 1:
        os.environ['MASTER_ADDR'] = args.distributed_master_addr
        os.environ['MASTER_PORT'] = args.distributed_master_port
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.distributed_world_size,
            rank=args.distributed_rank
        )
        logger.info(f"Distributed training initialized with {args.distributed_world_size} processes")

def main(script_args: KFGRPOScriptArguments, training_args: Any, model_args: Any) -> None:
    """Main training function."""
    # Setup environment
    setup_environment()
    
    # Setup configuration
    config = setup_training_config(script_args)
    
    # Setup experiment tracking
    setup_experiment_tracking(script_args, config)
    
    # Setup distributed training
    setup_distributed_training(script_args)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    try:
        # Load dataset and tokenizer
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            cache_dir=training_args.cache_dir,
            streaming=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.deepseek_config.model_name,
            trust_remote_code=True
        )
        
        # Initialize model with DeepSeek optimizations
        model = AutoModelForCausalLM.from_pretrained(
            script_args.deepseek_config.model_name,
            torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16 if script_args.fp16 else torch.float32,
            device_map="auto" if script_args.use_deepspeed else None,
            trust_remote_code=True
        )
        
        if script_args.use_deepseek_optimizations:
            setup_deepseek_optimizations(model, script_args.deepseek_config)
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=script_args.reward_funcs,
            args=script_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
        )
        
        # Train and evaluate
        final_loss = trainer.train()
        mlflow.log_metric("final_loss", final_loss)
        mlflow.pytorch.log_model(trainer.model, "model")
        
        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)
            
    except Exception as e:
        logger.error(f"Exception during training: {e}")
        sentry_sdk.capture_exception(e)
        raise
    finally:
        mlflow.end_run()
        if "wandb" in script_args.report_to:
            wandb.finish()

if __name__ == "__main__":
    args = tyro.cli(KFGRPOScriptArguments)
    main(args, args, args) 