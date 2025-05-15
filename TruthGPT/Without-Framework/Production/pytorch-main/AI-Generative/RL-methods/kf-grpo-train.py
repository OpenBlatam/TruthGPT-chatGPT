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
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, set_seed
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from rich.logging import RichHandler
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm.auto import tqdm
import gc
from functools import partial
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm.auto import tqdm
import gc
from functools import partial
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm.auto import tqdm
import gc
from functools import partial
import torch.backends.cudnn as cudnn

@dataclass
class KFGRPOScriptArguments(ScriptArguments):
    """Script arguments for the KF-GRPO training script with advanced optimizations."""
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    # Kalman Filter parameters
    process_noise: float = field(default=0.01, metadata={"help": "Process noise covariance (Q)"})
    measurement_noise: float = field(default=0.1, metadata={"help": "Measurement noise covariance (R)"})
    kalman_memory_size: int = field(default=1000, metadata={"help": "Size of Kalman filter memory buffer"})
    
    # CPPO parameters
    pruning_threshold: float = field(default=0.1, metadata={"help": "Threshold for sample pruning"})
    pruning_alpha: float = field(default=0.5, metadata={"help": "Alpha for dynamic K adjustment"})
    k_min: int = field(default=1, metadata={"help": "Minimum K value"})
    k_max: int = field(default=10, metadata={"help": "Maximum K value"})
    
    # AGPO parameters
    policy_clip_delta: float = field(default=0.2, metadata={"help": "Policy clipping delta"})
    length_penalty_lambda: float = field(default=0.1, metadata={"help": "Length penalty coefficient"})
    max_length: int = field(default=1000, metadata={"help": "Maximum sequence length for normalization"})
    
    # Advanced optimization parameters
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
    
    # Advanced scheduler
    use_one_cycle: bool = field(default=False, metadata={"help": "Use OneCycleLR scheduler"})
    div_factor: float = field(default=25.0, metadata={"help": "Initial learning rate divisor"})
    final_div_factor: float = field(default=1e4, metadata={"help": "Final learning rate divisor"})
    pct_start: float = field(default=0.3, metadata={"help": "Percentage of training for warmup"})

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

class KFGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[Any, List[Any]],
        args: Any = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, DatasetDict, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[Any] = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # Initialize distributed training
        self._setup_distributed()
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            memory_size=args.kalman_memory_size
        )
        
        # Initialize optimization components
        self.scaler = GradScaler() if args.use_amp else None
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self._metrics = {
            "kalman_reward": [],
            "pruned_samples": [],
            "length_penalty": [],
            "learning_rate": [],
            "gradient_norm": [],
            "memory_usage": [],
            "throughput": [],
            "gpu_utilization": []
        }
        
        # Enable memory optimizations
        self._setup_memory_optimizations()
        
        # Enable performance optimizations
        self._setup_performance_optimizations()
        
        # Compile model if enabled
        if args.use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

    def _setup_distributed(self):
        """Setup distributed training."""
        if self.args.distributed_world_size > 1:
            if torch.cuda.is_available():
                init_process_group(
                    backend=self.args.distributed_backend,
                    init_method=f"tcp://{self.args.distributed_master_addr}:{self.args.distributed_master_port}",
                    world_size=self.args.distributed_world_size,
                    rank=self.args.distributed_rank
                )
                self.model = DDP(self.model)
                self.train_sampler = DistributedSampler(self.train_dataset)
                self.eval_sampler = DistributedSampler(self.eval_dataset) if self.eval_dataset else None

    def _setup_memory_optimizations(self):
        """Setup memory optimizations."""
        if self.args.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if self.args.use_activation_checkpointing:
            self.model.config.use_activation_checkpointing = True
        
        if self.args.use_attention_slicing:
            self.model.config.use_attention_slicing = True
        
        if self.args.use_sequence_parallelism:
            self.model.config.use_sequence_parallelism = True
        
        if self.args.use_cpu_offload:
            self.model = self.model.cpu()
            self.model.half()

    def _setup_performance_optimizations(self):
        """Setup performance optimizations."""
        if self.args.use_cudnn_benchmark:
            cudnn.benchmark = True
        
        if self.args.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        if self.args.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay and parameter grouping."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.args.use_8bit_optimizer:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            return AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        if self.args.use_one_cycle:
            return OneCycleLR(
                self.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=self.args.num_train_epochs * len(self.train_dataset),
                pct_start=self.args.pct_start,
                div_factor=self.args.div_factor,
                final_div_factor=self.args.final_div_factor
            )
        elif self.args.lr_scheduler_type == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.num_train_epochs,
                T_mult=self.args.num_cycles
            )
        return None

    def train(self):
        """Enhanced training loop with optimizations."""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=len(self.train_dataset),
            desc="Training",
            disable=not self.is_local_process_zero()
        )
        
        for step, batch in enumerate(self.get_train_dataloader()):
            # Forward pass with mixed precision
            with autocast(enabled=self.args.use_amp):
                loss = self.compute_loss(self.model, batch)
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                # Clear memory
                if step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": total_loss / (step + 1),
                "lr": self.optimizer.param_groups[0]["lr"]
            })
            
            # Log metrics
            if step % self.args.logging_steps == 0:
                self._log_metrics()
                
                # Calculate throughput
                elapsed_time = time.time() - start_time
                samples_per_second = (step + 1) * self.args.train_batch_size / elapsed_time
                self._metrics["throughput"].append(samples_per_second)
                
                # Track GPU utilization
                if torch.cuda.is_available():
                    self._metrics["gpu_utilization"].append(
                        torch.cuda.utilization()
                    )
        
        progress_bar.close()
        return total_loss / len(self.train_dataset)

    def _log_metrics(self):
        """Log metrics to wandb with enhanced tracking."""
        if wandb.run is not None:
            metrics = {
                k: np.mean(v) for k, v in self._metrics.items()
            }
            
            # Add system metrics
            if torch.cuda.is_available():
                metrics.update({
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**2,
                    "gpu_utilization": torch.cuda.utilization(),
                })
            
            wandb.log(metrics)
            
            # Clear metrics after logging
            for v in self._metrics.values():
                v.clear()

def main(script_args: KFGRPOScriptArguments, training_args: Any, model_args: Any) -> None:
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(training_args.get_process_log_level())
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Initialize wandb with enhanced tracking
    if "wandb" in training_args.report_to:
        wandb.init(
            project="kf-grpo",
            config={
                **script_args.__dict__,
                **training_args.__dict__,
                **model_args.__dict__
            },
            settings=wandb.Settings(
                code_dir=".",
                disable_git=True,
                start_method="thread"
            )
        )
    
    # Load dataset and tokenizer with caching
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        cache_dir=training_args.cache_dir
    )
    tokenizer = get_tokenizer(model_args, training_args)
    
    # Initialize trainer with optimizations
    trainer = KFGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=script_args.reward_funcs,
        args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    
    # Train and evaluate
    trainer.train()
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((KFGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)