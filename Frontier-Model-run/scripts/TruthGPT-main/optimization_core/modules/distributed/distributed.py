"""
TruthGPT Distributed Training Module
Advanced distributed training utilities for TruthGPT models
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import os
import time
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTDistributedConfig:
    """Configuration for TruthGPT distributed training."""
    # Distributed settings
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Communication settings
    timeout: int = 1800  # 30 minutes
    broadcast_buffers: bool = True
    find_unused_parameters: bool = False
    
    # Performance settings
    enable_gradient_sync: bool = True
    gradient_sync_frequency: int = 1
    enable_all_reduce: bool = True
    
    # Advanced features
    enable_gradient_compression: bool = False
    compression_ratio: float = 0.1
    enable_async_gradient: bool = False
    
    # Monitoring
    enable_distributed_logging: bool = True
    log_interval: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'backend': self.backend,
            'init_method': self.init_method,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'timeout': self.timeout,
            'broadcast_buffers': self.broadcast_buffers,
            'find_unused_parameters': self.find_unused_parameters,
            'enable_gradient_sync': self.enable_gradient_sync,
            'gradient_sync_frequency': self.gradient_sync_frequency,
            'enable_all_reduce': self.enable_all_reduce,
            'enable_gradient_compression': self.enable_gradient_compression,
            'compression_ratio': self.compression_ratio,
            'enable_async_gradient': self.enable_async_gradient,
            'enable_distributed_logging': self.enable_distributed_logging,
            'log_interval': self.log_interval
        }

class TruthGPTDistributedManager:
    """Advanced distributed training manager for TruthGPT."""
    
    def __init__(self, config: TruthGPTDistributedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Distributed state
        self.is_initialized = False
        self.process_group = None
        self.device = None
        
        # Performance tracking
        self.communication_stats = {
            'all_reduce_time': 0.0,
            'broadcast_time': 0.0,
            'total_communications': 0
        }
    
    def initialize_distributed(self) -> bool:
        """Initialize distributed training."""
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=torch.distributed.constants.default_pg_timeout
            )
            
            # Setup device
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.config.local_rank}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")
            
            self.is_initialized = True
            self.logger.info(f"Distributed training initialized - Rank: {self.config.rank}, Device: {self.device}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            return False
    
    def cleanup_distributed(self) -> None:
        """Cleanup distributed training."""
        if self.is_initialized:
            try:
                dist.destroy_process_group()
                self.is_initialized = False
                self.logger.info("Distributed training cleaned up")
            except Exception as e:
                self.logger.error(f"Error during distributed cleanup: {e}")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=self.config.broadcast_buffers,
            find_unused_parameters=self.config.find_unused_parameters
        )
        
        self.logger.info("Model wrapped with DistributedDataParallel")
        return ddp_model
    
    def create_distributed_sampler(self, dataset: torch.utils.data.Dataset, 
                                 shuffle: bool = True) -> DistributedSampler:
        """Create distributed sampler for dataset."""
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle
        )
        
        self.logger.info("Distributed sampler created")
        return sampler
    
    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform all-reduce on tensor."""
        if not self.is_initialized:
            return tensor
        
        start_time = time.time()
        
        # Perform all-reduce
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / self.config.world_size
        
        # Update communication stats
        communication_time = time.time() - start_time
        self.communication_stats['all_reduce_time'] += communication_time
        self.communication_stats['total_communications'] += 1
        
        return tensor
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank."""
        if not self.is_initialized:
            return tensor
        
        start_time = time.time()
        
        # Perform broadcast
        dist.broadcast(tensor, src=src)
        
        # Update communication stats
        communication_time = time.time() - start_time
        self.communication_stats['broadcast_time'] += communication_time
        self.communication_stats['total_communications'] += 1
        
        return tensor
    
    def gather_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all ranks."""
        if not self.is_initialized:
            return [tensor]
        
        # Create list to hold gathered tensors
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        
        # Gather tensors
        dist.all_gather(gathered_tensors, tensor)
        
        return gathered_tensors
    
    def reduce_scatter_tensors(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        """Reduce and scatter tensors."""
        if not self.is_initialized:
            return tensor_list[0]
        
        # Create output tensor
        output_tensor = torch.zeros_like(tensor_list[0])
        
        # Reduce scatter
        dist.reduce_scatter(output_tensor, tensor_list)
        
        return output_tensor
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            'all_reduce_time': self.communication_stats['all_reduce_time'],
            'broadcast_time': self.communication_stats['broadcast_time'],
            'total_communications': self.communication_stats['total_communications'],
            'avg_all_reduce_time': self.communication_stats['all_reduce_time'] / max(1, self.communication_stats['total_communications']),
            'avg_broadcast_time': self.communication_stats['broadcast_time'] / max(1, self.communication_stats['total_communications'])
        }
    
    def log_communication_stats(self) -> None:
        """Log communication statistics."""
        stats = self.get_communication_stats()
        self.logger.info(f"Communication Stats: {stats}")

class TruthGPTGradientCompression:
    """Advanced gradient compression for distributed training."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compress_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compress gradients using top-k selection."""
        compressed_gradients = []
        
        for grad in gradients:
            if grad is None:
                compressed_gradients.append(None)
                continue
            
            # Flatten gradient
            flat_grad = grad.view(-1)
            
            # Select top-k values
            k = max(1, int(self.compression_ratio * flat_grad.numel()))
            _, top_k_indices = torch.topk(torch.abs(flat_grad), k)
            
            # Create compressed gradient
            compressed_grad = torch.zeros_like(flat_grad)
            compressed_grad[top_k_indices] = flat_grad[top_k_indices]
            
            # Reshape back to original shape
            compressed_grad = compressed_grad.view(grad.shape)
            compressed_gradients.append(compressed_grad)
        
        return compressed_gradients
    
    def decompress_gradients(self, compressed_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decompress gradients."""
        # For top-k compression, decompression is just using the compressed values
        return compressed_gradients

class TruthGPTDistributedOptimizer:
    """Advanced distributed optimizer for TruthGPT."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 distributed_manager: TruthGPTDistributedManager,
                 enable_compression: bool = False,
                 compression_ratio: float = 0.1):
        self.optimizer = optimizer
        self.distributed_manager = distributed_manager
        self.enable_compression = enable_compression
        self.compression_ratio = compression_ratio
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Setup gradient compression if enabled
        if self.enable_compression:
            self.gradient_compressor = TruthGPTGradientCompression(compression_ratio)
        else:
            self.gradient_compressor = None
    
    def step(self) -> None:
        """Perform distributed optimization step."""
        # Get gradients
        gradients = [p.grad for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
        
        if not gradients:
            return
        
        # Compress gradients if enabled
        if self.enable_compression and self.gradient_compressor:
            gradients = self.gradient_compressor.compress_gradients(gradients)
        
        # Synchronize gradients across processes
        for grad in gradients:
            if grad is not None:
                self.distributed_manager.all_reduce_tensor(grad)
        
        # Perform optimization step
        self.optimizer.step()
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

class TruthGPTDistributedTrainer:
    """Advanced distributed trainer for TruthGPT."""
    
    def __init__(self, config: TruthGPTDistributedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Distributed components
        self.distributed_manager = TruthGPTDistributedManager(config)
        self.distributed_optimizer = None
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def setup_distributed_training(self, model: nn.Module, 
                                  optimizer: torch.optim.Optimizer,
                                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
        """Setup distributed training."""
        # Initialize distributed training
        if not self.distributed_manager.initialize_distributed():
            raise RuntimeError("Failed to initialize distributed training")
        
        # Wrap model with DDP
        self.model = self.distributed_manager.wrap_model(model)
        
        # Setup distributed optimizer
        self.distributed_optimizer = TruthGPTDistributedOptimizer(
            optimizer, 
            self.distributed_manager,
            enable_compression=self.config.enable_gradient_compression,
            compression_ratio=self.config.compression_ratio
        )
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.logger.info("Distributed training setup completed")
    
    def train_step(self, loss: torch.Tensor) -> Dict[str, Any]:
        """Perform distributed training step."""
        # Backward pass
        loss.backward()
        
        # Distributed optimization step
        self.distributed_optimizer.step()
        self.distributed_optimizer.zero_grad()
        
        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
        
        # Get communication stats
        comm_stats = self.distributed_manager.get_communication_stats()
        
        return {
            'loss': loss.item(),
            'communication_stats': comm_stats
        }
    
    def cleanup(self) -> None:
        """Cleanup distributed training."""
        self.distributed_manager.cleanup_distributed()

# Factory functions
def create_truthgpt_distributed_manager(config: TruthGPTDistributedConfig) -> TruthGPTDistributedManager:
    """Create TruthGPT distributed manager."""
    return TruthGPTDistributedManager(config)

def create_truthgpt_distributed_trainer(config: TruthGPTDistributedConfig) -> TruthGPTDistributedTrainer:
    """Create TruthGPT distributed trainer."""
    return TruthGPTDistributedTrainer(config)

def setup_distributed_training(model: nn.Module, 
                             optimizer: torch.optim.Optimizer,
                             config: TruthGPTDistributedConfig) -> TruthGPTDistributedTrainer:
    """Quick setup for distributed training."""
    trainer = create_truthgpt_distributed_trainer(config)
    trainer.setup_distributed_training(model, optimizer)
    return trainer

# Example usage
if __name__ == "__main__":
    # Example TruthGPT distributed training
    print("🚀 TruthGPT Distributed Training Demo")
    print("=" * 50)
    
    # Create distributed configuration
    config = TruthGPTDistributedConfig(
        world_size=2,
        rank=0,
        local_rank=0,
        enable_gradient_compression=True,
        compression_ratio=0.1
    )
    
    # Create distributed trainer
    trainer = create_truthgpt_distributed_trainer(config)
    
    # Create sample model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 10000)
        
        def forward(self, x):
            return self.linear(x)
    
    model = TruthGPTModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Setup distributed training
    try:
        trainer.setup_distributed_training(model, optimizer)
        print("✅ Distributed training setup completed!")
    except RuntimeError as e:
        print(f"❌ Distributed training setup failed: {e}")
        print("Note: This demo requires multiple GPUs or proper distributed setup")
    
    print("✅ TruthGPT distributed training demo completed!")



