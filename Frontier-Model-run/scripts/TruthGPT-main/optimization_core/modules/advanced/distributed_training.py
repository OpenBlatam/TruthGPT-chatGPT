"""
Distributed Training for TruthGPT
Following deep learning best practices for multi-GPU and multi-node training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import os
import time
import logging
from contextlib import contextmanager


@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    use_ddp: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 25
    broadcast_buffers: bool = True


class DistributedTrainer:
    """Distributed training coordinator"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup distributed logging"""
        logger = logging.getLogger(f"DistributedTrainer_rank_{self.config.rank}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {self.config.rank}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize(self):
        """Initialize distributed training"""
        if self.is_initialized:
            return
        
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
                rank=self.config.rank
            )
            
            self.is_initialized = True
            self.logger.info(f"Distributed training initialized: rank {self.config.rank}/{self.config.world_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.is_initialized = False
            self.logger.info("Distributed training cleaned up")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with DDP"""
        if not self.config.use_ddp:
            return model
        
        if not self.is_initialized:
            self.initialize()
        
        # Move model to correct device
        device = torch.device(f"cuda:{self.config.local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            bucket_cap_mb=self.config.bucket_cap_mb,
            broadcast_buffers=self.config.broadcast_buffers
        )
        
        self.logger.info(f"Model wrapped with DDP on rank {self.config.rank}")
        return ddp_model
    
    def create_distributed_sampler(self, dataset, num_replicas: Optional[int] = None, 
                                  rank: Optional[int] = None) -> DistributedSampler:
        """Create distributed sampler"""
        if num_replicas is None:
            num_replicas = self.config.world_size
        if rank is None:
            rank = self.config.rank
        
        return DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True
        )
    
    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce tensor across all processes"""
        if not self.is_initialized:
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / self.config.world_size
        return tensor
    
    def all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """All-gather tensor from all processes"""
        if not self.is_initialized:
            return [tensor]
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def barrier(self):
        """Synchronize all processes"""
        if self.is_initialized:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.config.rank == 0
    
    def get_world_size(self) -> int:
        """Get world size"""
        return self.config.world_size
    
    def get_rank(self) -> int:
        """Get current rank"""
        return self.config.rank


class HorovodTrainer:
    """Horovod-based distributed trainer"""
    
    def __init__(self):
        try:
            import horovod.torch as hvd
            self.hvd = hvd
            self.is_available = True
        except ImportError:
            self.hvd = None
            self.is_available = False
            raise ImportError("Horovod not available. Install with: pip install horovod")
    
    def initialize(self):
        """Initialize Horovod"""
        if not self.is_available:
            raise RuntimeError("Horovod not available")
        
        self.hvd.init()
        torch.cuda.set_device(self.hvd.local_rank())
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with Horovod"""
        if not self.is_available:
            return model
        
        return self.hvd.DistributedOptimizer(model)
    
    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Wrap optimizer with Horovod"""
        if not self.is_available:
            return optimizer
        
        return self.hvd.DistributedOptimizer(optimizer)
    
    def get_rank(self) -> int:
        """Get Horovod rank"""
        if not self.is_available:
            return 0
        return self.hvd.rank()
    
    def get_world_size(self) -> int:
        """Get Horovod world size"""
        if not self.is_available:
            return 1
        return self.hvd.size()
    
    def is_main_process(self) -> bool:
        """Check if main process"""
        if not self.is_available:
            return True
        return self.hvd.rank() == 0


class RayTrainer:
    """Ray-based distributed trainer"""
    
    def __init__(self):
        try:
            import ray
            self.ray = ray
            self.is_available = True
        except ImportError:
            self.ray = None
            self.is_available = False
            raise ImportError("Ray not available. Install with: pip install ray")
    
    def initialize(self, num_workers: int = 4):
        """Initialize Ray"""
        if not self.is_available:
            raise RuntimeError("Ray not available")
        
        if not self.ray.is_initialized():
            self.ray.init(num_cpus=num_workers)
    
    def create_remote_trainer(self, trainer_class, *args, **kwargs):
        """Create remote trainer"""
        if not self.is_available:
            return trainer_class(*args, **kwargs)
        
        return self.ray.remote(trainer_class).remote(*args, **kwargs)
    
    def train_distributed(self, trainers: List, data_loaders: List):
        """Train with distributed Ray workers"""
        if not self.is_available:
            return
        
        # Train in parallel
        futures = []
        for trainer, data_loader in zip(trainers, data_loaders):
            future = trainer.train.remote(data_loader)
            futures.append(future)
        
        # Wait for completion
        self.ray.get(futures)
    
    def cleanup(self):
        """Cleanup Ray"""
        if self.is_available and self.ray.is_initialized():
            self.ray.shutdown()


class DistributedTrainingManager:
    """Manager for distributed training operations"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.trainer = DistributedTrainer(config)
        self.logger = logging.getLogger("DistributedTrainingManager")
    
    def setup_distributed_training(self, model: nn.Module, 
                                 optimizer: torch.optim.Optimizer,
                                 dataset) -> Tuple[nn.Module, torch.optim.Optimizer, DistributedSampler]:
        """Setup complete distributed training"""
        # Initialize distributed training
        self.trainer.initialize()
        
        # Wrap model
        ddp_model = self.trainer.wrap_model(model)
        
        # Create distributed sampler
        distributed_sampler = self.trainer.create_distributed_sampler(dataset)
        
        # Setup optimizer for distributed training
        if hasattr(optimizer, 'param_groups'):
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        param.grad = None
        
        self.logger.info("Distributed training setup completed")
        return ddp_model, optimizer, distributed_sampler
    
    def train_epoch(self, model: nn.Module, data_loader, optimizer: torch.optim.Optimizer,
                   loss_fn: Callable, device: torch.device) -> Dict[str, float]:
        """Train one epoch with distributed training"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient synchronization (handled by DDP)
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0 and self.trainer.is_main_process():
                self.logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Synchronize loss across processes
        if self.trainer.is_initialized:
            loss_tensor = torch.tensor(total_loss, device=device)
            loss_tensor = self.trainer.all_reduce_tensor(loss_tensor)
            total_loss = loss_tensor.item()
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate_epoch(self, model: nn.Module, data_loader, loss_fn: Callable,
                      device: torch.device) -> Dict[str, float]:
        """Validate one epoch with distributed training"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = loss_fn(outputs, batch)
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
        
        # Synchronize loss across processes
        if self.trainer.is_initialized:
            loss_tensor = torch.tensor(total_loss, device=device)
            loss_tensor = self.trainer.all_reduce_tensor(loss_tensor)
            total_loss = loss_tensor.item()
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                      epoch: int, loss: float, checkpoint_path: str):
        """Save checkpoint (only on main process)"""
        if self.trainer.is_main_process():
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': self.config
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def cleanup(self):
        """Cleanup distributed training"""
        self.trainer.cleanup()


@contextmanager
def distributed_training_context(config: DistributedConfig):
    """Context manager for distributed training"""
    manager = DistributedTrainingManager(config)
    try:
        manager.trainer.initialize()
        yield manager
    finally:
        manager.cleanup()



