"""
Optimization Modules
"""

from .imports import *
from .core import BaseModule

class OptimizationModule(BaseModule):
    """Base optimization module"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
    
    def _setup(self):
        """Setup optimization components"""
        self._create_optimizer()
        self._create_scheduler()
        if self.config.get("use_mixed_precision", False):
            self.scaler = amp.GradScaler()
    
    @abstractmethod
    def _create_optimizer(self):
        """Create optimizer"""
        pass
    
    @abstractmethod
    def _create_scheduler(self):
        """Create scheduler"""
        pass
    
    def optimize(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Optimize model"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            if self.scaler:
                with amp.autocast():
                    loss = self._compute_loss(model, batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(model, batch)
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    @abstractmethod
    def _compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss"""
        pass

class AdamWOptimizationModule(OptimizationModule):
    """AdamW optimization module"""
    
    def parameters(self):
        """Placeholder for parameters - strictly conceptual here since Module doesn't hold model parameters directly"""
        return []

    def _create_optimizer(self):
        """Create AdamW optimizer"""
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        # Note: self.parameters() empty here since model is external.
        # This module structure assumes optimizer is creating using external model params usually.
        # For now, following original logic but checking if parameters passed via config/init could be better.
        pass
    
    def _create_scheduler(self):
        """Create scheduler"""
        scheduler_type = self.config.get("scheduler", "cosine")
        
        if self.optimizer and scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 100),
                num_training_steps=self.config.get("total_steps", 1000)
            )
        else:
            self.scheduler = None
    
    def _compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss"""
        outputs = model(**batch)
        return outputs.loss

class LoRAOptimizationModule(OptimizationModule):
    """LoRA optimization module"""
    
    def parameters(self):
        return []

    def _create_optimizer(self):
        """Create optimizer for LoRA"""
        lr = self.config.get("learning_rate", 1e-4)
        # self.optimizer = AdamW(self.parameters(), lr=lr) 
        pass
    
    def _create_scheduler(self):
        """Create scheduler"""
        if self.optimizer:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.get("warmup_steps", 100),
                num_training_steps=self.config.get("total_steps", 1000)
            )
    
    def _compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss"""
        outputs = model(**batch)
        return outputs.loss

def create_optimization_module(optimization_type: str, config: Dict[str, Any]) -> OptimizationModule:
    """Create optimization module"""
    if optimization_type == "adamw":
        return AdamWOptimizationModule(config)
    elif optimization_type == "lora":
        return LoRAOptimizationModule(config)
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")

