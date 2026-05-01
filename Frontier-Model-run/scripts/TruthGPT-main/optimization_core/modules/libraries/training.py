"""
Training Modules
"""

from .imports import *
from .core import BaseModule
from .models import ModelModule
from .data import DataModule

class TrainingModule(BaseModule):
    """Base class for training modules"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_module = None
        self.data_module = None
        self.metrics = {}
        self.best_metric = float('inf')
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.early_stopping_counter = 0
    
    def _setup(self):
        """Setup training components"""
        self._setup_logging()
        self._setup_checkpointing()
    
    def _setup_logging(self):
        """Setup logging"""
        if self.config.get("use_wandb", False) and wandb:
            wandb.init(project=self.config.get("project_name", "truthgpt"))
        
        if self.config.get("use_tensorboard", False) and SummaryWriter:
            self.tb_writer = SummaryWriter(log_dir=self.config.get("log_dir", "runs"))
    
    def _setup_checkpointing(self):
        """Setup checkpointing"""
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train(self, model_module: ModelModule, data_module: DataModule, epochs: int):
        """Train the model"""
        self.model_module = model_module
        self.data_module = data_module
        
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Check early stopping
            if self._check_early_stopping(val_metrics):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics)
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        total_metrics = {}
        num_batches = 0
        
        dataloader = self.data_module.get_dataloader()
        if not dataloader:
            return {}

        for batch in dataloader:
            metrics = self.model_module.train_step(batch)
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value
            
            num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            if num_batches > 0:
                total_metrics[key] /= num_batches
        
        return total_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        total_metrics = {}
        num_batches = 0
        
        dataloader = self.data_module.get_dataloader()
        if not dataloader:
            return {}
            
        for batch in dataloader:
            metrics = self.model_module.eval_step(batch)
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value
            
            num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            if num_batches > 0:
                total_metrics[key] /= num_batches
        
        return total_metrics
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics"""
        self.logger.info(f"Epoch {epoch}: Train Loss: {train_metrics.get('loss', 0):.4f}, Val Loss: {val_metrics.get('loss', 0):.4f}")
        
        if hasattr(self, 'tb_writer'):
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f"Train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f"Val/{key}", value, epoch)
        
        if wandb and wandb.run:
            wandb.log({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check early stopping condition"""
        current_metric = val_metrics.get('loss', float('inf'))
        
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_module.model.state_dict(),
            'optimizer_state_dict': self.model_module.optimizer.state_dict(),
            'scheduler_state_dict': self.model_module.scheduler.state_dict() if self.model_module.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics.get('loss', float('inf')) < self.best_metric:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

class SupervisedTrainingModule(TrainingModule):
    """Supervised training module"""
    
    def _setup(self):
        """Setup supervised training"""
        super()._setup()
        self.metrics = ["accuracy", "f1", "precision", "recall"]

class UnsupervisedTrainingModule(TrainingModule):
    """Unsupervised training module"""
    
    def _setup(self):
        """Setup unsupervised training"""
        super()._setup()
        self.metrics = ["loss", "perplexity"]

def create_training_module(training_type: str, config: Dict[str, Any]) -> TrainingModule:
    """Create training module"""
    if training_type == "supervised":
        return SupervisedTrainingModule(config)
    elif training_type == "unsupervised":
        return UnsupervisedTrainingModule(config)
    else:
        raise ValueError(f"Unknown training type: {training_type}")

