"""
Professional experiment tracking with WandB and TensorBoard support.
"""
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False


class ExperimentTracker:
    """
    Professional experiment tracking with multiple backends.
    """
    
    def __init__(
        self,
        trackers: list[str] = None,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize experiment tracker.
        
        Args:
            trackers: List of trackers to use (wandb|tensorboard)
            project: Project name (for WandB)
            run_name: Run name
            log_dir: Log directory (for TensorBoard)
            **kwargs: Additional tracker arguments
        """
        self.trackers_enabled = trackers or []
        self.run_name = run_name
        self.wandb_run = None
        self.tensorboard_writer = None
        
        # Initialize WandB
        if "wandb" in self.trackers_enabled:
            if not _WANDB_AVAILABLE:
                logger.warning("WandB not available, skipping")
            else:
                try:
                    self.wandb_run = wandb.init(
                        project=project or "truthgpt",
                        name=run_name,
                        **kwargs
                    )
                    logger.info("WandB initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize WandB: {e}")
        
        # Initialize TensorBoard
        if "tensorboard" in self.trackers_enabled:
            if not _TENSORBOARD_AVAILABLE:
                logger.warning("TensorBoard not available, skipping")
            else:
                try:
                    log_dir = log_dir or Path("runs") / (run_name or "default")
                    self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
                    logger.info(f"TensorBoard initialized: {log_dir}")
                except Exception as e:
                    logger.error(f"Failed to initialize TensorBoard: {e}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to all enabled trackers.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.debug(f"Error logging to WandB: {e}")
        
        if self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step or 0)
                    elif isinstance(value, torch.Tensor):
                        self.tensorboard_writer.add_scalar(key, value.item(), step or 0)
            except Exception as e:
                logger.debug(f"Error logging to TensorBoard: {e}")
    
    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None) -> None:
        """
        Log histogram of values.
        
        Args:
            name: Histogram name
            values: Values to histogram
            step: Optional step number
        """
        if self.wandb_run:
            try:
                wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
            except Exception as e:
                logger.debug(f"Error logging histogram to WandB: {e}")
        
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_histogram(name, values, step or 0)
            except Exception as e:
                logger.debug(f"Error logging histogram to TensorBoard: {e}")
    
    def log_model(self, model: torch.nn.Module, input_shape: tuple) -> None:
        """
        Log model architecture.
        
        Args:
            model: Model to log
            input_shape: Input shape for visualization
        """
        if self.wandb_run:
            try:
                # WandB can log model architecture
                wandb.watch(model, log="all", log_freq=100)
            except Exception as e:
                logger.debug(f"Error logging model to WandB: {e}")
        
        if self.tensorboard_writer:
            try:
                # TensorBoard can log model graph
                dummy_input = torch.zeros(input_shape)
                self.tensorboard_writer.add_graph(model, dummy_input)
            except Exception as e:
                logger.debug(f"Error logging model to TensorBoard: {e}")
    
    def finish(self) -> None:
        """Finish tracking session."""
        if self.wandb_run:
            try:
                wandb.finish()
            except Exception as e:
                logger.debug(f"Error finishing WandB: {e}")
        
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
            except Exception as e:
                logger.debug(f"Error closing TensorBoard: {e}")



