"""
Enhanced callback system for training monitoring and logging.

Follows best practices for:
- Weights & Biases (wandb) integration
- TensorBoard logging
- Custom metrics tracking
- Error handling and graceful degradation
"""
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class for training events."""
    
    def on_log(self, state: Dict[str, Any]) -> None:
        """Called during training step logging.
        
        Args:
            state: Dictionary containing step, loss, learning_rate, tokens_per_sec, etc.
        """
        pass

    def on_eval(self, state: Dict[str, Any]) -> None:
        """Called during evaluation.
        
        Args:
            state: Dictionary containing step, val_loss, perplexity, improved, etc.
        """
        pass

    def on_save(self, state: Dict[str, Any]) -> None:
        """Called when checkpoint is saved.
        
        Args:
            state: Dictionary containing path, step, etc.
        """
        pass
    
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        """Called at the beginning of training.
        
        Args:
            state: Dictionary containing config, model info, etc.
        """
        pass
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        """Called at the end of training.
        
        Args:
            state: Dictionary containing final metrics, etc.
        """
        pass


class PrintLogger(Callback):
    """
    Simple print-based logger for console output.
    
    Provides clean, formatted output for training progress.
    """
    
    def on_log(self, state: Dict[str, Any]) -> None:
        """Print training step information."""
        step = state.get("step", 0)
        loss = state.get("loss")
        lr = state.get("learning_rate")
        tps = state.get("tokens_per_sec", 0)
        
        # Format output
        msg_parts = [f"step={step}"]
        if loss is not None:
            msg_parts.append(f"loss={loss:.4f}")
        if lr is not None:
            msg_parts.append(f"lr={lr:.2e}")
        if tps > 0:
            msg_parts.append(f"tokens/s={tps:.0f}")
        
        print(f"[train] {' '.join(msg_parts)}")

    def on_eval(self, state: Dict[str, Any]) -> None:
        """Print evaluation results."""
        step = state.get("step", 0)
        val_loss = state.get("val_loss")
        ppl = state.get("perplexity")
        improved = state.get("improved", False)
        
        msg_parts = [f"step={step}"]
        if val_loss is not None:
            msg_parts.append(f"val_loss={val_loss:.4f}")
        if ppl is not None:
            msg_parts.append(f"ppl={ppl:.2f}")
        if improved:
            msg_parts.append("✨ improved")
        
        print(f"[eval] {' '.join(msg_parts)}")

    def on_save(self, state: Dict[str, Any]) -> None:
        """Print checkpoint save information."""
        path = state.get("path", "unknown")
        print(f"[save] checkpoint saved -> {path}")


class WandbLogger(Callback):
    """
    Weights & Biases logger with enhanced error handling and metrics tracking.
    
    Follows W&B best practices for experiment tracking.
    """
    
    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ) -> None:
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            run_name: Run name
            config: Configuration dictionary to log
            tags: List of tags for the run
        """
        self._wandb = None
        self._enabled = False
        
        try:
            import wandb
            self._wandb = wandb
            
            # Initialize W&B run
            if not wandb.run:
                wandb.init(
                    project=project or "truthgpt",
                    name=run_name,
                    config=config or {},
                    tags=tags or [],
                    reinit=False,
                )
            else:
                # Update config if run already exists
                if config:
                    wandb.config.update(config)
            
            self._enabled = True
            logger.info(f"W&B initialized: project={project}, run={run_name}")
            
        except ImportError:
            logger.warning(
                "wandb not available. Install with: pip install wandb. "
                "Continuing without W&B logging."
            )
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}", exc_info=True)

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        """Log training start."""
        if not self._enabled or self._wandb is None:
            return
        
        try:
            # Log system information
            import torch
            if torch.cuda.is_available():
                self._wandb.config.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                })
            self._wandb.config.update({
                "pytorch_version": torch.__version__,
            })
        except Exception as e:
            logger.debug(f"Error logging system info to W&B: {e}")

    def on_log(self, state: Dict[str, Any]) -> None:
        """Log training metrics to W&B."""
        if not self._enabled or self._wandb is None:
            return
        
        try:
            # Extract metrics from state
            metrics = {
                "train/loss": state.get("loss"),
                "train/learning_rate": state.get("learning_rate"),
                "train/tokens_per_sec": state.get("tokens_per_sec"),
            }
            
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            if metrics:
                step = state.get("step", 0)
                self._wandb.log(metrics, step=step)
        except Exception as e:
            logger.debug(f"Error logging to W&B: {e}")

    def on_eval(self, state: Dict[str, Any]) -> None:
        """Log evaluation metrics to W&B."""
        if not self._enabled or self._wandb is None:
            return
        
        try:
            metrics = {
                "eval/val_loss": state.get("val_loss"),
                "eval/perplexity": state.get("perplexity"),
            }
            
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            if metrics:
                step = state.get("step", 0)
                self._wandb.log(metrics, step=step)
        except Exception as e:
            logger.debug(f"Error logging eval to W&B: {e}")

    def on_save(self, state: Dict[str, Any]) -> None:
        """Optionally log checkpoint artifacts."""
        if not self._enabled or self._wandb is None:
            return
        
        try:
            # W&B automatically tracks checkpoints if they're saved to the run directory
            # You can also manually log artifacts here if needed
            pass
        except Exception as e:
            logger.debug(f"Error in W&B save callback: {e}")
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        """Finish W&B run."""
        if not self._enabled or self._wandb is None:
            return
        
        try:
            # Log final metrics if provided
            if state:
                self._wandb.log(state)
            
            # Finish run (optional, W&B will finish automatically on exit)
            # wandb.finish()
        except Exception as e:
            logger.debug(f"Error finishing W&B run: {e}")


class TensorBoardLogger(Callback):
    """
    TensorBoard logger with enhanced metrics tracking.
    
    Follows TensorBoard best practices for experiment tracking.
    """
    
    def __init__(self, log_dir: Optional[str] = None) -> None:
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self._writer = None
        self._enabled = False
        self._log_dir = log_dir or "runs"
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self._log_dir)
            self._enabled = True
            logger.info(f"TensorBoard initialized: log_dir={self._log_dir}")
        except ImportError:
            logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard. "
                "Continuing without TensorBoard logging."
            )
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard: {e}", exc_info=True)

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        """Log training start."""
        if not self._enabled or self._writer is None:
            return
        
        try:
            # Log hyperparameters if provided
            if "config" in state:
                # Add hyperparameters as text (TensorBoard doesn't have native hparams support in all versions)
                # You can log them as text summaries or use add_hparams if available
                pass
        except Exception as e:
            logger.debug(f"Error in TensorBoard train_begin: {e}")

    def on_log(self, state: Dict[str, Any]) -> None:
        """Log training metrics to TensorBoard."""
        if not self._enabled or self._writer is None:
            return
        
        try:
            step = state.get("step", 0)
            
            # Log training metrics with proper naming
            if "loss" in state:
                self._writer.add_scalar("train/loss", state["loss"], global_step=step)
            if "learning_rate" in state:
                self._writer.add_scalar("train/learning_rate", state["learning_rate"], global_step=step)
            if "tokens_per_sec" in state:
                self._writer.add_scalar("train/tokens_per_sec", state["tokens_per_sec"], global_step=step)
            
            # Flush periodically (every 10 steps) for better real-time viewing
            if step % 10 == 0:
                self._writer.flush()
        except Exception as e:
            logger.debug(f"Error logging to TensorBoard: {e}")

    def on_eval(self, state: Dict[str, Any]) -> None:
        """Log evaluation metrics to TensorBoard."""
        if not self._enabled or self._writer is None:
            return
        
        try:
            step = state.get("step", 0)
            
            if "val_loss" in state:
                self._writer.add_scalar("eval/val_loss", state["val_loss"], global_step=step)
            if "perplexity" in state:
                self._writer.add_scalar("eval/perplexity", state["perplexity"], global_step=step)
            
            self._writer.flush()  # Always flush after eval
        except Exception as e:
            logger.debug(f"Error logging eval to TensorBoard: {e}")

    def on_save(self, state: Dict[str, Any]) -> None:
        """Log checkpoint information."""
        if not self._enabled or self._writer is None:
            return
        
        try:
            # TensorBoard doesn't directly track checkpoints, but you can log metadata
            pass
        except Exception as e:
            logger.debug(f"Error in TensorBoard save callback: {e}")
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        """Close TensorBoard writer."""
        if not self._enabled or self._writer is None:
            return
        
        try:
            self._writer.flush()
            self._writer.close()
            logger.info("TensorBoard writer closed")
        except Exception as e:
            logger.debug(f"Error closing TensorBoard writer: {e}")



