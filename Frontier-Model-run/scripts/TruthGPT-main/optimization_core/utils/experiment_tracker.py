"""
Experiment tracking and monitoring system
Following deep learning best practices for experiment management
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import torch
import numpy as np
from datetime import datetime
import wandb
import tensorboard
from tensorboard import SummaryWriter
import mlflow
import pandas as pd


class ExperimentMetrics(BaseModel):
    """Structured metrics for experiment tracking"""
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    learning_rate: float
    timestamp: str
    additional_metrics: Dict[str, float] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Experiment configuration"""
    experiment_name: str
    project_name: str = "truthgpt"
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)


class ExperimentTracker:
    """
    Comprehensive experiment tracking system
    Supports multiple backends: wandb, tensorboard, mlflow, local
    """
    
    def __init__(self, config: ExperimentConfig, backend: str = "local"):
        self.config = config
        self.backend = backend
        self.logger = self._setup_logging()
        
        # Initialize tracking backend
        self._initialize_backend()
        
        # Experiment state
        self.start_time = time.time()
        self.metrics_history = []
        self.best_metrics = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for experiment tracker"""
        logger = logging.getLogger("ExperimentTracker")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_backend(self):
        """Initialize the specified tracking backend"""
        try:
            if self.backend == "wandb":
                self._init_wandb()
            elif self.backend == "tensorboard":
                self._init_tensorboard()
            elif self.backend == "mlflow":
                self._init_mlflow()
            elif self.backend == "local":
                self._init_local()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            self.logger.error(f"Error initializing {self.backend} backend: {e}")
            raise
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                tags=self.config.tags,
                notes=self.config.notes,
                config=self.config.config
            )
            self.logger.info("Initialized Weights & Biases tracking")
        except Exception as e:
            self.logger.error(f"Error initializing wandb: {e}")
            raise
    
    def _init_tensorboard(self):
        """Initialize TensorBoard tracking"""
        try:
            log_dir = Path("runs") / self.config.experiment_name
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.logger.info(f"Initialized TensorBoard tracking at {log_dir}")
        except Exception as e:
            self.logger.error(f"Error initializing tensorboard: {e}")
            raise
    
    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            mlflow.set_experiment(self.config.experiment_name)
            self.run = mlflow.start_run()
            self.logger.info("Initialized MLflow tracking")
        except Exception as e:
            self.logger.error(f"Error initializing mlflow: {e}")
            raise
    
    def _init_local(self):
        """Initialize local file-based tracking"""
        try:
            self.experiment_dir = Path("experiments") / self.config.experiment_name
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Save experiment configuration
            config_path = self.experiment_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.model_dump(), f, indent=2)
            
            self.logger.info(f"Initialized local tracking at {self.experiment_dir}")
        except Exception as e:
            self.logger.error(f"Error initializing local tracking: {e}")
            raise
    
    def log_metrics(self, metrics: Union[ExperimentMetrics, Dict[str, float]], 
                   step: Optional[int] = None):
        """Log metrics to the tracking backend"""
        try:
            if isinstance(metrics, dict):
                metrics = ExperimentMetrics(
                    epoch=metrics.get('epoch', 0),
                    step=metrics.get('step', step or 0),
                    train_loss=metrics.get('train_loss', 0.0),
                    val_loss=metrics.get('val_loss', 0.0),
                    learning_rate=metrics.get('learning_rate', 0.0),
                    timestamp=datetime.now().isoformat(),
                    additional_metrics=metrics
                )
            
            # Store metrics in history
            self.metrics_history.append(metrics)
            
            # Update best metrics
            self._update_best_metrics(metrics)
            
            # Log to backend
            if self.backend == "wandb":
                self._log_wandb_metrics(metrics)
            elif self.backend == "tensorboard":
                self._log_tensorboard_metrics(metrics)
            elif self.backend == "mlflow":
                self._log_mlflow_metrics(metrics)
            elif self.backend == "local":
                self._log_local_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            raise
    
    def _log_wandb_metrics(self, metrics: ExperimentMetrics):
        """Log metrics to Weights & Biases"""
        log_dict = {
            'epoch': metrics.epoch,
            'step': metrics.step,
            'train_loss': metrics.train_loss,
            'val_loss': metrics.val_loss,
            'learning_rate': metrics.learning_rate
        }
        log_dict.update(metrics.additional_metrics)
        
        wandb.log(log_dict, step=metrics.step)
    
    def _log_tensorboard_metrics(self, metrics: ExperimentMetrics):
        """Log metrics to TensorBoard"""
        self.writer.add_scalar('Loss/Train', metrics.train_loss, metrics.step)
        self.writer.add_scalar('Loss/Validation', metrics.val_loss, metrics.step)
        self.writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.step)
        
        # Log additional metrics
        for key, value in metrics.additional_metrics.items():
            self.writer.add_scalar(f'Metrics/{key}', value, metrics.step)
    
    def _log_mlflow_metrics(self, metrics: ExperimentMetrics):
        """Log metrics to MLflow"""
        mlflow.log_metric("train_loss", metrics.train_loss, step=metrics.step)
        mlflow.log_metric("val_loss", metrics.val_loss, step=metrics.step)
        mlflow.log_metric("learning_rate", metrics.learning_rate, step=metrics.step)
        
        # Log additional metrics
        for key, value in metrics.additional_metrics.items():
            mlflow.log_metric(key, value, step=metrics.step)
    
    def _log_local_metrics(self, metrics: ExperimentMetrics):
        """Log metrics to local files"""
        metrics_file = self.experiment_dir / "metrics.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def _update_best_metrics(self, metrics: ExperimentMetrics):
        """Update best metrics tracking"""
        if not self.best_metrics or metrics.val_loss < self.best_metrics.get('val_loss', float('inf')):
            self.best_metrics = {
                'val_loss': metrics.val_loss,
                'train_loss': metrics.train_loss,
                'epoch': metrics.epoch,
                'step': metrics.step,
                'timestamp': metrics.timestamp
            }
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Log model architecture and parameters"""
        try:
            if self.backend == "wandb":
                wandb.watch(model, log="all", log_freq=100)
            elif self.backend == "tensorboard":
                # Log model graph (requires dummy input)
                dummy_input = torch.randn(1, 10, 768)  # Adjust based on model
                self.writer.add_graph(model, dummy_input)
            elif self.backend == "mlflow":
                # Save model as artifact
                model_path = self.experiment_dir / f"{model_name}.pt"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path))
            elif self.backend == "local":
                # Save model locally
                model_path = self.experiment_dir / f"{model_name}.pt"
                torch.save(model.state_dict(), model_path)
                
                # Save model info
                model_info = {
                    'total_parameters': sum(p.numel() for p in model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
                }
                
                info_path = self.experiment_dir / f"{model_name}_info.json"
                with open(info_path, 'w') as f:
                    json.dump(model_info, f, indent=2)
            
            self.logger.info(f"Logged model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error logging model: {e}")
            raise
    
    def log_artifacts(self, artifacts: Dict[str, Union[str, Path]]):
        """Log artifacts (files, images, etc.)"""
        try:
            for name, path in artifacts.items():
                if self.backend == "wandb":
                    wandb.save(str(path))
                elif self.backend == "tensorboard":
                    # For images, use add_image
                    if str(path).endswith(('.png', '.jpg', '.jpeg')):
                        from PIL import Image
                        img = Image.open(path)
                        self.writer.add_image(name, np.array(img))
                elif self.backend == "mlflow":
                    mlflow.log_artifact(str(path))
                elif self.backend == "local":
                    # Copy to experiment directory
                    dest_path = self.experiment_dir / name
                    if Path(path).is_file():
                        import shutil
                        shutil.copy2(path, dest_path)
            
            self.logger.info(f"Logged artifacts: {list(artifacts.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error logging artifacts: {e}")
            raise
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters"""
        try:
            if self.backend == "wandb":
                wandb.config.update(hyperparams)
            elif self.backend == "tensorboard":
                # Log as text
                hparam_str = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
                self.writer.add_text("Hyperparameters", hparam_str)
            elif self.backend == "mlflow":
                mlflow.log_params(hyperparams)
            elif self.backend == "local":
                # Save to JSON file
                hparam_path = self.experiment_dir / "hyperparameters.json"
                with open(hparam_path, 'w') as f:
                    json.dump(hyperparams, f, indent=2)
            
            self.logger.info("Logged hyperparameters")
            
        except Exception as e:
            self.logger.error(f"Error logging hyperparameters: {e}")
            raise
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        duration = time.time() - self.start_time
        
        return {
            'experiment_name': self.config.experiment_name,
            'project_name': self.config.project_name,
            'duration_seconds': duration,
            'total_steps': len(self.metrics_history),
            'best_metrics': self.best_metrics,
            'backend': self.backend,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat()
        }
    
    def save_experiment_summary(self):
        """Save experiment summary to file"""
        try:
            summary = self.get_experiment_summary()
            
            if self.backend == "local":
                summary_path = self.experiment_dir / "experiment_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
            elif self.backend == "wandb":
                wandb.log({"experiment_summary": summary})
            elif self.backend == "mlflow":
                mlflow.log_params(summary)
            
            self.logger.info("Saved experiment summary")
            
        except Exception as e:
            self.logger.error(f"Error saving experiment summary: {e}")
            raise
    
    def close(self):
        """Close the experiment tracking"""
        try:
            if self.backend == "wandb":
                wandb.finish()
            elif self.backend == "tensorboard":
                self.writer.close()
            elif self.backend == "mlflow":
                mlflow.end_run()
            
            # Save final summary
            self.save_experiment_summary()
            
            self.logger.info("Closed experiment tracking")
            
        except Exception as e:
            self.logger.error(f"Error closing experiment tracking: {e}")
            raise


# Utility functions for easy experiment tracking
def create_experiment_tracker(experiment_name: str, 
                            project_name: str = "truthgpt",
                            backend: str = "local",
                            tags: List[str] = None) -> ExperimentTracker:
    """Create an experiment tracker with default settings"""
    config = ExperimentConfig(
        experiment_name=experiment_name,
        project_name=project_name,
        tags=tags or []
    )
    
    return ExperimentTracker(config, backend)


def log_training_step(tracker: ExperimentTracker, 
                     epoch: int, step: int, 
                     train_loss: float, val_loss: float,
                     learning_rate: float, **kwargs):
    """Log a training step with common metrics"""
    metrics = ExperimentMetrics(
        epoch=epoch,
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=learning_rate,
        timestamp=datetime.now().isoformat(),
        additional_metrics=kwargs
    )
    
    tracker.log_metrics(metrics)


# Example usage
if __name__ == "__main__":
    # Create experiment tracker
    tracker = create_experiment_tracker(
        experiment_name="test_experiment",
        project_name="truthgpt",
        backend="local"
    )
    
    # Log some metrics
    log_training_step(
        tracker=tracker,
        epoch=1,
        step=100,
        train_loss=0.5,
        val_loss=0.6,
        learning_rate=1e-4,
        accuracy=0.95
    )
    
    # Close tracker
    tracker.close()



