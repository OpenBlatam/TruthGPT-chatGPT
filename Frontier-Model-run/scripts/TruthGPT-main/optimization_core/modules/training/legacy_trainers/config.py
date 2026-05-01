"""
Configuration module for trainer.

Separates configuration from implementation for better modularity.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any


class ModelConfig(BaseModel):
    """Configuration for model settings."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name_or_path: str = Field(default="gpt2", description="Model identifier")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    lora_enabled: bool = Field(default=False, description="Enable LoRA adapters")
    lora_r: int = Field(default=16, description="LoRA attention dimension")
    lora_alpha: int = Field(default=32, description="LoRA alpha scaling parameter")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    epochs: int = Field(default=3, description="Number of training epochs")
    train_batch_size: int = Field(default=8, description="Batch size for training")
    eval_batch_size: int = Field(default=8, description="Batch size for evaluation")
    grad_accum_steps: int = Field(default=2, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")
    learning_rate: float = Field(default=5e-5, description="Initial learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay factor")
    warmup_ratio: float = Field(default=0.06, description="Ratio of total steps for warmup")
    scheduler: str = Field(default="cosine", description="Learning rate scheduler type: cosine|linear")
    mixed_precision: str = Field(default="bf16", description="Mixed precision setting: none|fp16|bf16")
    early_stopping_patience: int = Field(default=2, description="Early stopping patience")
    log_interval: int = Field(default=50, description="Logging interval in steps")
    eval_interval: int = Field(default=500, description="Evaluation interval in steps")
    select_best_by: str = Field(default="loss", description="Metric to select best model: loss|ppl")
    optimizer_type: str = Field(default="adamw", description="Optimizer type: adamw|sgd|adafactor")


class HardwareConfig(BaseModel):
    """Configuration for hardware settings."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    device: str = Field(default="auto", description="Device type: auto|cuda|cpu|mps")
    multi_gpu: bool = Field(default=False, description="Enable multi-GPU training")
    ddp: bool = Field(default=False, description="Enable Distributed Data Parallel")
    allow_tf32: bool = Field(default=True, description="Allow TF32 matmul")
    torch_compile: bool = Field(default=False, description="Enable torch.compile")
    compile_mode: str = Field(default="default", description="Compile mode: default|reduce-overhead|max-autotune")
    fused_adamw: bool = Field(default=True, description="Enable fused AdamW if supported")
    detect_anomaly: bool = Field(default=False, description="Enable anomaly detection")
    use_profiler: bool = Field(default=False, description="Enable PyTorch profiler")
    num_workers: int = Field(default=4, description="Number of data loading workers")
    prefetch_factor: int = Field(default=2, description="Data loading prefetch factor")
    persistent_workers: bool = Field(default=True, description="Keep data loading workers alive")


class CheckpointConfig(BaseModel):
    """Configuration for checkpointing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    interval_steps: int = Field(default=1000, description="Checkpoint save interval in steps")
    keep_last: int = Field(default=3, description="Number of recent checkpoints to retain")
    save_safetensors: bool = Field(default=True, description="Save weights using safetensors")
    resume_enabled: bool = Field(default=False, description="Enable resuming from checkpoint")
    resume_checkpoint_dir: Optional[str] = Field(default=None, description="Directory to resume from")
    resume_from_checkpoint: Optional[str] = Field(default=None, description="Specific checkpoint path to resume from")


class EMAConfig(BaseModel):
    """Configuration for Exponential Moving Average."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    enabled: bool = Field(default=True, description="Enable EMA")
    decay: float = Field(default=0.999, description="EMA decay rate")


class TrainerConfig(BaseModel):
    """
    Complete trainer configuration composed of specialized configs.
    
    This follows the composition pattern for better modularity.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    seed: int = Field(default=42, description="Random seed")
    run_name: str = Field(default="run", description="Experiment run name")
    output_dir: str = Field(default="runs/run", description="Directory to save outputs")
    
    # Composition of specialized configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainerConfig":
        """
        Create TrainerConfig from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            TrainerConfig instance
        """
        # Extract top-level config
        seed = config_dict.get("seed", 42)
        run_name = config_dict.get("run_name", "run")
        output_dir = config_dict.get("output_dir", "runs/run")
        metadata = config_dict.get("metadata", {})
        
        # Extract model config
        model_dict = config_dict.get("model", {})
        model = ModelConfig(
            name_or_path=model_dict.get("name_or_path", "gpt2"),
            gradient_checkpointing=model_dict.get("gradient_checkpointing", True),
            lora_enabled=model_dict.get("lora", {}).get("enabled", False),
            lora_r=model_dict.get("lora", {}).get("r", 16),
            lora_alpha=model_dict.get("lora", {}).get("alpha", 32),
            lora_dropout=model_dict.get("lora", {}).get("dropout", 0.05),
        )
        
        # Extract training config
        training_dict = config_dict.get("training", {})
        training = TrainingConfig(
            epochs=training_dict.get("epochs", 3),
            train_batch_size=training_dict.get("train_batch_size", 8),
            eval_batch_size=training_dict.get("eval_batch_size", 8),
            grad_accum_steps=training_dict.get("grad_accum_steps", 2),
            max_grad_norm=training_dict.get("max_grad_norm", 1.0),
            learning_rate=training_dict.get("learning_rate", 5e-5),
            weight_decay=training_dict.get("weight_decay", 0.01),
            warmup_ratio=training_dict.get("warmup_ratio", 0.06),
            scheduler=training_dict.get("scheduler", "cosine"),
            mixed_precision=training_dict.get("mixed_precision", "bf16"),
            early_stopping_patience=training_dict.get("early_stopping_patience", 2),
            log_interval=training_dict.get("log_interval", 50),
            eval_interval=training_dict.get("eval_interval", 500),
            select_best_by=config_dict.get("eval", {}).get("select_best_by", "loss"),
        )
        
        # Extract hardware config
        hardware_dict = config_dict.get("hardware", {})
        training_dict_hw = training_dict  # Also check training dict for hardware settings
        hardware = HardwareConfig(
            device=hardware_dict.get("device", "auto"),
            multi_gpu=training_dict_hw.get("multi_gpu", False),
            ddp=training_dict_hw.get("ddp", False),
            allow_tf32=training_dict_hw.get("allow_tf32", True),
            torch_compile=training_dict_hw.get("torch_compile", False),
            compile_mode=training_dict_hw.get("compile_mode", "default"),
            fused_adamw=training_dict_hw.get("fused_adamw", True),
            detect_anomaly=training_dict_hw.get("detect_anomaly", False),
            use_profiler=training_dict_hw.get("use_profiler", False),
            num_workers=config_dict.get("data", {}).get("num_workers", 4),
            prefetch_factor=config_dict.get("data", {}).get("prefetch_factor", 2),
            persistent_workers=config_dict.get("data", {}).get("persistent_workers", True),
        )
        
        # Extract checkpoint config
        checkpoint_dict = config_dict.get("checkpoint", {})
        resume_dict = config_dict.get("resume", {})
        checkpoint = CheckpointConfig(
            interval_steps=checkpoint_dict.get("interval_steps", 1000),
            keep_last=checkpoint_dict.get("keep_last", 3),
            save_safetensors=training_dict_hw.get("save_safetensors", True),
            resume_enabled=resume_dict.get("enabled", False),
            resume_checkpoint_dir=resume_dict.get("checkpoint_dir"),
            resume_from_checkpoint=None,
        )
        
        # Extract EMA config
        ema_dict = config_dict.get("ema", {})
        ema = EMAConfig(
            enabled=ema_dict.get("enabled", True),
            decay=ema_dict.get("decay", 0.999),
        )
        
        return cls(
            seed=seed,
            run_name=run_name,
            output_dir=output_dir,
            model=model,
            training=training,
            hardware=hardware,
            checkpoint=checkpoint,
            ema=ema,
            metadata=metadata,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "seed": self.seed,
            "run_name": self.run_name,
            "output_dir": self.output_dir,
            "model": {
                "name_or_path": self.model.name_or_path,
                "gradient_checkpointing": self.model.gradient_checkpointing,
                "lora": {
                    "enabled": self.model.lora_enabled,
                    "r": self.model.lora_r,
                    "alpha": self.model.lora_alpha,
                    "dropout": self.model.lora_dropout,
                },
            },
            "training": {
                "epochs": self.training.epochs,
                "train_batch_size": self.training.train_batch_size,
                "eval_batch_size": self.training.eval_batch_size,
                "grad_accum_steps": self.training.grad_accum_steps,
                "max_grad_norm": self.training.max_grad_norm,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "warmup_ratio": self.training.warmup_ratio,
                "scheduler": self.training.scheduler,
                "mixed_precision": self.training.mixed_precision,
                "early_stopping_patience": self.training.early_stopping_patience,
                "log_interval": self.training.log_interval,
                "eval_interval": self.training.eval_interval,
            },
            "hardware": {
                "device": self.hardware.device,
                "multi_gpu": self.hardware.multi_gpu,
                "ddp": self.hardware.ddp,
                "allow_tf32": self.hardware.allow_tf32,
                "torch_compile": self.hardware.torch_compile,
                "compile_mode": self.hardware.compile_mode,
                "fused_adamw": self.hardware.fused_adamw,
                "detect_anomaly": self.hardware.detect_anomaly,
                "use_profiler": self.hardware.use_profiler,
            },
            "checkpoint": {
                "interval_steps": self.checkpoint.interval_steps,
                "keep_last": self.checkpoint.keep_last,
                "save_safetensors": self.checkpoint.save_safetensors,
            },
            "ema": {
                "enabled": self.ema.enabled,
                "decay": self.ema.decay,
            },
            "metadata": self.metadata,
        }




