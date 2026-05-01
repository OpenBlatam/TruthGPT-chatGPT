"""
Configuration management module with validation and loading.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    name_or_path: str = "gpt2"
    gradient_checkpointing: bool = True
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    attention_backend: str = "sdpa"
    kv_cache_type: str = "none"
    kv_cache_block_size: int = 128
    memory_policy: str = "adaptive"


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 8
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    scheduler: str = "cosine"
    mixed_precision: str = "bf16"  # none|fp16|bf16
    early_stopping_patience: int = 2
    log_interval: int = 50
    eval_interval: int = 500
    optimizer_type: str = "adamw"
    fused_adamw: bool = True
    allow_tf32: bool = True
    torch_compile: bool = False
    compile_mode: str = "default"
    detect_anomaly: bool = False
    use_profiler: bool = False
    save_safetensors: bool = True
    select_best_by: str = "loss"  # loss|ppl
    callbacks: list[str] = field(default_factory=lambda: ["print"])


@dataclass
class DataConfig:
    """Data loading configuration."""
    source: str = "hf"  # hf|jsonl|webdataset
    dataset: str = "wikitext"
    subset: Optional[str] = "wikitext-2-raw-v1"
    text_field: str = "text"
    streaming: bool = False
    collate: str = "lm"
    max_seq_len: int = 512
    bucket_by_length: bool = False
    bucket_bins: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = "auto"  # auto|cuda|cpu|mps
    multi_gpu: bool = False
    ddp: bool = False


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    interval_steps: int = 1000
    keep_last: int = 3
    enabled: bool = True


@dataclass
class EMAConfig:
    """EMA (Exponential Moving Average) configuration."""
    enabled: bool = True
    decay: float = 0.999


@dataclass
class ResumeConfig:
    """Resume training configuration."""
    enabled: bool = False
    checkpoint_dir: Optional[str] = None


@dataclass
class TrainerConfig:
    """
    Complete trainer configuration.
    Combines all sub-configurations.
    """
    seed: int = 42
    run_name: str = "run"
    output_dir: str = "runs/run"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    
    logging: Dict[str, Any] = field(default_factory=dict)
    eval: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainerConfig":
        """Create TrainerConfig from dictionary."""
        # Extract sub-configs
        model_dict = config_dict.get("model", {})
        training_dict = config_dict.get("training", {})
        data_dict = config_dict.get("data", {})
        hardware_dict = config_dict.get("hardware", {})
        ckpt_dict = config_dict.get("checkpoint", {})
        ema_dict = config_dict.get("ema", {})
        resume_dict = config_dict.get("resume", {})
        
        # Build sub-configs
        model_cfg = ModelConfig(
            name_or_path=model_dict.get("name_or_path", "gpt2"),
            gradient_checkpointing=model_dict.get("gradient_checkpointing", True),
            lora_enabled=model_dict.get("lora", {}).get("enabled", False),
            lora_r=model_dict.get("lora", {}).get("r", 16),
            lora_alpha=model_dict.get("lora", {}).get("alpha", 32),
            lora_dropout=model_dict.get("lora", {}).get("dropout", 0.05),
            attention_backend=model_dict.get("attention", {}).get("backend", "sdpa"),
            kv_cache_type=model_dict.get("kv_cache", {}).get("type", "none"),
            kv_cache_block_size=model_dict.get("kv_cache", {}).get("block_size", 128),
            memory_policy=model_dict.get("memory", {}).get("policy", "adaptive"),
        )
        
        training_cfg = TrainingConfig(
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
            optimizer_type=training_dict.get("optimizer", {}).get("type", "adamw"),
            fused_adamw=training_dict.get("fused_adamw", True),
            allow_tf32=training_dict.get("allow_tf32", True),
            torch_compile=training_dict.get("torch_compile", False),
            compile_mode=training_dict.get("compile_mode", "default"),
            detect_anomaly=training_dict.get("detect_anomaly", False),
            use_profiler=training_dict.get("use_profiler", False),
            save_safetensors=training_dict.get("save_safetensors", True),
            select_best_by=config_dict.get("eval", {}).get("select_best_by", "loss"),
            callbacks=training_dict.get("callbacks", ["print"]),
        )
        
        data_cfg = DataConfig(
            source=data_dict.get("source", "hf"),
            dataset=data_dict.get("dataset", "wikitext"),
            subset=data_dict.get("subset", "wikitext-2-raw-v1"),
            text_field=data_dict.get("text_field", "text"),
            streaming=data_dict.get("streaming", False),
            collate=data_dict.get("collate", "lm"),
            max_seq_len=data_dict.get("max_seq_len", 512),
            bucket_by_length=data_dict.get("bucket_by_length", False),
            bucket_bins=data_dict.get("bucket_bins", [64, 128, 256, 512]),
            num_workers=data_dict.get("num_workers", 4),
            prefetch_factor=data_dict.get("prefetch_factor", 2),
            persistent_workers=data_dict.get("persistent_workers", True),
        )
        
        hardware_cfg = HardwareConfig(
            device=hardware_dict.get("device", "auto"),
            multi_gpu=hardware_dict.get("multi_gpu", False),
            ddp=hardware_dict.get("ddp", False),
        )
        
        ckpt_cfg = CheckpointConfig(
            interval_steps=ckpt_dict.get("interval_steps", 1000),
            keep_last=ckpt_dict.get("keep_last", 3),
            enabled=True,
        )
        
        ema_cfg = EMAConfig(
            enabled=ema_dict.get("enabled", True),
            decay=ema_dict.get("decay", 0.999),
        )
        
        resume_cfg = ResumeConfig(
            enabled=resume_dict.get("enabled", False),
            checkpoint_dir=resume_dict.get("checkpoint_dir"),
        )
        
        return cls(
            seed=config_dict.get("seed", 42),
            run_name=config_dict.get("run_name", "run"),
            output_dir=config_dict.get("output_dir", "runs/run"),
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            hardware=hardware_cfg,
            checkpoint=ckpt_cfg,
            ema=ema_cfg,
            resume=resume_cfg,
            logging=config_dict.get("logging", {}),
            eval=config_dict.get("eval", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainerConfig to dictionary."""
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
                "attention": {"backend": self.model.attention_backend},
                "kv_cache": {
                    "type": self.model.kv_cache_type,
                    "block_size": self.model.kv_cache_block_size,
                },
                "memory": {"policy": self.model.memory_policy},
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
                "optimizer": {"type": self.training.optimizer_type},
                "fused_adamw": self.training.fused_adamw,
                "allow_tf32": self.training.allow_tf32,
                "torch_compile": self.training.torch_compile,
                "compile_mode": self.training.compile_mode,
                "detect_anomaly": self.training.detect_anomaly,
                "use_profiler": self.training.use_profiler,
                "save_safetensors": self.training.save_safetensors,
                "callbacks": self.training.callbacks,
            },
            "data": {
                "source": self.data.source,
                "dataset": self.data.dataset,
                "subset": self.data.subset,
                "text_field": self.data.text_field,
                "streaming": self.data.streaming,
                "collate": self.data.collate,
                "max_seq_len": self.data.max_seq_len,
                "bucket_by_length": self.data.bucket_by_length,
                "bucket_bins": self.data.bucket_bins,
                "num_workers": self.data.num_workers,
                "prefetch_factor": self.data.prefetch_factor,
                "persistent_workers": self.data.persistent_workers,
            },
            "hardware": {
                "device": self.hardware.device,
                "multi_gpu": self.hardware.multi_gpu,
                "ddp": self.hardware.ddp,
            },
            "checkpoint": {
                "interval_steps": self.checkpoint.interval_steps,
                "keep_last": self.checkpoint.keep_last,
            },
            "ema": {
                "enabled": self.ema.enabled,
                "decay": self.ema.decay,
            },
            "resume": {
                "enabled": self.resume.enabled,
                "checkpoint_dir": self.resume.checkpoint_dir,
            },
            "logging": self.logging,
            "eval": self.eval,
        }


class ConfigManager:
    """
    Configuration manager for loading and validating YAML configs.
    """
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file with validation.
        
        Args:
            path: Path to YAML file
        
        Returns:
            Configuration dictionary
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid or empty
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError(f"Empty or invalid YAML file: {path}")
            
            logger.info(f"Successfully loaded config from {path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error reading config file {path}: {e}", exc_info=True)
            raise
    
    @staticmethod
    def validate_config(config_dict: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["model", "training", "data"]
        
        for key in required_keys:
            if key not in config_dict:
                raise ValueError(f"Missing required configuration section: {key}")
        
        # Validate model config
        model = config_dict.get("model", {})
        if "name_or_path" not in model:
            raise ValueError("model.name_or_path is required")
        
        # Validate training config
        training = config_dict.get("training", {})
        if "epochs" in training and training["epochs"] < 1:
            raise ValueError("training.epochs must be >= 1")
        if "learning_rate" in training and training["learning_rate"] <= 0:
            raise ValueError("training.learning_rate must be > 0")
        
        # Validate data config
        data = config_dict.get("data", {})
        if "dataset" not in data:
            raise ValueError("data.dataset is required")
        
        logger.debug("Configuration validation passed")
        return True
    
    @classmethod
    def load_config(cls, path: str) -> TrainerConfig:
        """
        Load and validate configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
        
        Returns:
            TrainerConfig instance
        """
        config_dict = cls.load_yaml(path)
        cls.validate_config(config_dict)
        return TrainerConfig.from_dict(config_dict)



