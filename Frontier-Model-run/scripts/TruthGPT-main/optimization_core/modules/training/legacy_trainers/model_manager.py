"""
Model Manager — Pydantic-First Architecture.

Handles model loading, configuration, LoRA setup, and device placement.
Returns typed ``ModelLoadResult`` with parameter statistics and timing.
"""
import time
import logging
from enum import Enum
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field, ConfigDict, computed_field

from optimization_core.trainers.config import ModelConfig, HardwareConfig, TrainingConfig

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ParallelMode(str, Enum):
    """Multi-GPU training strategy."""
    NONE = "none"
    DATA_PARALLEL = "dp"
    DDP = "ddp"


class LoraTargetMap(BaseModel):
    """Architecture-aware LoRA target module mapping."""
    gpt_llama_mistral: List[str] = Field(
        default=["c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
    )
    bert_roberta: List[str] = Field(default=["query", "key", "value", "dense"])
    t5_ul2: List[str] = Field(default=["q", "k", "v", "o"])
    opt: List[str] = Field(default=["q_proj", "k_proj", "v_proj", "out_proj"])
    default: List[str] = Field(
        default=["c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
    )

    def resolve(self, model_type: str) -> List[str]:
        """Resolve target modules for a given HF model type string."""
        mt = model_type.lower()
        if any(k in mt for k in ("gpt", "llama", "mistral", "phi", "qwen")):
            return self.gpt_llama_mistral
        if any(k in mt for k in ("bert", "roberta")):
            return self.bert_roberta
        if any(k in mt for k in ("t5", "ul2")):
            return self.t5_ul2
        if "opt" in mt:
            return self.opt
        return self.default


class ModelLoadResult(BaseModel):
    """Typed output of model loading with statistics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_params: int = 0
    trainable_params: int = 0
    lora_applied: bool = False
    gradient_checkpointing: bool = False
    torch_compiled: bool = False
    load_dtype: str = "float32"
    elapsed_ms: float = 0.0

    @computed_field  # type: ignore[misc]
    @property
    def trainable_pct(self) -> float:
        if self.total_params == 0:
            return 0.0
        return round(self.trainable_params / self.total_params * 100, 2)


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Manages model loading, initialization, and configuration.

    Responsibilities:
    - Load tokenizer and model
    - Configure LoRA if needed
    - Enable gradient checkpointing
    - Apply torch.compile if requested
    - Handle device placement
    - Return typed ``ModelLoadResult`` with statistics
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
        training_config: TrainingConfig,
        device: torch.device,
    ):
        self.model_config = model_config
        self.hardware_config = hardware_config
        self.training_config = training_config
        self.device = device
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[nn.Module] = None
        self._parallel_mode = ParallelMode.NONE
        self._lora_target_map = LoraTargetMap()

    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token: %s", tokenizer.eos_token)
            self.tokenizer = tokenizer
            return tokenizer
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e, exc_info=True)
            raise

    def load_model(self) -> Tuple[nn.Module, ModelLoadResult]:
        """Load and configure model, returning typed result."""
        start = time.monotonic()

        # Determine dtype from training config
        load_dtype = None
        dtype_label = "float32"
        if self.device.type == "cuda":
            if self.training_config.mixed_precision == "bf16":
                load_dtype = torch.bfloat16
                dtype_label = "bfloat16"
            elif self.training_config.mixed_precision == "fp16":
                load_dtype = torch.float16
                dtype_label = "float16"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name_or_path,
                torch_dtype=load_dtype,
                device_map=None,
                trust_remote_code=False,
            )

            gc_enabled = False
            if self.model_config.gradient_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    gc_enabled = True
                    logger.info("Gradient checkpointing enabled")
                else:
                    logger.warning("Gradient checkpointing not available for this model")

            if hasattr(model, "config"):
                try:
                    model.config.use_cache = False
                except Exception:
                    pass

            lora_applied = False
            if self.model_config.lora_enabled:
                model = self._apply_lora(model)
                lora_applied = True

            model.to(self.device)

            compiled = False
            if self.hardware_config.torch_compile:
                model = self._compile_model(model)
                compiled = True

            self._initialize_weights(model)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Model loaded: %s total params, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

            self.model = model
            elapsed_ms = (time.monotonic() - start) * 1000

            result = ModelLoadResult(
                total_params=total_params,
                trainable_params=trainable_params,
                lora_applied=lora_applied,
                gradient_checkpointing=gc_enabled,
                torch_compiled=compiled,
                load_dtype=dtype_label,
                elapsed_ms=round(elapsed_ms, 2),
            )

            return model, result

        except Exception as e:
            logger.error("Failed to load model: %s", e, exc_info=True)
            raise

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA to model using architecture-aware target detection."""
        if not _PEFT_AVAILABLE:
            raise RuntimeError("PEFT not available. Install with: pip install peft")

        target_modules = self._detect_lora_target_modules(model)
        logger.info("Applying LoRA: r=%d, alpha=%d", self.model_config.lora_r, self.model_config.lora_alpha)

        try:
            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM if hasattr(TaskType, "CAUSAL_LM") else "CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            logger.info("LoRA successfully applied")
            return model
        except Exception as e:
            logger.error("Failed to apply LoRA: %s", e, exc_info=True)
            raise

    def _detect_lora_target_modules(self, model: nn.Module) -> List[str]:
        """Detect target modules for LoRA using the typed LoraTargetMap."""
        if hasattr(model, "config"):
            model_type = getattr(model.config, "model_type", "")
            modules = self._lora_target_map.resolve(model_type)
            logger.info("Auto-detected LoRA targets for '%s': %s", model_type, modules)
            return modules

        logger.warning("Could not auto-detect LoRA modules, using defaults")
        return self._lora_target_map.default

    def _compile_model(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile to model."""
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available")
            return model

        try:
            compiled = torch.compile(model, mode=self.hardware_config.compile_mode)
            logger.info("Model compiled with mode: %s", self.hardware_config.compile_mode)
            return compiled
        except Exception as e:
            logger.warning("Failed to compile model: %s", e)
            return model

    def _initialize_weights(self, model: nn.Module) -> None:
        """Initialize weights for new/modified layers."""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, "weight") and module.weight is not None:
                    weight_std = module.weight.std().item()
                    if weight_std < 1e-6:
                        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                        if hasattr(module, "bias") and module.bias is not None:
                            nn.init.constant_(module.bias, 0)

    def setup_parallel(self, mode: ParallelMode = ParallelMode.NONE) -> None:
        """
        Setup multi-GPU training.

        Args:
            mode: Parallel training strategy (NONE, DATA_PARALLEL, DDP).
        """
        if mode == ParallelMode.DDP:
            try:
                import torch.distributed as dist
                from torch.nn.parallel import DistributedDataParallel as DDP

                if dist.is_initialized():
                    local_rank = dist.get_rank() % torch.cuda.device_count()
                    self.device = torch.device(f"cuda:{local_rank}")
                    self.model.to(self.device)
                    self.model = DDP(
                        self.model,
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=False,
                    )
                    self._parallel_mode = ParallelMode.DDP
                    logger.info("Using DDP (rank %d)", dist.get_rank())
                else:
                    logger.warning("DDP requested but not initialized")
            except ImportError:
                logger.warning("Distributed training not available")

        elif mode == ParallelMode.DATA_PARALLEL and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self._parallel_mode = ParallelMode.DATA_PARALLEL
            logger.info("Using DataParallel with %d GPUs", torch.cuda.device_count())

    def get_model_for_operations(self) -> nn.Module:
        """Get the base model for operations (handles parallel wrappers)."""
        model = self.model
        if self._parallel_mode != ParallelMode.NONE:
            model = model.module
        if hasattr(model, "module"):
            model = model.module
        return model

    def get_total_params(self) -> Tuple[int, int]:
        """Get total and trainable parameter counts."""
        model = self.get_model_for_operations()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable



