"""
TensorRT-LLM Engine Configuration
=================================

Configuration classes for TensorRT-LLM engine initialization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

from optimization_core.core.config_base import ValidatedConfig, ConfigValidationError


@dataclass
class TensorRTLLMConfig(ValidatedConfig):
    """
    Configuration for TensorRT-LLM engine.
    
    Attributes:
        model_path: Path to model (HuggingFace format)
        engine_path: Path to save/load compiled engine
        precision: Precision mode (fp32, fp16, bf16, int8, fp8)
        max_batch_size: Maximum batch size
        max_seq_length: Maximum sequence length
        use_quantization: Enable quantization
        quantization_type: Quantization type (int8, fp8)
        builder_kwargs: Additional builder arguments
    """
    model_path: str
    engine_path: Optional[str] = None
    precision: str = "fp16"
    max_batch_size: int = 8
    max_seq_length: int = 512
    use_quantization: bool = False
    quantization_type: str = "int8"
    builder_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Valid precision modes
    VALID_PRECISIONS = {"fp32", "fp16", "bf16", "int8", "fp8"}
    
    # Valid quantization types
    VALID_QUANTIZATIONS = {"int8", "fp8"}
    
    def _validate(self) -> None:
        """Validate configuration after initialization."""
        super()._validate()
        
        self.validate_string(self.model_path, "model_path", min_length=1)
        self.validate_string(
            self.precision,
            "precision",
            allowed_values=self.VALID_PRECISIONS
        )
        
        if self.use_quantization:
            self.validate_string(
                self.quantization_type,
                "quantization_type",
                allowed_values=self.VALID_QUANTIZATIONS
            )
        
        self.validate_positive_int(self.max_batch_size, "max_batch_size")
        self.validate_positive_int(self.max_seq_length, "max_seq_length")
        
        # Validate model path exists
        if not Path(self.model_path).exists():
            raise ConfigValidationError(
                f"Model path does not exist: {self.model_path}"
            )


