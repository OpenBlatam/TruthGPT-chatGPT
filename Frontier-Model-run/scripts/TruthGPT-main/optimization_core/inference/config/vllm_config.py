"""
vLLM Engine Configuration
=========================

Configuration classes for vLLM engine initialization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from optimization_core.core.config_base import ValidatedConfig, ConfigValidationError


@dataclass
class VLLMConfig(ValidatedConfig):
    """
    Configuration for vLLM engine.
    
    Attributes:
        model: Model name or path (HuggingFace format)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        max_model_len: Maximum sequence length
        dtype: Data type (auto, float16, bfloat16, float32)
        quantization: Quantization method (awq, gptq, squeezellm, None)
        trust_remote_code: Trust remote code from HuggingFace
        engine_kwargs: Additional vLLM engine arguments
    """
    model: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    trust_remote_code: bool = False
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Valid data types
    VALID_DTYPES = {"auto", "float16", "bfloat16", "float32"}
    
    # Valid quantization methods
    VALID_QUANTIZATIONS = {"awq", "gptq", "squeezellm"}
    
    def _validate(self) -> None:
        """Validate configuration after initialization."""
        super()._validate()
        
        self.validate_string(self.model, "model", min_length=1)
        self.validate_positive_int(self.tensor_parallel_size, "tensor_parallel_size")
        self.validate_positive_float(
            self.gpu_memory_utilization,
            "gpu_memory_utilization",
            min_value=0.0,
            max_value=1.0
        )
        
        if self.max_model_len is not None:
            self.validate_positive_int(self.max_model_len, "max_model_len")
        
        self.validate_string(
            self.dtype,
            "dtype",
            allowed_values=self.VALID_DTYPES
        )
        
        if self.quantization is not None:
            self.validate_string(
                self.quantization,
                "quantization",
                allowed_values=self.VALID_QUANTIZATIONS
            )
    
    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Convert configuration to vLLM engine kwargs."""
        kwargs = {
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            **self.engine_kwargs
        }
        
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization
        
        return kwargs


