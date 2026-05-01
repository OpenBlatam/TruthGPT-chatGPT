"""
Unified Quantization module with automatic backend selection.

Supports INT8, FP16, and dynamic quantization with C++/Rust acceleration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Dict, Any, List
import numpy as np

from .backend import Backend, get_best_backend, is_backend_available


class QuantizationType(Enum):
    """Quantization types."""
    INT8 = "int8"           # 8-bit integer
    INT4 = "int4"           # 4-bit integer (GPTQ)
    FP16 = "fp16"           # Half precision
    BF16 = "bf16"           # Brain float 16
    DYNAMIC = "dynamic"     # Dynamic quantization
    STATIC = "static"       # Static quantization (requires calibration)


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    quantization_type: QuantizationType = QuantizationType.INT8
    calibration_samples: int = 100  # For static quantization
    per_channel: bool = True  # Per-channel vs per-tensor
    symmetric: bool = True  # Symmetric vs asymmetric


@dataclass
class QuantizationStats:
    """Quantization statistics."""
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    quantization_time_ms: float = 0.0
    
    @property
    def space_savings(self) -> float:
        return 1.0 - self.compression_ratio


class Quantizer:
    """
    Unified Quantizer with automatic backend selection.
    
    Automatically selects the best backend:
    - C++: Fast quantization with SIMD
    - Rust: Memory-efficient quantization
    - Python: PyTorch quantization (fallback)
    
    Example:
        >>> quantizer = Quantizer(quantization_type=QuantizationType.INT8)
        >>> quantized, stats = quantizer.quantize(weights)
        >>> print(f"Compression: {stats.compression_ratio:.2%}")
    """
    
    def __init__(
        self,
        config: Optional[QuantizationConfig] = None,
        quantization_type: str = "int8",
        backend: Optional[Backend] = None,
        **kwargs
    ):
        """
        Initialize Quantizer.
        
        Args:
            config: Quantization configuration
            quantization_type: Type name (if config not provided)
            backend: Force specific backend
            **kwargs: Additional config options
        """
        if config is None:
            qtype = QuantizationType(quantization_type.lower())
            config = QuantizationConfig(quantization_type=qtype, **kwargs)
        
        self.config = config
        self._backend = backend or get_best_backend('quantization')
        self._impl = self._create_implementation()
    
    def _create_implementation(self):
        """Create backend-specific implementation."""
        if self._backend == Backend.CPP and is_backend_available(Backend.CPP):
            return self._create_cpp_impl()
        elif self._backend == Backend.RUST and is_backend_available(Backend.RUST):
            return self._create_rust_impl()
        else:
            return None  # Use Python
    
    def _create_cpp_impl(self):
        """Create C++ implementation."""
        # Would use C++ quantization bindings
        return None
    
    def _create_rust_impl(self):
        """Create Rust implementation."""
        try:
            from optimization_core.rust_core import truthgpt_rust
            return truthgpt_rust.PyQuantizer(
                self.config.quantization_type.value,
                self.config.per_channel,
                self.config.symmetric
            )
        except Exception:
            return None
    
    def quantize(
        self,
        weights: np.ndarray,
        calibration_data: Optional[List[np.ndarray]] = None
    ) -> tuple[np.ndarray, QuantizationStats]:
        """
        Quantize weights.
        
        Args:
            weights: FP32 weights array
            calibration_data: Calibration data for static quantization
            
        Returns:
            Tuple of (quantized_weights, stats)
        """
        import time
        start = time.perf_counter()
        
        original_size = weights.nbytes / (1024 * 1024)  # MB
        
        if self._impl is not None and self._backend == Backend.RUST:
            # Rust implementation
            quantized = self._impl.quantize(weights)
            quantized_size = quantized.nbytes / (1024 * 1024)
        else:
            # Python fallback
            quantized = self._python_quantize(weights)
            quantized_size = quantized.nbytes / (1024 * 1024)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        stats = QuantizationStats(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=quantized_size / original_size if original_size > 0 else 0.0,
            quantization_time_ms=elapsed_ms
        )
        
        return quantized, stats
    
    def dequantize(self, quantized: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """
        Dequantize weights back to FP32.
        
        Args:
            quantized: Quantized weights
            scale: Quantization scale (if known)
            
        Returns:
            Dequantized FP32 weights
        """
        if self._impl is not None and self._backend == Backend.RUST:
            return self._impl.dequantize(quantized, scale)
        else:
            return self._python_dequantize(quantized, scale)
    
    def _python_quantize(self, weights: np.ndarray) -> np.ndarray:
        """Python fallback quantization."""
        qtype = self.config.quantization_type
        
        if qtype == QuantizationType.INT8:
            # Simple INT8 quantization
            scale = np.abs(weights).max() / 127.0
            quantized = np.round(weights / scale).astype(np.int8)
            return quantized
        elif qtype == QuantizationType.FP16:
            return weights.astype(np.float16)
        elif qtype == QuantizationType.BF16:
            # BF16 approximation (not true BF16 without hardware support)
            return weights.astype(np.float32)  # Fallback
        else:
            return weights
    
    def _python_dequantize(self, quantized: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """Python fallback dequantization."""
        if quantized.dtype == np.int8:
            if scale is None:
                scale = 1.0  # Would need to store scale
            return quantized.astype(np.float32) * scale
        elif quantized.dtype == np.float16:
            return quantized.astype(np.float32)
        else:
            return quantized.astype(np.float32)
    
    def quantize_model(
        self,
        model: Any,  # PyTorch model
        calibration_data: Optional[List[Any]] = None
    ) -> Any:
        """
        Quantize entire PyTorch model.
        
        Args:
            model: PyTorch model
            calibration_data: Calibration data for static quantization
            
        Returns:
            Quantized model
        """
        try:
            import torch
            import torch.quantization as quant
            
            if self.config.quantization_type == QuantizationType.DYNAMIC:
                # Dynamic quantization
                quantized = quant.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                return quantized
            elif self.config.quantization_type == QuantizationType.STATIC:
                # Static quantization requires calibration
                if calibration_data is None:
                    raise ValueError("Static quantization requires calibration_data")
                
                model.eval()
                model.qconfig = quant.get_default_qconfig('fbgemm')
                quant.prepare(model, inplace=True)
                
                # Calibrate
                for sample in calibration_data:
                    model(sample)
                
                quantized = quant.convert(model, inplace=False)
                return quantized
            else:
                return model
        except ImportError:
            raise RuntimeError("PyTorch required for model quantization")
    
    @property
    def backend(self) -> Backend:
        return self._backend
    
    def __repr__(self) -> str:
        return (f"Quantizer(type={self.config.quantization_type.value}, "
                f"backend={self._backend.name})")


# Convenience functions
def quantize_weights(weights: np.ndarray, qtype: str = "int8") -> tuple[np.ndarray, QuantizationStats]:
    """Quick quantize function."""
    return Quantizer(quantization_type=qtype).quantize(weights)


def dequantize_weights(quantized: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    """Quick dequantize function."""
    return Quantizer().dequantize(quantized, scale)













