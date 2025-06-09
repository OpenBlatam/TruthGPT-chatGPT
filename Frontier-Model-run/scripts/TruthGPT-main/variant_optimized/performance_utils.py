"""
Performance utilities for model optimization and profiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import time
import warnings
from contextlib import contextmanager

class PerformanceProfiler:
    """Detailed performance profiler for model operations."""
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
        
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            self.profiles[name] = {
                'time_ms': (end_time - start_time) * 1000,
                'memory_delta_mb': (end_memory - start_memory) / 1024 / 1024
            }
    
    def get_profile_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all profiles."""
        return self.profiles.copy()

class OptimizedAttention(nn.Module):
    """Memory-efficient attention implementation."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attn_output)

class OptimizedMLP(nn.Module):
    """Memory-efficient MLP with optional activation checkpointing."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        intermediate = self.dropout(intermediate)
        return self.down_proj(intermediate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            try:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(self._forward_impl, x, use_reentrant=False)
            except (ImportError, AttributeError):
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)

class MemoryEfficientEmbedding(nn.Module):
    """Memory-efficient embedding with optional weight sharing."""
    
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))
        self.padding_idx = padding_idx
        
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight, self.padding_idx)

class GradientCheckpointingWrapper(nn.Module):
    """Wrapper for applying gradient checkpointing to any module."""
    
    def __init__(self, module: nn.Module, use_checkpointing: bool = True):
        super().__init__()
        self.module = module
        self.use_checkpointing = use_checkpointing
    
    def forward(self, *args, **kwargs):
        if self.use_checkpointing and self.training:
            try:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(self.module, *args, use_reentrant=False, **kwargs)
            except (ImportError, AttributeError):
                return self.module(*args, **kwargs)
        else:
            return self.module(*args, **kwargs)

class DynamicBatchProcessor:
    """Dynamic batch processing for optimal GPU utilization."""
    
    def __init__(self, max_batch_size: int = 32, max_memory_mb: float = 8000):
        self.max_batch_size = max_batch_size
        self.max_memory_mb = max_memory_mb
        self.optimal_batch_size = 1
        
    def find_optimal_batch_size(self, model: nn.Module, input_generator, start_size: int = 1) -> int:
        """Find optimal batch size for the given model and input."""
        model.eval()
        batch_size = start_size
        
        while batch_size <= self.max_batch_size:
            try:
                with torch.no_grad():
                    input_data = input_generator(batch_size)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        start_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    
                    if isinstance(input_data, dict):
                        _ = model(**input_data)
                    elif isinstance(input_data, (list, tuple)):
                        _ = model(*input_data)
                    else:
                        _ = model(input_data)
                    
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        if current_memory - start_memory > self.max_memory_mb:
                            break
                    
                    self.optimal_batch_size = batch_size
                    batch_size *= 2
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise e
        
        return self.optimal_batch_size

class ModelOptimizer:
    """Utility class for applying various optimizations to models."""
    
    @staticmethod
    def apply_weight_sharing(model: nn.Module, share_embeddings: bool = True) -> nn.Module:
        """Apply weight sharing optimizations."""
        if share_embeddings:
            embedding_layers = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Embedding, MemoryEfficientEmbedding)):
                    embedding_layers.append((name, module))
            
            if len(embedding_layers) > 1:
                warnings.warn(f"Found {len(embedding_layers)} embedding layers, consider weight sharing")
        
        return model
    
    @staticmethod
    def apply_gradient_checkpointing(model: nn.Module, layer_types: Optional[List[type]] = None) -> nn.Module:
        """Apply gradient checkpointing to specified layer types."""
        if layer_types is None:
            layer_types = [nn.TransformerEncoderLayer, nn.TransformerDecoderLayer]
        
        for name, module in model.named_children():
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                setattr(model, name, GradientCheckpointingWrapper(module))
            else:
                ModelOptimizer.apply_gradient_checkpointing(module, layer_types)
        
        return model
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """Apply inference-specific optimizations."""
        model.eval()
        
        for module in model.modules():
            if hasattr(module, 'use_checkpointing'):
                module.use_checkpointing = False
        
        if hasattr(torch, 'jit') and hasattr(torch.jit, 'optimize_for_inference'):
            try:
                model = torch.jit.optimize_for_inference(model)
            except Exception as e:
                warnings.warn(f"Failed to apply JIT optimization: {e}")
        
        return model
    
    @staticmethod
    def apply_quantization(model: nn.Module, quantization_type: str = "dynamic") -> nn.Module:
        """Apply quantization to the model."""
        if quantization_type == "dynamic":
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                warnings.warn(f"Failed to apply dynamic quantization: {e}")
        
        return model

def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Estimate FLOPs for a model given input shape."""
    total_flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features
            if hasattr(input[0], 'shape') and len(input[0].shape) > 2:
                batch_size = input[0].shape[0]
                seq_len = input[0].shape[1] if len(input[0].shape) > 2 else 1
                total_flops *= batch_size * seq_len
        
        elif isinstance(module, nn.Conv2d):
            output_dims = output.shape[2] * output.shape[3]
            kernel_dims = module.kernel_size[0] * module.kernel_size[1]
            total_flops += output_dims * kernel_dims * module.in_channels * module.out_channels
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(flop_count_hook))
    
    try:
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            model(dummy_input)
    finally:
        for hook in hooks:
            hook.remove()
    
    return total_flops
