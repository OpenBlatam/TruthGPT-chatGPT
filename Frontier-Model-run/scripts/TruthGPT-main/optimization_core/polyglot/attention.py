"""
Unified Attention Interface

Provides Python interface to Rust, C++, and Julia attention implementations.
"""
from typing import Optional, Tuple
import logging
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    from truthgpt_rust import flash_attention_block
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    import _cpp_core as cpp_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

try:
    from julia import TruthGPTCore
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False

def attention(
    q,
    k,
    v,
    backend: Optional[str] = None,
    use_flash: bool = True,
    causal: bool = True,
    **kwargs
):
    """
    Unified attention computation.
    
    Automatically selects best available backend:
    1. C++ (GPU, fastest)
    2. Rust (CPU, fast)
    3. Julia (scientific computing)
    4. PyTorch (fallback)
    """
    if backend is None:
        if CPP_AVAILABLE and TORCH_AVAILABLE and q.is_cuda:
            backend = "cpp"
        elif RUST_AVAILABLE:
            backend = "rust"
        elif JULIA_AVAILABLE:
            backend = "julia"
        else:
            backend = "pytorch"
    
    if backend == "cpp" and CPP_AVAILABLE:
        return _attention_cpp(q, k, v, use_flash, causal, **kwargs)
    elif backend == "rust" and RUST_AVAILABLE:
        return _attention_rust(q, k, v, causal, **kwargs)
    elif backend == "julia" and JULIA_AVAILABLE:
        return _attention_julia(q, k, v, causal, **kwargs)
    else:
        return _attention_pytorch(q, k, v, causal, **kwargs)

def _attention_cpp(q, k, v, use_flash, causal, **kwargs):
    """C++ attention backend."""
    if not CPP_AVAILABLE:
        return _attention_pytorch(q, k, v, causal, **kwargs)
    
    try:
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        q_np = q.detach().cpu().numpy().astype(np.float32)
        k_np = k.detach().cpu().numpy().astype(np.float32)
        v_np = v.detach().cpu().numpy().astype(np.float32)
        
        config = cpp_core.attention.AttentionConfig()
        config.num_heads = num_heads
        config.head_dim = head_dim
        config.use_flash = use_flash
        config.use_causal_mask = causal
        
        if use_flash:
            attn = cpp_core.attention.FlashAttention(config)
        else:
            attn = cpp_core.attention.ScaledDotProductAttention(config)
        
        q_flat = q_np.reshape(-1, seq_len, head_dim)
        k_flat = k_np.reshape(-1, seq_len, head_dim)
        v_flat = v_np.reshape(-1, seq_len, head_dim)
        
        result = attn.forward(q_flat, k_flat, v_flat, batch_size, seq_len)
        
        output = torch.from_numpy(result).to(q.device)
        return output.reshape(batch_size, num_heads, seq_len, head_dim)
    except Exception as e:
        logger.warning(f"C++ attention failed: {e}, falling back to PyTorch")
        return _attention_pytorch(q, k, v, causal, **kwargs)

def _attention_rust(q, k, v, causal, **kwargs):
    """Rust attention backend."""
    if not RUST_AVAILABLE:
        return _attention_pytorch(q, k, v, causal, **kwargs)
    
    try:
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        q_np = q.detach().cpu().numpy().astype(np.float32)
        k_np = k.detach().cpu().numpy().astype(np.float32)
        v_np = v.detach().cpu().numpy().astype(np.float32)
        
        result = flash_attention_block(
            q_np.flatten(),
            k_np.flatten(),
            v_np.flatten(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            causal,
        )
        
        output = torch.from_numpy(result).to(q.device)
        return output.reshape(batch_size, num_heads, seq_len, head_dim)
    except Exception as e:
        logger.warning(f"Rust attention failed: {e}, falling back to PyTorch")
        return _attention_pytorch(q, k, v, causal, **kwargs)

def _attention_julia(q, k, v, causal, **kwargs):
    """Julia attention backend."""
    if not JULIA_AVAILABLE:
        return _attention_pytorch(q, k, v, causal, **kwargs)
    
    try:
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        q_np = q.detach().cpu().numpy().astype(np.float32)
        k_np = k.detach().cpu().numpy().astype(np.float32)
        v_np = v.detach().cpu().numpy().astype(np.float32)
        
        config = TruthGPTCore.AttentionConfig(
            num_heads=num_heads,
            head_dim=head_dim,
            use_flash=True,
            use_causal=causal,
        )
        
        q_jl = TruthGPTCore.to_float32(q_np)
        k_jl = TruthGPTCore.to_float32(k_np)
        v_jl = TruthGPTCore.to_float32(v_np)
        
        result = TruthGPTCore.flash_attention(q_jl, k_jl, v_jl, config)
        
        output = torch.from_numpy(np.array(result)).to(q.device)
        return output.reshape(batch_size, num_heads, seq_len, head_dim)
    except Exception as e:
        logger.warning(f"Julia attention failed: {e}, falling back to PyTorch")
        return _attention_pytorch(q, k, v, causal, **kwargs)

def _attention_pytorch(q, k, v, causal, **kwargs):
    """PyTorch attention fallback."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    scale = (q.shape[-1]) ** -0.5
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        scores = scores + mask
    
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    return output

__all__ = ["attention"]













