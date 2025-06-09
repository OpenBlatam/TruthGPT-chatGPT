"""
Computational Efficiency Optimizations for TruthGPT Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

class FusedAttention(nn.Module):
    """Fused attention implementation for better efficiency."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        try:
            from optimization_core.cuda_kernels import OptimizedLinear
            self.qkv_proj = OptimizedLinear(hidden_size, hidden_size * 3, bias=False)
            self.out_proj = OptimizedLinear(hidden_size, hidden_size, bias=False)
        except ImportError:
            self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fused QKV projection and attention computation."""
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attn_output)

class BatchOptimizer:
    """Optimizations for batch processing and inference."""
    
    @staticmethod
    def optimize_batch_size(model: nn.Module, max_memory_mb: float = 8000) -> int:
        """Determine optimal batch size based on available memory."""
        optimal_batch_size = 1
        hidden_size = getattr(model, 'hidden_size', 512)
        test_input = torch.randn(1, 512, hidden_size)
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            try:
                test_batch = test_input.repeat(batch_size, 1, 1)
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        _ = model(test_batch)
                    else:
                        break
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
                    if memory_used > max_memory_mb:
                        break
                
                optimal_batch_size = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                break
        
        return optimal_batch_size
    
    @staticmethod
    def profile_inference_time(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
        """Profile inference time with multiple runs for statistical accuracy."""
        import time
        
        model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
        
        return {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }

class ComputationalOptimizer:
    """Main computational optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_fused_attention = config.get('use_fused_attention', True)
        self.enable_kernel_fusion = config.get('enable_kernel_fusion', True)
        self.optimize_batch_size = config.get('optimize_batch_size', True)
        self.use_flash_attention = config.get('use_flash_attention', True)
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply computational optimizations to model."""
        if self.use_fused_attention:
            model = self.replace_attention_layers(model)
        
        if self.enable_kernel_fusion and torch.cuda.is_available():
            model = torch.jit.script(model)
        
        return model
    
    def replace_attention_layers(self, model: nn.Module) -> nn.Module:
        """Replace standard attention with fused attention where possible."""
        for name, module in model.named_children():
            if hasattr(module, 'attention') and hasattr(module.attention, 'hidden_size'):
                hidden_size = module.attention.hidden_size
                num_heads = getattr(module.attention, 'num_heads', 8)
                dropout = getattr(module.attention, 'dropout', 0.1)
                
                fused_attn = FusedAttention(hidden_size, num_heads, dropout)
                setattr(module, 'attention', fused_attn)
            else:
                self.replace_attention_layers(module)
        
        return model

def create_computational_optimizer(config: Dict[str, Any]) -> ComputationalOptimizer:
    """Create computational optimizer from configuration."""
    return ComputationalOptimizer(config)
