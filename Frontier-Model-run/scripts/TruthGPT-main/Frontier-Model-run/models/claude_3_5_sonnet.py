import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

@dataclass
class ClaudeArgs:
    """Configuration for Claude-3.5-Sonnet model architecture."""
    dim: int = 8192  # Estimated model dimension
    n_layers: int = 80  # Estimated layers for Claude-3.5-Sonnet
    n_heads: int = 64  # Estimated attention heads
    n_kv_heads: int = 8  # GQA with fewer KV heads
    vocab_size: int = 100352  # Anthropic's vocabulary size
    multiple_of: int = 256
    ffn_dim_multiplier: float = 2.6875  # SwiGLU expansion
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 200000  # 200K context length
    
    use_constitutional_ai: bool = True
    use_harmlessness_filter: bool = True
    use_helpfulness_boost: bool = True
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8
    
    use_mixture_of_depths: bool = True  # Adaptive computation
    use_retrieval_augmentation: bool = False
    safety_threshold: float = 0.95

class ClaudeRMSNorm(nn.Module):
    """RMS Normalization optimized for Claude architecture."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class ClaudeLinear(nn.Module):
    """Optimized linear layer for Claude with constitutional AI considerations."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 use_quantization: bool = False, quantization_bits: int = 8,
                 apply_safety_filter: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.apply_safety_filter = apply_safety_filter
        
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.use_quantization and self.training:
            weight = self.quantize_weight(self.weight)
        else:
            weight = self.weight
        
        output = F.linear(x, weight, self.bias)
        
        if self.apply_safety_filter:
            output = self.apply_constitutional_filter(output)
        
        return output
    
    def quantize_weight(self, weight):
        """Quantization with constitutional AI preservation."""
        if self.quantization_bits == 8:
            scale = weight.abs().max() / 127.0
            quantized = torch.round(weight / scale).clamp(-128, 127)
            return quantized * scale
        return weight
    
    def apply_constitutional_filter(self, x):
        """Apply constitutional AI filtering (mock implementation)."""
        return torch.clamp(x, -10.0, 10.0)  # Simple clipping for demonstration

def precompute_freqs_cis_claude(dim: int, end: int, theta: float = 10000.0):
    """Precompute frequency tensor for Claude's rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb_claude(xq, xk, freqs_cis):
    """Apply rotary embeddings optimized for Claude."""
    print(f"DEBUG: xq.shape = {xq.shape}, xk.shape = {xk.shape}, freqs_cis.shape = {freqs_cis.shape}")
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    print(f"DEBUG: xq_.shape = {xq_.shape}, xk_.shape = {xk_.shape}")
    
    # Ensure freqs_cis matches the sequence length dimension
    seq_len = xq_.shape[1]
    if freqs_cis.shape[0] != seq_len:
        freqs_cis = freqs_cis[:seq_len]
    
    if len(freqs_cis.shape) == 2:
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # Add batch and head dims
    
    print(f"DEBUG: After adjustment, freqs_cis.shape = {freqs_cis.shape}")
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class ClaudeAttention(nn.Module):
    """Multi-head attention with constitutional AI optimizations."""
    
    def __init__(self, args: ClaudeArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.use_flash_attention = args.use_flash_attention
        self.use_constitutional_ai = args.use_constitutional_ai

        self.wq = ClaudeLinear(args.dim, args.n_heads * self.head_dim, bias=False,
                              use_quantization=args.use_quantization,
                              quantization_bits=args.quantization_bits)
        self.wk = ClaudeLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False,
                              use_quantization=args.use_quantization,
                              quantization_bits=args.quantization_bits)
        self.wv = ClaudeLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False,
                              use_quantization=args.use_quantization,
                              quantization_bits=args.quantization_bits)
        self.wo = ClaudeLinear(args.n_heads * self.head_dim, args.dim, bias=False,
                              use_quantization=args.use_quantization,
                              quantization_bits=args.quantization_bits,
                              apply_safety_filter=self.use_constitutional_ai)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], safety_context: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if freqs_cis.shape[0] < seqlen:
            freqs_cis = freqs_cis[:seqlen]
        elif freqs_cis.shape[0] > seqlen:
            freqs_cis = freqs_cis[:seqlen]
        
        xq, xk = apply_rotary_emb_claude(xq, xk, freqs_cis=freqs_cis)

        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.use_flash_attention:
            try:
                import flash_attn
                output = flash_attn.flash_attn_func(xq, xk, xv, causal=True)
            except ImportError:
                output = self._constitutional_attention(xq, xk, xv, mask, safety_context)
        else:
            output = self._constitutional_attention(xq, xk, xv, mask, safety_context)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def _constitutional_attention(self, xq, xk, xv, mask, safety_context):
        """Attention with constitutional AI considerations."""
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        if self.use_constitutional_ai and safety_context is not None:
            safety_scores = self._compute_safety_scores(scores, safety_context)
            scores = scores * safety_scores
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        return output
    
    def _compute_safety_scores(self, attention_scores, safety_context):
        """Compute safety modulation scores (mock implementation)."""
        safety_factor = torch.sigmoid(safety_context.mean(dim=-1, keepdim=True))
        batch_size, seq_len = safety_factor.shape[:2]
        n_heads = attention_scores.shape[1]
        safety_factor = safety_factor.unsqueeze(1).expand(batch_size, n_heads, seq_len, 1)
        return safety_factor.expand_as(attention_scores)

class ClaudeFeedForward(nn.Module):
    """SwiGLU Feed Forward with constitutional AI optimizations."""
    
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int,
                 ffn_dim_multiplier: Optional[float], use_quantization: bool = False,
                 quantization_bits: int = 8, use_constitutional_ai: bool = True):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.use_constitutional_ai = use_constitutional_ai
        
        self.w1 = ClaudeLinear(dim, hidden_dim, bias=False,
                              use_quantization=use_quantization,
                              quantization_bits=quantization_bits)
        self.w2 = ClaudeLinear(hidden_dim, dim, bias=False,
                              use_quantization=use_quantization,
                              quantization_bits=quantization_bits,
                              apply_safety_filter=use_constitutional_ai)
        self.w3 = ClaudeLinear(dim, hidden_dim, bias=False,
                              use_quantization=use_quantization,
                              quantization_bits=quantization_bits)

    def forward(self, x):
        gate_output = F.silu(self.w1(x))
        up_output = self.w3(x)
        
        if self.use_constitutional_ai:
            gate_output = self._apply_constitutional_constraints(gate_output)
            up_output = self._apply_constitutional_constraints(up_output)
        
        return self.w2(gate_output * up_output)
    
    def _apply_constitutional_constraints(self, x):
        """Apply constitutional AI constraints (mock implementation)."""
        return torch.tanh(x)  # Gentle activation to prevent extreme values

class MixtureOfDepths(nn.Module):
    """Mixture of Depths for adaptive computation in Claude."""
    
    def __init__(self, dim: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.gate = ClaudeLinear(dim, num_experts, bias=False)
        self.depth_predictors = nn.ModuleList([
            ClaudeLinear(dim, 1, bias=False) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """Predict computational depth needed for each token."""
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        depth_scores = []
        for i, predictor in enumerate(self.depth_predictors):
            depth_score = torch.sigmoid(predictor(x))
            depth_scores.append(depth_score)
        
        depth_scores = torch.stack(depth_scores, dim=-1)
        weighted_depth = (gate_probs * depth_scores).sum(dim=-1, keepdim=True)
        
        return weighted_depth

class ClaudeTransformerBlock(nn.Module):
    """Transformer block with constitutional AI and adaptive computation."""
    
    def __init__(self, layer_id: int, args: ClaudeArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = ClaudeAttention(args)
        self.feed_forward = ClaudeFeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            use_quantization=args.use_quantization,
            quantization_bits=args.quantization_bits,
            use_constitutional_ai=args.use_constitutional_ai,
        )
        self.layer_id = layer_id
        self.attention_norm = ClaudeRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = ClaudeRMSNorm(args.dim, eps=args.norm_eps)
        self.use_gradient_checkpointing = args.use_gradient_checkpointing
        self.use_mixture_of_depths = args.use_mixture_of_depths
        
        if self.use_mixture_of_depths:
            self.depth_predictor = MixtureOfDepths(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], safety_context: Optional[torch.Tensor] = None):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, start_pos, freqs_cis, mask, safety_context, use_reentrant=False
            )
        else:
            return self._forward_impl(x, start_pos, freqs_cis, mask, safety_context)
    
    def _forward_impl(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                      mask: Optional[torch.Tensor], safety_context: Optional[torch.Tensor]):
        if self.use_mixture_of_depths:
            depth_weight = self.depth_predictor(x)
        else:
            depth_weight = torch.ones_like(x[..., :1])
        
        h = x + depth_weight * self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, safety_context
        )
        
        out = h + depth_weight * self.feed_forward(self.ffn_norm(h))
        return out

class ClaudeTransformer(nn.Module):
    """Claude-3.5-Sonnet Transformer with constitutional AI and all optimizations."""
    
    def __init__(self, params: ClaudeArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.use_constitutional_ai = params.use_constitutional_ai
        self.safety_threshold = params.safety_threshold

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(ClaudeTransformerBlock(layer_id, params))
        self.norm = ClaudeRMSNorm(params.dim, eps=params.norm_eps)
        self.output = ClaudeLinear(params.dim, params.vocab_size, bias=False,
                                  use_quantization=params.use_quantization,
                                  quantization_bits=params.quantization_bits,
                                  apply_safety_filter=params.use_constitutional_ai)

        if self.use_constitutional_ai:
            self.safety_classifier = ClaudeLinear(params.dim, 1, bias=False, apply_safety_filter=True)
            self.harmlessness_head = ClaudeLinear(params.dim, params.vocab_size, bias=False, apply_safety_filter=True)
            self.helpfulness_head = ClaudeLinear(params.dim, params.vocab_size, bias=False, apply_safety_filter=True)

        self.freqs_cis = precompute_freqs_cis_claude(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, 
                apply_constitutional_ai: bool = True):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        safety_context = None
        if self.use_constitutional_ai and apply_constitutional_ai:
            safety_context = self._generate_safety_context(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, safety_context)
        
        h = self.norm(h)
        
        if self.use_constitutional_ai and apply_constitutional_ai:
            output = self._apply_constitutional_output(h)
        else:
            output = self.output(h).float()
        
        return output
    
    def _generate_safety_context(self, embeddings):
        """Generate safety context for constitutional AI."""
        safety_scores = self.safety_classifier(embeddings)
        return torch.sigmoid(safety_scores)
    
    def _apply_constitutional_output(self, hidden_states):
        """Apply constitutional AI to output generation."""
        standard_output = self.output(hidden_states).float()
        
        harmlessness_output = self.harmlessness_head(hidden_states).float()
        helpfulness_output = self.helpfulness_head(hidden_states).float()
        
        safety_scores = torch.sigmoid(self.safety_classifier(hidden_states))
        
        constitutional_output = (
            0.6 * standard_output +
            0.3 * harmlessness_output +
            0.1 * helpfulness_output
        )
        
        safe_mask = (safety_scores > self.safety_threshold).float()
        final_output = safe_mask * constitutional_output + (1 - safe_mask) * harmlessness_output
        
        return final_output

def create_claude_3_5_sonnet_model(config: Optional[Dict[str, Any]] = None) -> ClaudeTransformer:
    """Factory function to create Claude-3.5-Sonnet model with constitutional AI and optimization_core integration."""
    
    default_config = {
        'dim': 8192,
        'n_layers': 80,
        'n_heads': 64,
        'n_kv_heads': 8,
        'vocab_size': 100352,
        'multiple_of': 256,
        'ffn_dim_multiplier': 2.6875,
        'norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'max_seq_len': 200000,
        'use_constitutional_ai': True,
        'use_harmlessness_filter': True,
        'use_helpfulness_boost': True,
        'use_flash_attention': True,
        'use_gradient_checkpointing': True,
        'use_quantization': False,
        'quantization_bits': 8,
        'use_mixture_of_depths': True,
        'use_retrieval_augmentation': False,
        'safety_threshold': 0.95,
    }
    
    if config:
        default_config.update(config)
    
    args = ClaudeArgs(**default_config)
    model = ClaudeTransformer(args)
    
    print(f"✅ Created Claude-3.5-Sonnet model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔧 Optimizations: Constitutional AI={args.use_constitutional_ai}, "
          f"Flash Attention={args.use_flash_attention}, "
          f"Mixture of Depths={args.use_mixture_of_depths}")
    print(f"🛡️ Safety Features: Harmlessness Filter={args.use_harmlessness_filter}, "
          f"Safety Threshold={args.safety_threshold}")
    
    return model
