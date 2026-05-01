"""
Advanced Transformer Model Module
Highly modular transformer implementation with cutting-edge features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import logging
from enum import Enum
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)

class AttentionType(Enum):
    """Attention mechanism types"""
    STANDARD = "standard"
    FLASH = "flash"
    MEMORY_EFFICIENT = "memory_efficient"
    SPARSE = "sparse"
    LOCAL = "local"
    GLOBAL = "global"

class PositionalEncodingType(Enum):
    """Positional encoding types"""
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROTARY = "rotary"
    ALIBI = "alibi"
    RELATIVE = "relative"

class TransformerConfig(BaseModel):
    """Transformer configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    attention_type: AttentionType = AttentionType.STANDARD
    positional_encoding: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL
    max_seq_length: int = 512
    use_bias: bool = True
    layer_norm_eps: float = 1e-6
    use_pre_norm: bool = False
    use_post_norm: bool = True
    use_residual: bool = True
    use_ffn: bool = True
    use_attention: bool = True
    use_embedding: bool = True
    use_output: bool = True
    vocab_size: int = 30000
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    unk_token_id: int = 3

    @model_validator(mode='after')
    def validate_dimensions(self) -> 'TransformerConfig':
        """Ensure d_model is divisible by n_heads."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        seq_length = x.size(0)
        return x + self.pe[:seq_length, :]

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(max_seq_length, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to input"""
        seq_length = x.size(0)
        positions = torch.arange(seq_length, device=x.device)
        pos_embeddings = self.embedding(positions).unsqueeze(1)
        return x + pos_embeddings

class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE)"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create frequency matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional encoding"""
        t = torch.arange(seq_length, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = torch.cos(emb)
        sin = torch.sin(emb)
        
        return cos, sin

class AlibiPositionalEncoding(nn.Module):
    """ALiBi (Attention with Linear Biases) positional encoding"""
    
    def __init__(self, n_heads: int, max_seq_length: int = 5000):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        
        # Create ALiBi slopes
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """Get ALiBi slopes"""
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2])
        
        return torch.tensor(slopes)
    
    def forward(self, attention_scores: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Apply ALiBi biases to attention scores"""
        # Create distance matrix
        positions = torch.arange(seq_length, device=attention_scores.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Apply slopes to distance
        slopes = self.slopes.unsqueeze(0).unsqueeze(0)
        biases = slopes * distance.unsqueeze(0)
        
        return attention_scores + biases

class MultiHeadAttention(nn.Module):
    """Multi-head attention with various optimizations"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 attention_type: AttentionType = AttentionType.STANDARD,
                 use_bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.attention_type = attention_type
        self.use_bias = use_bias
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Dropout
        self.dropout_layer = Dropout(dropout)
        
        # Attention type specific components
        if attention_type == AttentionType.FLASH:
            self._setup_flash_attention()
        elif attention_type == AttentionType.MEMORY_EFFICIENT:
            self._setup_memory_efficient_attention()
        elif attention_type == AttentionType.SPARSE:
            self._setup_sparse_attention()
    
    def _setup_flash_attention(self):
        """Setup flash attention"""
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
        except ImportError:
            logger.warning("Flash attention not available, falling back to standard attention")
            self.attention_type = AttentionType.STANDARD
    
    def _setup_memory_efficient_attention(self):
        """Setup memory efficient attention"""
        self.scale = 1.0 / math.sqrt(self.d_k)
    
    def _setup_sparse_attention(self):
        """Setup sparse attention"""
        self.sparse_pattern = self._create_sparse_pattern()
    
    def _create_sparse_pattern(self) -> torch.Tensor:
        """Create sparse attention pattern"""
        # Create a simple sparse pattern (can be customized)
        pattern = torch.ones(self.n_heads, self.d_k, self.d_k)
        # Add sparsity (example: keep only diagonal and nearby elements)
        for i in range(self.d_k):
            for j in range(self.d_k):
                if abs(i - j) > 2:  # Keep only nearby positions
                    pattern[:, i, j] = 0
        return pattern
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        batch_size, seq_length = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        if self.attention_type == AttentionType.FLASH:
            attn_output = self._flash_attention(Q, K, V, mask)
        elif self.attention_type == AttentionType.MEMORY_EFFICIENT:
            attn_output = self._memory_efficient_attention(Q, K, V, mask)
        elif self.attention_type == AttentionType.SPARSE:
            attn_output = self._sparse_attention(Q, K, V, mask)
        else:
            attn_output = self._standard_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        
        return output, None  # Return attention weights if needed
    
    def _flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flash attention implementation"""
        if hasattr(self, 'flash_attn_func'):
            return self.flash_attn_func(Q, K, V, dropout_p=self.dropout if self.training else 0.0)
        else:
            return self._standard_attention(Q, K, V, mask)
    
    def _memory_efficient_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory efficient attention implementation"""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output
    
    def _sparse_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sparse attention implementation"""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sparse pattern
        if hasattr(self, 'sparse_pattern'):
            scores = scores * self.sparse_pattern.to(scores.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output
    
    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention implementation"""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output

class FeedForwardNetwork(nn.Module):
    """Feed-forward network with various activations"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "relu", use_bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.use_bias = use_bias
        
        # Linear layers
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)
        
        # Dropout
        self.dropout_layer = Dropout(dropout)
        
        # Activation function
        self.activation_fn = self._get_activation_fn(activation)
    
    def _get_activation_fn(self, activation: str):
        """Get activation function"""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "swish":
            return F.silu
        elif activation == "mish":
            return self._mish
        elif activation == "glu":
            return self._glu
        else:
            return F.relu
    
    def _mish(self, x: torch.Tensor) -> torch.Tensor:
        """Mish activation function"""
        return x * torch.tanh(F.softplus(x))
    
    def _glu(self, x: torch.Tensor) -> torch.Tensor:
        """Gated Linear Unit"""
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with various configurations"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.dropout = config.dropout
        self.use_pre_norm = config.use_pre_norm
        self.use_post_norm = config.use_post_norm
        self.use_residual = config.use_residual
        self.use_ffn = config.use_ffn
        self.use_attention = config.use_attention
        
        # Attention
        if self.use_attention:
            self.attention = MultiHeadAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout,
                attention_type=config.attention_type,
                use_bias=config.use_bias
            )
        
        # Feed-forward network
        if self.use_ffn:
            self.ffn = FeedForwardNetwork(
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.dropout,
                activation=config.activation,
                use_bias=config.use_bias
            )
        
        # Layer normalization
        if self.use_pre_norm:
            self.norm1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
            self.norm2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        if self.use_post_norm:
            self.norm1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
            self.norm2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Attention
        if self.use_attention:
            if self.use_pre_norm:
                x_norm = self.norm1(x)
                attn_output, _ = self.attention(x_norm, x_norm, x_norm, mask)
                if self.use_residual:
                    x = x + self.dropout(attn_output)
                else:
                    x = self.dropout(attn_output)
            else:
                attn_output, _ = self.attention(x, x, x, mask)
                if self.use_residual:
                    x = x + self.dropout(attn_output)
                else:
                    x = self.dropout(attn_output)
                
                if self.use_post_norm:
                    x = self.norm1(x)
        
        # Feed-forward network
        if self.use_ffn:
            if self.use_pre_norm:
                x_norm = self.norm2(x)
                ffn_output = self.ffn(x_norm)
                if self.use_residual:
                    x = x + self.dropout(ffn_output)
                else:
                    x = self.dropout(ffn_output)
            else:
                ffn_output = self.ffn(x)
                if self.use_residual:
                    x = x + self.dropout(ffn_output)
                else:
                    x = self.dropout(ffn_output)
                
                if self.use_post_norm:
                    x = self.norm2(x)
        
        return x

class TransformerModel(nn.Module):
    """Advanced Transformer model with modular components"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        
        # Embedding
        if config.use_embedding:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
            self.embedding_dropout = Dropout(config.dropout)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output layer
        if config.use_output:
            self.output_layer = nn.Linear(config.d_model, config.vocab_size, bias=config.use_bias)
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, config: TransformerConfig):
        """Create positional encoding"""
        if config.positional_encoding == PositionalEncodingType.SINUSOIDAL:
            return SinusoidalPositionalEncoding(config.d_model, config.max_seq_length)
        elif config.positional_encoding == PositionalEncodingType.LEARNED:
            return LearnedPositionalEncoding(config.d_model, config.max_seq_length)
        elif config.positional_encoding == PositionalEncodingType.ROTARY:
            return RotaryPositionalEncoding(config.d_model, config.max_seq_length)
        elif config.positional_encoding == PositionalEncodingType.ALIBI:
            return AlibiPositionalEncoding(config.n_heads, config.max_seq_length)
        else:
            return None
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size, seq_length = input_ids.size()
        
        # Embedding
        if self.config.use_embedding:
            x = self.embedding(input_ids)
            x = self.embedding_dropout(x)
        else:
            x = input_ids
        
        # Positional encoding
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, RotaryPositionalEncoding):
                cos, sin = self.pos_encoding(x, seq_length)
                # Apply rotary positional encoding to attention
                x = self._apply_rotary_encoding(x, cos, sin)
            else:
                x = self.pos_encoding(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Output
        outputs = {}
        if self.config.use_output:
            logits = self.output_layer(x)
            outputs["logits"] = logits
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                outputs["loss"] = loss
        
        return outputs
    
    def _apply_rotary_encoding(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional encoding"""
        # This is a simplified implementation
        # In practice, you would apply RoPE to the query and key vectors in attention
        return x
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                 do_sample: bool = True) -> torch.Tensor:
        """Generate text"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                logits = outputs["logits"][:, -1, :] / temperature
                
                if do_sample:
                    # Top-k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        logits = torch.full_like(logits, -float('inf'))
                        logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = -float('inf')
                    
                    # Sample from distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# Factory functions
def create_transformer_model(config: TransformerConfig) -> TransformerModel:
    """Create transformer model"""
    return TransformerModel(config)

def create_transformer_config(**kwargs) -> TransformerConfig:
    """Create transformer configuration"""
    return TransformerConfig(**kwargs)

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = create_transformer_config(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        vocab_size=30000,
        attention_type=AttentionType.FLASH,
        positional_encoding=PositionalEncodingType.ROTARY
    )
    
    # Create model
    model = create_transformer_model(config)
    
    # Example forward pass
    batch_size, seq_length = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(input_ids, attention_mask)
    print(f"Model outputs: {list(outputs.keys())}")
    print(f"Logits shape: {outputs['logits'].shape}")
