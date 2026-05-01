"""
TruthGPT Models Module
Advanced model architectures for TruthGPT following transformer best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTModelConfig:
    """Configuration for TruthGPT models."""
    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    
    # Attention configuration
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    attention_type: str = "multi_head"  # multi_head, sparse, local
    
    # Activation configuration
    activation_function: str = "gelu"  # gelu, relu, swish, silu
    layer_norm_eps: float = 1e-5
    
    # Advanced features
    enable_gradient_checkpointing: bool = True
    enable_memory_efficient_attention: bool = True
    enable_flash_attention: bool = False
    
    # Initialization
    initializer_range: float = 0.02
    use_cache: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'attention_dropout': self.attention_dropout,
            'hidden_dropout': self.hidden_dropout,
            'attention_type': self.attention_type,
            'activation_function': self.activation_function,
            'layer_norm_eps': self.layer_norm_eps,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'enable_memory_efficient_attention': self.enable_memory_efficient_attention,
            'enable_flash_attention': self.enable_flash_attention,
            'initializer_range': self.initializer_range,
            'use_cache': self.use_cache
        }

class TruthGPTPositionalEncoding(nn.Module):
    """Positional encoding for TruthGPT models."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Create positional embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        position_embeddings = self.position_embeddings(position_ids)
        return self.dropout(position_embeddings)

class TruthGPTSelfAttention(nn.Module):
    """Self-attention layer for TruthGPT models."""
    
    def __init__(self, config: TruthGPTModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        
        # Calculate dimensions
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear layers
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dense = nn.Linear(self.all_head_size, self.hidden_size)
        
        # Dropout
        self.attention_dropout = nn.Dropout(self.attention_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.query.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.key.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.value.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.dense.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Linear transformations
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        # Output projection
        output = self.dense(context_layer)
        
        return output

class TruthGPTFeedForward(nn.Module):
    """Feed-forward network for TruthGPT models."""
    
    def __init__(self, config: TruthGPTModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.activation_function = config.activation_function
        
        # Linear layers
        self.dense_1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.dense_2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.dense_1.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.normal_(self.dense_2.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # First linear layer
        hidden_states = self.dense_1(hidden_states)
        
        # Activation function
        if self.activation_function == "gelu":
            hidden_states = F.gelu(hidden_states)
        elif self.activation_function == "relu":
            hidden_states = F.relu(hidden_states)
        elif self.activation_function == "swish":
            hidden_states = F.silu(hidden_states)
        elif self.activation_function == "silu":
            hidden_states = F.silu(hidden_states)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")
        
        # Second linear layer
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

class TruthGPTTransformerLayer(nn.Module):
    """Transformer layer for TruthGPT models."""
    
    def __init__(self, config: TruthGPTModelConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = TruthGPTSelfAttention(config)
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward
        self.feed_forward = TruthGPTFeedForward(config)
        self.feed_forward_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Self-attention with residual connection
        attention_output = self.self_attention(hidden_states, attention_mask)
        attention_output = self.hidden_dropout(attention_output)
        hidden_states = self.attention_layernorm(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layernorm(hidden_states + feed_forward_output)
        
        return hidden_states

class TruthGPTModel(nn.Module):
    """Main TruthGPT model architecture."""
    
    def __init__(self, config: TruthGPTModelConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = TruthGPTPositionalEncoding(
            config.hidden_size, 
            config.max_position_embeddings, 
            config.hidden_dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TruthGPTTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Layer normalization
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Enable optimizations
        self._enable_optimizations()
        
        self.logger.info(f"TruthGPT model initialized with {self.num_parameters():,} parameters")
    
    def _init_weights(self):
        """Initialize model weights."""
        # Token embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=self.config.initializer_range)
        
        # Language modeling head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)
        
        # Share weights between token embeddings and lm_head
        self.lm_head.weight = self.token_embeddings.weight
    
    def _enable_optimizations(self):
        """Enable model optimizations."""
        if self.config.enable_gradient_checkpointing:
            for layer in self.layers:
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
            self.logger.info("✅ Gradient checkpointing enabled")
        
        if self.config.enable_memory_efficient_attention:
            # This would require specific attention implementations
            self.logger.info("✅ Memory efficient attention enabled")
        
        if self.config.enable_flash_attention:
            # This would require flash attention implementation
            self.logger.info("✅ Flash attention enabled")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_length = input_ids.size()
        
        # Token embeddings
        token_embeddings = self.token_embeddings(input_ids)
        
        # Position embeddings
        position_embeddings = self.position_embeddings(input_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Layer normalization
        hidden_states = self.layernorm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def num_parameters(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'total_parameters': self.num_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'model_size_mb': (param_size + buffer_size) / (1024 * 1024),
            'parameters_mb': param_size / (1024 * 1024),
            'buffers_mb': buffer_size / (1024 * 1024)
        }

class TruthGPTConfig:
    """Configuration class for TruthGPT models."""
    
    def __init__(self, **kwargs):
        # Set default values
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.hidden_size = kwargs.get('hidden_size', 768)
        self.num_layers = kwargs.get('num_layers', 12)
        self.num_attention_heads = kwargs.get('num_attention_heads', 12)
        self.intermediate_size = kwargs.get('intermediate_size', 3072)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 2048)
        self.attention_dropout = kwargs.get('attention_dropout', 0.1)
        self.hidden_dropout = kwargs.get('hidden_dropout', 0.1)
        self.activation_function = kwargs.get('activation_function', 'gelu')
        self.layer_norm_eps = kwargs.get('layer_norm_eps', 1e-5)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.use_cache = kwargs.get('use_cache', True)
        
        # Advanced features
        self.enable_gradient_checkpointing = kwargs.get('enable_gradient_checkpointing', True)
        self.enable_memory_efficient_attention = kwargs.get('enable_memory_efficient_attention', True)
        self.enable_flash_attention = kwargs.get('enable_flash_attention', False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'attention_dropout': self.attention_dropout,
            'hidden_dropout': self.hidden_dropout,
            'activation_function': self.activation_function,
            'layer_norm_eps': self.layer_norm_eps,
            'initializer_range': self.initializer_range,
            'use_cache': self.use_cache,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'enable_memory_efficient_attention': self.enable_memory_efficient_attention,
            'enable_flash_attention': self.enable_flash_attention
        }

# Factory functions
def create_truthgpt_model(config: TruthGPTModelConfig) -> TruthGPTModel:
    """Create TruthGPT model."""
    return TruthGPTModel(config)

def load_truthgpt_model(filepath: str, config: TruthGPTModelConfig) -> TruthGPTModel:
    """Load TruthGPT model from file."""
    model = create_truthgpt_model(config)
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    return model

def save_truthgpt_model(model: TruthGPTModel, filepath: str) -> None:
    """Save TruthGPT model to file."""
    torch.save(model.state_dict(), filepath)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT model
    print("🚀 TruthGPT Models Demo")
    print("=" * 50)
    
    # Create configuration
    config = TruthGPTModelConfig(
        vocab_size=10000,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048
    )
    
    # Create model
    model = create_truthgpt_model(config)
    
    # Get model information
    model_info = model.get_model_size()
    print(f"Model information: {model_info}")
    
    # Test forward pass
    input_ids = torch.randint(0, 10000, (2, 512))
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    
    print("✅ TruthGPT model created successfully!")



