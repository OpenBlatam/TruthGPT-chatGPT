"""
Ultra-fast modular transformer model
Following deep learning best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import math

from .attention import MultiHeadAttention, AttentionConfig
from .feedforward import FeedForward, FeedForwardConfig
from .normalization import LayerNorm, NormalizationConfig


@dataclass
class TransformerConfig:
    """Transformer configuration"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    use_flash_attention: bool = True


class TransformerBlock(nn.Module):
    """Ultra-fast transformer block"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Attention
        attention_config = AttentionConfig(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention
        )
        self.attention = MultiHeadAttention(attention_config)
        
        # Feedforward
        feedforward_config = FeedForwardConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            activation=config.hidden_act
        )
        self.feedforward = FeedForward(feedforward_config)
        
        # Layer normalization
        norm_config = NormalizationConfig(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.attention_norm = LayerNorm(norm_config)
        self.ffn_norm = LayerNorm(norm_config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Self-attention
        attn_output = self.attention(x, attention_mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = self.attention_norm(x)
        
        # Feedforward
        ffn_output = self.feedforward(x)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output
        x = self.ffn_norm(x)
        
        return x


class TransformerModel(nn.Module):
    """Ultra-fast transformer model"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = LayerNorm(NormalizationConfig(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps
        ))
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': [hidden_states],  # Simplified for speed
            'attentions': None  # Simplified for speed
        }
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_p: float = 0.9,
                 do_sample: bool = True) -> torch.Tensor:
        """Generate text"""
        self.eval()
        
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length):
                # Get model outputs
                outputs = self.forward(generated)
                logits = outputs['last_hidden_state'][:, -1, :] / temperature
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated



