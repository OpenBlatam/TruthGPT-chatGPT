"""
Enhanced text generation with advanced AI capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterator
import math

@dataclass
class EnhancedTextGeneratorArgs:
    """Configuration for enhanced text generator."""
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50000
    max_sequence_length: int = 2048
    dropout: float = 0.1
    
    generation_modes: Optional[List[str]] = None
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    enable_streaming: bool = True
    enable_conditional_generation: bool = True
    enable_brand_conditioning: bool = True
    enable_viral_conditioning: bool = True
    
    def __post_init__(self):
        if self.generation_modes is None:
            self.generation_modes = ['creative', 'formal', 'casual', 'technical']

class AdvancedAttentionLayer(nn.Module):
    """Advanced attention with optimizations for text generation."""
    
    def __init__(self, args: EnhancedTextGeneratorArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = args.hidden_size // args.num_heads
        
        self.qkv_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=False)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q, k = self.rotary_emb(q, k)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value[0], past_key_value[1]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        present_key_value = (k, v)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.o_proj(attn_output), (k, v)

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding for improved sequence modeling."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        
        q_embed = (q * cos_emb) + (self._rotate_half(q) * sin_emb)
        k_embed = (k * cos_emb) + (self._rotate_half(k) * sin_emb)
        
        return q_embed, k_embed
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class ConditionalGenerationLayer(nn.Module):
    """Layer for conditional text generation based on context."""
    
    def __init__(self, args: EnhancedTextGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.brand_conditioning = nn.Linear(768, args.hidden_size) if args.enable_brand_conditioning else None
        self.viral_conditioning = nn.Linear(512, args.hidden_size) if args.enable_viral_conditioning else None
        self.mode_embedding = nn.Embedding(len(args.generation_modes), args.hidden_size)
        
        self.conditioning_fusion = nn.Sequential(
            nn.Linear(args.hidden_size * 3, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
    def forward(self, hidden_states: torch.Tensor, brand_profile: Optional[torch.Tensor] = None,
                viral_features: Optional[torch.Tensor] = None, generation_mode: Optional[torch.Tensor] = None) -> torch.Tensor:
        conditioning_vectors = [hidden_states]
        
        if self.brand_conditioning is not None and brand_profile is not None:
            brand_cond = self.brand_conditioning(brand_profile)
            if brand_cond.dim() == 2:
                brand_cond = brand_cond.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            conditioning_vectors.append(brand_cond)
        else:
            conditioning_vectors.append(torch.zeros_like(hidden_states))
        
        if self.viral_conditioning is not None and viral_features is not None:
            viral_cond = self.viral_conditioning(viral_features)
            if viral_cond.dim() == 2:
                viral_cond = viral_cond.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            conditioning_vectors.append(viral_cond)
        else:
            conditioning_vectors.append(torch.zeros_like(hidden_states))
        
        if generation_mode is not None:
            mode_emb = self.mode_embedding(generation_mode)
            if mode_emb.dim() == 2:
                mode_emb = mode_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            conditioning_vectors[-1] = conditioning_vectors[-1] + mode_emb
        
        combined = torch.cat(conditioning_vectors, dim=-1)
        return self.conditioning_fusion(combined)

class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with conditional generation."""
    
    def __init__(self, args: EnhancedTextGeneratorArgs):
        super().__init__()
        self.attention = AdvancedAttentionLayer(args)
        self.conditional_layer = ConditionalGenerationLayer(args)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size * 4, args.hidden_size),
            nn.Dropout(args.dropout)
        )
        
        self.input_layernorm = nn.LayerNorm(args.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(args.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None, **conditioning_kwargs) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, present_key_value = self.attention(
            hidden_states, attention_mask, past_key_value
        )
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.conditional_layer(hidden_states, **conditioning_kwargs)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states, present_key_value

class EnhancedTextGenerator(nn.Module):
    """Enhanced text generator with advanced AI capabilities."""
    
    def __init__(self, args: EnhancedTextGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_positions = nn.Embedding(args.max_sequence_length, args.hidden_size)
        
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(args) for _ in range(args.num_layers)
        ])
        
        self.norm = nn.LayerNorm(args.hidden_size)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor]]] = None, **conditioning_kwargs) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        if past_key_values is None:
            past_length = 0
            past_key_values = [(None, None)] * len(self.layers)
        else:
            past_length = past_key_values[0][0].shape[-2]
        
        position_ids = torch.arange(past_length, seq_len + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
        hidden_states = self.dropout(hidden_states)
        
        present_key_values = []
        for i, layer in enumerate(self.layers):
            hidden_states, present_key_value = layer(
                hidden_states, attention_mask, past_key_values[i], **conditioning_kwargs
            )
            present_key_values.append(present_key_value)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'past_key_values': present_key_values,
            'hidden_states': hidden_states
        }
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                temperature: Optional[float] = None, top_k: Optional[int] = None, top_p: Optional[float] = None,
                **conditioning_kwargs) -> torch.Tensor:
        """Generate text with advanced sampling strategies."""
        if temperature is None:
            temperature = self.args.temperature
        if top_k is None:
            top_k = self.args.top_k
        if top_p is None:
            top_p = self.args.top_p
        
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.forward(
                    generated[:, -1:] if past_key_values is not None else generated,
                    past_key_values=past_key_values,
                    **conditioning_kwargs
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                past_key_values = outputs['past_key_values']
                
                filtered_logits = self._apply_sampling_filters(logits, top_k, top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                if next_token.item() == self.args.vocab_size - 1:  # EOS token
                    break
        
        return generated
    
    def _apply_sampling_filters(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Apply top-k and nucleus sampling filters."""
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return logits

class StreamingTextGenerator:
    """Streaming text generator for real-time applications."""
    
    def __init__(self, model: EnhancedTextGenerator, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_stream(self, prompt: str, max_length: int = 100, **kwargs) -> Iterator[str]:
        """Generate text in streaming fashion."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for streaming generation")
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        past_key_values = None
        generated_text = prompt
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids[:, -1:] if past_key_values is not None else input_ids,
                    past_key_values=past_key_values,
                    **kwargs
                )
                
                logits = outputs['logits'][:, -1, :]
                past_key_values = outputs['past_key_values']
                
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                next_token = self.tokenizer.decode(next_token_id[0])
                
                generated_text += next_token
                input_ids = next_token_id
                
                yield next_token
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

class ConditionalTextGenerator:
    """Wrapper for conditional text generation with brand and viral context."""
    
    def __init__(self, model: EnhancedTextGenerator):
        self.model = model
        
    def generate_brand_consistent_text(self, prompt: str, brand_profile: torch.Tensor,
                                     generation_mode: str = 'formal', **kwargs) -> str:
        """Generate text consistent with brand profile."""
        mode_id = torch.tensor([self.model.args.generation_modes.index(generation_mode)])
        
        input_ids = torch.tensor([[1, 2, 3]])  # Mock tokenization
        
        generated = self.model.generate(
            input_ids,
            brand_profile=brand_profile,
            generation_mode=mode_id,
            **kwargs
        )
        
        return "Generated brand-consistent text"  # Mock output
    
    def generate_viral_content(self, viral_features: torch.Tensor, content_type: str = 'social_post', **kwargs) -> str:
        """Generate viral content based on viral features."""
        input_ids = torch.tensor([[1, 2, 3]])  # Mock tokenization
        
        generated = self.model.generate(
            input_ids,
            viral_features=viral_features,
            **kwargs
        )
        
        return "Generated viral content"  # Mock output

def create_enhanced_text_generator(config: Dict[str, Any]) -> EnhancedTextGenerator:
    """Create enhanced text generator from configuration."""
    args = EnhancedTextGeneratorArgs(
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_layers', 12),
        num_heads=config.get('num_heads', 12),
        vocab_size=config.get('vocab_size', 50000),
        max_sequence_length=config.get('max_sequence_length', 2048),
        dropout=config.get('dropout', 0.1),
        generation_modes=config.get('generation_modes', ['creative', 'formal', 'casual', 'technical']),
        temperature=config.get('temperature', 0.8),
        top_k=config.get('top_k', 50),
        top_p=config.get('top_p', 0.9),
        repetition_penalty=config.get('repetition_penalty', 1.1),
        enable_streaming=config.get('enable_streaming', True),
        enable_conditional_generation=config.get('enable_conditional_generation', True),
        enable_brand_conditioning=config.get('enable_brand_conditioning', True),
        enable_viral_conditioning=config.get('enable_viral_conditioning', True)
    )
    
    return EnhancedTextGenerator(args)

def create_text_generator(config: Dict[str, Any]) -> EnhancedTextGenerator:
    """Create text generator from configuration (alias for create_enhanced_text_generator)."""
    return create_enhanced_text_generator(config)
