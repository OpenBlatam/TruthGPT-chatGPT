"""
Enterprise TruthGPT Model - PyTorch Core
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from .config import AdapterConfig, EnterpriseModelInfo


class EnterpriseTruthGPTModel(nn.Module):
    """Enterprise TruthGPT model core with advanced optimizations."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding = self._create_embedding()
        self.layers = nn.ModuleList([
            self._create_transformer_layer(i)
            for i in range(config.num_layers)
        ])
        self.output = self._create_output_layer()
        
        # Enterprise optimizations
        self._apply_enterprise_optimizations()
    
    def _create_embedding(self) -> nn.Module:
        """Create embedding layer."""
        return nn.Sequential(
            nn.Embedding(
                self.config.vocab_size,
                self.config.hidden_size,
                max_norm=1.0
            ),
            nn.Dropout(self.config.dropout)
        )
    
    def _create_transformer_layer(self, layer_idx: int) -> nn.ModuleDict:
        """Create transformer layer."""
        return nn.ModuleDict({
            'attention': self._create_attention_layer(),
            'feed_forward': self._create_feed_forward_layer(),
            'layer_norm1': nn.LayerNorm(
                self.config.hidden_size,
                eps=self.config.layer_norm_eps
            ),
            'layer_norm2': nn.LayerNorm(
                self.config.hidden_size,
                eps=self.config.layer_norm_eps
            ),
            'dropout1': nn.Dropout(self.config.dropout),
            'dropout2': nn.Dropout(self.config.activation_dropout)
        })
    
    def _create_attention_layer(self) -> nn.Module:
        """Create attention layer with enterprise optimizations."""
        if self.config.use_flash_attention:
            # Use Flash Attention for better performance
            return self._create_flash_attention()
        else:
            return self._create_standard_attention()
    
    def _create_flash_attention(self) -> nn.Module:
        """Create flash attention layer."""
        try:
            from flash_attn import flash_attn_func
            # Simplified mock for structure
            return nn.Module()
        except ImportError:
            self.logger.warning("Flash Attention not available, using standard attention")
            return self._create_standard_attention()
    
    def _create_standard_attention(self) -> nn.Module:
        """Create standard attention layer."""
        return nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout,
            batch_first=False
        )
    
    def _create_feed_forward_layer(self) -> nn.Module:
        """Create feed-forward layer."""
        return nn.Sequential(
            nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size * 4
            ),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.hidden_size * 4,
                self.config.hidden_size
            )
        )
    
    def _create_output_layer(self) -> nn.Module:
        """Create output layer."""
        return nn.Sequential(
            nn.LayerNorm(
                self.config.hidden_size,
                eps=self.config.layer_norm_eps
            ),
            nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False
            )
        )
    
    def _apply_enterprise_optimizations(self):
        """Apply enterprise optimizations."""
        if self.config.use_mixed_precision:
            self.half()
        
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            # Note: This usually happens outside or via a wrapper
            pass
        
        if self.config.use_quantization:
            # Dynamic quantization for inference
            try:
                self = torch.quantization.quantize_dynamic(
                    self,
                    {nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as e:
                self.logger.warning(f"Failed to apply dynamic quantization: {e}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with enterprise optimizations."""
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            residual = x
            x = layer['layer_norm1'](x)
            x, _ = layer['attention'](x, x, x)
            x = layer['dropout1'](x)
            x = x + residual
            
            residual = x
            x = layer['layer_norm2'](x)
            x = layer['feed_forward'](x)
            x = layer['dropout2'](x)
            x = x + residual
        
        return self.output(x)
    
    def optimize_for_inference(self):
        """Optimize model for inference."""
        self.eval()
        self.requires_grad_(False)
        try:
            torch.jit.optimize_for_inference(self)
        except Exception:
            pass
            
    def get_model_info(self) -> EnterpriseModelInfo:
        """Get typed model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return EnterpriseModelInfo(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            attention_heads=self.config.attention_heads,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            use_flash_attention=self.config.use_flash_attention,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            use_mixed_precision=self.config.use_mixed_precision,
            use_data_parallel=self.config.use_data_parallel,
            use_quantization=self.config.use_quantization
        )

