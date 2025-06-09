"""
Cross-modal generation capabilities for multi-modal content creation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class CrossModalGeneratorArgs:
    """Configuration for cross-modal generator."""
    hidden_size: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    dropout: float = 0.1
    
    text_encoder_dim: int = 768
    image_encoder_dim: int = 512
    video_encoder_dim: int = 1024
    
    enable_text_to_image: bool = True
    enable_image_to_text: bool = True
    enable_video_to_content: bool = True
    enable_multi_modal_coherence: bool = True

class CrossModalAttention(nn.Module):
    """Cross-modal attention for different modality fusion."""
    
    def __init__(self, args: CrossModalGeneratorArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = args.hidden_size // args.num_heads
        
        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.k_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.v_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attn_output)

class ModalityEncoder(nn.Module):
    """Encoder for different modalities."""
    
    def __init__(self, input_dim: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class CrossModalFusionLayer(nn.Module):
    """Layer for fusing different modalities."""
    
    def __init__(self, args: CrossModalGeneratorArgs):
        super().__init__()
        self.cross_attention = CrossModalAttention(args)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size * 4, args.hidden_size),
            nn.Dropout(args.dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(args.hidden_size)
        self.layer_norm2 = nn.LayerNorm(args.hidden_size)
        
    def forward(self, query_modality: torch.Tensor, key_modality: torch.Tensor,
                value_modality: torch.Tensor) -> torch.Tensor:
        residual = query_modality
        query_modality = self.layer_norm1(query_modality)
        
        attn_output = self.cross_attention(query_modality, key_modality, value_modality)
        query_modality = residual + attn_output
        
        residual = query_modality
        query_modality = self.layer_norm2(query_modality)
        ff_output = self.feed_forward(query_modality)
        query_modality = residual + ff_output
        
        return query_modality

class TextToImageGenerator(nn.Module):
    """Generate images from text descriptions."""
    
    def __init__(self, args: CrossModalGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.text_encoder = ModalityEncoder(args.text_encoder_dim, args.hidden_size, args.dropout)
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(args) for _ in range(args.num_layers // 2)
        ])
        
        self.image_decoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size * 2, args.image_encoder_dim),
            nn.Tanh()
        )
        
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        encoded_text = self.text_encoder(text_features)
        
        fused_features = encoded_text
        for layer in self.fusion_layers:
            fused_features = layer(fused_features, encoded_text, encoded_text)
        
        image_features = self.image_decoder(fused_features)
        
        return image_features

class ImageToTextGenerator(nn.Module):
    """Generate text descriptions from images."""
    
    def __init__(self, args: CrossModalGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.image_encoder = ModalityEncoder(args.image_encoder_dim, args.hidden_size, args.dropout)
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(args) for _ in range(args.num_layers // 2)
        ])
        
        self.text_decoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size * 2, args.text_encoder_dim),
            nn.LayerNorm(args.text_encoder_dim)
        )
        
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        encoded_image = self.image_encoder(image_features)
        
        fused_features = encoded_image
        for layer in self.fusion_layers:
            fused_features = layer(fused_features, encoded_image, encoded_image)
        
        text_features = self.text_decoder(fused_features)
        
        return text_features

class VideoToContentGenerator(nn.Module):
    """Generate content from video features."""
    
    def __init__(self, args: CrossModalGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.video_encoder = ModalityEncoder(args.video_encoder_dim, args.hidden_size, args.dropout)
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=args.hidden_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            batch_first=True
        )
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(args) for _ in range(args.num_layers // 2)
        ])
        
        self.content_decoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size * 2, args.text_encoder_dim + args.image_encoder_dim)
        )
        
    def forward(self, video_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = video_features.shape
        
        encoded_video = self.video_encoder(video_features)
        
        temporal_features, _ = self.temporal_attention(encoded_video, encoded_video, encoded_video)
        
        fused_features = temporal_features
        for layer in self.fusion_layers:
            fused_features = layer(fused_features, temporal_features, temporal_features)
        
        pooled_features = fused_features.mean(dim=1)
        
        content_features = self.content_decoder(pooled_features)
        
        text_features = content_features[:, :self.args.text_encoder_dim]
        image_features = content_features[:, self.args.text_encoder_dim:]
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'fused_features': pooled_features
        }

class CrossModalGenerator(nn.Module):
    """Main cross-modal generator combining all capabilities."""
    
    def __init__(self, args: CrossModalGeneratorArgs):
        super().__init__()
        self.args = args
        
        if args.enable_text_to_image:
            self.text_to_image = TextToImageGenerator(args)
        
        if args.enable_image_to_text:
            self.image_to_text = ImageToTextGenerator(args)
        
        if args.enable_video_to_content:
            self.video_to_content = VideoToContentGenerator(args)
        
        if args.enable_multi_modal_coherence:
            self.coherence_scorer = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, 1),
                nn.Sigmoid()
            )
        
    def forward(self, modality_type: str, input_features: torch.Tensor, 
                target_modality: Optional[str] = None) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        if modality_type == 'text' and hasattr(self, 'text_to_image'):
            outputs['generated_image'] = self.text_to_image(input_features)
        
        if modality_type == 'image' and hasattr(self, 'image_to_text'):
            outputs['generated_text'] = self.image_to_text(input_features)
        
        if modality_type == 'video' and hasattr(self, 'video_to_content'):
            video_outputs = self.video_to_content(input_features)
            outputs.update(video_outputs)
        
        return outputs
    
    def compute_coherence_score(self, modality1: torch.Tensor, modality2: torch.Tensor) -> torch.Tensor:
        """Compute coherence score between two modalities."""
        if not hasattr(self, 'coherence_scorer'):
            return torch.tensor(0.0)
        
        combined = torch.cat([modality1, modality2], dim=-1)
        return self.coherence_scorer(combined)
    
    def generate_cross_modal_content(self, input_features: torch.Tensor, 
                                   source_modality: str, target_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """Generate content across multiple modalities."""
        results = {}
        
        for target in target_modalities:
            if source_modality == 'text' and target == 'image' and hasattr(self, 'text_to_image'):
                results['image'] = self.text_to_image(input_features)
            elif source_modality == 'image' and target == 'text' and hasattr(self, 'image_to_text'):
                results['text'] = self.image_to_text(input_features)
            elif source_modality == 'video' and hasattr(self, 'video_to_content'):
                video_results = self.video_to_content(input_features)
                if target == 'text':
                    results['text'] = video_results['text_features']
                elif target == 'image':
                    results['image'] = video_results['image_features']
        
        return results

def create_cross_modal_generator(config: Dict[str, Any]) -> CrossModalGenerator:
    """Create cross-modal generator from configuration."""
    args = CrossModalGeneratorArgs(
        hidden_size=config.get('hidden_size', 1024),
        num_layers=config.get('num_layers', 8),
        num_heads=config.get('num_heads', 16),
        dropout=config.get('dropout', 0.1),
        text_encoder_dim=config.get('text_encoder_dim', 768),
        image_encoder_dim=config.get('image_encoder_dim', 512),
        video_encoder_dim=config.get('video_encoder_dim', 1024),
        enable_text_to_image=config.get('enable_text_to_image', True),
        enable_image_to_text=config.get('enable_image_to_text', True),
        enable_video_to_content=config.get('enable_video_to_content', True),
        enable_multi_modal_coherence=config.get('enable_multi_modal_coherence', True)
    )
    
    return CrossModalGenerator(args)
