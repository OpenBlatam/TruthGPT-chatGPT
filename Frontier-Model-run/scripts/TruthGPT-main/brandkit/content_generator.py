"""
Content generation model for creating brand-consistent assets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings

@dataclass
class ContentGeneratorArgs:
    """Configuration for the content generator model."""
    hidden_size: int = 768
    num_layers: int = 6
    num_attention_heads: int = 12
    dropout: float = 0.1
    max_sequence_length: int = 1024
    
    text_vocab_size: int = 50000
    image_feature_dim: int = 512
    layout_dim: int = 256
    
    brand_profile_dim: int = 768
    style_conditioning_dim: int = 128
    
    content_types: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.content_types is None:
            self.content_types = ['social_post', 'blog_header', 'advertisement', 'logo_variant', 'color_scheme']

class BrandConditioningLayer(nn.Module):
    """Conditions content generation on brand profile."""
    
    def __init__(self, args: ContentGeneratorArgs):
        super().__init__()
        self.args = args
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            self.brand_projector = nn.Sequential(
                OptimizedLinear(args.brand_profile_dim, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                OptimizedLinear(args.hidden_size, args.style_conditioning_dim)
            )
        except ImportError:
            self.brand_projector = nn.Sequential(
                nn.Linear(args.brand_profile_dim, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, args.style_conditioning_dim)
            )
        
        try:
            from optimization_core import OptimizedLayerNorm
            self.brand_norm = OptimizedLayerNorm(args.style_conditioning_dim)
        except ImportError:
            self.brand_norm = nn.LayerNorm(args.style_conditioning_dim)
        
        self.content_type_embedding = nn.Embedding(
            len(args.content_types) if args.content_types is not None else 5, 
            args.style_conditioning_dim
        )
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            self.conditioning_fusion = nn.Sequential(
                OptimizedLinear(args.style_conditioning_dim * 2, args.hidden_size),
                nn.ReLU(),
                OptimizedLinear(args.hidden_size, args.hidden_size)
            )
        except ImportError:
            self.conditioning_fusion = nn.Sequential(
                nn.Linear(args.style_conditioning_dim * 2, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
            )
        
    def forward(self, brand_profile, content_type_ids):
        """
        Generate conditioning vectors for content generation.
        
        Args:
            brand_profile: (batch_size, brand_profile_dim) - Brand characteristics
            content_type_ids: (batch_size,) - Content type indices
        """
        brand_conditioning = self.brand_projector(brand_profile)
        brand_conditioning = self.brand_norm(brand_conditioning)
        
        content_type_emb = self.content_type_embedding(content_type_ids)
        
        combined_conditioning = torch.cat([brand_conditioning, content_type_emb], dim=-1)
        conditioning_vector = self.conditioning_fusion(combined_conditioning)
        
        return conditioning_vector

class TextGenerator(nn.Module):
    """Generates brand-consistent text content."""
    
    def __init__(self, args: ContentGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.text_embedding = nn.Embedding(args.text_vocab_size, args.hidden_size)
        self.pos_encoding = nn.Parameter(
            torch.randn(args.max_sequence_length, args.hidden_size)
        )
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=args.hidden_size,
                nhead=args.num_attention_heads,
                dim_feedforward=args.hidden_size * 4,
                dropout=args.dropout,
                batch_first=True
            ) for _ in range(args.num_layers)
        ])
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            self.output_projection = OptimizedLinear(args.hidden_size, args.text_vocab_size)
        except ImportError:
            self.output_projection = nn.Linear(args.hidden_size, args.text_vocab_size)
        
    def forward(self, input_ids, conditioning_vector, attention_mask=None):
        """
        Generate text conditioned on brand profile.
        
        Args:
            input_ids: (batch_size, seq_len) - Input token IDs
            conditioning_vector: (batch_size, hidden_size) - Brand conditioning
            attention_mask: (batch_size, seq_len) - Attention mask
        """
        batch_size, seq_len = input_ids.shape
        
        token_emb = self.text_embedding(input_ids)
        pos_emb = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        brand_conditioning = conditioning_vector.unsqueeze(1).expand(-1, seq_len, -1)
        
        hidden_states = token_emb + pos_emb + brand_conditioning
        
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, hidden_states)
        
        logits = self.output_projection(hidden_states)
        
        return logits

class ImageLayoutGenerator(nn.Module):
    """Generates image layouts and visual compositions."""
    
    def __init__(self, args: ContentGeneratorArgs):
        super().__init__()
        self.args = args
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            self.layout_generator = nn.Sequential(
                OptimizedLinear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                OptimizedLinear(args.hidden_size, args.layout_dim),
                nn.ReLU(),
                OptimizedLinear(args.layout_dim, args.image_feature_dim)
            )
        except ImportError:
            self.layout_generator = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, args.layout_dim),
                nn.ReLU(),
                nn.Linear(args.layout_dim, args.image_feature_dim)
            )
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            self.color_generator = nn.Sequential(
                OptimizedLinear(args.hidden_size, args.hidden_size // 2),
                nn.ReLU(),
                OptimizedLinear(args.hidden_size // 2, 15)  # 5 colors × 3 RGB values
            )
        except ImportError:
            self.color_generator = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(args.hidden_size // 2, 15)  # 5 colors × 3 RGB values
            )
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            self.typography_generator = nn.Sequential(
                OptimizedLinear(args.hidden_size, args.hidden_size // 2),
                nn.ReLU(),
                OptimizedLinear(args.hidden_size // 2, 64)  # Typography parameters
            )
        except ImportError:
            self.typography_generator = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(args.hidden_size // 2, 64)  # Typography parameters
            )
        
    def forward(self, conditioning_vector):
        """
        Generate visual layout components.
        
        Args:
            conditioning_vector: (batch_size, hidden_size) - Brand conditioning
        """
        layout_features = self.layout_generator(conditioning_vector)
        
        color_logits = self.color_generator(conditioning_vector)
        color_scheme = color_logits.view(-1, 5, 3)  # 5 colors, RGB
        color_scheme = torch.sigmoid(color_scheme)  # Normalize to [0,1]
        
        typography_params = self.typography_generator(conditioning_vector)
        
        return {
            'layout_features': layout_features,
            'color_scheme': color_scheme,
            'typography_params': typography_params
        }

class ContentGenerator(nn.Module):
    """Main content generator model."""
    
    def __init__(self, args: ContentGeneratorArgs):
        super().__init__()
        self.args = args
        
        self.brand_conditioning = BrandConditioningLayer(args)
        self.text_generator = TextGenerator(args)
        self.image_generator = ImageLayoutGenerator(args)
        
        try:
            from optimization_core.enhanced_mlp import OptimizedLinear
            self.quality_scorer = nn.Sequential(
                OptimizedLinear(args.hidden_size, args.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                OptimizedLinear(args.hidden_size // 2, 1),
                nn.Sigmoid()
            )
        except ImportError:
            self.quality_scorer = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
    def forward(self, brand_profile, content_type_ids, input_ids=None, generate_images=True):
        """
        Generate brand-consistent content.
        
        Args:
            brand_profile: (batch_size, brand_profile_dim) - Brand characteristics
            content_type_ids: (batch_size,) - Content type indices
            input_ids: (batch_size, seq_len) - Input tokens for text generation
            generate_images: Whether to generate image layouts
        """
        conditioning_vector = self.brand_conditioning(brand_profile, content_type_ids)
        
        outputs = {
            'conditioning_vector': conditioning_vector
        }
        
        if input_ids is not None:
            text_logits = self.text_generator(input_ids, conditioning_vector)
            outputs['text_logits'] = text_logits
        
        if generate_images:
            image_outputs = self.image_generator(conditioning_vector)
            outputs.update(image_outputs)
        
        quality_score = self.quality_scorer(conditioning_vector)
        outputs['quality_score'] = quality_score
        
        return outputs
    
    def generate_content_assets(self, brand_profile, content_types, max_length=100):
        """
        Generate complete content assets for given brand profile.
        
        Args:
            brand_profile: Brand characteristics tensor
            content_types: List of content type names
            max_length: Maximum text length for generation
        """
        content_assets = {}
        
        with torch.no_grad():
            for content_type in content_types:
                if content_type not in self.args.content_types:
                    warnings.warn(f"Unknown content type: {content_type}")
                    continue
                
                content_type_id = torch.tensor([self.args.content_types.index(content_type)])
                
                conditioning = self.brand_conditioning(brand_profile.unsqueeze(0), content_type_id)
                
                image_outputs = self.image_generator(conditioning)
                
                mock_input_ids = torch.randint(0, 1000, (1, 20))  # Mock tokens
                text_outputs = self.text_generator(mock_input_ids, conditioning)
                
                quality = self.quality_scorer(conditioning)
                
                content_assets[content_type] = {
                    'layout_features': image_outputs['layout_features'].squeeze(0).tolist(),
                    'color_scheme': image_outputs['color_scheme'].squeeze(0).tolist(),
                    'typography_params': image_outputs['typography_params'].squeeze(0).tolist(),
                    'text_logits': text_outputs.squeeze(0).tolist(),
                    'quality_score': quality.item(),
                    'conditioning_vector': conditioning.squeeze(0).tolist()
                }
        
        return content_assets

def create_content_generator_model(config: Dict[str, Any]) -> ContentGenerator:
    """Create a content generator model from configuration."""
    args = ContentGeneratorArgs(
        hidden_size=config.get('hidden_size', 768),
        num_layers=config.get('num_layers', 6),
        num_attention_heads=config.get('num_attention_heads', 12),
        dropout=config.get('dropout', 0.1),
        max_sequence_length=config.get('max_sequence_length', 1024),
        
        text_vocab_size=config.get('text_vocab_size', 50000),
        image_feature_dim=config.get('image_feature_dim', 512),
        layout_dim=config.get('layout_dim', 256),
        
        brand_profile_dim=config.get('brand_profile_dim', 768),
        style_conditioning_dim=config.get('style_conditioning_dim', 128),
        
        content_types=config.get('content_types', None)
    )
    
    model = ContentGenerator(args)
    
    try:
        from enhanced_model_optimizer import create_universal_optimizer
        optimizer = create_universal_optimizer({
            'enable_fp16': True,
            'enable_gradient_checkpointing': True,
            'use_advanced_normalization': True,
            'use_enhanced_mlp': True
        })
        model = optimizer.optimize_model(model, "Content-Generator")
    except ImportError:
        pass
    
    return model
