"""
Optimized brandkit models with enhanced performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from .performance_utils import OptimizedAttention, OptimizedMLP, GradientCheckpointingWrapper

@dataclass
class OptimizedBrandKitArgs:
    """Configuration for optimized brandkit models."""
    hidden_size: int = 768
    num_layers: int = 8
    num_attention_heads: int = 12
    dropout: float = 0.1
    max_sequence_length: int = 2048
    
    color_palette_size: int = 16
    typography_features: int = 64
    layout_features: int = 128
    
    tone_categories: int = 10
    sentiment_dim: int = 32
    style_dim: int = 64
    
    visual_feature_dim: int = 1024
    text_feature_dim: int = 768
    metadata_feature_dim: int = 256
    
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_cached_embeddings: bool = True
    use_efficient_cross_attention: bool = True

class OptimizedVisualAnalyzer(nn.Module):
    """Optimized visual analyzer with efficient processing."""
    
    def __init__(self, args: OptimizedBrandKitArgs):
        super().__init__()
        self.args = args
        
        self.color_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, args.color_palette_size)
        )
        
        self.typography_encoder = nn.Sequential(
            nn.Linear(args.typography_features, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, args.hidden_size // 2)
        )
        
        self.layout_encoder = nn.Sequential(
            nn.Linear(args.layout_features, 256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, args.hidden_size // 2)
        )
        
        self.visual_fusion = nn.Sequential(
            nn.Linear(args.color_palette_size + args.hidden_size, args.visual_feature_dim),
            nn.LayerNorm(args.visual_feature_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        if args.use_cached_embeddings:
            self.embedding_cache = {}
        
    def forward(self, colors, typography_features, layout_features):
        batch_size = colors.shape[0]
        
        colors_flat = colors.view(-1, 3)
        color_embeddings = self.color_encoder(colors_flat)
        color_embeddings = color_embeddings.view(batch_size, -1, self.args.color_palette_size)
        color_summary = color_embeddings.mean(dim=1)
        
        typo_emb = self.typography_encoder(typography_features)
        layout_emb = self.layout_encoder(layout_features)
        
        visual_structure = torch.cat([typo_emb, layout_emb], dim=-1)
        visual_combined = torch.cat([color_summary, visual_structure], dim=-1)
        visual_features = self.visual_fusion(visual_combined)
        
        return {
            'visual_features': visual_features,
            'color_palette': color_summary,
            'typography_embedding': typo_emb,
            'layout_embedding': layout_emb
        }

class OptimizedTextAnalyzer(nn.Module):
    """Optimized text analyzer with efficient processing."""
    
    def __init__(self, args: OptimizedBrandKitArgs):
        super().__init__()
        self.args = args
        
        self.text_encoder = nn.Sequential(
            nn.Linear(args.text_feature_dim, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LayerNorm(args.hidden_size)
        )
        
        self.tone_classifier = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size // 2, args.tone_categories)
        )
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.sentiment_dim),
            nn.Tanh()
        )
        
        self.style_encoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.style_dim),
            nn.ReLU()
        )
        
    def forward(self, text_features):
        encoded_text = self.text_encoder(text_features)
        pooled_text = encoded_text.mean(dim=1)
        
        tone_logits = self.tone_classifier(pooled_text)
        sentiment_emb = self.sentiment_encoder(pooled_text)
        style_emb = self.style_encoder(pooled_text)
        
        return {
            'text_features': pooled_text,
            'tone_logits': tone_logits,
            'sentiment_embedding': sentiment_emb,
            'style_embedding': style_emb
        }

class OptimizedBrandFusionLayer(nn.Module):
    """Optimized brand fusion with efficient cross-attention."""
    
    def __init__(self, args: OptimizedBrandKitArgs):
        super().__init__()
        self.args = args
        
        if args.use_efficient_cross_attention:
            self.cross_attention = OptimizedAttention(
                args.hidden_size, args.num_attention_heads, args.dropout, args.use_flash_attention
            )
        else:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=args.hidden_size,
                num_heads=args.num_attention_heads,
                dropout=args.dropout,
                batch_first=True
            )
        
        self.visual_proj = nn.Linear(args.visual_feature_dim, args.hidden_size)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_size + args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LayerNorm(args.hidden_size)
        )
        
    def forward(self, visual_features, text_features):
        visual_expanded = visual_features.unsqueeze(1)
        text_expanded = text_features.unsqueeze(1)
        
        visual_aligned = self.visual_proj(visual_expanded)
        
        if self.args.use_efficient_cross_attention:
            combined_features = torch.cat([visual_aligned, text_expanded], dim=1)
            attended_features = self.cross_attention(combined_features)
            attended_visual, attended_text = attended_features.chunk(2, dim=1)
        else:
            attended_visual, _ = self.cross_attention(visual_aligned, text_expanded, text_expanded)
            attended_text, _ = self.cross_attention(text_expanded, visual_aligned, visual_aligned)
        
        combined = torch.cat([
            attended_visual.squeeze(1), 
            attended_text.squeeze(1)
        ], dim=-1)
        
        fused_features = self.fusion_layer(combined)
        
        return fused_features

class OptimizedBrandAnalyzer(nn.Module):
    """Optimized brand analyzer model."""
    
    def __init__(self, args: OptimizedBrandKitArgs):
        super().__init__()
        self.args = args
        
        self.visual_analyzer = OptimizedVisualAnalyzer(args)
        self.text_analyzer = OptimizedTextAnalyzer(args)
        self.brand_fusion = OptimizedBrandFusionLayer(args)
        
        self.brand_profile_head = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size)
        )
        
        self.consistency_head = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, colors, typography_features, layout_features, text_features):
        visual_analysis = self.visual_analyzer(colors, typography_features, layout_features)
        text_analysis = self.text_analyzer(text_features)
        
        fused_features = self.brand_fusion(
            visual_analysis['visual_features'],
            text_analysis['text_features']
        )
        
        brand_profile = self.brand_profile_head(fused_features)
        consistency_score = self.consistency_head(fused_features)
        
        return {
            'brand_profile': brand_profile,
            'consistency_score': consistency_score,
            'visual_analysis': visual_analysis,
            'text_analysis': text_analysis,
            'fused_features': fused_features
        }
    
    def extract_brand_kit(self, website_data):
        """Extract comprehensive brand kit with caching."""
        cache_key = hash(str(website_data))
        
        if self.args.use_cached_embeddings and hasattr(self, '_brand_cache'):
            if cache_key in self._brand_cache:
                return self._brand_cache[cache_key]
        
        with torch.no_grad():
            outputs = self.forward(**website_data)
            
            color_palette = outputs['visual_analysis']['color_palette']
            dominant_colors = torch.topk(color_palette, k=5, dim=-1).indices
            
            typography_emb = outputs['visual_analysis']['typography_embedding']
            tone_probs = F.softmax(outputs['text_analysis']['tone_logits'], dim=-1)
            dominant_tone = torch.argmax(tone_probs, dim=-1)[0]
            consistency = outputs['consistency_score'][0].item()
            
            brand_kit = {
                'color_palette': dominant_colors.tolist(),
                'typography_profile': typography_emb[0].tolist(),
                'tone_profile': {
                    'dominant_tone': dominant_tone.item(),
                    'tone_distribution': tone_probs[0].tolist()
                },
                'style_embedding': outputs['text_analysis']['style_embedding'][0].tolist(),
                'sentiment_profile': outputs['text_analysis']['sentiment_embedding'][0].tolist(),
                'brand_consistency_score': consistency,
                'brand_profile': outputs['brand_profile'][0].tolist()
            }
            
            if self.args.use_cached_embeddings:
                if not hasattr(self, '_brand_cache'):
                    self._brand_cache = {}
                self._brand_cache[cache_key] = brand_kit
            
            return brand_kit

class OptimizedContentGenerator(nn.Module):
    """Optimized content generator with efficient processing."""
    
    def __init__(self, args: OptimizedBrandKitArgs):
        super().__init__()
        self.args = args
        
        self.brand_conditioning = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        self.content_type_embedding = nn.Embedding(5, args.hidden_size)
        
        self.generator_layers = nn.ModuleList([
            GradientCheckpointingWrapper(
                OptimizedMLP(args.hidden_size, args.hidden_size * 4, args.dropout, args.use_gradient_checkpointing),
                args.use_gradient_checkpointing
            ) for _ in range(4)
        ])
        
        self.layout_head = nn.Sequential(
            nn.Linear(args.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(args.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 5 * 3)
        )
        
        self.typography_head = nn.Sequential(
            nn.Linear(args.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(args.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, brand_profile, content_type_ids, generate_images=True):
        batch_size = brand_profile.shape[0]
        
        conditioned_brand = self.brand_conditioning(brand_profile)
        content_type_emb = self.content_type_embedding(content_type_ids)
        
        combined = conditioned_brand + content_type_emb
        
        for layer in self.generator_layers:
            combined = layer(combined) + combined
        
        layout_features = self.layout_head(combined)
        color_scheme = self.color_head(combined).view(batch_size, 5, 3)
        typography_params = self.typography_head(combined)
        quality_score = self.quality_head(combined)
        
        outputs = {
            'layout_features': layout_features,
            'color_scheme': color_scheme,
            'typography_params': typography_params,
            'quality_score': quality_score
        }
        
        return outputs
    
    def generate_content_assets(self, brand_profile, content_types):
        """Generate content assets with batch processing."""
        content_type_map = {
            'social_post': 0, 'blog_header': 1, 'advertisement': 2, 
            'logo_variant': 3, 'color_scheme': 4
        }
        
        assets = {}
        
        for content_type in content_types:
            if content_type not in content_type_map:
                continue
                
            type_id = torch.tensor([content_type_map[content_type]], dtype=torch.long)
            
            with torch.no_grad():
                outputs = self.forward(brand_profile.unsqueeze(0), type_id)
                
                assets[content_type] = {
                    'layout_features': outputs['layout_features'][0],
                    'color_scheme': outputs['color_scheme'][0],
                    'typography_params': outputs['typography_params'][0],
                    'quality_score': outputs['quality_score'][0].item()
                }
        
        return assets

def create_optimized_brand_analyzer_model(config: Dict[str, Any]) -> OptimizedBrandAnalyzer:
    """Create an optimized brand analyzer model from configuration."""
    args = OptimizedBrandKitArgs(**config)
    return OptimizedBrandAnalyzer(args)

def create_optimized_content_generator_model(config: Dict[str, Any]) -> OptimizedContentGenerator:
    """Create an optimized content generator model from configuration."""
    args = OptimizedBrandKitArgs(**config)
    return OptimizedContentGenerator(args)
