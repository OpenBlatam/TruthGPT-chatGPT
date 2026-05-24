"""
Brand Kit Ads Model - Native Implementation for Website Brand Analysis and Ad Generation

This model specializes in:
1. Website analysis and brand kit extraction (colors, typography, design patterns)
2. Brand identity understanding and style recognition
3. Targeted advertising content generation
4. Image title and description creation
5. Multi-modal brand-aware content creation

Architecture combines:
- Vision transformer for visual brand analysis
- Language model for content generation
- Brand embedding space for style consistency
- Multi-modal fusion for comprehensive understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import json
import yaml
import requests
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass
import colorsys
import re
from urllib.parse import urljoin, urlparse
import webcolors


@dataclass
class BrandKitExtraction:
    """Data structure for extracted brand kit information"""
    primary_colors: List[str]
    secondary_colors: List[str]
    accent_colors: List[str]
    typography: Dict[str, Any]
    logo_elements: List[Dict[str, Any]]
    design_patterns: Dict[str, Any]
    brand_personality: Dict[str, float]
    visual_hierarchy: Dict[str, Any]
    spacing_patterns: Dict[str, Any]
    brand_voice: Dict[str, Any]


@dataclass
class AdContent:
    """Generated advertising content structure"""
    headline: str
    subheadline: str
    body_text: str
    call_to_action: str
    image_descriptions: List[str]
    brand_alignment_score: float
    target_audience: str
    content_type: str
    visual_suggestions: Dict[str, Any]


class BrandKitAdsConfig(PretrainedConfig):
    """Configuration for Brand Kit Ads Model"""
    
    model_type = "brand_kit_ads"
    
    def __init__(
        self,
        # Base language model config
        vocab_size: int = 50257,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 8192,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 4096,
        layer_norm_eps: float = 1e-5,
        
        # Vision transformer config
        vision_hidden_size: int = 768,
        vision_num_layers: int = 12,
        vision_num_heads: int = 12,
        vision_patch_size: int = 16,
        vision_image_size: int = 224,
        
        # Brand analysis config
        brand_embedding_size: int = 512,
        color_embedding_size: int = 128,
        typography_embedding_size: int = 256,
        design_pattern_size: int = 384,
        
        # Multi-modal fusion config
        fusion_hidden_size: int = 1024,
        fusion_num_layers: int = 6,
        
        # Brand kit extraction config
        max_colors_extract: int = 20,
        color_clustering_threshold: float = 0.15,
        typography_analysis_depth: int = 5,
        
        # Ad generation config
        max_ad_length: int = 512,
        num_ad_variants: int = 3,
        brand_consistency_weight: float = 0.3,
        creativity_weight: float = 0.4,
        target_relevance_weight: float = 0.3,
        
        # Training config
        use_brand_consistency_loss: bool = True,
        use_visual_alignment_loss: bool = True,
        use_content_quality_loss: bool = True,
        brand_loss_weight: float = 0.2,
        visual_loss_weight: float = 0.15,
        quality_loss_weight: float = 0.1,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Base model config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        
        # Vision config
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        
        # Brand analysis config
        self.brand_embedding_size = brand_embedding_size
        self.color_embedding_size = color_embedding_size
        self.typography_embedding_size = typography_embedding_size
        self.design_pattern_size = design_pattern_size
        
        # Multi-modal fusion
        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_num_layers = fusion_num_layers
        
        # Brand kit extraction
        self.max_colors_extract = max_colors_extract
        self.color_clustering_threshold = color_clustering_threshold
        self.typography_analysis_depth = typography_analysis_depth
        
        # Ad generation
        self.max_ad_length = max_ad_length
        self.num_ad_variants = num_ad_variants
        self.brand_consistency_weight = brand_consistency_weight
        self.creativity_weight = creativity_weight
        self.target_relevance_weight = target_relevance_weight
        
        # Training
        self.use_brand_consistency_loss = use_brand_consistency_loss
        self.use_visual_alignment_loss = use_visual_alignment_loss
        self.use_content_quality_loss = use_content_quality_loss
        self.brand_loss_weight = brand_loss_weight
        self.visual_loss_weight = visual_loss_weight
        self.quality_loss_weight = quality_loss_weight

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('model', {}))


class VisionPatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.patch_size = config.vision_patch_size
        self.hidden_size = config.vision_hidden_size
        
        self.projection = nn.Conv2d(
            3, self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch_size, 3, height, width]
        batch_size = images.shape[0]
        
        # Extract patches and project
        patches = self.projection(images)  # [batch_size, hidden_size, num_patches_h, num_patches_w]
        patches = patches.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_size]
        
        return patches


class VisionTransformerLayer(nn.Module):
    """Single layer of vision transformer"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.hidden_size = config.vision_hidden_size
        self.num_heads = config.vision_num_heads
        
        self.attention = nn.MultiheadAttention(
            self.hidden_size, self.num_heads, 
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states


class VisionTransformer(nn.Module):
    """Vision transformer for image analysis"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = VisionPatchEmbedding(config)
        
        # Position embeddings
        num_patches = (config.vision_image_size // config.vision_patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config.vision_hidden_size)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vision_hidden_size))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            VisionTransformerLayer(config) for _ in range(config.vision_num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.vision_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        
        # Extract patches
        patch_embeddings = self.patch_embedding(images)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        embeddings = self.norm(embeddings)
        
        return embeddings


class ColorAnalyzer(nn.Module):
    """Analyze and extract color information from images"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.config = config
        
        # Color feature extractor
        self.color_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 64, config.color_embedding_size)
        )
        
        # Color clustering head
        self.color_cluster_head = nn.Linear(
            config.color_embedding_size, 
            config.max_colors_extract * 3  # RGB values
        )
        
        # Color importance weights
        self.color_importance_head = nn.Linear(
            config.color_embedding_size,
            config.max_colors_extract
        )
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract color features
        color_features = self.color_conv(images)
        
        # Predict color clusters
        color_clusters = self.color_cluster_head(color_features)
        color_clusters = color_clusters.view(-1, self.config.max_colors_extract, 3)
        color_clusters = torch.sigmoid(color_clusters)  # Normalize to [0, 1]
        
        # Predict color importance
        color_importance = torch.softmax(
            self.color_importance_head(color_features), dim=-1
        )
        
        return {
            'color_features': color_features,
            'color_clusters': color_clusters,
            'color_importance': color_importance
        }
    
    def extract_dominant_colors(self, image_np: np.ndarray, num_colors: int = 10) -> List[str]:
        """Extract dominant colors using traditional CV methods"""
        # Reshape image for clustering
        data = image_np.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert to hex colors
        colors = []
        for center in centers:
            color = tuple(map(int, center))
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            colors.append(hex_color)
        
        return colors


class TypographyAnalyzer(nn.Module):
    """Analyze typography and text styling"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.config = config
        
        # Typography feature extractor
        self.typography_encoder = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.typography_embedding_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.typography_embedding_size, config.typography_embedding_size)
        )
        
        # Font style classifier
        self.font_style_head = nn.Linear(config.typography_embedding_size, 10)  # serif, sans-serif, etc.
        
        # Font weight classifier
        self.font_weight_head = nn.Linear(config.typography_embedding_size, 9)  # 100-900
        
        # Text hierarchy analyzer
        self.hierarchy_head = nn.Linear(config.typography_embedding_size, 5)  # h1-h5, body
        
    def forward(self, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use CLS token for typography analysis
        cls_features = vision_features[:, 0]  # [batch_size, vision_hidden_size]
        
        # Extract typography features
        typography_features = self.typography_encoder(cls_features)
        
        # Classify font properties
        font_style = torch.softmax(self.font_style_head(typography_features), dim=-1)
        font_weight = torch.softmax(self.font_weight_head(typography_features), dim=-1)
        text_hierarchy = torch.softmax(self.hierarchy_head(typography_features), dim=-1)
        
        return {
            'typography_features': typography_features,
            'font_style': font_style,
            'font_weight': font_weight,
            'text_hierarchy': text_hierarchy
        }


class DesignPatternAnalyzer(nn.Module):
    """Analyze design patterns and visual elements"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.config = config
        
        # Design pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.design_pattern_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.design_pattern_size, config.design_pattern_size)
        )
        
        # Layout pattern classifier
        self.layout_head = nn.Linear(config.design_pattern_size, 8)  # grid, flex, etc.
        
        # Visual style classifier
        self.style_head = nn.Linear(config.design_pattern_size, 12)  # modern, classic, etc.
        
        # Spacing pattern analyzer
        self.spacing_head = nn.Linear(config.design_pattern_size, 6)  # tight, normal, loose, etc.
        
    def forward(self, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Use CLS token for design analysis
        cls_features = vision_features[:, 0]
        
        # Extract design pattern features
        pattern_features = self.pattern_encoder(cls_features)
        
        # Classify design patterns
        layout_pattern = torch.softmax(self.layout_head(pattern_features), dim=-1)
        visual_style = torch.softmax(self.style_head(pattern_features), dim=-1)
        spacing_pattern = torch.softmax(self.spacing_head(pattern_features), dim=-1)
        
        return {
            'pattern_features': pattern_features,
            'layout_pattern': layout_pattern,
            'visual_style': visual_style,
            'spacing_pattern': spacing_pattern
        }


class BrandEmbedding(nn.Module):
    """Create unified brand embedding from all analysis components"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.config = config
        
        # Combine all feature types
        total_input_size = (
            config.color_embedding_size +
            config.typography_embedding_size +
            config.design_pattern_size +
            config.vision_hidden_size  # CLS token
        )
        
        self.brand_fusion = nn.Sequential(
            nn.Linear(total_input_size, config.brand_embedding_size * 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.brand_embedding_size * 2, config.brand_embedding_size),
            nn.Tanh()  # Normalize brand embedding
        )
        
        # Brand personality classifier
        self.personality_head = nn.Linear(config.brand_embedding_size, 16)  # professional, playful, etc.
        
        # Brand voice classifier
        self.voice_head = nn.Linear(config.brand_embedding_size, 12)  # formal, casual, etc.
        
    def forward(
        self, 
        vision_features: torch.Tensor,
        color_features: torch.Tensor,
        typography_features: torch.Tensor,
        pattern_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        # Combine all features
        cls_features = vision_features[:, 0]  # CLS token
        combined_features = torch.cat([
            cls_features,
            color_features,
            typography_features,
            pattern_features
        ], dim=-1)
        
        # Create unified brand embedding
        brand_embedding = self.brand_fusion(combined_features)
        
        # Classify brand characteristics
        brand_personality = torch.softmax(self.personality_head(brand_embedding), dim=-1)
        brand_voice = torch.softmax(self.voice_head(brand_embedding), dim=-1)
        
        return {
            'brand_embedding': brand_embedding,
            'brand_personality': brand_personality,
            'brand_voice': brand_voice
        }


class LanguageModelLayer(nn.Module):
    """Transformer layer for language generation"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        self.attention = nn.MultiheadAttention(
            self.hidden_size, self.num_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, self.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask
        )
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states


class MultiModalFusion(nn.Module):
    """Fuse brand information with language generation"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.config = config
        
        # Project brand embedding to language model space
        self.brand_projection = nn.Linear(
            config.brand_embedding_size, 
            config.hidden_size
        )
        
        # Cross-attention between brand and language
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.fusion_hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.fusion_hidden_size * 4,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            ) for _ in range(config.fusion_num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.fusion_hidden_size, config.hidden_size)
        
    def forward(
        self,
        language_features: torch.Tensor,
        brand_embedding: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, seq_len = language_features.shape[:2]
        
        # Project brand embedding
        brand_features = self.brand_projection(brand_embedding)  # [batch_size, hidden_size]
        brand_features = brand_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Cross-attention
        fused_features, _ = self.cross_attention(
            language_features, brand_features, brand_features
        )
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            fused_features = layer(fused_features)
        
        # Project to output space
        output = self.output_projection(fused_features)
        
        return output


class AdContentGenerator(nn.Module):
    """Generate advertising content based on brand analysis"""
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__()
        self.config = config
        
        # Content type embeddings
        self.content_type_embeddings = nn.Embedding(10, config.hidden_size)  # headline, body, etc.
        
        # Target audience embeddings
        self.audience_embeddings = nn.Embedding(20, config.hidden_size)  # demographics
        
        # Brand-aware generation head
        self.generation_head = nn.Sequential(
            nn.Linear(config.hidden_size + config.brand_embedding_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
        # Content quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Brand alignment scorer
        self.alignment_scorer = nn.Sequential(
            nn.Linear(config.hidden_size + config.brand_embedding_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        language_features: torch.Tensor,
        brand_embedding: torch.Tensor,
        content_type_ids: Optional[torch.Tensor] = None,
        audience_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = language_features.shape[:2]
        
        # Add content type and audience information
        if content_type_ids is not None:
            content_type_emb = self.content_type_embeddings(content_type_ids)
            language_features = language_features + content_type_emb.unsqueeze(1)
        
        if audience_ids is not None:
            audience_emb = self.audience_embeddings(audience_ids)
            language_features = language_features + audience_emb.unsqueeze(1)
        
        # Combine with brand embedding
        brand_expanded = brand_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        combined_features = torch.cat([language_features, brand_expanded], dim=-1)
        
        # Generate content logits
        logits = self.generation_head(combined_features)
        
        # Score content quality and brand alignment
        quality_scores = self.quality_scorer(language_features)
        alignment_scores = self.alignment_scorer(combined_features)
        
        return {
            'logits': logits,
            'quality_scores': quality_scores,
            'alignment_scores': alignment_scores
        }


class BrandKitAdsModel(PreTrainedModel):
    """
    Complete Brand Kit Ads Model for website analysis and ad generation
    """
    
    config_class = BrandKitAdsConfig
    
    def __init__(self, config: BrandKitAdsConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Vision components
        self.vision_transformer = VisionTransformer(config)
        self.color_analyzer = ColorAnalyzer(config)
        self.typography_analyzer = TypographyAnalyzer(config)
        self.design_analyzer = DesignPatternAnalyzer(config)
        self.brand_embedding = BrandEmbedding(config)
        
        # Language model components
        self.language_layers = nn.ModuleList([
            LanguageModelLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Multi-modal fusion
        self.multimodal_fusion = MultiModalFusion(config)
        
        # Ad generation
        self.ad_generator = AdContentGenerator(config)
        
        # Output layers
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.init_weights()
        
    def get_input_embeddings(self):
        return self.token_embeddings
    
    def set_input_embeddings(self, value):
        self.token_embeddings = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def analyze_website_brand(
        self, 
        images: torch.Tensor,
        return_detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze website images to extract brand kit information
        
        Args:
            images: Tensor of shape [batch_size, 3, height, width]
            return_detailed: Whether to return detailed analysis
            
        Returns:
            Dictionary containing brand analysis results
        """
        
        # Vision analysis
        vision_features = self.vision_transformer(images)
        
        # Component analysis
        color_analysis = self.color_analyzer(images)
        typography_analysis = self.typography_analyzer(vision_features)
        design_analysis = self.design_analyzer(vision_features)
        
        # Create unified brand embedding
        brand_analysis = self.brand_embedding(
            vision_features,
            color_analysis['color_features'],
            typography_analysis['typography_features'],
            design_analysis['pattern_features']
        )
        
        results = {
            'brand_embedding': brand_analysis['brand_embedding'],
            'brand_personality': brand_analysis['brand_personality'],
            'brand_voice': brand_analysis['brand_voice']
        }
        
        if return_detailed:
            results.update({
                'color_analysis': color_analysis,
                'typography_analysis': typography_analysis,
                'design_analysis': design_analysis,
                'vision_features': vision_features
            })
        
        return results
    
    def extract_brand_kit_from_url(self, url: str) -> BrandKitExtraction:
        """
        Extract brand kit from website URL
        
        Args:
            url: Website URL to analyze
            
        Returns:
            BrandKitExtraction object with extracted information
        """
        try:
            # This would integrate with web scraping and screenshot tools
            # For now, return a placeholder structure
            return BrandKitExtraction(
                primary_colors=['#1a1a1a', '#ffffff', '#007bff'],
                secondary_colors=['#6c757d', '#28a745'],
                accent_colors=['#ffc107', '#dc3545'],
                typography={
                    'primary_font': 'Arial, sans-serif',
                    'secondary_font': 'Georgia, serif',
                    'font_sizes': [12, 14, 16, 18, 24, 32, 48],
                    'line_heights': [1.2, 1.4, 1.6],
                    'font_weights': [400, 600, 700]
                },
                logo_elements=[],
                design_patterns={
                    'layout': 'grid',
                    'spacing': 'normal',
                    'border_radius': 4,
                    'shadows': True
                },
                brand_personality={
                    'professional': 0.8,
                    'modern': 0.7,
                    'trustworthy': 0.9,
                    'innovative': 0.6
                },
                visual_hierarchy={
                    'header_prominence': 0.9,
                    'content_structure': 0.8,
                    'call_to_action_visibility': 0.7
                },
                spacing_patterns={
                    'margin': 16,
                    'padding': 12,
                    'gap': 8
                },
                brand_voice={
                    'tone': 'professional',
                    'formality': 0.7,
                    'friendliness': 0.6
                }
            )
        except Exception as e:
            print(f"Error extracting brand kit from {url}: {e}")
            return None
    
    def generate_ad_content(
        self,
        brand_embedding: torch.Tensor,
        prompt: str,
        content_type: str = "social_media",
        target_audience: str = "general",
        max_length: int = 512,
        num_variants: int = 3,
        temperature: float = 0.8
    ) -> List[AdContent]:
        """
        Generate advertising content based on brand analysis
        
        Args:
            brand_embedding: Brand embedding from analysis
            prompt: Content generation prompt
            content_type: Type of content to generate
            target_audience: Target audience description
            max_length: Maximum content length
            num_variants: Number of content variants
            temperature: Generation temperature
            
        Returns:
            List of AdContent objects
        """
        
        # Content type mapping
        content_type_map = {
            'social_media': 0, 'email': 1, 'banner': 2, 'video': 3,
            'blog': 4, 'newsletter': 5, 'landing_page': 6
        }
        
        # Audience mapping
        audience_map = {
            'general': 0, 'young_adults': 1, 'professionals': 2,
            'families': 3, 'seniors': 4, 'students': 5
        }
        
        content_type_id = torch.tensor([content_type_map.get(content_type, 0)])
        audience_id = torch.tensor([audience_map.get(target_audience, 0)])
        
        # Generate multiple variants
        ad_contents = []
        
        for i in range(num_variants):
            # This would implement the actual generation logic
            # For now, return placeholder content
            ad_content = AdContent(
                headline=f"Discover Amazing {content_type.replace('_', ' ').title()} - Variant {i+1}",
                subheadline="Experience the difference with our innovative solution",
                body_text="Transform your experience with our cutting-edge approach that delivers exceptional results.",
                call_to_action="Learn More Today",
                image_descriptions=[
                    "Hero image showing product in action",
                    "Lifestyle image with target audience",
                    "Close-up of key features"
                ],
                brand_alignment_score=0.85 + (i * 0.05),
                target_audience=target_audience,
                content_type=content_type,
                visual_suggestions={
                    'color_scheme': 'primary_brand_colors',
                    'typography': 'brand_fonts',
                    'layout': 'clean_modern',
                    'imagery_style': 'professional_lifestyle'
                }
            )
            ad_contents.append(ad_content)
        
        return ad_contents
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        brand_embedding: Optional[torch.Tensor] = None,
        content_type_ids: Optional[torch.Tensor] = None,
        audience_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Handle different input modes
        if images is not None and brand_embedding is None:
            # Extract brand embedding from images
            brand_analysis = self.analyze_website_brand(images, return_detailed=False)
            brand_embedding = brand_analysis['brand_embedding']
        
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            
            # Token embeddings
            token_embeds = self.token_embeddings(input_ids)
            
            # Position embeddings
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            position_embeds = self.position_embeddings(position_ids)
            
            # Combine embeddings
            hidden_states = token_embeds + position_embeds
            hidden_states = self.dropout(hidden_states)
            
            # Apply language model layers
            for layer in self.language_layers:
                hidden_states = layer(hidden_states, attention_mask)
            
            # Multi-modal fusion if brand embedding available
            if brand_embedding is not None:
                hidden_states = self.multimodal_fusion(hidden_states, brand_embedding)
            
            # Generate ad content
            if brand_embedding is not None:
                ad_outputs = self.ad_generator(
                    hidden_states, brand_embedding, 
                    content_type_ids, audience_ids
                )
                logits = ad_outputs['logits']
            else:
                # Standard language modeling
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
        else:
            # Image-only mode for brand analysis
            if images is not None:
                brand_analysis = self.analyze_website_brand(images)
                return brand_analysis
            else:
                raise ValueError("Must provide either input_ids or images")
        
        loss = None
        if labels is not None:
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss = lm_loss
            
            # Additional losses if in training mode
            if self.training and brand_embedding is not None:
                total_loss = lm_loss
                
                if self.config.use_brand_consistency_loss:
                    # Brand consistency loss would be implemented here
                    brand_loss = torch.tensor(0.0, device=lm_loss.device)
                    total_loss += self.config.brand_loss_weight * brand_loss
                
                if self.config.use_visual_alignment_loss:
                    # Visual alignment loss would be implemented here
                    visual_loss = torch.tensor(0.0, device=lm_loss.device)
                    total_loss += self.config.visual_loss_weight * visual_loss
                
                if self.config.use_content_quality_loss:
                    # Content quality loss would be implemented here
                    quality_loss = torch.tensor(0.0, device=lm_loss.device)
                    total_loss += self.config.quality_loss_weight * quality_loss
                
                loss = total_loss
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )
    
    def generate_with_brand_awareness(
        self,
        input_ids: torch.Tensor,
        brand_embedding: torch.Tensor,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        content_type: str = "social_media",
        target_audience: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content with brand awareness
        
        Args:
            input_ids: Input token IDs
            brand_embedding: Brand embedding from analysis
            max_length: Maximum generation length
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            content_type: Type of content to generate
            target_audience: Target audience
            
        Returns:
            Dictionary with generated content and metadata
        """
        
        self.eval()
        
        with torch.no_grad():
            # Content type and audience IDs
            content_type_map = {
                'social_media': 0, 'email': 1, 'banner': 2, 'video': 3,
                'blog': 4, 'newsletter': 5, 'landing_page': 6
            }
            audience_map = {
                'general': 0, 'young_adults': 1, 'professionals': 2,
                'families': 3, 'seniors': 4, 'students': 5
            }
            
            content_type_ids = torch.tensor([content_type_map.get(content_type, 0)])
            audience_ids = torch.tensor([audience_map.get(target_audience, 0)])
            
            # Generate content
            generated_ids = input_ids.clone()
            
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self.forward(
                    input_ids=generated_ids,
                    brand_embedding=brand_embedding,
                    content_type_ids=content_type_ids,
                    audience_ids=audience_ids
                )
                
                logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for end token
                if next_token.item() == self.config.eos_token_id:
                    break
            
            return {
                'generated_ids': generated_ids,
                'content_type': content_type,
                'target_audience': target_audience,
                'brand_alignment_score': 0.85  # Placeholder
            }


# Utility functions for web scraping and brand analysis
class WebsiteBrandExtractor:
    """Utility class for extracting brand information from websites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_colors_from_css(self, css_content: str) -> List[str]:
        """Extract color values from CSS content"""
        color_patterns = [
            r'#[0-9a-fA-F]{6}',  # Hex colors
            r'#[0-9a-fA-F]{3}',   # Short hex colors
            r'rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)',  # RGB colors
            r'rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\d.]+\s*\)',  # RGBA colors
        ]
        
        colors = []
        for pattern in color_patterns:
            matches = re.findall(pattern, css_content, re.IGNORECASE)
            colors.extend(matches)
        
        return list(set(colors))  # Remove duplicates
    
    def extract_fonts_from_css(self, css_content: str) -> List[str]:
        """Extract font families from CSS content"""
        font_pattern = r'font-family\s*:\s*([^;]+)'
        matches = re.findall(font_pattern, css_content, re.IGNORECASE)
        
        fonts = []
        for match in matches:
            # Clean up font names
            font_list = [font.strip().strip('"\'') for font in match.split(',')]
            fonts.extend(font_list)
        
        return list(set(fonts))
    
    def analyze_website_structure(self, html_content: str) -> Dict[str, Any]:
        """Analyze website structure and hierarchy"""
        # This would use BeautifulSoup or similar to analyze HTML structure
        # For now, return placeholder data
        return {
            'has_header': True,
            'has_navigation': True,
            'has_footer': True,
            'content_sections': 5,
            'heading_hierarchy': ['h1', 'h2', 'h3'],
            'layout_type': 'responsive'
        }


# Export main components
__all__ = [
    'BrandKitAdsConfig',
    'BrandKitAdsModel',
    'BrandKitExtraction',
    'AdContent',
    'WebsiteBrandExtractor'
]