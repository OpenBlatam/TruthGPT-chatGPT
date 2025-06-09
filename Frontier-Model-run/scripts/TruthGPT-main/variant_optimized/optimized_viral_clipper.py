"""
Optimized viral video clipper with enhanced performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from .performance_utils import OptimizedAttention, OptimizedMLP, GradientCheckpointingWrapper

@dataclass
class OptimizedViralClipperArgs:
    """Configuration for optimized viral video clipper model."""
    hidden_size: int = 512
    num_layers: int = 6
    num_attention_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 1000
    
    max_duration: int = 3600
    clip_duration: int = 30
    min_clip_duration: int = 10
    max_clip_duration: int = 60
    
    engagement_threshold: float = 0.8
    view_velocity_threshold: int = 1000
    comment_ratio_threshold: float = 0.05
    like_ratio_threshold: float = 0.1
    
    visual_feature_dim: int = 2048
    audio_feature_dim: int = 512
    text_feature_dim: int = 768
    engagement_feature_dim: int = 64
    
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_efficient_fusion: bool = True
    use_streaming_inference: bool = True

class OptimizedMultiModalEncoder(nn.Module):
    """Optimized multi-modal encoder with efficient feature fusion."""
    
    def __init__(self, args: OptimizedViralClipperArgs):
        super().__init__()
        self.args = args
        
        self.visual_proj = nn.Sequential(
            nn.Linear(args.visual_feature_dim, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(args.audio_feature_dim, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(args.text_feature_dim, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        self.engagement_proj = nn.Sequential(
            nn.Linear(args.engagement_feature_dim, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        if args.use_efficient_fusion:
            self.fusion_attention = OptimizedAttention(
                args.hidden_size, args.num_attention_heads, args.dropout, args.use_flash_attention
            )
        
        self.pos_encoding = nn.Parameter(
            torch.randn(args.max_sequence_length, args.hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, visual_features, audio_features, text_features, engagement_features):
        batch_size, seq_len = visual_features.shape[:2]
        
        visual_emb = self.visual_proj(visual_features)
        audio_emb = self.audio_proj(audio_features)
        text_emb = self.text_proj(text_features)
        engagement_emb = self.engagement_proj(engagement_features)
        
        if self.args.use_efficient_fusion:
            modalities = torch.stack([visual_emb, audio_emb, text_emb, engagement_emb], dim=2)
            modalities = modalities.view(batch_size * seq_len, 4, self.args.hidden_size)
            fused = self.fusion_attention(modalities).mean(dim=1)
            combined = fused.view(batch_size, seq_len, self.args.hidden_size)
        else:
            combined = visual_emb + audio_emb + text_emb + engagement_emb
        
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        combined = combined + pos_enc
        
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        return combined

class OptimizedViralDetectorLayer(nn.Module):
    """Optimized transformer layer for viral content detection."""
    
    def __init__(self, args: OptimizedViralClipperArgs):
        super().__init__()
        self.attention = OptimizedAttention(
            args.hidden_size, args.num_attention_heads, args.dropout, args.use_flash_attention
        )
        self.feed_forward = OptimizedMLP(
            args.hidden_size, args.hidden_size * 4, args.dropout, args.use_gradient_checkpointing
        )
        self.layer_norm1 = nn.LayerNorm(args.hidden_size)
        self.layer_norm2 = nn.LayerNorm(args.hidden_size)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.layer_norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x

class StreamingInferenceBuffer:
    """Buffer for streaming inference on long video sequences."""
    
    def __init__(self, buffer_size: int = 100, overlap: int = 10):
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.buffer = []
        self.results = []
        
    def add_chunk(self, chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """Add a chunk and return processable segment if buffer is full."""
        self.buffer.append(chunk)
        
        if len(self.buffer) >= self.buffer_size:
            segment = torch.cat(self.buffer, dim=1)
            self.buffer = self.buffer[-self.overlap:]
            return segment
        
        return None
    
    def flush(self) -> Optional[torch.Tensor]:
        """Flush remaining buffer contents."""
        if self.buffer:
            segment = torch.cat(self.buffer, dim=1)
            self.buffer = []
            return segment
        return None

class OptimizedViralClipper(nn.Module):
    """Optimized viral video clipper model with streaming capabilities."""
    
    def __init__(self, args: OptimizedViralClipperArgs):
        super().__init__()
        self.args = args
        
        self.encoder = OptimizedMultiModalEncoder(args)
        
        self.layers = nn.ModuleList([
            GradientCheckpointingWrapper(
                OptimizedViralDetectorLayer(args),
                args.use_gradient_checkpointing
            ) for _ in range(args.num_layers)
        ])
        
        self.virality_head = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.segment_head = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size // 2, 2)
        )
        
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        
        if args.use_streaming_inference:
            self.streaming_buffer = StreamingInferenceBuffer()
        
    def forward(self, visual_features, audio_features, text_features, engagement_features, mask=None):
        x = self.encoder(visual_features, audio_features, text_features, engagement_features)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.layer_norm(x)
        
        virality_scores = self.virality_head(x)
        segment_logits = self.segment_head(x)
        
        return {
            'virality_scores': virality_scores,
            'segment_logits': segment_logits,
            'hidden_states': x
        }
    
    def predict_viral_segments_streaming(self, feature_stream, threshold=None):
        """Predict viral segments using streaming inference."""
        if not self.args.use_streaming_inference:
            raise ValueError("Streaming inference not enabled")
        
        if threshold is None:
            threshold = self.args.engagement_threshold
        
        all_segments = []
        
        for features in feature_stream:
            chunk_to_process = self.streaming_buffer.add_chunk(features)
            
            if chunk_to_process is not None:
                with torch.no_grad():
                    outputs = self.forward(**chunk_to_process)
                    segments = self._extract_segments(outputs, threshold)
                    all_segments.extend(segments)
        
        final_chunk = self.streaming_buffer.flush()
        if final_chunk is not None:
            with torch.no_grad():
                outputs = self.forward(**final_chunk)
                segments = self._extract_segments(outputs, threshold)
                all_segments.extend(segments)
        
        return all_segments
    
    def _extract_segments(self, outputs, threshold):
        """Extract viral segments from model outputs."""
        virality_scores = outputs['virality_scores'].squeeze(-1)
        segment_logits = outputs['segment_logits']
        
        viral_mask = virality_scores > threshold
        start_probs = F.softmax(segment_logits[:, :, 0], dim=-1)
        end_probs = F.softmax(segment_logits[:, :, 1], dim=-1)
        
        segments = []
        for batch_idx in range(virality_scores.shape[0]):
            batch_viral = viral_mask[batch_idx]
            batch_scores = virality_scores[batch_idx]
            
            viral_indices = torch.where(batch_viral)[0]
            if len(viral_indices) > 0:
                segment_groups = self._group_consecutive_indices(viral_indices)
                
                for group in segment_groups:
                    start_idx = group[0]
                    end_idx = group[-1]
                    avg_score = batch_scores[start_idx:end_idx+1].mean().item()
                    segments.append((start_idx, end_idx, avg_score))
        
        return segments
    
    def _group_consecutive_indices(self, indices):
        """Group consecutive indices into segments."""
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices[0].item()]
        
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] == 1:
                current_group.append(indices[i].item())
            else:
                groups.append(current_group)
                current_group = [indices[i].item()]
        groups.append(current_group)
        
        return groups
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the model."""
        total_params = sum(p.numel() for p in self.parameters())
        total_size_mb = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 / 1024
        
        metrics = {
            'total_parameters': total_params,
            'model_size_mb': total_size_mb,
            'supports_streaming': self.args.use_streaming_inference,
            'uses_flash_attention': self.args.use_flash_attention,
            'uses_gradient_checkpointing': self.args.use_gradient_checkpointing,
            'max_sequence_length': self.args.max_sequence_length
        }
        
        return metrics

def create_optimized_viral_clipper_model(config: Dict[str, Any]) -> OptimizedViralClipper:
    """Create an optimized viral video clipper model from configuration."""
    args = OptimizedViralClipperArgs(
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 6),
        num_attention_heads=config.get('num_attention_heads', 8),
        dropout=config.get('dropout', 0.1),
        max_sequence_length=config.get('max_sequence_length', 1000),
        
        max_duration=config.get('max_duration', 3600),
        clip_duration=config.get('clip_duration', 30),
        min_clip_duration=config.get('min_clip_duration', 10),
        max_clip_duration=config.get('max_clip_duration', 60),
        
        engagement_threshold=config.get('engagement_threshold', 0.8),
        view_velocity_threshold=config.get('view_velocity_threshold', 1000),
        comment_ratio_threshold=config.get('comment_ratio_threshold', 0.05),
        like_ratio_threshold=config.get('like_ratio_threshold', 0.1),
        
        visual_feature_dim=config.get('visual_feature_dim', 2048),
        audio_feature_dim=config.get('audio_feature_dim', 512),
        text_feature_dim=config.get('text_feature_dim', 768),
        engagement_feature_dim=config.get('engagement_feature_dim', 64),
        
        use_flash_attention=config.get('use_flash_attention', True),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', True),
        use_efficient_fusion=config.get('use_efficient_fusion', True),
        use_streaming_inference=config.get('use_streaming_inference', True)
    )
    
    return OptimizedViralClipper(args)
