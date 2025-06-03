#!/usr/bin/env python3
"""
Multi-Modal Transformer Implementation
Cross-modal understanding and generation without using DeepSeek.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
from dataclasses import dataclass
import torchvision.transforms as transforms
from torchvision.models import resnet50, ViT_B_16_Weights
import torchaudio
import librosa


@dataclass
class MultiModalTransformerConfig:
    """Configuration for Multi-Modal Transformer model."""
    vocab_size: int = 50257
    max_position_embeddings: int = 8192
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 16384
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # Multi-modal parameters
    modalities: List[str] = None  # ['text', 'vision', 'audio']
    
    # Vision parameters
    vision_encoder_type: str = "resnet"  # "resnet", "vit", "clip"
    image_size: int = 224
    vision_hidden_size: int = 2048
    num_vision_tokens: int = 196  # 14x14 for ViT
    vision_patch_size: int = 16
    
    # Audio parameters
    audio_encoder_type: str = "wav2vec"  # "wav2vec", "mel_spectrogram"
    audio_sample_rate: int = 16000
    audio_hidden_size: int = 768
    num_audio_tokens: int = 100
    mel_bins: int = 80
    
    # Cross-modal fusion
    use_cross_modal_attention: bool = True
    cross_modal_layers: List[int] = None  # Which layers to apply cross-modal attention
    fusion_method: str = "attention"  # "attention", "concat", "gating"
    
    # Unified embedding space
    use_unified_embedding: bool = True
    unified_embedding_dim: int = 4096
    
    # Modality-specific parameters
    use_modality_embeddings: bool = True
    use_modality_specific_layers: bool = False
    
    # Advanced features
    use_rotary_embeddings: bool = True
    use_rms_norm: bool = True
    use_pre_norm: bool = True


class VisionEncoder(nn.Module):
    """Vision encoder for processing images."""
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size
        
        if config.vision_encoder_type == "resnet":
            # ResNet-based encoder
            self.backbone = resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()  # Remove final classification layer
            
            # Spatial feature extraction
            self.spatial_pool = nn.AdaptiveAvgPool2d((14, 14))  # 14x14 spatial features
            self.feature_proj = nn.Linear(2048, config.hidden_size)
            
        elif config.vision_encoder_type == "vit":
            # Vision Transformer encoder
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.backbone.heads = nn.Identity()  # Remove classification head
            
            # Patch embeddings
            self.patch_embed = nn.Conv2d(
                3, config.hidden_size, 
                kernel_size=config.vision_patch_size, 
                stride=config.vision_patch_size
            )
            
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Position embeddings for vision tokens
        self.vision_pos_embed = nn.Parameter(
            torch.randn(1, config.num_vision_tokens, config.hidden_size)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process images and return vision tokens.
        
        Args:
            images: [batch_size, channels, height, width]
            
        Returns:
            vision_tokens: [batch_size, num_vision_tokens, hidden_size]
        """
        batch_size = images.shape[0]
        
        if self.config.vision_encoder_type == "resnet":
            # Extract features using ResNet
            with torch.no_grad():
                features = self.backbone(images)  # [batch_size, 2048]
            
            # Reshape to spatial features (assuming conv features before global pooling)
            # This is a simplified approach - in practice, you'd extract conv features
            features = features.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 2048, 1, 1]
            features = features.expand(-1, -1, 14, 14)  # [batch_size, 2048, 14, 14]
            
            # Flatten spatial dimensions
            features = features.flatten(2).transpose(1, 2)  # [batch_size, 196, 2048]
            
            # Project to hidden size
            vision_tokens = self.feature_proj(features)  # [batch_size, 196, hidden_size]
            
        elif self.config.vision_encoder_type == "vit":
            # Patch embedding
            patches = self.patch_embed(images)  # [batch_size, hidden_size, H/P, W/P]
            patches = patches.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_size]
            vision_tokens = patches
        
        # Add position embeddings
        vision_tokens = vision_tokens + self.vision_pos_embed[:, :vision_tokens.shape[1], :]
        
        return vision_tokens


class AudioEncoder(nn.Module):
    """Audio encoder for processing audio signals."""
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.audio_sample_rate
        self.hidden_size = config.hidden_size
        self.audio_hidden_size = config.audio_hidden_size
        
        if config.audio_encoder_type == "wav2vec":
            # Wav2Vec2-based encoder (simplified)
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 512, kernel_size=10, stride=5),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, stride=2),
                nn.ReLU(),
            )
            
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=config.hidden_dropout_prob
                ),
                num_layers=6
            )
            
            self.feature_proj = nn.Linear(512, config.hidden_size)
            
        elif config.audio_encoder_type == "mel_spectrogram":
            # Mel-spectrogram based encoder
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.audio_sample_rate,
                n_mels=config.mel_bins,
                n_fft=1024,
                hop_length=256
            )
            
            # CNN for mel-spectrogram processing
            self.conv_encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((config.num_audio_tokens, 1))
            )
            
            self.feature_proj = nn.Linear(256, config.hidden_size)
        
        # Position embeddings for audio tokens
        self.audio_pos_embed = nn.Parameter(
            torch.randn(1, config.num_audio_tokens, config.hidden_size)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process audio and return audio tokens.
        
        Args:
            audio: [batch_size, audio_length] or [batch_size, channels, audio_length]
            
        Returns:
            audio_tokens: [batch_size, num_audio_tokens, hidden_size]
        """
        batch_size = audio.shape[0]
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        if self.config.audio_encoder_type == "wav2vec":
            # Convolutional feature extraction
            features = self.conv_layers(audio)  # [batch_size, 512, seq_len]
            features = features.transpose(1, 2)  # [batch_size, seq_len, 512]
            
            # Transformer encoding
            features = self.transformer_encoder(features.transpose(0, 1))  # [seq_len, batch_size, 512]
            features = features.transpose(0, 1)  # [batch_size, seq_len, 512]
            
            # Subsample to fixed number of tokens
            if features.shape[1] > self.config.num_audio_tokens:
                indices = torch.linspace(0, features.shape[1] - 1, self.config.num_audio_tokens, dtype=torch.long)
                features = features[:, indices, :]
            elif features.shape[1] < self.config.num_audio_tokens:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, 
                    self.config.num_audio_tokens - features.shape[1], 
                    features.shape[2],
                    device=features.device
                )
                features = torch.cat([features, padding], dim=1)
            
            # Project to hidden size
            audio_tokens = self.feature_proj(features)
            
        elif self.config.audio_encoder_type == "mel_spectrogram":
            # Convert to mel-spectrogram
            mel_spec = self.mel_transform(audio.squeeze(1))  # [batch_size, mel_bins, time]
            mel_spec = mel_spec.unsqueeze(1)  # [batch_size, 1, mel_bins, time]
            
            # CNN encoding
            features = self.conv_encoder(mel_spec)  # [batch_size, 256, num_audio_tokens, 1]
            features = features.squeeze(-1).transpose(1, 2)  # [batch_size, num_audio_tokens, 256]
            
            # Project to hidden size
            audio_tokens = self.feature_proj(features)
        
        # Add position embeddings
        audio_tokens = audio_tokens + self.audio_pos_embed
        
        return audio_tokens


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Attention projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(
        self, 
        query_states: torch.Tensor, 
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cross-modal attention between different modalities."""
        batch_size, query_len, _ = query_states.shape
        _, kv_len, _ = key_value_states.shape
        
        # Project to Q, K, V
        queries = self.q_proj(query_states)
        keys = self.k_proj(key_value_states)
        values = self.v_proj(key_value_states)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, values)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, query_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class ModalityFusion(nn.Module):
    """Fusion mechanism for combining multiple modalities."""
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        
        if config.fusion_method == "attention":
            self.cross_modal_attention = CrossModalAttention(config)
        elif config.fusion_method == "gating":
            self.gate_network = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1),
                nn.Sigmoid()
            )
        elif config.fusion_method == "concat":
            self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self, 
        primary_features: torch.Tensor, 
        secondary_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse features from different modalities."""
        
        if self.fusion_method == "attention":
            # Use cross-modal attention
            fused_features = self.cross_modal_attention(primary_features, secondary_features)
            return primary_features + fused_features
            
        elif self.fusion_method == "gating":
            # Gated fusion
            combined = torch.cat([primary_features, secondary_features], dim=-1)
            gate = self.gate_network(combined)
            return gate * primary_features + (1 - gate) * secondary_features
            
        elif self.fusion_method == "concat":
            # Concatenation and projection
            combined = torch.cat([primary_features, secondary_features], dim=-1)
            return self.fusion_proj(combined)
        
        else:
            # Simple addition
            return primary_features + secondary_features


class MultiModalTransformerLayer(nn.Module):
    """Transformer layer with multi-modal capabilities."""
    
    def __init__(self, config: MultiModalTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Self attention
        from ..native_transformer.model import AdaptiveAttention
        self.self_attn = AdaptiveAttention(config)
        
        # MLP
        from ..native_transformer.model import NativeMLP
        self.mlp = NativeMLP(config)
        
        # Cross-modal components (only for specified layers)
        self.use_cross_modal = (
            config.cross_modal_layers is None or 
            layer_idx in config.cross_modal_layers
        )
        
        if self.use_cross_modal and config.use_cross_modal_attention:
            self.modality_fusion = ModalityFusion(config)
        
        # Normalization layers
        if config.use_rms_norm:
            from ..native_transformer.model import RMSNorm
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if self.use_cross_modal:
                self.post_cross_modal_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if self.use_cross_modal:
                self.post_cross_modal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        residual = hidden_states
        
        # Pre-norm
        if self.config.use_pre_norm:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]
        
        # Residual connection
        hidden_states = residual + attn_output
        
        if not self.config.use_pre_norm:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Cross-modal fusion
        cross_modal_info = {}
        if self.use_cross_modal and self.config.use_cross_modal_attention:
            cross_modal_residual = hidden_states
            
            if self.config.use_pre_norm:
                hidden_states = self.post_cross_modal_layernorm(hidden_states)
            
            # Fuse with vision features
            if vision_features is not None:
                hidden_states = self.modality_fusion(hidden_states, vision_features)
                cross_modal_info['vision_fused'] = True
            
            # Fuse with audio features
            if audio_features is not None:
                hidden_states = self.modality_fusion(hidden_states, audio_features)
                cross_modal_info['audio_fused'] = True
            
            # Residual connection
            hidden_states = cross_modal_residual + hidden_states
            
            if not self.config.use_pre_norm:
                hidden_states = self.post_cross_modal_layernorm(hidden_states)
        
        # MLP
        residual = hidden_states
        if self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + mlp_output
        
        if not self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        outputs = (hidden_states, cross_modal_info)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[-1],)
        
        return outputs


class MultiModalTransformerModel(nn.Module):
    """Multi-Modal Transformer Model."""
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size - 1
        self.vocab_size = config.vocab_size
        
        # Text embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Position embeddings (if not using RoPE)
        if not config.use_rotary_embeddings:
            self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Modality encoders
        self.modalities = config.modalities or ['text']
        
        if 'vision' in self.modalities:
            self.vision_encoder = VisionEncoder(config)
        
        if 'audio' in self.modalities:
            self.audio_encoder = AudioEncoder(config)
        
        # Modality embeddings
        if config.use_modality_embeddings:
            self.modality_embeddings = nn.Embedding(len(self.modalities), config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MultiModalTransformerLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        if config.use_rms_norm:
            from ..native_transformer.model import RMSNorm
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        
        # Process text input
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Add position embeddings if not using RoPE
        if not self.config.use_rotary_embeddings:
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            position_embeddings = self.embed_positions(position_ids)
            hidden_states = inputs_embeds + position_embeddings
        else:
            hidden_states = inputs_embeds
        
        # Add modality embeddings
        if self.config.use_modality_embeddings:
            text_modality_id = self.modalities.index('text')
            modality_embedding = self.modality_embeddings(
                torch.full((batch_size, seq_length), text_modality_id, device=hidden_states.device)
            )
            hidden_states = hidden_states + modality_embedding
        
        # Process other modalities
        vision_features = None
        audio_features = None
        
        if 'vision' in self.modalities and images is not None:
            vision_features = self.vision_encoder(images)
            
            # Add modality embeddings to vision features
            if self.config.use_modality_embeddings:
                vision_modality_id = self.modalities.index('vision')
                vision_modality_embedding = self.modality_embeddings(
                    torch.full(
                        (vision_features.shape[0], vision_features.shape[1]), 
                        vision_modality_id, 
                        device=vision_features.device
                    )
                )
                vision_features = vision_features + vision_modality_embedding
        
        if 'audio' in self.modalities and audio is not None:
            audio_features = self.audio_encoder(audio)
            
            # Add modality embeddings to audio features
            if self.config.use_modality_embeddings:
                audio_modality_id = self.modalities.index('audio')
                audio_modality_embedding = self.modality_embeddings(
                    torch.full(
                        (audio_features.shape[0], audio_features.shape[1]), 
                        audio_modality_id, 
                        device=audio_features.device
                    )
                )
                audio_features = audio_features + audio_modality_embedding
        
        # Transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_modal_info = []
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                vision_features=vision_features,
                audio_features=audio_features,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            cross_modal_info = layer_outputs[1]
            all_cross_modal_info.append(cross_modal_info)
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[2] if len(layer_outputs) > 2 else None,)
        
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "cross_modal_info": all_cross_modal_info,
            "vision_features": vision_features,
            "audio_features": audio_features,
        }


class MultiModalTransformerForCausalLM(nn.Module):
    """Multi-Modal Transformer Model for Causal Language Modeling."""
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        self.model = MultiModalTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        
        # Decoder outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            images=images,
            audio=audio,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "cross_modal_info": outputs.get("cross_modal_info", []),
            "vision_features": outputs.get("vision_features"),
            "audio_features": outputs.get("audio_features"),
        }