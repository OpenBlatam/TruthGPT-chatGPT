"""
Viral Video Clips Model - Native Implementation

A revolutionary AI model that extracts YouTube videos, analyzes content, and automatically
creates viral short-form clips with intelligent editing, captions, logos, animated subtitles,
and viral effects.

Key Features:
- YouTube video extraction and analysis
- Intelligent scene detection and highlight identification
- Automatic clip generation (15-60 seconds optimal for viral content)
- Dynamic caption generation with emotion and context awareness
- Logo placement and branding integration
- Animated subtitle generation with motion tracking
- Viral effect application (zoom, transitions, overlays)
- Multi-platform optimization (TikTok, Instagram Reels, YouTube Shorts)
- Engagement prediction and optimization
- Trend analysis and viral pattern recognition

Architecture:
- Video Understanding Transformer for scene analysis
- Audio Processing Module for speech and music analysis
- Highlight Detection Network for viral moment identification
- Caption Generation Model with emotion and context awareness
- Visual Effects Engine for transitions and animations
- Brand Integration Module for logo and watermark placement
- Viral Pattern Analyzer for trend-based optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import cv2
import numpy as np
import librosa
import whisper
import yt_dlp
import moviepy.editor as mp
from moviepy.video.fx import resize, fadein, fadeout
from moviepy.video.tools.subtitles import SubtitlesClip
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import json
import yaml
import os
import re
import time
import requests
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import subprocess
import hashlib
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiohttp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import face_recognition
import mediapipe as mp_pose
import tensorflow as tf


@dataclass
class ViralVideoClipsConfig:
    """Configuration for Viral Video Clips Model"""
    
    # Model architecture
    model_size: str = "medium"
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    
    # Video processing
    video_resolution: Tuple[int, int] = (1080, 1920)  # Vertical for mobile
    target_fps: int = 30
    max_video_duration: int = 3600  # 1 hour max
    clip_duration_range: Tuple[int, int] = (15, 60)  # 15-60 seconds
    num_clips_to_generate: int = 15
    
    # Audio processing
    audio_sample_rate: int = 16000
    audio_chunk_duration: float = 30.0
    speech_detection_threshold: float = 0.5
    music_detection_threshold: float = 0.3
    
    # Visual analysis
    scene_change_threshold: float = 0.3
    face_detection_confidence: float = 0.5
    object_detection_confidence: float = 0.6
    motion_detection_sensitivity: float = 0.4
    
    # Caption generation
    caption_max_length: int = 100
    caption_words_per_second: float = 3.0
    caption_font_size: int = 48
    caption_animation_duration: float = 0.5
    
    # Viral optimization
    engagement_prediction_threshold: float = 0.7
    trend_analysis_window_days: int = 7
    viral_pattern_confidence: float = 0.8
    platform_optimization: List[str] = field(default_factory=lambda: ["tiktok", "instagram", "youtube_shorts"])
    
    # Effects and transitions
    transition_duration: float = 0.3
    zoom_intensity_range: Tuple[float, float] = (1.1, 1.5)
    effect_probability: float = 0.6
    logo_opacity: float = 0.8
    watermark_position: str = "bottom_right"
    
    # Performance optimization
    use_gpu_acceleration: bool = True
    max_concurrent_processes: int = 4
    cache_extracted_features: bool = True
    temp_dir: str = "/tmp/viral_clips"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ViralVideoClipsConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('model', {}))


@dataclass
class VideoClip:
    """Represents a generated viral video clip"""
    
    clip_id: str
    start_time: float
    end_time: float
    duration: float
    title: str
    description: str
    captions: List[Dict[str, Any]]
    viral_score: float
    engagement_prediction: float
    platform_optimizations: Dict[str, Any]
    effects_applied: List[str]
    audio_features: Dict[str, Any]
    visual_features: Dict[str, Any]
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoAnalysis:
    """Complete analysis of source video"""
    
    video_id: str
    title: str
    duration: float
    resolution: Tuple[int, int]
    fps: int
    audio_features: Dict[str, Any]
    visual_features: Dict[str, Any]
    scene_changes: List[float]
    highlight_moments: List[Dict[str, Any]]
    speech_segments: List[Dict[str, Any]]
    face_detections: List[Dict[str, Any]]
    object_detections: List[Dict[str, Any]]
    motion_analysis: Dict[str, Any]
    viral_potential: float
    trending_topics: List[str]
    recommended_clips: List[Dict[str, Any]]


class VideoUnderstandingTransformer(nn.Module):
    """Transformer model for video understanding and analysis"""
    
    def __init__(self, config: ViralVideoClipsConfig):
        super().__init__()
        self.config = config
        
        # Video feature extraction
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.AdaptiveAvgPool3d((8, 7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 8 * 7 * 7, config.hidden_size)
        )
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            VideoTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Scene change detection
        self.scene_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Highlight detection
        self.highlight_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Viral potential predictor
        self.viral_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, video_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for video understanding
        
        Args:
            video_frames: (batch_size, channels, time, height, width)
        
        Returns:
            Dictionary with analysis results
        """
        batch_size, channels, time_steps, height, width = video_frames.shape
        
        # Extract video features
        video_features = self.video_encoder(video_frames)  # (batch_size, hidden_size)
        
        # Reshape for temporal processing
        video_features = video_features.unsqueeze(1).repeat(1, time_steps, 1)  # (batch_size, time_steps, hidden_size)
        
        # Apply temporal attention layers
        hidden_states = video_features
        for layer in self.temporal_layers:
            hidden_states = layer(hidden_states)
        
        # Detect scene changes
        scene_changes = self.scene_detector(hidden_states)  # (batch_size, time_steps, 1)
        
        # Detect highlights
        highlights = self.highlight_detector(hidden_states)  # (batch_size, time_steps, 1)
        
        # Predict viral potential
        pooled_features = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        viral_potential = self.viral_predictor(pooled_features)  # (batch_size, 1)
        
        return {
            'video_features': hidden_states,
            'scene_changes': scene_changes.squeeze(-1),
            'highlights': highlights.squeeze(-1),
            'viral_potential': viral_potential.squeeze(-1),
            'pooled_features': pooled_features
        }


class VideoTransformerLayer(nn.Module):
    """Single transformer layer for video processing"""
    
    def __init__(self, config: ViralVideoClipsConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(0.1)
        )
        
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attn_output
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output
        
        return hidden_states


class AudioProcessingModule(nn.Module):
    """Module for audio analysis and processing"""
    
    def __init__(self, config: ViralVideoClipsConfig):
        super().__init__()
        self.config = config
        
        # Audio feature extraction
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100),
            
            nn.Flatten(),
            nn.Linear(256 * 100, config.hidden_size)
        )
        
        # Speech detection
        self.speech_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Music detection
        self.music_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Emotion detection
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size // 2, 8)  # 8 emotions
        )
        
        # Initialize Whisper for speech recognition
        self.whisper_model = None
    
    def forward(self, audio_waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process audio and extract features
        
        Args:
            audio_waveform: (batch_size, audio_length)
        
        Returns:
            Dictionary with audio analysis results
        """
        # Add channel dimension if needed
        if audio_waveform.dim() == 2:
            audio_waveform = audio_waveform.unsqueeze(1)  # (batch_size, 1, audio_length)
        
        # Extract audio features
        audio_features = self.audio_encoder(audio_waveform)  # (batch_size, hidden_size)
        
        # Detect speech
        speech_probability = self.speech_detector(audio_features)
        
        # Detect music
        music_probability = self.music_detector(audio_features)
        
        # Classify emotions
        emotion_logits = self.emotion_classifier(audio_features)
        emotion_probabilities = F.softmax(emotion_logits, dim=-1)
        
        return {
            'audio_features': audio_features,
            'speech_probability': speech_probability.squeeze(-1),
            'music_probability': music_probability.squeeze(-1),
            'emotion_probabilities': emotion_probabilities
        }
    
    def transcribe_speech(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe speech using Whisper"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base")
        
        result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
        
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'confidence': segment.get('avg_logprob', 0.0),
                'words': segment.get('words', [])
            })
        
        return segments


class HighlightDetectionNetwork(nn.Module):
    """Network for detecting viral highlights in videos"""
    
    def __init__(self, config: ViralVideoClipsConfig):
        super().__init__()
        self.config = config
        
        # Multi-modal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Highlight scoring
        self.highlight_scorer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Viral pattern detector
        self.viral_pattern_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size // 2, 10),  # 10 viral patterns
            nn.Sigmoid()
        )
        
        # Engagement predictor
        self.engagement_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        video_features: torch.Tensor, 
        audio_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Detect highlights using multi-modal features
        
        Args:
            video_features: (batch_size, sequence_length, hidden_size)
            audio_features: (batch_size, hidden_size)
        
        Returns:
            Dictionary with highlight detection results
        """
        batch_size, seq_len, hidden_size = video_features.shape
        
        # Expand audio features to match video sequence length
        audio_features_expanded = audio_features.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Fuse video and audio features
        fused_features = torch.cat([video_features, audio_features_expanded], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        
        # Score highlights
        highlight_scores = self.highlight_scorer(fused_features).squeeze(-1)
        
        # Detect viral patterns
        pooled_features = fused_features.mean(dim=1)
        viral_patterns = self.viral_pattern_detector(pooled_features)
        
        # Predict engagement
        engagement_scores = self.engagement_predictor(pooled_features).squeeze(-1)
        
        return {
            'highlight_scores': highlight_scores,
            'viral_patterns': viral_patterns,
            'engagement_scores': engagement_scores,
            'fused_features': fused_features
        }


class CaptionGenerationModel(nn.Module):
    """Model for generating dynamic captions with emotion and context"""
    
    def __init__(self, config: ViralVideoClipsConfig):
        super().__init__()
        self.config = config
        
        # Caption generation transformer
        self.caption_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Emotion-aware caption head
        self.caption_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 50000)  # Vocabulary size
        )
        
        # Caption style classifier
        self.style_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 5)  # 5 caption styles
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        target_captions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate captions based on video and audio context
        
        Args:
            video_features: (batch_size, sequence_length, hidden_size)
            audio_features: (batch_size, hidden_size)
            target_captions: (batch_size, caption_length) for training
        
        Returns:
            Dictionary with caption generation results
        """
        batch_size, seq_len, hidden_size = video_features.shape
        
        # Encode context
        audio_expanded = audio_features.unsqueeze(1).repeat(1, seq_len, 1)
        context_features = torch.cat([video_features, audio_expanded], dim=-1)
        context_encoded = self.context_encoder(context_features)
        
        # Generate captions
        if target_captions is not None:
            # Training mode
            caption_embeddings = self._embed_captions(target_captions)
            caption_output = self.caption_transformer(
                caption_embeddings,
                context_encoded
            )
        else:
            # Inference mode
            caption_output = self._generate_captions(context_encoded)
        
        # Generate caption logits
        caption_logits = self.caption_head(caption_output)
        
        # Classify caption style
        pooled_context = context_encoded.mean(dim=1)
        style_logits = self.style_classifier(pooled_context)
        
        return {
            'caption_logits': caption_logits,
            'style_logits': style_logits,
            'context_features': context_encoded
        }
    
    def _embed_captions(self, captions: torch.Tensor) -> torch.Tensor:
        """Embed caption tokens"""
        # Simple embedding (would use proper tokenizer in practice)
        embedding = nn.Embedding(50000, self.config.hidden_size)
        return embedding(captions)
    
    def _generate_captions(self, context: torch.Tensor) -> torch.Tensor:
        """Generate captions autoregressively"""
        batch_size, seq_len, hidden_size = context.shape
        max_caption_length = self.config.caption_max_length
        
        # Start with special token
        generated = torch.zeros(batch_size, 1, hidden_size, device=context.device)
        
        for _ in range(max_caption_length):
            output = self.caption_transformer(generated, context)
            generated = torch.cat([generated, output[:, -1:, :]], dim=1)
        
        return generated


class VisualEffectsEngine(nn.Module):
    """Engine for applying visual effects and transitions"""
    
    def __init__(self, config: ViralVideoClipsConfig):
        super().__init__()
        self.config = config
        
        # Effect parameter predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 20)  # 20 effect parameters
        )
        
        # Transition type classifier
        self.transition_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 8)  # 8 transition types
        )
        
        # Zoom intensity predictor
        self.zoom_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict visual effects parameters
        
        Args:
            features: (batch_size, hidden_size)
        
        Returns:
            Dictionary with effect parameters
        """
        effect_params = self.effect_predictor(features)
        transition_logits = self.transition_classifier(features)
        zoom_intensity = self.zoom_predictor(features)
        
        return {
            'effect_parameters': effect_params,
            'transition_logits': transition_logits,
            'zoom_intensity': zoom_intensity
        }


class ViralVideoClipsModel(nn.Module):
    """Main model for viral video clips generation"""
    
    def __init__(self, config: ViralVideoClipsConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.video_transformer = VideoUnderstandingTransformer(config)
        self.audio_processor = AudioProcessingModule(config)
        self.highlight_detector = HighlightDetectionNetwork(config)
        self.caption_generator = CaptionGenerationModel(config)
        self.effects_engine = VisualEffectsEngine(config)
        
        # YouTube downloader
        self.youtube_downloader = None
        
        # Video processing tools
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Face detection
        self.face_detector = None
        
        # Pose detection
        self.pose_detector = None
    
    def extract_youtube_video(self, url: str) -> str:
        """Extract video from YouTube URL"""
        if self.youtube_downloader is None:
            self.youtube_downloader = yt_dlp.YoutubeDL({
                'format': 'best[height<=1080]',
                'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
                'writesubtitles': True,
                'writeautomaticsub': True,
            })
        
        try:
            info = self.youtube_downloader.extract_info(url, download=True)
            video_path = self.youtube_downloader.prepare_filename(info)
            return video_path
        except Exception as e:
            logging.error(f"Error extracting YouTube video: {e}")
            return None
    
    def analyze_video(self, video_path: str) -> VideoAnalysis:
        """Comprehensive video analysis"""
        
        # Load video
        video = mp.VideoFileClip(video_path)
        
        # Extract basic info
        video_info = {
            'duration': video.duration,
            'fps': video.fps,
            'resolution': (video.w, video.h)
        }
        
        # Extract frames for analysis
        frames = self._extract_frames(video, max_frames=100)
        video_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0)
        
        # Extract audio
        audio_path = str(self.temp_dir / "temp_audio.wav")
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        # Analyze video
        with torch.no_grad():
            video_analysis = self.video_transformer(video_tensor)
            
            # Analyze audio
            audio_waveform, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)
            audio_tensor = torch.from_numpy(audio_waveform).float().unsqueeze(0)
            audio_analysis = self.audio_processor(audio_tensor)
            
            # Detect highlights
            highlight_analysis = self.highlight_detector(
                video_analysis['video_features'],
                audio_analysis['audio_features']
            )
        
        # Transcribe speech
        speech_segments = self.audio_processor.transcribe_speech(audio_path)
        
        # Detect faces and objects
        face_detections = self._detect_faces(frames)
        object_detections = self._detect_objects(frames)
        
        # Analyze motion
        motion_analysis = self._analyze_motion(frames)
        
        # Detect scene changes
        scene_changes = self._detect_scene_changes(
            video_analysis['scene_changes'].numpy(),
            video_info['duration']
        )
        
        # Find highlight moments
        highlight_moments = self._find_highlight_moments(
            highlight_analysis['highlight_scores'].numpy(),
            video_info['duration']
        )
        
        # Calculate viral potential
        viral_potential = float(video_analysis['viral_potential'].item())
        
        # Clean up
        video.close()
        os.remove(audio_path)
        
        return VideoAnalysis(
            video_id=self._generate_video_id(video_path),
            title=Path(video_path).stem,
            duration=video_info['duration'],
            resolution=video_info['resolution'],
            fps=video_info['fps'],
            audio_features=self._tensor_to_dict(audio_analysis),
            visual_features=self._tensor_to_dict(video_analysis),
            scene_changes=scene_changes,
            highlight_moments=highlight_moments,
            speech_segments=speech_segments,
            face_detections=face_detections,
            object_detections=object_detections,
            motion_analysis=motion_analysis,
            viral_potential=viral_potential,
            trending_topics=self._extract_trending_topics(speech_segments),
            recommended_clips=self._recommend_clips(highlight_moments, video_info['duration'])
        )
    
    def generate_viral_clips(
        self,
        video_path: str,
        video_analysis: VideoAnalysis,
        num_clips: Optional[int] = None
    ) -> List[VideoClip]:
        """Generate viral clips from analyzed video"""
        
        if num_clips is None:
            num_clips = self.config.num_clips_to_generate
        
        # Load video
        video = mp.VideoFileClip(video_path)
        
        # Select best moments for clips
        clip_moments = self._select_clip_moments(video_analysis, num_clips)
        
        clips = []
        for i, moment in enumerate(clip_moments):
            clip = self._create_viral_clip(
                video=video,
                start_time=moment['start'],
                end_time=moment['end'],
                clip_index=i,
                video_analysis=video_analysis
            )
            clips.append(clip)
        
        video.close()
        return clips
    
    def _create_viral_clip(
        self,
        video: mp.VideoFileClip,
        start_time: float,
        end_time: float,
        clip_index: int,
        video_analysis: VideoAnalysis
    ) -> VideoClip:
        """Create a single viral clip with all effects"""
        
        # Extract clip
        clip_video = video.subclip(start_time, end_time)
        
        # Resize for mobile (vertical format)
        if clip_video.w > clip_video.h:
            # Landscape to portrait conversion
            clip_video = clip_video.resize(height=self.config.video_resolution[1])
            clip_video = clip_video.crop(
                x_center=clip_video.w/2,
                width=self.config.video_resolution[0]
            )
        else:
            clip_video = clip_video.resize(self.config.video_resolution)
        
        # Generate captions
        captions = self._generate_clip_captions(
            clip_video, start_time, end_time, video_analysis
        )
        
        # Add captions to video
        if captions:
            clip_video = self._add_captions_to_video(clip_video, captions)
        
        # Apply visual effects
        clip_video = self._apply_visual_effects(clip_video, video_analysis)
        
        # Add logo/watermark
        clip_video = self._add_logo(clip_video)
        
        # Add transitions
        clip_video = self._add_transitions(clip_video)
        
        # Calculate viral score
        viral_score = self._calculate_viral_score(
            start_time, end_time, video_analysis
        )
        
        # Generate metadata
        clip_id = f"{video_analysis.video_id}_clip_{clip_index:02d}"
        title = self._generate_clip_title(captions, video_analysis)
        description = self._generate_clip_description(captions, video_analysis)
        
        # Save clip
        clip_path = str(self.temp_dir / f"{clip_id}.mp4")
        clip_video.write_videofile(
            clip_path,
            fps=self.config.target_fps,
            verbose=False,
            logger=None
        )
        
        # Generate thumbnail
        thumbnail_path = str(self.temp_dir / f"{clip_id}_thumb.jpg")
        clip_video.save_frame(thumbnail_path, t=clip_video.duration/2)
        
        clip_video.close()
        
        return VideoClip(
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            title=title,
            description=description,
            captions=captions,
            viral_score=viral_score,
            engagement_prediction=viral_score * 0.9,  # Simplified
            platform_optimizations=self._generate_platform_optimizations(),
            effects_applied=["resize", "captions", "logo", "transitions"],
            audio_features={},
            visual_features={},
            file_path=clip_path,
            thumbnail_path=thumbnail_path,
            metadata={
                'original_video': video_analysis.video_id,
                'clip_index': clip_index,
                'generation_time': datetime.now().isoformat()
            }
        )
    
    def _extract_frames(self, video: mp.VideoFileClip, max_frames: int = 100) -> np.ndarray:
        """Extract frames from video for analysis"""
        duration = video.duration
        frame_times = np.linspace(0, duration - 0.1, max_frames)
        
        frames = []
        for t in frame_times:
            frame = video.get_frame(t)
            # Resize for processing
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        return np.array(frames)
    
    def _detect_faces(self, frames: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in video frames"""
        face_detections = []
        
        for i, frame in enumerate(frames):
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_detections.append({
                    'frame_index': i,
                    'bbox': [left, top, right, bottom],
                    'confidence': 0.9  # face_recognition doesn't provide confidence
                })
        
        return face_detections
    
    def _detect_objects(self, frames: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in video frames"""
        # Simplified object detection (would use YOLO or similar in practice)
        object_detections = []
        
        for i, frame in enumerate(frames):
            # Placeholder for object detection
            # In practice, would use a pre-trained model like YOLO
            object_detections.append({
                'frame_index': i,
                'objects': [],
                'confidence': 0.0
            })
        
        return object_detections
    
    def _analyze_motion(self, frames: np.ndarray) -> Dict[str, Any]:
        """Analyze motion in video"""
        if len(frames) < 2:
            return {'motion_intensity': 0.0, 'motion_vectors': []}
        
        motion_intensities = []
        
        for i in range(1, len(frames)):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, None, None
            )
            
            # Calculate motion intensity
            if flow[0] is not None:
                motion_intensity = np.mean(np.linalg.norm(flow[1], axis=1))
            else:
                motion_intensity = 0.0
            
            motion_intensities.append(motion_intensity)
        
        return {
            'motion_intensity': np.mean(motion_intensities),
            'motion_vectors': motion_intensities
        }
    
    def _detect_scene_changes(self, scene_scores: np.ndarray, duration: float) -> List[float]:
        """Detect scene changes from model predictions"""
        threshold = self.config.scene_change_threshold
        scene_changes = []
        
        # Find peaks in scene change scores
        peaks, _ = find_peaks(scene_scores, height=threshold, distance=10)
        
        # Convert frame indices to timestamps
        for peak in peaks:
            timestamp = (peak / len(scene_scores)) * duration
            scene_changes.append(timestamp)
        
        return scene_changes
    
    def _find_highlight_moments(self, highlight_scores: np.ndarray, duration: float) -> List[Dict[str, Any]]:
        """Find highlight moments from model predictions"""
        threshold = self.config.engagement_prediction_threshold
        highlights = []
        
        # Find peaks in highlight scores
        peaks, properties = find_peaks(
            highlight_scores, 
            height=threshold, 
            distance=20,
            width=5
        )
        
        for i, peak in enumerate(peaks):
            start_frame = max(0, peak - 15)  # 15 frames before peak
            end_frame = min(len(highlight_scores), peak + 15)  # 15 frames after peak
            
            start_time = (start_frame / len(highlight_scores)) * duration
            end_time = (end_frame / len(highlight_scores)) * duration
            
            highlights.append({
                'start': start_time,
                'end': end_time,
                'peak_time': (peak / len(highlight_scores)) * duration,
                'score': float(highlight_scores[peak]),
                'width': properties['widths'][i] if 'widths' in properties else 1.0
            })
        
        # Sort by score (highest first)
        highlights.sort(key=lambda x: x['score'], reverse=True)
        
        return highlights
    
    def _extract_trending_topics(self, speech_segments: List[Dict[str, Any]]) -> List[str]:
        """Extract trending topics from speech"""
        # Combine all speech text
        all_text = " ".join([seg['text'] for seg in speech_segments])
        
        # Simple keyword extraction (would use NLP in practice)
        keywords = []
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = {}
        
        for word in words:
            if word not in common_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:10]]
        
        return keywords
    
    def _recommend_clips(self, highlight_moments: List[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
        """Recommend optimal clips based on highlights"""
        recommendations = []
        
        for i, highlight in enumerate(highlight_moments[:self.config.num_clips_to_generate]):
            # Determine clip duration based on content
            min_duration, max_duration = self.config.clip_duration_range
            
            # Adjust clip boundaries
            clip_duration = min(max_duration, max(min_duration, highlight['end'] - highlight['start'] + 10))
            
            start_time = max(0, highlight['peak_time'] - clip_duration / 2)
            end_time = min(duration, start_time + clip_duration)
            
            # Adjust start if end exceeds duration
            if end_time >= duration:
                end_time = duration
                start_time = max(0, end_time - clip_duration)
            
            recommendations.append({
                'clip_index': i,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'highlight_score': highlight['score'],
                'reason': f"High engagement moment (score: {highlight['score']:.2f})"
            })
        
        return recommendations
    
    def _select_clip_moments(self, video_analysis: VideoAnalysis, num_clips: int) -> List[Dict[str, float]]:
        """Select the best moments for viral clips"""
        # Use recommended clips from analysis
        recommended = video_analysis.recommended_clips[:num_clips]
        
        # Ensure no overlap between clips
        selected_moments = []
        for rec in recommended:
            # Check for overlap with existing clips
            overlap = False
            for existing in selected_moments:
                if (rec['start'] < existing['end'] and rec['end'] > existing['start']):
                    overlap = True
                    break
            
            if not overlap:
                selected_moments.append({
                    'start': rec['start'],
                    'end': rec['end'],
                    'score': rec['highlight_score']
                })
        
        # Fill remaining slots if needed
        while len(selected_moments) < num_clips and len(selected_moments) < len(video_analysis.highlight_moments):
            for highlight in video_analysis.highlight_moments:
                if len(selected_moments) >= num_clips:
                    break
                
                # Check if this highlight is already used
                used = False
                for existing in selected_moments:
                    if abs(highlight['peak_time'] - (existing['start'] + existing['end'])/2) < 30:
                        used = True
                        break
                
                if not used:
                    duration = min(60, max(15, highlight['end'] - highlight['start'] + 10))
                    start_time = max(0, highlight['peak_time'] - duration / 2)
                    end_time = min(video_analysis.duration, start_time + duration)
                    
                    selected_moments.append({
                        'start': start_time,
                        'end': end_time,
                        'score': highlight['score']
                    })
        
        return selected_moments
    
    def _generate_clip_captions(
        self,
        clip_video: mp.VideoFileClip,
        start_time: float,
        end_time: float,
        video_analysis: VideoAnalysis
    ) -> List[Dict[str, Any]]:
        """Generate captions for a clip"""
        captions = []
        
        # Find speech segments that overlap with clip
        for segment in video_analysis.speech_segments:
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Check if segment overlaps with clip
            if seg_start < end_time and seg_end > start_time:
                # Adjust timing relative to clip
                caption_start = max(0, seg_start - start_time)
                caption_end = min(clip_video.duration, seg_end - start_time)
                
                if caption_end > caption_start:
                    captions.append({
                        'start': caption_start,
                        'end': caption_end,
                        'text': segment['text'],
                        'confidence': segment['confidence'],
                        'style': 'dynamic',
                        'animation': 'fade_in_out',
                        'position': 'bottom_center'
                    })
        
        # Add viral-style captions if no speech
        if not captions:
            captions = self._generate_viral_captions(clip_video.duration)
        
        return captions
    
    def _generate_viral_captions(self, duration: float) -> List[Dict[str, Any]]:
        """Generate viral-style captions for clips without speech"""
        viral_phrases = [
            "Wait for it...",
            "You won't believe this!",
            "This is insane!",
            "Watch till the end!",
            "Mind = Blown 🤯",
            "This changes everything!",
            "Plot twist incoming...",
            "The best part is coming!",
            "This is why I love the internet",
            "Share if you agree!"
        ]
        
        captions = []
        num_captions = min(3, int(duration / 5))  # One caption every 5 seconds
        
        for i in range(num_captions):
            start_time = (i * duration) / num_captions
            end_time = min(duration, start_time + 3)  # 3 second duration
            
            captions.append({
                'start': start_time,
                'end': end_time,
                'text': viral_phrases[i % len(viral_phrases)],
                'confidence': 1.0,
                'style': 'viral',
                'animation': 'zoom_in',
                'position': 'center'
            })
        
        return captions
    
    def _add_captions_to_video(self, video: mp.VideoFileClip, captions: List[Dict[str, Any]]) -> mp.VideoFileClip:
        """Add animated captions to video"""
        caption_clips = []
        
        for caption in captions:
            # Create text clip
            text_clip = mp.TextClip(
                caption['text'],
                fontsize=self.config.caption_font_size,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2
            ).set_start(caption['start']).set_duration(caption['end'] - caption['start'])
            
            # Position caption
            if caption['position'] == 'bottom_center':
                text_clip = text_clip.set_position(('center', 'bottom')).set_margin(50)
            elif caption['position'] == 'center':
                text_clip = text_clip.set_position('center')
            else:
                text_clip = text_clip.set_position(('center', 'bottom')).set_margin(50)
            
            # Add animation
            if caption['animation'] == 'fade_in_out':
                text_clip = text_clip.fadein(0.3).fadeout(0.3)
            elif caption['animation'] == 'zoom_in':
                text_clip = text_clip.resize(lambda t: 1 + 0.1 * t)
            
            caption_clips.append(text_clip)
        
        # Composite video with captions
        if caption_clips:
            video = mp.CompositeVideoClip([video] + caption_clips)
        
        return video
    
    def _apply_visual_effects(self, video: mp.VideoFileClip, video_analysis: VideoAnalysis) -> mp.VideoFileClip:
        """Apply visual effects to enhance viral potential"""
        
        # Apply zoom effect based on viral potential
        if video_analysis.viral_potential > 0.7:
            zoom_factor = 1.1 + (video_analysis.viral_potential - 0.7) * 0.4
            video = video.resize(zoom_factor)
        
        # Add fade in/out
        video = video.fadein(0.5).fadeout(0.5)
        
        # Apply color grading for better engagement
        video = video.fx(mp.vfx.colorx, 1.1)  # Slight saturation boost
        
        return video
    
    def _add_logo(self, video: mp.VideoFileClip) -> mp.VideoFileClip:
        """Add logo/watermark to video"""
        # Create a simple text logo (would use actual logo image in practice)
        logo = mp.TextClip(
            "ViralClips",
            fontsize=24,
            color='white',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=1
        ).set_duration(video.duration).set_opacity(self.config.logo_opacity)
        
        # Position logo
        if self.config.watermark_position == 'bottom_right':
            logo = logo.set_position(('right', 'bottom')).set_margin(20)
        elif self.config.watermark_position == 'top_right':
            logo = logo.set_position(('right', 'top')).set_margin(20)
        else:
            logo = logo.set_position(('right', 'bottom')).set_margin(20)
        
        # Composite with video
        video = mp.CompositeVideoClip([video, logo])
        
        return video
    
    def _add_transitions(self, video: mp.VideoFileClip) -> mp.VideoFileClip:
        """Add transitions to video"""
        # Simple fade in/out transitions
        transition_duration = self.config.transition_duration
        
        if video.duration > transition_duration * 2:
            video = video.fadein(transition_duration).fadeout(transition_duration)
        
        return video
    
    def _calculate_viral_score(
        self,
        start_time: float,
        end_time: float,
        video_analysis: VideoAnalysis
    ) -> float:
        """Calculate viral potential score for a clip"""
        
        # Base score from video analysis
        base_score = video_analysis.viral_potential
        
        # Boost score if clip contains highlights
        highlight_boost = 0.0
        for highlight in video_analysis.highlight_moments:
            if highlight['start'] <= end_time and highlight['end'] >= start_time:
                highlight_boost += highlight['score'] * 0.2
        
        # Boost score if clip contains speech
        speech_boost = 0.0
        for segment in video_analysis.speech_segments:
            if segment['start'] <= end_time and segment['end'] >= start_time:
                speech_boost += 0.1
        
        # Boost score if clip contains faces
        face_boost = 0.0
        if video_analysis.face_detections:
            face_boost = 0.1
        
        # Combine scores
        viral_score = min(1.0, base_score + highlight_boost + speech_boost + face_boost)
        
        return viral_score
    
    def _generate_clip_title(self, captions: List[Dict[str, Any]], video_analysis: VideoAnalysis) -> str:
        """Generate engaging title for clip"""
        
        # Use first caption as base
        if captions and captions[0]['text']:
            base_text = captions[0]['text'][:50]
        else:
            base_text = "Amazing moment"
        
        # Add viral elements
        viral_prefixes = [
            "🔥", "😱", "🤯", "💯", "⚡", "🚀", "👀", "🎯"
        ]
        
        viral_suffixes = [
            "you won't believe this!",
            "this is insane!",
            "watch till the end!",
            "mind blown!",
            "viral moment!"
        ]
        
        prefix = np.random.choice(viral_prefixes)
        suffix = np.random.choice(viral_suffixes)
        
        title = f"{prefix} {base_text} - {suffix}"
        
        return title[:100]  # Limit length
    
    def _generate_clip_description(self, captions: List[Dict[str, Any]], video_analysis: VideoAnalysis) -> str:
        """Generate description for clip"""
        
        # Combine caption texts
        caption_text = " ".join([cap['text'] for cap in captions[:3]])
        
        # Add trending topics
        topics = ", ".join(video_analysis.trending_topics[:5])
        
        description = f"{caption_text}\n\n"
        if topics:
            description += f"Topics: {topics}\n\n"
        
        description += "#viral #trending #shorts #fyp #amazing"
        
        return description
    
    def _generate_platform_optimizations(self) -> Dict[str, Any]:
        """Generate platform-specific optimizations"""
        return {
            'tiktok': {
                'aspect_ratio': '9:16',
                'duration': '15-60s',
                'hashtags': ['#fyp', '#viral', '#trending'],
                'effects': ['zoom', 'transitions']
            },
            'instagram': {
                'aspect_ratio': '9:16',
                'duration': '15-90s',
                'hashtags': ['#reels', '#viral', '#explore'],
                'effects': ['captions', 'music']
            },
            'youtube_shorts': {
                'aspect_ratio': '9:16',
                'duration': '15-60s',
                'hashtags': ['#shorts', '#viral'],
                'effects': ['thumbnails', 'captions']
            }
        }
    
    def _tensor_to_dict(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Convert tensor dictionary to serializable format"""
        result = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy().tolist()
            else:
                result[key] = value
        return result
    
    def _generate_video_id(self, video_path: str) -> str:
        """Generate unique video ID"""
        return hashlib.md5(video_path.encode()).hexdigest()[:12]
    
    def process_youtube_url(self, url: str) -> Tuple[VideoAnalysis, List[VideoClip]]:
        """Complete pipeline: extract, analyze, and generate clips"""
        
        # Extract video
        video_path = self.extract_youtube_video(url)
        if not video_path:
            raise ValueError("Failed to extract video from YouTube")
        
        try:
            # Analyze video
            video_analysis = self.analyze_video(video_path)
            
            # Generate clips
            clips = self.generate_viral_clips(video_path, video_analysis)
            
            return video_analysis, clips
            
        finally:
            # Clean up original video
            if os.path.exists(video_path):
                os.remove(video_path)
    
    def save_clips_metadata(self, clips: List[VideoClip], output_path: str):
        """Save clips metadata to JSON file"""
        metadata = []
        for clip in clips:
            clip_data = {
                'clip_id': clip.clip_id,
                'start_time': clip.start_time,
                'end_time': clip.end_time,
                'duration': clip.duration,
                'title': clip.title,
                'description': clip.description,
                'viral_score': clip.viral_score,
                'engagement_prediction': clip.engagement_prediction,
                'file_path': clip.file_path,
                'thumbnail_path': clip.thumbnail_path,
                'captions': clip.captions,
                'effects_applied': clip.effects_applied,
                'platform_optimizations': clip.platform_optimizations,
                'metadata': clip.metadata
            }
            metadata.append(clip_data)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = ViralVideoClipsConfig()
    
    # Initialize model
    model = ViralVideoClipsModel(config)
    
    # Example: Process a YouTube video
    youtube_url = "https://www.youtube.com/watch?v=example"
    
    try:
        # Process video and generate clips
        video_analysis, clips = model.process_youtube_url(youtube_url)
        
        print(f"Generated {len(clips)} viral clips")
        print(f"Video viral potential: {video_analysis.viral_potential:.2f}")
        
        # Save metadata
        model.save_clips_metadata(clips, "clips_metadata.json")
        
        # Print clip information
        for i, clip in enumerate(clips):
            print(f"\nClip {i+1}:")
            print(f"  Title: {clip.title}")
            print(f"  Duration: {clip.duration:.1f}s")
            print(f"  Viral Score: {clip.viral_score:.2f}")
            print(f"  File: {clip.file_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")