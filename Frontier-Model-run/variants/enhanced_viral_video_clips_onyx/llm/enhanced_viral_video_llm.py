"""
Enhanced Viral Video Clips Model - Main LLM Implementation
Enterprise-grade video processing with Onyx-inspired architecture
"""

import os
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Iterator, Tuple
from pathlib import Path
from collections.abc import Sequence
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoProcessor,
    pipeline, Trainer, TrainingArguments
)
import cv2
import librosa
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import yt_dlp
from langchain.schema.language_model import LanguageModelInput
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage

from ..interfaces.video_llm_interface import (
    VideoLLM, VideoLLMConfig, VideoProcessingMode, PlatformType,
    VideoMetadata, ClipSegment, ViralClipOutput,
    VideoLLMException, VideoProcessingError, ModelNotLoadedError
)
from ..configs.video_model_configs import (
    get_model_config, get_platform_config, get_processing_config
)

logger = logging.getLogger(__name__)


class VideoUnderstandingTransformer(nn.Module):
    """Advanced video understanding transformer with multi-modal capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Video encoder (3D CNN + Transformer)
        self.video_encoder = self._build_video_encoder()
        
        # Audio encoder
        self.audio_encoder = self._build_audio_encoder()
        
        # Text encoder for captions and metadata
        self.text_encoder = self._build_text_encoder()
        
        # Multi-modal fusion layer
        self.fusion_layer = self._build_fusion_layer()
        
        # Task-specific heads
        self.viral_predictor = self._build_viral_predictor()
        self.highlight_detector = self._build_highlight_detector()
        self.emotion_classifier = self._build_emotion_classifier()
        self.caption_generator = self._build_caption_generator()
        
        # Platform optimization layers
        self.platform_optimizers = self._build_platform_optimizers()
        
    def _build_video_encoder(self) -> nn.Module:
        """Build 3D CNN + Transformer for video understanding"""
        return nn.Sequential(
            # 3D CNN for spatial-temporal features
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.AdaptiveAvgPool3d((8, 7, 7)),  # Temporal x Spatial pooling
            nn.Flatten(start_dim=2),  # Keep batch and temporal dims
        )
    
    def _build_audio_encoder(self) -> nn.Module:
        """Build audio encoder for speech and music analysis"""
        return nn.Sequential(
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
            nn.AdaptiveAvgPool1d(512),
        )
    
    def _build_text_encoder(self) -> nn.Module:
        """Build text encoder for captions and metadata"""
        return nn.Sequential(
            nn.Embedding(50000, 512),  # Vocabulary size, embedding dim
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1
                ),
                num_layers=6
            ),
            nn.AdaptiveAvgPool1d(512),
        )
    
    def _build_fusion_layer(self) -> nn.Module:
        """Build multi-modal fusion layer"""
        return nn.Sequential(
            nn.Linear(256 * 7 * 7 + 256 + 512, 1024),  # Video + Audio + Text
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def _build_viral_predictor(self) -> nn.Module:
        """Build viral potential predictor"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(PlatformType)),  # Score per platform
            nn.Sigmoid()
        )
    
    def _build_highlight_detector(self) -> nn.Module:
        """Build highlight detection network"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Highlight score
            nn.Sigmoid()
        )
    
    def _build_emotion_classifier(self) -> nn.Module:
        """Build emotion classification network"""
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "excitement"]
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(emotions)),
            nn.Softmax(dim=-1)
        )
    
    def _build_caption_generator(self) -> nn.Module:
        """Build caption generation network"""
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 50000),  # Vocabulary size
        )
    
    def _build_platform_optimizers(self) -> nn.ModuleDict:
        """Build platform-specific optimization networks"""
        optimizers = {}
        for platform in PlatformType:
            optimizers[platform.value] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),  # Platform-specific features
            )
        return nn.ModuleDict(optimizers)
    
    def forward(self, video_frames, audio_features, text_tokens):
        """Forward pass through the model"""
        # Encode modalities
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)
        text_features = self.text_encoder(text_tokens)
        
        # Flatten video features
        batch_size, temporal_dim = video_features.shape[:2]
        video_features = video_features.view(batch_size, temporal_dim, -1)
        
        # Concatenate features
        combined_features = torch.cat([
            video_features.mean(dim=1),  # Average over temporal dimension
            audio_features.squeeze(-1),
            text_features.squeeze(-1)
        ], dim=-1)
        
        # Fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Task-specific outputs
        viral_scores = self.viral_predictor(fused_features)
        highlight_scores = self.highlight_detector(fused_features)
        emotions = self.emotion_classifier(fused_features)
        
        return {
            "viral_scores": viral_scores,
            "highlight_scores": highlight_scores,
            "emotions": emotions,
            "fused_features": fused_features
        }


class EnhancedViralVideoLLM(VideoLLM):
    """
    Enhanced Viral Video Clips LLM with enterprise-grade capabilities
    Inspired by Onyx architecture patterns
    """
    
    def __init__(self, config: VideoLLMConfig):
        self._config = config
        self.model = None
        self.tokenizer = None
        self.whisper_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
        # Initialize model components
        self._initialize_model()
        
        logger.info(f"Initialized EnhancedViralVideoLLM with variant: {config.model_variant}")
    
    @property
    def config(self) -> VideoLLMConfig:
        """Get the model configuration"""
        return self._config
    
    @property
    def requires_warm_up(self) -> bool:
        """This model benefits from warm-up"""
        return True
    
    @property
    def requires_api_key(self) -> bool:
        """This model doesn't require an API key"""
        return False
    
    def _initialize_model(self) -> None:
        """Initialize the model components"""
        try:
            # Get model configuration
            model_config = get_model_config(self._config.model_variant)
            
            # Initialize video understanding transformer
            transformer_config = {
                "variant": self._config.model_variant,
                "parameters": model_config["parameters"],
                "memory_requirement": model_config["memory_requirement"]
            }
            
            self.model = VideoUnderstandingTransformer(transformer_config)
            
            # Initialize tokenizer for text processing
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize Whisper for speech recognition
            whisper_model_size = {
                "small": "base",
                "medium": "small", 
                "large": "medium"
            }.get(self._config.model_variant, "base")
            
            self.whisper_model = whisper.load_model(whisper_model_size)
            
            # Move models to device
            self.model.to(self.device)
            
            self.is_loaded = True
            logger.info("Model components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise ModelNotLoadedError(f"Model initialization failed: {e}")
    
    def log_model_configs(self) -> None:
        """Log model configuration details"""
        logger.info("=== Enhanced Viral Video LLM Configuration ===")
        logger.info(f"Provider: {self._config.model_provider}")
        logger.info(f"Model: {self._config.model_name}")
        logger.info(f"Variant: {self._config.model_variant}")
        logger.info(f"Processing Mode: {self._config.video_processing_mode}")
        logger.info(f"Target Platforms: {[p.value for p in self._config.target_platforms]}")
        logger.info(f"Temperature: {self._config.temperature}")
        logger.info(f"Max Clips: {self._config.max_clips_per_video}")
        logger.info(f"Viral Threshold: {self._config.viral_threshold}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 50)
    
    def warm_up(self) -> None:
        """Warm up the model for faster inference"""
        if not self.is_loaded:
            raise ModelNotLoadedError("Model not loaded")
        
        logger.info("Warming up Enhanced Viral Video LLM...")
        
        try:
            # Create dummy inputs for warm-up
            dummy_video = torch.randn(1, 3, 16, 224, 224).to(self.device)
            dummy_audio = torch.randn(1, 1, 16000).to(self.device)
            dummy_text = torch.randint(0, 1000, (1, 50)).to(self.device)
            
            # Run forward pass
            with torch.no_grad():
                _ = self.model(dummy_video, dummy_audio, dummy_text)
            
            logger.info("Model warm-up completed")
            
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def extract_video_features(
        self,
        video_path: str,
        extract_audio: bool = True,
        extract_frames: bool = True,
        frame_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Extract comprehensive features from video"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        features = {}
        
        try:
            # Load video
            video_clip = VideoFileClip(video_path)
            
            # Extract metadata
            features["metadata"] = VideoMetadata(
                duration=video_clip.duration,
                fps=video_clip.fps,
                resolution=(video_clip.w, video_clip.h),
                aspect_ratio=f"{video_clip.w}:{video_clip.h}",
                file_size=os.path.getsize(video_path),
                format=Path(video_path).suffix.lower(),
                audio_channels=video_clip.audio.nchannels if video_clip.audio else 0,
                audio_sample_rate=video_clip.audio.fps if video_clip.audio else 0,
                bitrate=0,  # Would need additional analysis
                codec="unknown"  # Would need additional analysis
            )
            
            if extract_frames:
                features["frames"] = self._extract_frame_features(video_clip, frame_interval)
            
            if extract_audio and video_clip.audio:
                features["audio"] = self._extract_audio_features(video_clip)
            
            # Extract motion features
            features["motion"] = self._extract_motion_features(video_clip)
            
            # Extract scene changes
            features["scenes"] = self._detect_scene_changes(video_clip)
            
            video_clip.close()
            
            logger.info(f"Extracted features from video: {video_path}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract video features: {e}")
            raise VideoProcessingError(f"Feature extraction failed: {e}")
    
    def _extract_frame_features(self, video_clip, frame_interval: float) -> Dict[str, Any]:
        """Extract features from video frames"""
        frames = []
        timestamps = []
        
        # Extract frames at specified intervals
        for t in np.arange(0, video_clip.duration, frame_interval):
            frame = video_clip.get_frame(t)
            frames.append(frame)
            timestamps.append(t)
        
        return {
            "frames": frames,
            "timestamps": timestamps,
            "count": len(frames),
            "interval": frame_interval
        }
    
    def _extract_audio_features(self, video_clip) -> Dict[str, Any]:
        """Extract audio features and transcription"""
        # Extract audio array
        audio_array = video_clip.audio.to_soundarray()
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Transcribe with Whisper
        transcript_result = self.whisper_model.transcribe(audio_array)
        
        # Extract audio features with librosa
        mfccs = librosa.feature.mfcc(y=audio_array, sr=video_clip.audio.fps, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=video_clip.audio.fps)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array)
        
        return {
            "transcript": transcript_result["text"],
            "segments": transcript_result["segments"],
            "mfccs": mfccs.tolist(),
            "spectral_centroids": spectral_centroids.tolist(),
            "zero_crossing_rate": zero_crossing_rate.tolist(),
            "duration": len(audio_array) / video_clip.audio.fps,
            "sample_rate": video_clip.audio.fps
        }
    
    def _extract_motion_features(self, video_clip) -> Dict[str, Any]:
        """Extract motion and activity features"""
        motion_scores = []
        timestamps = []
        
        prev_frame = None
        for t in np.arange(0, video_clip.duration, 1.0):  # 1 second intervals
            frame = video_clip.get_frame(t)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame, gray_frame, None, None
                )[0]
                
                # Calculate motion intensity
                if flow is not None:
                    motion_intensity = np.mean(np.sqrt(flow[:, 0]**2 + flow[:, 1]**2))
                else:
                    motion_intensity = 0.0
                
                motion_scores.append(motion_intensity)
                timestamps.append(t)
            
            prev_frame = gray_frame
        
        return {
            "motion_scores": motion_scores,
            "timestamps": timestamps,
            "average_motion": np.mean(motion_scores) if motion_scores else 0.0,
            "max_motion": np.max(motion_scores) if motion_scores else 0.0
        }
    
    def _detect_scene_changes(self, video_clip) -> Dict[str, Any]:
        """Detect scene changes in video"""
        scene_changes = []
        prev_frame = None
        
        for t in np.arange(0, video_clip.duration, 0.5):  # 0.5 second intervals
            frame = video_clip.get_frame(t)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray_frame)
                diff_score = np.mean(diff)
                
                # Threshold for scene change detection
                if diff_score > 30:  # Adjustable threshold
                    scene_changes.append(t)
            
            prev_frame = gray_frame
        
        return {
            "scene_changes": scene_changes,
            "scene_count": len(scene_changes) + 1,
            "average_scene_length": video_clip.duration / (len(scene_changes) + 1) if scene_changes else video_clip.duration
        }
    
    def analyze_viral_potential(
        self,
        video_features: Dict[str, Any],
        target_platforms: List[PlatformType]
    ) -> Dict[str, float]:
        """Analyze viral potential for different platforms"""
        
        if not self.is_loaded:
            raise ModelNotLoadedError("Model not loaded")
        
        try:
            # Prepare features for model
            model_input = self._prepare_model_input(video_features)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**model_input)
                viral_scores = outputs["viral_scores"].cpu().numpy()[0]
            
            # Map scores to platforms
            platform_scores = {}
            for i, platform in enumerate(PlatformType):
                if platform in target_platforms:
                    platform_scores[platform.value] = float(viral_scores[i])
            
            return platform_scores
            
        except Exception as e:
            logger.error(f"Failed to analyze viral potential: {e}")
            raise VideoProcessingError(f"Viral analysis failed: {e}")
    
    def detect_highlights(
        self,
        video_features: Dict[str, Any],
        min_duration: float = 15.0,
        max_duration: float = 60.0
    ) -> List[ClipSegment]:
        """Detect highlight segments in video"""
        
        if not self.is_loaded:
            raise ModelNotLoadedError("Model not loaded")
        
        try:
            highlights = []
            metadata = video_features["metadata"]
            
            # Analyze video in segments
            segment_duration = 10.0  # Analyze 10-second segments
            for start_time in np.arange(0, metadata.duration - min_duration, segment_duration):
                end_time = min(start_time + max_duration, metadata.duration)
                
                if end_time - start_time < min_duration:
                    continue
                
                # Extract segment features
                segment_features = self._extract_segment_features(
                    video_features, start_time, end_time
                )
                
                # Prepare for model
                model_input = self._prepare_model_input(segment_features)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**model_input)
                    highlight_score = float(outputs["highlight_scores"].cpu().numpy()[0])
                    emotions = outputs["emotions"].cpu().numpy()[0]
                
                # Create clip segment if score is above threshold
                if highlight_score > self._config.viral_threshold:
                    segment = ClipSegment(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        viral_score=highlight_score,
                        engagement_prediction=highlight_score * 0.9,  # Estimate
                        content_type="highlight",
                        emotions=self._get_top_emotions(emotions),
                        objects_detected=[],  # Would need object detection
                        faces_count=0,  # Would need face detection
                        motion_intensity=video_features.get("motion", {}).get("average_motion", 0.0),
                        audio_features=segment_features.get("audio", {}),
                        transcript=segment_features.get("audio", {}).get("transcript", "")
                    )
                    highlights.append(segment)
            
            # Sort by viral score and limit to max clips
            highlights.sort(key=lambda x: x.viral_score, reverse=True)
            return highlights[:self._config.max_clips_per_video]
            
        except Exception as e:
            logger.error(f"Failed to detect highlights: {e}")
            raise VideoProcessingError(f"Highlight detection failed: {e}")
    
    def _extract_segment_features(
        self, 
        video_features: Dict[str, Any], 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Any]:
        """Extract features for a specific video segment"""
        # This is a simplified version - would need more sophisticated segment extraction
        return {
            "metadata": video_features["metadata"],
            "audio": video_features.get("audio", {}),
            "motion": video_features.get("motion", {}),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time
        }
    
    def _prepare_model_input(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare features for model input"""
        # Create dummy tensors - in real implementation, would process actual features
        batch_size = 1
        
        video_frames = torch.randn(batch_size, 3, 16, 224, 224).to(self.device)
        audio_features = torch.randn(batch_size, 1, 16000).to(self.device)
        
        # Process text if available
        text = features.get("audio", {}).get("transcript", "")
        if text:
            tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=50, truncation=True, padding=True)
        else:
            tokens = torch.randint(0, 1000, (batch_size, 50))
        
        text_tokens = tokens.to(self.device)
        
        return {
            "video_frames": video_frames,
            "audio_features": audio_features,
            "text_tokens": text_tokens
        }
    
    def _get_top_emotions(self, emotion_scores: np.ndarray, top_k: int = 3) -> List[str]:
        """Get top emotions from emotion scores"""
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "excitement"]
        top_indices = np.argsort(emotion_scores)[-top_k:][::-1]
        return [emotions[i] for i in top_indices]
    
    def generate_captions(
        self,
        audio_features: Dict[str, Any],
        video_context: Dict[str, Any],
        style: str = "viral"
    ) -> List[Dict[str, Any]]:
        """Generate captions with timing and styling"""
        
        captions = []
        transcript = audio_features.get("transcript", "")
        segments = audio_features.get("segments", [])
        
        if not segments:
            return captions
        
        for segment in segments:
            # Generate viral-style caption
            original_text = segment["text"].strip()
            if not original_text:
                continue
            
            viral_caption = self._generate_viral_caption(original_text, style)
            
            caption = {
                "start_time": segment["start"],
                "end_time": segment["end"],
                "duration": segment["end"] - segment["start"],
                "original_text": original_text,
                "viral_text": viral_caption,
                "style": style,
                "animation": self._get_caption_animation(style),
                "position": self._get_caption_position(style),
                "font_size": self._get_caption_font_size(style),
                "color": self._get_caption_color(style)
            }
            captions.append(caption)
        
        return captions
    
    def _generate_viral_caption(self, text: str, style: str) -> str:
        """Generate viral-style caption from original text"""
        # Simple viral caption generation - could be enhanced with LLM
        viral_patterns = {
            "viral": [
                "🔥 {text} 🔥",
                "✨ {text} ✨",
                "💯 {text}",
                "{text} 🚀",
                "🎯 {text}"
            ],
            "engaging": [
                "Wait for it... {text}",
                "You won't believe {text}",
                "This is why {text}",
                "POV: {text}",
                "When {text} 😱"
            ]
        }
        
        patterns = viral_patterns.get(style, viral_patterns["viral"])
        import random
        pattern = random.choice(patterns)
        return pattern.format(text=text)
    
    def _get_caption_animation(self, style: str) -> str:
        """Get caption animation style"""
        animations = {
            "viral": "bounce",
            "engaging": "fade",
            "professional": "slide",
            "fun": "zoom"
        }
        return animations.get(style, "fade")
    
    def _get_caption_position(self, style: str) -> str:
        """Get caption position"""
        positions = {
            "viral": "bottom",
            "engaging": "center",
            "professional": "bottom",
            "fun": "top"
        }
        return positions.get(style, "bottom")
    
    def _get_caption_font_size(self, style: str) -> int:
        """Get caption font size"""
        sizes = {
            "viral": 48,
            "engaging": 36,
            "professional": 32,
            "fun": 44
        }
        return sizes.get(style, 36)
    
    def _get_caption_color(self, style: str) -> str:
        """Get caption color"""
        colors = {
            "viral": "yellow",
            "engaging": "white",
            "professional": "white",
            "fun": "rainbow"
        }
        return colors.get(style, "white")
    
    def optimize_for_platform(
        self,
        clip_segment: ClipSegment,
        platform: PlatformType,
        video_path: str
    ) -> ViralClipOutput:
        """Optimize clip for specific platform"""
        
        platform_config = get_platform_config(platform)
        
        # Generate optimized clip
        clip_id = f"clip_{int(time.time())}_{platform.value}"
        
        # Generate platform-specific caption
        viral_caption = self._generate_platform_caption(clip_segment, platform)
        
        # Generate hashtags
        hashtags = self._generate_hashtags(clip_segment, platform)
        
        # Determine effects to apply
        effects = self._get_platform_effects(platform)
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(clip_segment, platform)
        
        # Generate output path
        output_path = f"./output/{clip_id}.{platform_config['format'] if 'format' in platform_config else 'mp4'}"
        
        return ViralClipOutput(
            clip_id=clip_id,
            segment=clip_segment,
            optimized_for_platform=platform,
            caption_text=viral_caption,
            hashtags=hashtags,
            effects_applied=effects,
            thumbnail_timestamp=clip_segment.start_time + clip_segment.duration * 0.3,
            viral_potential=clip_segment.viral_score,
            engagement_score=engagement_score,
            output_path=output_path,
            metadata={
                "platform_config": platform_config,
                "optimization_timestamp": time.time(),
                "model_variant": self._config.model_variant
            }
        )
    
    def _generate_platform_caption(self, segment: ClipSegment, platform: PlatformType) -> str:
        """Generate platform-specific caption"""
        base_text = segment.transcript or "Amazing moment!"
        
        platform_styles = {
            PlatformType.TIKTOK: "🔥 {text} #viral #fyp",
            PlatformType.INSTAGRAM_REELS: "✨ {text} ✨ #reels #trending",
            PlatformType.YOUTUBE_SHORTS: "🎯 {text} #shorts #youtube",
            PlatformType.FACEBOOK_REELS: "💯 {text} #facebook #reels",
            PlatformType.TWITTER_X: "{text} 🚀 #viral"
        }
        
        template = platform_styles.get(platform, "{text}")
        return template.format(text=base_text[:100])  # Limit length
    
    def _generate_hashtags(self, segment: ClipSegment, platform: PlatformType) -> List[str]:
        """Generate platform-specific hashtags"""
        base_hashtags = ["viral", "trending", "amazing"]
        
        platform_hashtags = {
            PlatformType.TIKTOK: ["fyp", "foryou", "tiktok", "viral", "trending"],
            PlatformType.INSTAGRAM_REELS: ["reels", "instagram", "explore", "viral", "trending"],
            PlatformType.YOUTUBE_SHORTS: ["shorts", "youtube", "viral", "trending", "subscribe"],
            PlatformType.FACEBOOK_REELS: ["facebook", "reels", "viral", "share", "like"],
            PlatformType.TWITTER_X: ["twitter", "viral", "trending", "retweet", "share"]
        }
        
        return platform_hashtags.get(platform, base_hashtags)[:5]
    
    def _get_platform_effects(self, platform: PlatformType) -> List[str]:
        """Get effects to apply for platform"""
        platform_effects = {
            PlatformType.TIKTOK: ["quick_cuts", "zoom_effects", "speed_ramp"],
            PlatformType.INSTAGRAM_REELS: ["smooth_transitions", "color_grading"],
            PlatformType.YOUTUBE_SHORTS: ["thumbnails", "end_screens"],
            PlatformType.FACEBOOK_REELS: ["community_focus", "share_prompts"],
            PlatformType.TWITTER_X: ["minimal_effects", "news_style"]
        }
        
        return platform_effects.get(platform, ["basic_effects"])
    
    def _calculate_engagement_score(self, segment: ClipSegment, platform: PlatformType) -> float:
        """Calculate expected engagement score for platform"""
        base_score = segment.viral_score
        
        # Platform-specific multipliers
        platform_multipliers = {
            PlatformType.TIKTOK: 1.2,
            PlatformType.INSTAGRAM_REELS: 1.1,
            PlatformType.YOUTUBE_SHORTS: 1.0,
            PlatformType.FACEBOOK_REELS: 0.9,
            PlatformType.TWITTER_X: 0.8
        }
        
        multiplier = platform_multipliers.get(platform, 1.0)
        return min(base_score * multiplier, 1.0)
    
    def apply_viral_effects(
        self,
        video_path: str,
        effects_config: Dict[str, Any]
    ) -> str:
        """Apply viral effects to video"""
        
        try:
            # Load video
            video_clip = VideoFileClip(video_path)
            
            # Apply effects based on configuration
            processed_clip = video_clip
            
            effects = effects_config.get("effects", [])
            
            for effect in effects:
                if effect == "speed_ramp":
                    processed_clip = self._apply_speed_ramp(processed_clip)
                elif effect == "zoom_effects":
                    processed_clip = self._apply_zoom_effects(processed_clip)
                elif effect == "color_grading":
                    processed_clip = self._apply_color_grading(processed_clip)
                # Add more effects as needed
            
            # Generate output path
            output_path = effects_config.get("output_path", "output_with_effects.mp4")
            
            # Write processed video
            processed_clip.write_videofile(output_path, codec='libx264')
            
            # Cleanup
            video_clip.close()
            processed_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to apply viral effects: {e}")
            raise VideoProcessingError(f"Effects application failed: {e}")
    
    def _apply_speed_ramp(self, clip):
        """Apply speed ramp effect"""
        # Simple speed ramp - could be enhanced
        duration = clip.duration
        if duration > 10:
            # Speed up middle section
            part1 = clip.subclip(0, duration * 0.3)
            part2 = clip.subclip(duration * 0.3, duration * 0.7).fx(lambda c: c.speedx(1.5))
            part3 = clip.subclip(duration * 0.7, duration)
            return CompositeVideoClip([part1, part2, part3])
        return clip
    
    def _apply_zoom_effects(self, clip):
        """Apply zoom effects"""
        # Simple zoom effect
        return clip.resize(lambda t: 1 + 0.1 * np.sin(2 * np.pi * t / clip.duration))
    
    def _apply_color_grading(self, clip):
        """Apply color grading"""
        # Simple color adjustment
        return clip.fx(lambda c: c.colorx(1.1))
    
    def batch_process_videos(
        self,
        video_paths: List[str],
        processing_config: Dict[str, Any]
    ) -> List[List[ViralClipOutput]]:
        """Process multiple videos in batch"""
        
        results = []
        
        for video_path in video_paths:
            try:
                logger.info(f"Processing video: {video_path}")
                
                # Extract features
                features = self.extract_video_features(video_path)
                
                # Detect highlights
                highlights = self.detect_highlights(features)
                
                # Generate clips for each platform
                video_clips = []
                for highlight in highlights:
                    for platform in self._config.target_platforms:
                        clip_output = self.optimize_for_platform(highlight, platform, video_path)
                        video_clips.append(clip_output)
                
                results.append(video_clips)
                
            except Exception as e:
                logger.error(f"Failed to process video {video_path}: {e}")
                results.append([])  # Empty result for failed video
        
        return results
    
    def _process_video_with_prompt(
        self,
        prompt: LanguageModelInput,
        video_path: str,
        processing_mode: VideoProcessingMode,
        target_platforms: List[PlatformType],
        **kwargs
    ) -> BaseMessage:
        """Process video with text prompt"""
        
        try:
            # Extract video features
            features = self.extract_video_features(video_path)
            
            # Process based on mode
            if processing_mode == VideoProcessingMode.VIRAL_CLIPS:
                highlights = self.detect_highlights(features)
                clips = []
                for highlight in highlights:
                    for platform in target_platforms:
                        clip = self.optimize_for_platform(highlight, platform, video_path)
                        clips.append(clip)
                
                response = f"Generated {len(clips)} viral clips from video. "
                response += f"Found {len(highlights)} highlight segments. "
                response += f"Optimized for platforms: {[p.value for p in target_platforms]}"
                
            elif processing_mode == VideoProcessingMode.HIGHLIGHTS:
                highlights = self.detect_highlights(features)
                response = f"Detected {len(highlights)} highlight segments in video."
                
            elif processing_mode == VideoProcessingMode.CAPTIONS:
                captions = self.generate_captions(features.get("audio", {}), features)
                response = f"Generated {len(captions)} caption segments."
                
            else:
                response = f"Processed video with mode: {processing_mode.value}"
            
            return AIMessage(content=response)
            
        except Exception as e:
            error_msg = f"Video processing failed: {e}"
            logger.error(error_msg)
            return AIMessage(content=error_msg)
    
    def _process_text_only(self, prompt: LanguageModelInput, **kwargs) -> BaseMessage:
        """Process text-only requests"""
        
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list) and len(prompt) > 0:
            # Get the last human message
            human_messages = [msg for msg in prompt if isinstance(msg, HumanMessage)]
            text = human_messages[-1].content if human_messages else "No text provided"
        else:
            text = "No text provided"
        
        # Simple text processing response
        response = f"Enhanced Viral Video LLM received text: {text[:100]}..."
        response += f"\nModel variant: {self._config.model_variant}"
        response += f"\nSupported platforms: {[p.value for p in self._config.target_platforms]}"
        response += "\nTo process videos, please provide a video file path."
        
        return AIMessage(content=response)
    
    def _stream_video_processing(
        self,
        prompt: LanguageModelInput,
        video_path: str,
        processing_mode: VideoProcessingMode,
        target_platforms: List[PlatformType],
        **kwargs
    ) -> Iterator[AIMessageChunk]:
        """Stream video processing updates"""
        
        yield AIMessageChunk(content="🎬 Starting video processing...\n")
        
        try:
            yield AIMessageChunk(content="📊 Extracting video features...\n")
            features = self.extract_video_features(video_path)
            
            yield AIMessageChunk(content="🔍 Analyzing viral potential...\n")
            viral_scores = self.analyze_viral_potential(features, target_platforms)
            
            yield AIMessageChunk(content="✨ Detecting highlights...\n")
            highlights = self.detect_highlights(features)
            
            yield AIMessageChunk(content=f"🎯 Found {len(highlights)} highlight segments\n")
            
            for i, highlight in enumerate(highlights):
                yield AIMessageChunk(content=f"🎬 Processing clip {i+1}/{len(highlights)}...\n")
                
                for platform in target_platforms:
                    clip = self.optimize_for_platform(highlight, platform, video_path)
                    yield AIMessageChunk(content=f"✅ Optimized for {platform.value}\n")
            
            yield AIMessageChunk(content="🎉 Video processing completed!\n")
            
        except Exception as e:
            yield AIMessageChunk(content=f"❌ Error: {e}\n")
    
    def _stream_text_processing(
        self,
        prompt: LanguageModelInput,
        **kwargs
    ) -> Iterator[AIMessageChunk]:
        """Stream text processing updates"""
        
        yield AIMessageChunk(content="💬 Processing text input...\n")
        yield AIMessageChunk(content="🤖 Enhanced Viral Video LLM ready\n")
        yield AIMessageChunk(content="📹 Provide a video file to start processing\n")