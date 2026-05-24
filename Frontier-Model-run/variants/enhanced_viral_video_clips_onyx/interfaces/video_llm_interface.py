"""
Enhanced Viral Video Clips Model - LLM Interface Layer
Inspired by Onyx architecture for enterprise-grade video processing
"""

import abc
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum

from langchain.schema.language_model import LanguageModelInput
from langchain_core.messages import BaseMessage, AIMessageChunk
from pydantic import BaseModel
import torch
import numpy as np


class VideoProcessingMode(Enum):
    """Video processing modes for different use cases"""
    VIRAL_CLIPS = "viral_clips"
    HIGHLIGHTS = "highlights"
    SUMMARIES = "summaries"
    CAPTIONS = "captions"
    EFFECTS = "effects"
    PLATFORM_OPTIMIZATION = "platform_optimization"


class PlatformType(Enum):
    """Supported social media platforms"""
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    YOUTUBE_SHORTS = "youtube_shorts"
    FACEBOOK_REELS = "facebook_reels"
    TWITTER_X = "twitter_x"
    SNAPCHAT = "snapchat"
    LINKEDIN = "linkedin"


@dataclass
class VideoMetadata:
    """Video metadata structure"""
    duration: float
    fps: int
    resolution: tuple[int, int]
    aspect_ratio: str
    file_size: int
    format: str
    audio_channels: int
    audio_sample_rate: int
    bitrate: int
    codec: str


@dataclass
class ClipSegment:
    """Individual clip segment with timing and metadata"""
    start_time: float
    end_time: float
    duration: float
    viral_score: float
    engagement_prediction: float
    content_type: str
    emotions: List[str]
    objects_detected: List[str]
    faces_count: int
    motion_intensity: float
    audio_features: Dict[str, Any]
    transcript: Optional[str] = None


@dataclass
class ViralClipOutput:
    """Output structure for generated viral clips"""
    clip_id: str
    segment: ClipSegment
    optimized_for_platform: PlatformType
    caption_text: str
    hashtags: List[str]
    effects_applied: List[str]
    thumbnail_timestamp: float
    viral_potential: float
    engagement_score: float
    output_path: str
    metadata: Dict[str, Any]


class VideoLLMConfig(BaseModel):
    """Configuration for Video LLM models"""
    model_provider: str
    model_name: str
    model_variant: str  # small, medium, large
    temperature: float
    max_input_tokens: int
    max_output_tokens: int
    
    # Video-specific configurations
    video_processing_mode: VideoProcessingMode
    target_platforms: List[PlatformType]
    clip_duration_range: tuple[int, int]  # min, max seconds
    max_clips_per_video: int
    viral_threshold: float
    
    # Model capabilities
    supports_audio_analysis: bool
    supports_face_detection: bool
    supports_object_detection: bool
    supports_emotion_analysis: bool
    supports_motion_analysis: bool
    supports_text_overlay: bool
    supports_effects: bool
    
    # Performance settings
    batch_size: int
    gpu_memory_limit: Optional[int]
    cpu_threads: int
    enable_caching: bool
    
    # API settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    credentials_file: Optional[str] = None
    
    # Advanced settings
    custom_config: Dict[str, Any] = {}
    
    model_config = {"protected_namespaces": ()}


class VideoLLM(abc.ABC):
    """
    Abstract base class for Video LLM models
    Inspired by Onyx LLM interface with video-specific capabilities
    """
    
    @property
    def requires_warm_up(self) -> bool:
        """Is this model running in memory and needs an initial call to warm it up?"""
        return True
    
    @property
    def requires_api_key(self) -> bool:
        """Does this model require an API key?"""
        return False
    
    @property
    def supports_streaming(self) -> bool:
        """Does this model support streaming responses?"""
        return True
    
    @property
    @abc.abstractmethod
    def config(self) -> VideoLLMConfig:
        """Get the model configuration"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def log_model_configs(self) -> None:
        """Log model configuration details"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def warm_up(self) -> None:
        """Warm up the model for faster inference"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def extract_video_features(
        self,
        video_path: str,
        extract_audio: bool = True,
        extract_frames: bool = True,
        frame_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Extract comprehensive features from video"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def analyze_viral_potential(
        self,
        video_features: Dict[str, Any],
        target_platforms: List[PlatformType]
    ) -> Dict[str, float]:
        """Analyze viral potential for different platforms"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def detect_highlights(
        self,
        video_features: Dict[str, Any],
        min_duration: float = 15.0,
        max_duration: float = 60.0
    ) -> List[ClipSegment]:
        """Detect highlight segments in video"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def generate_captions(
        self,
        audio_features: Dict[str, Any],
        video_context: Dict[str, Any],
        style: str = "viral"
    ) -> List[Dict[str, Any]]:
        """Generate captions with timing and styling"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def optimize_for_platform(
        self,
        clip_segment: ClipSegment,
        platform: PlatformType,
        video_path: str
    ) -> ViralClipOutput:
        """Optimize clip for specific platform"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def apply_viral_effects(
        self,
        video_path: str,
        effects_config: Dict[str, Any]
    ) -> str:
        """Apply viral effects to video"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def batch_process_videos(
        self,
        video_paths: List[str],
        processing_config: Dict[str, Any]
    ) -> List[List[ViralClipOutput]]:
        """Process multiple videos in batch"""
        raise NotImplementedError
    
    def invoke(
        self,
        prompt: LanguageModelInput,
        video_path: Optional[str] = None,
        processing_mode: VideoProcessingMode = VideoProcessingMode.VIRAL_CLIPS,
        target_platforms: Optional[List[PlatformType]] = None,
        **kwargs
    ) -> BaseMessage:
        """
        Main invoke method for video processing
        Compatible with LangChain interface
        """
        self._precall(prompt, video_path)
        
        if video_path:
            return self._process_video_with_prompt(
                prompt, video_path, processing_mode, target_platforms or [], **kwargs
            )
        else:
            return self._process_text_only(prompt, **kwargs)
    
    def stream(
        self,
        prompt: LanguageModelInput,
        video_path: Optional[str] = None,
        processing_mode: VideoProcessingMode = VideoProcessingMode.VIRAL_CLIPS,
        target_platforms: Optional[List[PlatformType]] = None,
        **kwargs
    ) -> Iterator[AIMessageChunk]:
        """
        Streaming interface for real-time processing updates
        """
        self._precall(prompt, video_path)
        
        if video_path:
            yield from self._stream_video_processing(
                prompt, video_path, processing_mode, target_platforms or [], **kwargs
            )
        else:
            yield from self._stream_text_processing(prompt, **kwargs)
    
    @abc.abstractmethod
    def _process_video_with_prompt(
        self,
        prompt: LanguageModelInput,
        video_path: str,
        processing_mode: VideoProcessingMode,
        target_platforms: List[PlatformType],
        **kwargs
    ) -> BaseMessage:
        """Process video with text prompt"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _process_text_only(
        self,
        prompt: LanguageModelInput,
        **kwargs
    ) -> BaseMessage:
        """Process text-only requests"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _stream_video_processing(
        self,
        prompt: LanguageModelInput,
        video_path: str,
        processing_mode: VideoProcessingMode,
        target_platforms: List[PlatformType],
        **kwargs
    ) -> Iterator[AIMessageChunk]:
        """Stream video processing updates"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _stream_text_processing(
        self,
        prompt: LanguageModelInput,
        **kwargs
    ) -> Iterator[AIMessageChunk]:
        """Stream text processing updates"""
        raise NotImplementedError
    
    def _precall(self, prompt: LanguageModelInput, video_path: Optional[str] = None) -> None:
        """Pre-processing validation and logging"""
        if self.config.model_provider == "disabled":
            raise Exception("Video processing is disabled")
        
        # Log the request
        self._log_request(prompt, video_path)
    
    def _log_request(self, prompt: LanguageModelInput, video_path: Optional[str]) -> None:
        """Log the processing request"""
        import logging
        logger = logging.getLogger(__name__)
        
        if video_path:
            logger.info(f"Processing video: {video_path}")
        
        if isinstance(prompt, str):
            logger.debug(f"Prompt: {prompt[:200]}...")
        elif isinstance(prompt, list):
            logger.debug(f"Messages: {len(prompt)} messages")


class VideoLLMException(Exception):
    """Base exception for Video LLM operations"""
    pass


class VideoProcessingError(VideoLLMException):
    """Exception raised during video processing"""
    pass


class ModelNotLoadedError(VideoLLMException):
    """Exception raised when model is not properly loaded"""
    pass


class UnsupportedFormatError(VideoLLMException):
    """Exception raised for unsupported video formats"""
    pass


class InsufficientResourcesError(VideoLLMException):
    """Exception raised when insufficient computational resources"""
    pass


# Tool choice options for video processing
VideoToolChoiceOptions = Literal["required"] | Literal["auto"] | Literal["none"]

# Type aliases for better code readability
VideoFeatures = Dict[str, Any]
ProcessingConfig = Dict[str, Any]
EffectsConfig = Dict[str, Any]
PlatformConfig = Dict[str, Any]