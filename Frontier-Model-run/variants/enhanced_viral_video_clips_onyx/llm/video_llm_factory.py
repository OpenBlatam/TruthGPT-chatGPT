"""
Enhanced Viral Video Clips Model - LLM Factory
Inspired by Onyx factory pattern for dynamic model creation and management
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import logging
from pathlib import Path

from ..interfaces.video_llm_interface import (
    VideoLLM, VideoLLMConfig, VideoProcessingMode, PlatformType,
    VideoLLMException, ModelNotLoadedError
)
from ..configs.video_model_configs import (
    get_model_config, get_platform_config, get_processing_config,
    VIDEO_MODEL_VARIANT, DEFAULT_PROCESSING_CONFIG
)

logger = logging.getLogger(__name__)


class VideoLLMRegistry:
    """Registry for video LLM implementations"""
    
    _implementations: Dict[str, Type[VideoLLM]] = {}
    _instances: Dict[str, VideoLLM] = {}
    
    @classmethod
    def register(cls, provider_name: str, implementation: Type[VideoLLM]) -> None:
        """Register a video LLM implementation"""
        cls._implementations[provider_name] = implementation
        logger.info(f"Registered video LLM provider: {provider_name}")
    
    @classmethod
    def get_implementation(cls, provider_name: str) -> Type[VideoLLM]:
        """Get video LLM implementation by provider name"""
        if provider_name not in cls._implementations:
            raise ValueError(f"Unknown video LLM provider: {provider_name}")
        return cls._implementations[provider_name]
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers"""
        return list(cls._implementations.keys())
    
    @classmethod
    def get_instance(cls, provider_name: str, config: VideoLLMConfig) -> VideoLLM:
        """Get or create video LLM instance"""
        instance_key = f"{provider_name}_{config.model_variant}_{id(config)}"
        
        if instance_key not in cls._instances:
            implementation = cls.get_implementation(provider_name)
            cls._instances[instance_key] = implementation(config)
            logger.info(f"Created new video LLM instance: {instance_key}")
        
        return cls._instances[instance_key]
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached instances"""
        cls._instances.clear()
        logger.info("Cleared all video LLM instances")


def create_video_llm_config(
    model_provider: str = "enhanced_viral_clips",
    model_name: str = "viral-video-clips-v1",
    model_variant: str = None,
    processing_mode: VideoProcessingMode = VideoProcessingMode.VIRAL_CLIPS,
    target_platforms: List[PlatformType] = None,
    temperature: float = 0.3,
    custom_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> VideoLLMConfig:
    """Create a video LLM configuration"""
    
    # Use default variant if not specified
    variant = model_variant or VIDEO_MODEL_VARIANT
    
    # Get model configuration for the variant
    model_config = get_model_config(variant)
    
    # Default platforms if not specified
    if target_platforms is None:
        target_platforms = [
            PlatformType.TIKTOK,
            PlatformType.INSTAGRAM_REELS,
            PlatformType.YOUTUBE_SHORTS
        ]
    
    # Create configuration
    config = VideoLLMConfig(
        model_provider=model_provider,
        model_name=model_name,
        model_variant=variant,
        temperature=temperature,
        max_input_tokens=kwargs.get("max_input_tokens", 8192),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
        
        # Video-specific settings
        video_processing_mode=processing_mode,
        target_platforms=target_platforms,
        clip_duration_range=kwargs.get("clip_duration_range", (15, 60)),
        max_clips_per_video=kwargs.get("max_clips_per_video", 15),
        viral_threshold=kwargs.get("viral_threshold", 0.7),
        
        # Model capabilities based on variant
        supports_audio_analysis=True,
        supports_face_detection=variant in ["medium", "large"],
        supports_object_detection=variant in ["medium", "large"],
        supports_emotion_analysis=variant in ["medium", "large"],
        supports_motion_analysis=True,
        supports_text_overlay=True,
        supports_effects=True,
        
        # Performance settings based on variant
        batch_size=model_config.get("max_concurrent_videos", 2),
        gpu_memory_limit=kwargs.get("gpu_memory_limit"),
        cpu_threads=kwargs.get("cpu_threads", 4),
        enable_caching=kwargs.get("enable_caching", True),
        
        # API settings
        api_key=kwargs.get("api_key"),
        api_base=kwargs.get("api_base"),
        api_version=kwargs.get("api_version"),
        deployment_name=kwargs.get("deployment_name"),
        credentials_file=kwargs.get("credentials_file"),
        
        # Custom configuration
        custom_config=custom_config or {}
    )
    
    return config


def get_default_video_llm(
    model_variant: str = None,
    processing_mode: VideoProcessingMode = VideoProcessingMode.VIRAL_CLIPS,
    target_platforms: List[PlatformType] = None,
    **kwargs
) -> VideoLLM:
    """Get default video LLM instance"""
    
    config = create_video_llm_config(
        model_variant=model_variant,
        processing_mode=processing_mode,
        target_platforms=target_platforms,
        **kwargs
    )
    
    return VideoLLMRegistry.get_instance("enhanced_viral_clips", config)


def get_video_llm_for_platform(
    platform: PlatformType,
    model_variant: str = None,
    **kwargs
) -> VideoLLM:
    """Get video LLM optimized for specific platform"""
    
    # Get platform-specific configuration
    platform_config = get_platform_config(platform)
    
    # Adjust settings based on platform requirements
    if platform == PlatformType.TIKTOK:
        # TikTok requires fast processing and high viral potential
        processing_mode = VideoProcessingMode.VIRAL_CLIPS
        variant = model_variant or "medium"
        viral_threshold = 0.8
    elif platform == PlatformType.YOUTUBE_SHORTS:
        # YouTube Shorts benefits from high quality and longer clips
        processing_mode = VideoProcessingMode.VIRAL_CLIPS
        variant = model_variant or "large"
        viral_threshold = 0.7
    elif platform == PlatformType.INSTAGRAM_REELS:
        # Instagram Reels needs balanced quality and speed
        processing_mode = VideoProcessingMode.VIRAL_CLIPS
        variant = model_variant or "medium"
        viral_threshold = 0.75
    else:
        # Default settings for other platforms
        processing_mode = VideoProcessingMode.VIRAL_CLIPS
        variant = model_variant or "medium"
        viral_threshold = 0.7
    
    config = create_video_llm_config(
        model_variant=variant,
        processing_mode=processing_mode,
        target_platforms=[platform],
        viral_threshold=viral_threshold,
        clip_duration_range=(
            platform_config["min_duration"],
            platform_config["max_duration"]
        ),
        **kwargs
    )
    
    return VideoLLMRegistry.get_instance("enhanced_viral_clips", config)


def get_video_llm_for_batch_processing(
    video_count: int,
    total_duration: float,
    target_platforms: List[PlatformType] = None,
    **kwargs
) -> VideoLLM:
    """Get video LLM optimized for batch processing"""
    
    # Choose model variant based on workload
    if video_count <= 5 and total_duration <= 1800:  # 30 minutes total
        variant = "large"  # High quality for small batches
    elif video_count <= 20 and total_duration <= 7200:  # 2 hours total
        variant = "medium"  # Balanced for medium batches
    else:
        variant = "small"  # Fast processing for large batches
    
    config = create_video_llm_config(
        model_variant=variant,
        processing_mode=VideoProcessingMode.VIRAL_CLIPS,
        target_platforms=target_platforms,
        batch_size=min(video_count, 8),
        enable_caching=True,
        **kwargs
    )
    
    return VideoLLMRegistry.get_instance("enhanced_viral_clips", config)


def get_video_llm_for_real_time(
    platform: PlatformType = PlatformType.TIKTOK,
    **kwargs
) -> VideoLLM:
    """Get video LLM optimized for real-time processing"""
    
    config = create_video_llm_config(
        model_variant="small",  # Fast processing
        processing_mode=VideoProcessingMode.HIGHLIGHTS,
        target_platforms=[platform],
        max_clips_per_video=5,  # Fewer clips for speed
        viral_threshold=0.6,  # Lower threshold for speed
        batch_size=1,  # Real-time processing
        enable_caching=True,
        **kwargs
    )
    
    return VideoLLMRegistry.get_instance("enhanced_viral_clips", config)


class VideoLLMManager:
    """Manager for video LLM instances and lifecycle"""
    
    def __init__(self):
        self._active_instances: Dict[str, VideoLLM] = {}
        self._instance_usage: Dict[str, int] = {}
        self._max_instances = 10
    
    def get_or_create_llm(
        self,
        provider: str,
        config: VideoLLMConfig,
        instance_id: Optional[str] = None
    ) -> VideoLLM:
        """Get or create video LLM instance with lifecycle management"""
        
        if instance_id is None:
            instance_id = f"{provider}_{config.model_variant}_{id(config)}"
        
        # Check if instance already exists
        if instance_id in self._active_instances:
            self._instance_usage[instance_id] += 1
            return self._active_instances[instance_id]
        
        # Create new instance if under limit
        if len(self._active_instances) < self._max_instances:
            llm = VideoLLMRegistry.get_instance(provider, config)
            self._active_instances[instance_id] = llm
            self._instance_usage[instance_id] = 1
            
            # Warm up the model if needed
            if llm.requires_warm_up:
                logger.info(f"Warming up video LLM instance: {instance_id}")
                llm.warm_up()
            
            return llm
        
        # Remove least used instance to make room
        least_used_id = min(self._instance_usage, key=self._instance_usage.get)
        self.remove_instance(least_used_id)
        
        # Create new instance
        return self.get_or_create_llm(provider, config, instance_id)
    
    def remove_instance(self, instance_id: str) -> None:
        """Remove video LLM instance"""
        if instance_id in self._active_instances:
            del self._active_instances[instance_id]
            del self._instance_usage[instance_id]
            logger.info(f"Removed video LLM instance: {instance_id}")
    
    def clear_all_instances(self) -> None:
        """Clear all video LLM instances"""
        self._active_instances.clear()
        self._instance_usage.clear()
        VideoLLMRegistry.clear_instances()
        logger.info("Cleared all video LLM instances")
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """Get statistics about active instances"""
        return {
            "active_instances": len(self._active_instances),
            "max_instances": self._max_instances,
            "instance_usage": dict(self._instance_usage),
            "total_usage": sum(self._instance_usage.values())
        }


# Global manager instance
_video_llm_manager = VideoLLMManager()


def get_video_llm_manager() -> VideoLLMManager:
    """Get the global video LLM manager"""
    return _video_llm_manager


def create_video_llm(
    provider: str = "enhanced_viral_clips",
    config: Optional[VideoLLMConfig] = None,
    **kwargs
) -> VideoLLM:
    """Create video LLM with automatic configuration"""
    
    if config is None:
        config = create_video_llm_config(
            model_provider=provider,
            **kwargs
        )
    
    return _video_llm_manager.get_or_create_llm(provider, config)


def get_available_providers() -> List[str]:
    """Get list of available video LLM providers"""
    return VideoLLMRegistry.list_providers()


def validate_video_llm_config(config: VideoLLMConfig) -> bool:
    """Validate video LLM configuration"""
    
    # Check required fields
    required_fields = [
        "model_provider", "model_name", "model_variant",
        "video_processing_mode", "target_platforms"
    ]
    
    for field in required_fields:
        if not hasattr(config, field) or getattr(config, field) is None:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Validate model variant
    if config.model_variant not in ["small", "medium", "large"]:
        raise ValueError(f"Invalid model variant: {config.model_variant}")
    
    # Validate platforms
    for platform in config.target_platforms:
        if not isinstance(platform, PlatformType):
            raise ValueError(f"Invalid platform type: {platform}")
    
    # Validate numeric ranges
    if config.temperature < 0 or config.temperature > 2:
        raise ValueError("Temperature must be between 0 and 2")
    
    if config.viral_threshold < 0 or config.viral_threshold > 1:
        raise ValueError("Viral threshold must be between 0 and 1")
    
    min_duration, max_duration = config.clip_duration_range
    if min_duration <= 0 or max_duration <= min_duration:
        raise ValueError("Invalid clip duration range")
    
    return True


# Auto-register default implementation when module is imported
def _register_default_implementation():
    """Register default video LLM implementation"""
    try:
        from .enhanced_viral_video_llm import EnhancedViralVideoLLM
        VideoLLMRegistry.register("enhanced_viral_clips", EnhancedViralVideoLLM)
    except ImportError:
        logger.warning("Default video LLM implementation not available")


# Register on import
_register_default_implementation()