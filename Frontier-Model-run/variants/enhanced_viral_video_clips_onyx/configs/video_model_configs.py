"""
Enhanced Viral Video Clips Model - Configuration Management
Inspired by Onyx configuration system with video-specific settings
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..interfaces.video_llm_interface import VideoProcessingMode, PlatformType


#####
# Video Model Configurations
#####

# Default video processing model
DEFAULT_VIDEO_MODEL = "enhanced-viral-video-clips-v1"
VIDEO_MODEL = os.environ.get("VIDEO_MODEL") or DEFAULT_VIDEO_MODEL

# Model variants with different capabilities and resource requirements
VIDEO_MODEL_VARIANTS = {
    "small": {
        "parameters": "3B",
        "memory_requirement": "8GB",
        "processing_speed": "5000 frames/s",
        "max_video_length": 600,  # 10 minutes
        "max_concurrent_videos": 4,
        "supports_real_time": True,
        "gpu_required": False
    },
    "medium": {
        "parameters": "8B", 
        "memory_requirement": "16GB",
        "processing_speed": "2500 frames/s",
        "max_video_length": 1800,  # 30 minutes
        "max_concurrent_videos": 2,
        "supports_real_time": True,
        "gpu_required": True
    },
    "large": {
        "parameters": "15B",
        "memory_requirement": "32GB", 
        "processing_speed": "1200 frames/s",
        "max_video_length": 3600,  # 60 minutes
        "max_concurrent_videos": 1,
        "supports_real_time": False,
        "gpu_required": True
    }
}

# Current model variant
VIDEO_MODEL_VARIANT = os.environ.get("VIDEO_MODEL_VARIANT", "medium")

# Video processing settings
VIDEO_PROCESSING_BATCH_SIZE = int(os.environ.get("VIDEO_PROCESSING_BATCH_SIZE") or 4)
VIDEO_FRAME_EXTRACTION_FPS = float(os.environ.get("VIDEO_FRAME_EXTRACTION_FPS") or 1.0)
VIDEO_MAX_DURATION = int(os.environ.get("VIDEO_MAX_DURATION") or 3600)  # 1 hour
VIDEO_MIN_CLIP_DURATION = float(os.environ.get("VIDEO_MIN_CLIP_DURATION") or 15.0)
VIDEO_MAX_CLIP_DURATION = float(os.environ.get("VIDEO_MAX_CLIP_DURATION") or 60.0)
VIDEO_MAX_CLIPS_PER_VIDEO = int(os.environ.get("VIDEO_MAX_CLIPS_PER_VIDEO") or 15)

# Viral detection thresholds
VIRAL_SCORE_THRESHOLD = float(os.environ.get("VIRAL_SCORE_THRESHOLD") or 0.7)
ENGAGEMENT_PREDICTION_THRESHOLD = float(os.environ.get("ENGAGEMENT_PREDICTION_THRESHOLD") or 0.8)
HIGHLIGHT_DETECTION_SENSITIVITY = float(os.environ.get("HIGHLIGHT_DETECTION_SENSITIVITY") or 0.6)

# Platform-specific configurations
PLATFORM_CONFIGS = {
    PlatformType.TIKTOK: {
        "aspect_ratio": "9:16",
        "max_duration": 60,
        "min_duration": 15,
        "optimal_duration": 30,
        "resolution": (1080, 1920),
        "fps": 30,
        "bitrate": "2M",
        "audio_required": True,
        "captions_style": "large_bold",
        "effects_intensity": "high",
        "trending_hashtags_count": 5,
        "viral_elements": ["quick_cuts", "zoom_effects", "trending_sounds", "text_overlays"]
    },
    PlatformType.INSTAGRAM_REELS: {
        "aspect_ratio": "9:16",
        "max_duration": 90,
        "min_duration": 15,
        "optimal_duration": 30,
        "resolution": (1080, 1920),
        "fps": 30,
        "bitrate": "3.5M",
        "audio_required": True,
        "captions_style": "clean_modern",
        "effects_intensity": "medium",
        "trending_hashtags_count": 8,
        "viral_elements": ["smooth_transitions", "color_grading", "music_sync", "story_hooks"]
    },
    PlatformType.YOUTUBE_SHORTS: {
        "aspect_ratio": "9:16",
        "max_duration": 60,
        "min_duration": 15,
        "optimal_duration": 45,
        "resolution": (1080, 1920),
        "fps": 60,
        "bitrate": "5M",
        "audio_required": True,
        "captions_style": "youtube_standard",
        "effects_intensity": "medium",
        "trending_hashtags_count": 3,
        "viral_elements": ["thumbnails", "end_screens", "subscribe_prompts", "engagement_hooks"]
    },
    PlatformType.FACEBOOK_REELS: {
        "aspect_ratio": "9:16",
        "max_duration": 90,
        "min_duration": 15,
        "optimal_duration": 30,
        "resolution": (1080, 1920),
        "fps": 30,
        "bitrate": "4M",
        "audio_required": True,
        "captions_style": "facebook_standard",
        "effects_intensity": "low",
        "trending_hashtags_count": 5,
        "viral_elements": ["community_focus", "share_prompts", "reaction_hooks", "local_trends"]
    },
    PlatformType.TWITTER_X: {
        "aspect_ratio": "16:9",
        "max_duration": 140,
        "min_duration": 10,
        "optimal_duration": 30,
        "resolution": (1280, 720),
        "fps": 30,
        "bitrate": "2M",
        "audio_required": False,
        "captions_style": "minimal",
        "effects_intensity": "low",
        "trending_hashtags_count": 3,
        "viral_elements": ["news_hooks", "trending_topics", "thread_integration", "retweet_prompts"]
    }
}

# Audio processing settings
AUDIO_SAMPLE_RATE = int(os.environ.get("AUDIO_SAMPLE_RATE") or 44100)
AUDIO_CHANNELS = int(os.environ.get("AUDIO_CHANNELS") or 2)
AUDIO_BITRATE = os.environ.get("AUDIO_BITRATE", "128k")
ENABLE_AUDIO_ENHANCEMENT = (os.environ.get("ENABLE_AUDIO_ENHANCEMENT") or "true").lower() == "true"
ENABLE_NOISE_REDUCTION = (os.environ.get("ENABLE_NOISE_REDUCTION") or "true").lower() == "true"

# Caption generation settings
CAPTION_LANGUAGE = os.environ.get("CAPTION_LANGUAGE", "en")
CAPTION_MAX_CHARS_PER_LINE = int(os.environ.get("CAPTION_MAX_CHARS_PER_LINE") or 40)
CAPTION_MAX_LINES = int(os.environ.get("CAPTION_MAX_LINES") or 2)
CAPTION_ANIMATION_DURATION = float(os.environ.get("CAPTION_ANIMATION_DURATION") or 0.3)
ENABLE_EMOTION_BASED_STYLING = (os.environ.get("ENABLE_EMOTION_BASED_STYLING") or "true").lower() == "true"

# Effects and transitions
AVAILABLE_TRANSITIONS = [
    "fade", "slide", "zoom", "wipe", "dissolve", "cut", "push", "reveal"
]
AVAILABLE_EFFECTS = [
    "speed_ramp", "slow_motion", "time_lapse", "reverse", "loop", "freeze_frame",
    "color_grading", "saturation_boost", "contrast_enhance", "vintage_filter",
    "glitch_effect", "particle_overlay", "light_leaks", "vignette"
]
DEFAULT_TRANSITION_DURATION = float(os.environ.get("DEFAULT_TRANSITION_DURATION") or 0.5)
DEFAULT_EFFECT_INTENSITY = float(os.environ.get("DEFAULT_EFFECT_INTENSITY") or 0.7)

# AI model settings
AI_TEMPERATURE = float(os.environ.get("AI_TEMPERATURE") or 0.3)
AI_MAX_TOKENS = int(os.environ.get("AI_MAX_TOKENS") or 2048)
AI_TOP_P = float(os.environ.get("AI_TOP_P") or 0.9)
AI_FREQUENCY_PENALTY = float(os.environ.get("AI_FREQUENCY_PENALTY") or 0.1)
AI_PRESENCE_PENALTY = float(os.environ.get("AI_PRESENCE_PENALTY") or 0.1)

# Performance and resource management
GPU_MEMORY_FRACTION = float(os.environ.get("GPU_MEMORY_FRACTION") or 0.8)
CPU_THREADS = int(os.environ.get("CPU_THREADS") or 4)
ENABLE_MODEL_CACHING = (os.environ.get("ENABLE_MODEL_CACHING") or "true").lower() == "true"
CACHE_SIZE_GB = int(os.environ.get("CACHE_SIZE_GB") or 10)
ENABLE_PARALLEL_PROCESSING = (os.environ.get("ENABLE_PARALLEL_PROCESSING") or "true").lower() == "true"

# Output settings
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "mp4")
OUTPUT_QUALITY = os.environ.get("OUTPUT_QUALITY", "high")  # low, medium, high, ultra
OUTPUT_DIRECTORY = os.environ.get("OUTPUT_DIRECTORY", "./output/viral_clips")
ENABLE_WATERMARK = (os.environ.get("ENABLE_WATERMARK") or "false").lower() == "true"
WATERMARK_OPACITY = float(os.environ.get("WATERMARK_OPACITY") or 0.3)

# API and external service settings
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ENABLE_EXTERNAL_APIS = (os.environ.get("ENABLE_EXTERNAL_APIS") or "true").lower() == "true"

# Logging and monitoring
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
ENABLE_PERFORMANCE_MONITORING = (os.environ.get("ENABLE_PERFORMANCE_MONITORING") or "true").lower() == "true"
ENABLE_USAGE_ANALYTICS = (os.environ.get("ENABLE_USAGE_ANALYTICS") or "false").lower() == "true"
LOG_VIDEO_PROCESSING_STEPS = (os.environ.get("LOG_VIDEO_PROCESSING_STEPS") or "true").lower() == "true"

# Security settings
ENABLE_INPUT_VALIDATION = (os.environ.get("ENABLE_INPUT_VALIDATION") or "true").lower() == "true"
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB") or 500)
ALLOWED_VIDEO_FORMATS = os.environ.get("ALLOWED_VIDEO_FORMATS", "mp4,avi,mov,mkv,webm").split(",")
ENABLE_CONTENT_FILTERING = (os.environ.get("ENABLE_CONTENT_FILTERING") or "true").lower() == "true"

# Advanced model configurations
ENABLE_MULTI_MODAL_ANALYSIS = (os.environ.get("ENABLE_MULTI_MODAL_ANALYSIS") or "true").lower() == "true"
ENABLE_FACE_DETECTION = (os.environ.get("ENABLE_FACE_DETECTION") or "true").lower() == "true"
ENABLE_OBJECT_DETECTION = (os.environ.get("ENABLE_OBJECT_DETECTION") or "true").lower() == "true"
ENABLE_EMOTION_ANALYSIS = (os.environ.get("ENABLE_EMOTION_ANALYSIS") or "true").lower() == "true"
ENABLE_MOTION_ANALYSIS = (os.environ.get("ENABLE_MOTION_ANALYSIS") or "true").lower() == "true"
ENABLE_SCENE_DETECTION = (os.environ.get("ENABLE_SCENE_DETECTION") or "true").lower() == "true"

# Experimental features
ENABLE_EXPERIMENTAL_FEATURES = (os.environ.get("ENABLE_EXPERIMENTAL_FEATURES") or "false").lower() == "true"
ENABLE_AI_VOICE_CLONING = (os.environ.get("ENABLE_AI_VOICE_CLONING") or "false").lower() == "true"
ENABLE_DEEPFAKE_DETECTION = (os.environ.get("ENABLE_DEEPFAKE_DETECTION") or "true").lower() == "true"
ENABLE_REAL_TIME_PROCESSING = (os.environ.get("ENABLE_REAL_TIME_PROCESSING") or "false").lower() == "true"


def get_model_config(variant: str = None) -> Dict[str, Any]:
    """Get configuration for specific model variant"""
    variant = variant or VIDEO_MODEL_VARIANT
    if variant not in VIDEO_MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    
    return VIDEO_MODEL_VARIANTS[variant]


def get_platform_config(platform: PlatformType) -> Dict[str, Any]:
    """Get configuration for specific platform"""
    if platform not in PLATFORM_CONFIGS:
        raise ValueError(f"Unsupported platform: {platform}")
    
    return PLATFORM_CONFIGS[platform]


def get_processing_config(
    mode: VideoProcessingMode,
    platforms: List[PlatformType],
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get comprehensive processing configuration"""
    config = {
        "mode": mode,
        "platforms": platforms,
        "model_variant": VIDEO_MODEL_VARIANT,
        "batch_size": VIDEO_PROCESSING_BATCH_SIZE,
        "frame_extraction_fps": VIDEO_FRAME_EXTRACTION_FPS,
        "max_duration": VIDEO_MAX_DURATION,
        "min_clip_duration": VIDEO_MIN_CLIP_DURATION,
        "max_clip_duration": VIDEO_MAX_CLIP_DURATION,
        "max_clips_per_video": VIDEO_MAX_CLIPS_PER_VIDEO,
        "viral_threshold": VIRAL_SCORE_THRESHOLD,
        "engagement_threshold": ENGAGEMENT_PREDICTION_THRESHOLD,
        "ai_settings": {
            "temperature": AI_TEMPERATURE,
            "max_tokens": AI_MAX_TOKENS,
            "top_p": AI_TOP_P,
            "frequency_penalty": AI_FREQUENCY_PENALTY,
            "presence_penalty": AI_PRESENCE_PENALTY
        },
        "performance": {
            "gpu_memory_fraction": GPU_MEMORY_FRACTION,
            "cpu_threads": CPU_THREADS,
            "enable_caching": ENABLE_MODEL_CACHING,
            "enable_parallel": ENABLE_PARALLEL_PROCESSING
        },
        "features": {
            "multi_modal_analysis": ENABLE_MULTI_MODAL_ANALYSIS,
            "face_detection": ENABLE_FACE_DETECTION,
            "object_detection": ENABLE_OBJECT_DETECTION,
            "emotion_analysis": ENABLE_EMOTION_ANALYSIS,
            "motion_analysis": ENABLE_MOTION_ANALYSIS,
            "scene_detection": ENABLE_SCENE_DETECTION
        },
        "output": {
            "format": OUTPUT_FORMAT,
            "quality": OUTPUT_QUALITY,
            "directory": OUTPUT_DIRECTORY,
            "enable_watermark": ENABLE_WATERMARK,
            "watermark_opacity": WATERMARK_OPACITY
        }
    }
    
    # Add platform-specific configurations
    config["platform_configs"] = {
        platform: get_platform_config(platform) for platform in platforms
    }
    
    # Merge custom configuration
    if custom_config:
        config.update(custom_config)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    required_keys = ["mode", "platforms", "model_variant"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate model variant
    if config["model_variant"] not in VIDEO_MODEL_VARIANTS:
        raise ValueError(f"Invalid model variant: {config['model_variant']}")
    
    # Validate platforms
    for platform in config["platforms"]:
        if platform not in PLATFORM_CONFIGS:
            raise ValueError(f"Unsupported platform: {platform}")
    
    # Validate numeric ranges
    if config.get("viral_threshold", 0) < 0 or config.get("viral_threshold", 1) > 1:
        raise ValueError("Viral threshold must be between 0 and 1")
    
    if config.get("min_clip_duration", 0) <= 0:
        raise ValueError("Minimum clip duration must be positive")
    
    if config.get("max_clip_duration", 0) <= config.get("min_clip_duration", 0):
        raise ValueError("Maximum clip duration must be greater than minimum")
    
    return True


def load_custom_config(config_path: str) -> Dict[str, Any]:
    """Load custom configuration from file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        if config_file.suffix.lower() == '.json':
            return json.load(f)
        elif config_file.suffix.lower() in ['.yml', '.yaml']:
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        if output_file.suffix.lower() == '.json':
            json.dump(config, f, indent=2, default=str)
        elif output_file.suffix.lower() in ['.yml', '.yaml']:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output file format: {output_file.suffix}")


# Export commonly used configurations
DEFAULT_PROCESSING_CONFIG = get_processing_config(
    VideoProcessingMode.VIRAL_CLIPS,
    [PlatformType.TIKTOK, PlatformType.INSTAGRAM_REELS, PlatformType.YOUTUBE_SHORTS]
)

FAST_PROCESSING_CONFIG = get_processing_config(
    VideoProcessingMode.HIGHLIGHTS,
    [PlatformType.TIKTOK],
    {"model_variant": "small", "batch_size": 8}
)

HIGH_QUALITY_CONFIG = get_processing_config(
    VideoProcessingMode.VIRAL_CLIPS,
    list(PlatformType),
    {"model_variant": "large", "output": {"quality": "ultra"}}
)