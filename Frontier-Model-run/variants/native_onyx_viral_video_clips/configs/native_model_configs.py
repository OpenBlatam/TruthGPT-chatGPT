"""
Native Model Configurations
Configuration management for native AI models without external APIs

This module provides configuration classes for managing native transformer models,
encoders, and processing pipelines.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
import torch
from enum import Enum

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Model size options"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


class Platform(Enum):
    """Social media platforms"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    TWITTER = "twitter"


@dataclass
class VideoEncoderConfig:
    """Configuration for video encoder"""
    model_name: str = "openai/clip-vit-base-patch32"
    model_size: ModelSize = ModelSize.MEDIUM
    max_frames: int = 32
    frame_size: tuple = (224, 224)
    batch_size: int = 8
    feature_dim: int = 512
    device: str = "auto"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TextEncoderConfig:
    """Configuration for text encoder"""
    model_name: str = "gpt2"
    model_size: ModelSize = ModelSize.MEDIUM
    max_length: int = 512
    generation_max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    feature_dim: int = 768
    device: str = "auto"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adjust feature dim based on model size
        size_dims = {
            ModelSize.SMALL: 512,
            ModelSize.MEDIUM: 768,
            ModelSize.LARGE: 1024,
            ModelSize.XLARGE: 1536
        }
        self.feature_dim = size_dims.get(self.model_size, 768)


@dataclass
class AudioEncoderConfig:
    """Configuration for audio encoder"""
    model_name: str = "openai/whisper-base"
    model_size: ModelSize = ModelSize.MEDIUM
    sample_rate: int = 16000
    max_duration: float = 30.0
    feature_dim: int = 512
    device: str = "auto"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ViralClassifierConfig:
    """Configuration for viral score classifier"""
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.3
    activation: str = "relu"
    output_activation: str = "sigmoid"
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PlatformConfig:
    """Configuration for specific social media platform"""
    name: str
    aspect_ratio: tuple = (9, 16)
    max_duration: float = 60.0
    min_duration: float = 15.0
    resolution: tuple = (1080, 1920)
    fps: int = 30
    viral_weight: float = 1.0
    effects_enabled: bool = True
    captions_required: bool = True
    hashtags_enabled: bool = True
    
    # Platform-specific features
    quick_cuts: bool = False
    zoom_effects: bool = False
    speed_ramps: bool = False
    color_grading: bool = False
    transitions: bool = False
    thumbnails: bool = False
    end_screens: bool = False


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline"""
    segment_duration: float = 10.0
    overlap_duration: float = 2.0
    max_segments: int = 10
    min_viral_score: float = 0.5
    motion_threshold: float = 0.3
    audio_analysis: bool = True
    face_detection: bool = True
    object_detection: bool = True
    scene_classification: bool = True
    emotion_analysis: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    gpu_memory_fraction: float = 0.8


@dataclass
class NativeModelConfig:
    """Main configuration for native video LLM"""
    # Model configurations
    video_encoder: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    audio_encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    viral_classifier: ViralClassifierConfig = field(default_factory=ViralClassifierConfig)
    
    # Processing configuration
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Platform configurations
    platforms: Dict[str, PlatformConfig] = field(default_factory=dict)
    
    # General settings
    model_size: ModelSize = ModelSize.MEDIUM
    cache_enabled: bool = True
    cache_dir: str = "./cache"
    log_level: str = "INFO"
    
    def __post_init__(self):
        # Initialize platform configurations if empty
        if not self.platforms:
            self.platforms = self._get_default_platform_configs()
        
        # Update model sizes based on global setting
        self.video_encoder.model_size = self.model_size
        self.text_encoder.model_size = self.model_size
        self.audio_encoder.model_size = self.model_size
    
    def _get_default_platform_configs(self) -> Dict[str, PlatformConfig]:
        """Get default platform configurations"""
        return {
            "tiktok": PlatformConfig(
                name="tiktok",
                aspect_ratio=(9, 16),
                max_duration=60.0,
                min_duration=15.0,
                resolution=(1080, 1920),
                fps=30,
                viral_weight=1.0,
                quick_cuts=True,
                zoom_effects=True,
                speed_ramps=True,
                hashtags_enabled=True
            ),
            "instagram": PlatformConfig(
                name="instagram",
                aspect_ratio=(9, 16),
                max_duration=90.0,
                min_duration=15.0,
                resolution=(1080, 1920),
                fps=30,
                viral_weight=0.9,
                color_grading=True,
                transitions=True,
                hashtags_enabled=True
            ),
            "youtube": PlatformConfig(
                name="youtube",
                aspect_ratio=(9, 16),
                max_duration=60.0,
                min_duration=15.0,
                resolution=(1080, 1920),
                fps=30,
                viral_weight=0.8,
                thumbnails=True,
                end_screens=True,
                captions_required=True
            ),
            "facebook": PlatformConfig(
                name="facebook",
                aspect_ratio=(9, 16),
                max_duration=90.0,
                min_duration=15.0,
                resolution=(1080, 1920),
                fps=30,
                viral_weight=0.7,
                transitions=True,
                captions_required=True
            ),
            "twitter": PlatformConfig(
                name="twitter",
                aspect_ratio=(16, 9),
                max_duration=140.0,
                min_duration=10.0,
                resolution=(1280, 720),
                fps=30,
                viral_weight=0.6,
                effects_enabled=False,
                captions_required=False
            )
        }
    
    def get_platform_config(self, platform: str) -> PlatformConfig:
        """Get configuration for specific platform"""
        return self.platforms.get(platform, self.platforms["tiktok"])
    
    def update_model_size(self, size: ModelSize):
        """Update model size for all components"""
        self.model_size = size
        self.video_encoder.model_size = size
        self.text_encoder.model_size = size
        self.audio_encoder.model_size = size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "video_encoder": self.video_encoder.__dict__,
            "text_encoder": self.text_encoder.__dict__,
            "audio_encoder": self.audio_encoder.__dict__,
            "viral_classifier": self.viral_classifier.__dict__,
            "processing": self.processing.__dict__,
            "platforms": {k: v.__dict__ for k, v in self.platforms.items()},
            "model_size": self.model_size.value,
            "cache_enabled": self.cache_enabled,
            "cache_dir": self.cache_dir,
            "log_level": self.log_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NativeModelConfig":
        """Create config from dictionary"""
        config = cls()
        
        # Update video encoder
        if "video_encoder" in config_dict:
            for key, value in config_dict["video_encoder"].items():
                if hasattr(config.video_encoder, key):
                    setattr(config.video_encoder, key, value)
        
        # Update text encoder
        if "text_encoder" in config_dict:
            for key, value in config_dict["text_encoder"].items():
                if hasattr(config.text_encoder, key):
                    setattr(config.text_encoder, key, value)
        
        # Update audio encoder
        if "audio_encoder" in config_dict:
            for key, value in config_dict["audio_encoder"].items():
                if hasattr(config.audio_encoder, key):
                    setattr(config.audio_encoder, key, value)
        
        # Update viral classifier
        if "viral_classifier" in config_dict:
            for key, value in config_dict["viral_classifier"].items():
                if hasattr(config.viral_classifier, key):
                    setattr(config.viral_classifier, key, value)
        
        # Update processing
        if "processing" in config_dict:
            for key, value in config_dict["processing"].items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
        
        # Update platforms
        if "platforms" in config_dict:
            for platform_name, platform_config in config_dict["platforms"].items():
                config.platforms[platform_name] = PlatformConfig(**platform_config)
        
        # Update general settings
        if "model_size" in config_dict:
            config.model_size = ModelSize(config_dict["model_size"])
        
        for key in ["cache_enabled", "cache_dir", "log_level"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def save(self, path: str):
        """Save configuration to file"""
        config_dict = self.to_dict()
        
        # Convert enums to strings
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj
        
        config_dict = convert_enums(config_dict)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "NativeModelConfig":
        """Load configuration from file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {path}")
        return cls.from_dict(config_dict)


class ConfigManager:
    """Configuration manager for native models"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._configs: Dict[str, NativeModelConfig] = {}
    
    def get_config(self, name: str = "default") -> NativeModelConfig:
        """Get configuration by name"""
        if name not in self._configs:
            config_path = self.config_dir / f"{name}.yaml"
            if config_path.exists():
                self._configs[name] = NativeModelConfig.load(str(config_path))
            else:
                self._configs[name] = NativeModelConfig()
                self.save_config(name, self._configs[name])
        
        return self._configs[name]
    
    def save_config(self, name: str, config: NativeModelConfig):
        """Save configuration"""
        config_path = self.config_dir / f"{name}.yaml"
        config.save(str(config_path))
        self._configs[name] = config
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        configs = []
        for config_file in self.config_dir.glob("*.yaml"):
            configs.append(config_file.stem)
        return configs
    
    def create_config_for_size(self, size: ModelSize, name: Optional[str] = None) -> NativeModelConfig:
        """Create configuration for specific model size"""
        if name is None:
            name = f"native_{size.value}"
        
        config = NativeModelConfig()
        config.update_model_size(size)
        
        # Adjust model names based on size
        if size == ModelSize.SMALL:
            config.video_encoder.model_name = "openai/clip-vit-base-patch16"
            config.text_encoder.model_name = "gpt2"
            config.audio_encoder.model_name = "openai/whisper-tiny"
        elif size == ModelSize.MEDIUM:
            config.video_encoder.model_name = "openai/clip-vit-base-patch32"
            config.text_encoder.model_name = "gpt2-medium"
            config.audio_encoder.model_name = "openai/whisper-base"
        elif size == ModelSize.LARGE:
            config.video_encoder.model_name = "openai/clip-vit-large-patch14"
            config.text_encoder.model_name = "gpt2-large"
            config.audio_encoder.model_name = "openai/whisper-large"
        elif size == ModelSize.XLARGE:
            config.video_encoder.model_name = "openai/clip-vit-large-patch14-336"
            config.text_encoder.model_name = "gpt2-xl"
            config.audio_encoder.model_name = "openai/whisper-large-v2"
        
        self.save_config(name, config)
        return config
    
    def get_environment_config(self) -> NativeModelConfig:
        """Get configuration from environment variables"""
        config = NativeModelConfig()
        
        # Override with environment variables
        if os.getenv("NATIVE_MODEL_SIZE"):
            config.update_model_size(ModelSize(os.getenv("NATIVE_MODEL_SIZE")))
        
        if os.getenv("NATIVE_CACHE_DIR"):
            config.cache_dir = os.getenv("NATIVE_CACHE_DIR")
        
        if os.getenv("NATIVE_LOG_LEVEL"):
            config.log_level = os.getenv("NATIVE_LOG_LEVEL")
        
        if os.getenv("NATIVE_GPU_MEMORY_FRACTION"):
            config.processing.gpu_memory_fraction = float(os.getenv("NATIVE_GPU_MEMORY_FRACTION"))
        
        return config


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions
def get_config(name: str = "default") -> NativeModelConfig:
    """Get configuration by name"""
    return config_manager.get_config(name)


def get_platform_config(platform: str, config_name: str = "default") -> PlatformConfig:
    """Get platform-specific configuration"""
    config = get_config(config_name)
    return config.get_platform_config(platform)


def create_small_config() -> NativeModelConfig:
    """Create small model configuration"""
    return config_manager.create_config_for_size(ModelSize.SMALL)


def create_medium_config() -> NativeModelConfig:
    """Create medium model configuration"""
    return config_manager.create_config_for_size(ModelSize.MEDIUM)


def create_large_config() -> NativeModelConfig:
    """Create large model configuration"""
    return config_manager.create_config_for_size(ModelSize.LARGE)


def create_xlarge_config() -> NativeModelConfig:
    """Create extra large model configuration"""
    return config_manager.create_config_for_size(ModelSize.XLARGE)