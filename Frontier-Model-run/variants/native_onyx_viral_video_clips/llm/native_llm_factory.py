"""
Native LLM Factory
Factory pattern for creating native video LLM instances without external APIs

This module provides a factory for creating and managing native transformer-based
video understanding models using only local AI models.
"""

import logging
from typing import Dict, Type, Optional, Any, List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from dataclasses import dataclass

from ..interfaces.native_video_interface import (
    NativeVideoLLMInterface,
    StreamingNativeVideoLLM,
    CLIPVideoEncoder,
    GPT2TextEncoder,
    WhisperAudioEncoder
)
from ..configs.native_model_configs import NativeModelConfig, ModelSize

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a native model"""
    name: str
    description: str
    model_class: Type[NativeVideoLLMInterface]
    size: ModelSize
    memory_gb: float
    features: List[str]
    supported_platforms: List[str]


class NativeLLMRegistry:
    """Registry for native LLM models"""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._register_default_models()
    
    def register_model(self, model_info: ModelInfo):
        """Register a new model"""
        self._models[model_info.name] = model_info
        logger.info(f"Registered native model: {model_info.name}")
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get model information"""
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """List available models"""
        return list(self._models.keys())
    
    def get_models_by_size(self, size: ModelSize) -> List[str]:
        """Get models by size"""
        return [name for name, info in self._models.items() if info.size == size]
    
    def get_models_by_memory(self, max_memory_gb: float) -> List[str]:
        """Get models that fit in memory limit"""
        return [name for name, info in self._models.items() if info.memory_gb <= max_memory_gb]
    
    def _register_default_models(self):
        """Register default native models"""
        # Small models
        self.register_model(ModelInfo(
            name="native_small",
            description="Small native video LLM for fast processing",
            model_class=StreamingNativeVideoLLM,
            size=ModelSize.SMALL,
            memory_gb=4.0,
            features=["video_analysis", "caption_generation", "viral_prediction"],
            supported_platforms=["tiktok", "instagram", "youtube"]
        ))
        
        # Medium models
        self.register_model(ModelInfo(
            name="native_medium",
            description="Medium native video LLM for balanced performance",
            model_class=StreamingNativeVideoLLM,
            size=ModelSize.MEDIUM,
            memory_gb=8.0,
            features=["video_analysis", "caption_generation", "viral_prediction", "highlight_detection"],
            supported_platforms=["tiktok", "instagram", "youtube", "facebook", "twitter"]
        ))
        
        # Large models
        self.register_model(ModelInfo(
            name="native_large",
            description="Large native video LLM for high-quality analysis",
            model_class=StreamingNativeVideoLLM,
            size=ModelSize.LARGE,
            memory_gb=16.0,
            features=["video_analysis", "caption_generation", "viral_prediction", "highlight_detection", "emotion_analysis"],
            supported_platforms=["tiktok", "instagram", "youtube", "facebook", "twitter"]
        ))
        
        # Extra large models
        self.register_model(ModelInfo(
            name="native_xlarge",
            description="Extra large native video LLM for maximum quality",
            model_class=StreamingNativeVideoLLM,
            size=ModelSize.XLARGE,
            memory_gb=32.0,
            features=["video_analysis", "caption_generation", "viral_prediction", "highlight_detection", "emotion_analysis", "object_detection"],
            supported_platforms=["tiktok", "instagram", "youtube", "facebook", "twitter"]
        ))


class NativeLLMFactory:
    """Factory for creating native video LLM instances"""
    
    def __init__(self):
        self.registry = NativeLLMRegistry()
        self._instances: Dict[str, NativeVideoLLMInterface] = {}
    
    def create_llm(
        self,
        model_name: str = "native_medium",
        config: Optional[NativeModelConfig] = None,
        **kwargs
    ) -> NativeVideoLLMInterface:
        """Create a native video LLM instance"""
        
        # Get model info
        model_info = self.registry.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create configuration if not provided
        if config is None:
            config = NativeModelConfig()
            config.update_model_size(model_info.size)
        
        # Create instance key for caching
        instance_key = f"{model_name}_{id(config)}"
        
        # Return cached instance if available
        if instance_key in self._instances:
            return self._instances[instance_key]
        
        # Create new instance
        logger.info(f"Creating native LLM: {model_name}")
        
        try:
            # Create the model instance
            llm = model_info.model_class()
            
            # Configure the model
            self._configure_llm(llm, config, model_info)
            
            # Cache the instance
            self._instances[instance_key] = llm
            
            logger.info(f"Successfully created native LLM: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create native LLM {model_name}: {e}")
            raise
    
    def create_llm_by_size(
        self,
        size: ModelSize,
        config: Optional[NativeModelConfig] = None,
        **kwargs
    ) -> NativeVideoLLMInterface:
        """Create LLM by model size"""
        models = self.registry.get_models_by_size(size)
        if not models:
            raise ValueError(f"No models available for size: {size}")
        
        # Use the first available model of the requested size
        return self.create_llm(models[0], config, **kwargs)
    
    def create_llm_for_memory(
        self,
        max_memory_gb: float,
        config: Optional[NativeModelConfig] = None,
        **kwargs
    ) -> NativeVideoLLMInterface:
        """Create LLM that fits in memory limit"""
        models = self.registry.get_models_by_memory(max_memory_gb)
        if not models:
            raise ValueError(f"No models available for memory limit: {max_memory_gb}GB")
        
        # Sort by memory usage (descending) to get the largest model that fits
        model_infos = [(name, self.registry.get_model_info(name)) for name in models]
        model_infos.sort(key=lambda x: x[1].memory_gb, reverse=True)
        
        return self.create_llm(model_infos[0][0], config, **kwargs)
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        return [self.registry.get_model_info(name) for name in self.registry.list_models()]
    
    def get_model_recommendations(
        self,
        use_case: str = "general",
        memory_limit_gb: Optional[float] = None,
        platforms: Optional[List[str]] = None
    ) -> List[str]:
        """Get model recommendations based on use case"""
        
        available_models = self.registry.list_models()
        
        # Filter by memory limit
        if memory_limit_gb:
            available_models = [
                name for name in available_models
                if self.registry.get_model_info(name).memory_gb <= memory_limit_gb
            ]
        
        # Filter by platforms
        if platforms:
            available_models = [
                name for name in available_models
                if all(platform in self.registry.get_model_info(name).supported_platforms for platform in platforms)
            ]
        
        # Sort by use case preferences
        if use_case == "speed":
            # Prefer smaller, faster models
            available_models.sort(key=lambda x: self.registry.get_model_info(x).memory_gb)
        elif use_case == "quality":
            # Prefer larger, higher-quality models
            available_models.sort(key=lambda x: self.registry.get_model_info(x).memory_gb, reverse=True)
        elif use_case == "balanced":
            # Prefer medium-sized models
            available_models.sort(key=lambda x: abs(self.registry.get_model_info(x).memory_gb - 8.0))
        
        return available_models
    
    def _configure_llm(
        self,
        llm: NativeVideoLLMInterface,
        config: NativeModelConfig,
        model_info: ModelInfo
    ):
        """Configure LLM instance"""
        
        # Initialize encoders with configuration
        if hasattr(llm, 'video_encoder') and llm.video_encoder:
            self._configure_video_encoder(llm.video_encoder, config.video_encoder)
        
        if hasattr(llm, 'text_encoder') and llm.text_encoder:
            self._configure_text_encoder(llm.text_encoder, config.text_encoder)
        
        if hasattr(llm, 'audio_encoder') and llm.audio_encoder:
            self._configure_audio_encoder(llm.audio_encoder, config.audio_encoder)
        
        # Configure viral classifier if available
        if hasattr(llm, 'viral_classifier') and llm.viral_classifier:
            self._configure_viral_classifier(llm.viral_classifier, config.viral_classifier)
        
        logger.info(f"Configured LLM with {model_info.size.value} model size")
    
    def _configure_video_encoder(self, encoder: CLIPVideoEncoder, config):
        """Configure video encoder"""
        if hasattr(encoder, 'model_name'):
            encoder.model_name = config.model_name
        if hasattr(encoder, 'max_frames'):
            encoder.max_frames = config.max_frames
        if hasattr(encoder, 'batch_size'):
            encoder.batch_size = config.batch_size
    
    def _configure_text_encoder(self, encoder: GPT2TextEncoder, config):
        """Configure text encoder"""
        if hasattr(encoder, 'model_name'):
            encoder.model_name = config.model_name
        if hasattr(encoder, 'max_length'):
            encoder.max_length = config.max_length
        if hasattr(encoder, 'temperature'):
            encoder.temperature = config.temperature
    
    def _configure_audio_encoder(self, encoder: WhisperAudioEncoder, config):
        """Configure audio encoder"""
        if hasattr(encoder, 'model_name'):
            encoder.model_name = config.model_name
        if hasattr(encoder, 'sample_rate'):
            encoder.sample_rate = config.sample_rate
    
    def _configure_viral_classifier(self, classifier: nn.Module, config):
        """Configure viral classifier"""
        # Classifier configuration is handled during model creation
        pass
    
    def clear_cache(self):
        """Clear cached instances"""
        self._instances.clear()
        logger.info("Cleared LLM instance cache")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of cached instances"""
        usage = {}
        for key, instance in self._instances.items():
            if hasattr(instance, 'get_memory_usage'):
                usage[key] = instance.get_memory_usage()
            else:
                # Estimate memory usage
                usage[key] = self._estimate_memory_usage(instance)
        return usage
    
    def _estimate_memory_usage(self, instance: NativeVideoLLMInterface) -> float:
        """Estimate memory usage of an instance"""
        total_params = 0
        
        # Count parameters in encoders
        if hasattr(instance, 'video_encoder') and instance.video_encoder:
            if hasattr(instance.video_encoder, 'model'):
                total_params += sum(p.numel() for p in instance.video_encoder.model.parameters())
        
        if hasattr(instance, 'text_encoder') and instance.text_encoder:
            if hasattr(instance.text_encoder, 'model'):
                total_params += sum(p.numel() for p in instance.text_encoder.model.parameters())
        
        if hasattr(instance, 'audio_encoder') and instance.audio_encoder:
            if hasattr(instance.audio_encoder, 'model'):
                total_params += sum(p.numel() for p in instance.audio_encoder.model.parameters())
        
        if hasattr(instance, 'viral_classifier') and instance.viral_classifier:
            total_params += sum(p.numel() for p in instance.viral_classifier.parameters())
        
        # Estimate memory usage (4 bytes per parameter for float32)
        memory_gb = (total_params * 4) / (1024 ** 3)
        return memory_gb


# Global factory instance
native_llm_factory = NativeLLMFactory()


# Convenience functions
def create_native_llm(
    model_name: str = "native_medium",
    config: Optional[NativeModelConfig] = None,
    **kwargs
) -> NativeVideoLLMInterface:
    """Create a native video LLM instance"""
    return native_llm_factory.create_llm(model_name, config, **kwargs)


def create_small_llm(config: Optional[NativeModelConfig] = None) -> NativeVideoLLMInterface:
    """Create small native LLM"""
    return native_llm_factory.create_llm_by_size(ModelSize.SMALL, config)


def create_medium_llm(config: Optional[NativeModelConfig] = None) -> NativeVideoLLMInterface:
    """Create medium native LLM"""
    return native_llm_factory.create_llm_by_size(ModelSize.MEDIUM, config)


def create_large_llm(config: Optional[NativeModelConfig] = None) -> NativeVideoLLMInterface:
    """Create large native LLM"""
    return native_llm_factory.create_llm_by_size(ModelSize.LARGE, config)


def create_xlarge_llm(config: Optional[NativeModelConfig] = None) -> NativeVideoLLMInterface:
    """Create extra large native LLM"""
    return native_llm_factory.create_llm_by_size(ModelSize.XLARGE, config)


def create_llm_for_memory(max_memory_gb: float, config: Optional[NativeModelConfig] = None) -> NativeVideoLLMInterface:
    """Create LLM that fits in memory limit"""
    return native_llm_factory.create_llm_for_memory(max_memory_gb, config)


def get_model_recommendations(
    use_case: str = "general",
    memory_limit_gb: Optional[float] = None,
    platforms: Optional[List[str]] = None
) -> List[str]:
    """Get model recommendations"""
    return native_llm_factory.get_model_recommendations(use_case, memory_limit_gb, platforms)


def list_available_models() -> List[ModelInfo]:
    """List available native models"""
    return native_llm_factory.get_available_models()


def clear_model_cache():
    """Clear model cache"""
    native_llm_factory.clear_cache()


def get_memory_usage() -> Dict[str, float]:
    """Get memory usage of cached models"""
    return native_llm_factory.get_memory_usage()