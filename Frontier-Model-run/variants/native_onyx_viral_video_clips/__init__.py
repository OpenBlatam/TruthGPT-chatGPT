"""
Native Onyx Viral Video Clips Model
Enterprise-grade viral video processing with pure native AI models

This package provides a comprehensive solution for viral video analysis and clip
generation using only native transformer models and encoders without external API
dependencies. Built with Onyx architecture patterns for enterprise scalability.
"""

__version__ = "1.0.0"
__author__ = "Native Onyx Viral Video Team"
__description__ = "Enterprise-grade viral video processing with pure native AI models"

# Core interfaces
from .interfaces.native_video_interface import (
    NativeVideoLLMInterface,
    VideoAnalysisResult,
    VideoSegment,
    CLIPVideoEncoder,
    GPT2TextEncoder,
    WhisperAudioEncoder,
    StreamingNativeVideoLLM,
    create_native_video_llm,
    analyze_video_native,
    generate_captions_native,
    predict_viral_score_native
)

# Enhanced LLM models
from .llm.enhanced_native_viral_llm import (
    NativeViralVideoModel,
    VideoUnderstandingTransformer,
    create_enhanced_native_viral_llm
)

# Factory and model management
from .llm.native_llm_factory import (
    create_native_llm,
    create_small_llm,
    create_medium_llm,
    create_large_llm,
    create_xlarge_llm,
    create_llm_for_memory,
    get_model_recommendations,
    list_available_models,
    clear_model_cache,
    get_memory_usage,
    ModelInfo,
    NativeLLMFactory
)

# Configuration management
from .configs.native_model_configs import (
    NativeModelConfig,
    VideoEncoderConfig,
    TextEncoderConfig,
    AudioEncoderConfig,
    ViralClassifierConfig,
    PlatformConfig,
    ProcessingConfig,
    ModelSize,
    Platform,
    get_config,
    get_platform_config,
    create_small_config,
    create_medium_config,
    create_large_config,
    create_xlarge_config,
    ConfigManager
)

# Video processing tools
from .tools.native_video_tools import (
    NativeVideoProcessor,
    YouTubeDownloader,
    VideoAnalyzer,
    VideoClip,
    ProcessingResult,
    process_video_to_viral_clips,
    download_youtube_video,
    analyze_video
)

# Intelligent agent
from .agents.native_viral_agent import (
    NativeViralVideoAgent,
    AgentTask,
    AgentCapabilities,
    create_viral_agent,
    quick_youtube_to_clips,
    quick_video_to_clips,
    quick_video_analysis
)

# Convenience functions for quick usage
async def process_youtube_video(
    url: str,
    platforms: list = None,
    model_size: str = "medium",
    output_dir: str = "./output"
) -> dict:
    """
    Quick function to process YouTube video to viral clips
    
    Args:
        url: YouTube video URL
        platforms: List of target platforms (default: tiktok, instagram, youtube)
        model_size: Model size (small, medium, large, xlarge)
        output_dir: Output directory for clips
    
    Returns:
        Dictionary with processing results
    """
    if platforms is None:
        platforms = ["tiktok", "instagram", "youtube"]
    
    # Create configuration based on model size
    if model_size == "small":
        config = create_small_config()
    elif model_size == "large":
        config = create_large_config()
    elif model_size == "xlarge":
        config = create_xlarge_config()
    else:
        config = create_medium_config()
    
    # Create and use agent
    agent = NativeViralVideoAgent(config, output_dir)
    await agent.initialize()
    
    return await agent.process_youtube_video(url, platforms)


async def process_local_video(
    video_path: str,
    platforms: list = None,
    model_size: str = "medium",
    output_dir: str = "./output"
) -> dict:
    """
    Quick function to process local video to viral clips
    
    Args:
        video_path: Path to local video file
        platforms: List of target platforms (default: tiktok, instagram, youtube)
        model_size: Model size (small, medium, large, xlarge)
        output_dir: Output directory for clips
    
    Returns:
        Dictionary with processing results
    """
    if platforms is None:
        platforms = ["tiktok", "instagram", "youtube"]
    
    # Create configuration based on model size
    if model_size == "small":
        config = create_small_config()
    elif model_size == "large":
        config = create_large_config()
    elif model_size == "xlarge":
        config = create_xlarge_config()
    else:
        config = create_medium_config()
    
    # Create and use agent
    agent = NativeViralVideoAgent(config, output_dir)
    await agent.initialize()
    
    return await agent.process_local_video(video_path, platforms)


async def analyze_video_viral_potential(
    video_path: str,
    model_size: str = "medium"
) -> dict:
    """
    Quick function to analyze video viral potential
    
    Args:
        video_path: Path to video file
        model_size: Model size (small, medium, large, xlarge)
    
    Returns:
        Dictionary with analysis results
    """
    # Create configuration based on model size
    if model_size == "small":
        config = create_small_config()
    elif model_size == "large":
        config = create_large_config()
    elif model_size == "xlarge":
        config = create_xlarge_config()
    else:
        config = create_medium_config()
    
    # Create and use agent
    agent = NativeViralVideoAgent(config)
    await agent.initialize()
    
    return await agent.analyze_video_only(video_path)


def get_supported_platforms() -> list:
    """Get list of supported platforms"""
    return ["tiktok", "instagram", "youtube", "facebook", "twitter"]


def get_available_model_sizes() -> list:
    """Get list of available model sizes"""
    return ["small", "medium", "large", "xlarge"]


def get_model_memory_requirements() -> dict:
    """Get memory requirements for different model sizes"""
    return {
        "small": "4GB",
        "medium": "8GB", 
        "large": "16GB",
        "xlarge": "32GB"
    }


def print_package_info():
    """Print package information"""
    print(f"\n🎬 {PACKAGE_INFO['name']} v{PACKAGE_INFO['version']}")
    print(f"📝 {PACKAGE_INFO['description']}")
    print(f"👨‍💻 {PACKAGE_INFO['author']}")
    
    print("\n✨ Key Features:")
    for feature in PACKAGE_INFO['features']:
        print(f"  • {feature}")
    
    print(f"\n📱 Supported Platforms: {', '.join(get_supported_platforms())}")
    print(f"🤖 Model Sizes: {', '.join(get_available_model_sizes())}")
    
    print("\n💾 Memory Requirements:")
    for size, memory in get_model_memory_requirements().items():
        print(f"  • {size.capitalize()}: {memory}")
    
    print("\n🚀 Quick Start:")
    print("  import asyncio")
    print("  from native_onyx_viral_video_clips import process_youtube_video")
    print("  ")
    print("  async def main():")
    print("      result = await process_youtube_video('https://youtube.com/watch?v=...')")
    print("      print(f'Generated {result[\"total_clips\"]} clips!')")
    print("  ")
    print("  asyncio.run(main())")


# Package metadata
PACKAGE_INFO = {
    "name": "native_onyx_viral_video_clips",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "Pure native AI models (no external APIs)",
        "Enterprise Onyx architecture patterns",
        "Multi-modal video understanding",
        "Viral potential prediction",
        "Intelligent highlight detection",
        "Automated clip generation",
        "Multi-platform optimization",
        "YouTube video processing",
        "Real-time analysis",
        "Batch processing support",
        "Advanced video effects",
        "Configurable model sizes",
        "Memory-efficient processing",
        "Comprehensive logging",
        "Task management system"
    ],
    "supported_platforms": get_supported_platforms(),
    "model_sizes": get_available_model_sizes(),
    "memory_requirements": get_model_memory_requirements(),
    "architecture": "Native Onyx Enterprise",
    "dependencies": "Native only (no external LLM APIs)"
}


# Export main components
__all__ = [
    # Core interfaces
    "NativeVideoLLMInterface",
    "VideoAnalysisResult", 
    "VideoSegment",
    "CLIPVideoEncoder",
    "GPT2TextEncoder",
    "WhisperAudioEncoder",
    "StreamingNativeVideoLLM",
    "create_native_video_llm",
    "analyze_video_native",
    "generate_captions_native",
    "predict_viral_score_native",
    
    # Enhanced models
    "NativeViralVideoModel",
    "VideoUnderstandingTransformer",
    "create_enhanced_native_viral_llm",
    
    # Factory and management
    "create_native_llm",
    "create_small_llm",
    "create_medium_llm", 
    "create_large_llm",
    "create_xlarge_llm",
    "create_llm_for_memory",
    "get_model_recommendations",
    "list_available_models",
    "clear_model_cache",
    "get_memory_usage",
    "ModelInfo",
    "NativeLLMFactory",
    
    # Configuration
    "NativeModelConfig",
    "VideoEncoderConfig",
    "TextEncoderConfig",
    "AudioEncoderConfig",
    "ViralClassifierConfig",
    "PlatformConfig",
    "ProcessingConfig",
    "ModelSize",
    "Platform",
    "get_config",
    "get_platform_config",
    "create_small_config",
    "create_medium_config",
    "create_large_config",
    "create_xlarge_config",
    "ConfigManager",
    
    # Tools
    "NativeVideoProcessor",
    "YouTubeDownloader",
    "VideoAnalyzer",
    "VideoClip",
    "ProcessingResult",
    "process_video_to_viral_clips",
    "download_youtube_video",
    "analyze_video",
    
    # Agent
    "NativeViralVideoAgent",
    "AgentTask",
    "AgentCapabilities",
    "create_viral_agent",
    "quick_youtube_to_clips",
    "quick_video_to_clips",
    "quick_video_analysis",
    
    # Convenience functions
    "process_youtube_video",
    "process_local_video",
    "analyze_video_viral_potential",
    "get_supported_platforms",
    "get_available_model_sizes",
    "get_model_memory_requirements",
    "print_package_info",
    
    # Package info
    "PACKAGE_INFO"
]

# Version check
import sys
if sys.version_info < (3.8):
    raise RuntimeError("Native Onyx Viral Video Clips requires Python 3.8 or higher")

# Optional: Print info on import
import os
if os.environ.get("NATIVE_ONYX_SHOW_INFO", "false").lower() == "true":
    print_package_info()