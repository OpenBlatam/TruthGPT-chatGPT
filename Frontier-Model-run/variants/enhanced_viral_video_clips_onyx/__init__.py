"""
Enhanced Viral Video Clips Model with Onyx-Inspired Architecture
Revolutionary AI-powered video processing for viral content creation

This package provides enterprise-grade video processing capabilities inspired by
the Onyx architecture, featuring:

- Advanced LLM interfaces for video understanding
- Factory pattern for dynamic model creation
- Comprehensive configuration management
- Modular tool system for video processing
- Intelligent agent framework for workflow automation
- Multi-platform optimization (TikTok, Instagram, YouTube, etc.)
"""

__version__ = "1.0.0"
__author__ = "Enhanced Viral Video Clips Team"
__description__ = "Enterprise-grade viral video processing with Onyx-inspired architecture"

# Core interfaces
from .interfaces.video_llm_interface import (
    VideoLLM,
    VideoLLMConfig,
    VideoProcessingMode,
    PlatformType,
    VideoMetadata,
    ClipSegment,
    ViralClipOutput,
    VideoLLMException,
    VideoProcessingError,
    ModelNotLoadedError,
    UnsupportedFormatError,
    InsufficientResourcesError
)

# Configuration management
from .configs.video_model_configs import (
    get_model_config,
    get_platform_config,
    get_processing_config,
    validate_config,
    load_custom_config,
    save_config,
    DEFAULT_PROCESSING_CONFIG,
    FAST_PROCESSING_CONFIG,
    HIGH_QUALITY_CONFIG
)

# LLM factory and management
from .llm.video_llm_factory import (
    VideoLLMRegistry,
    VideoLLMManager,
    create_video_llm_config,
    get_default_video_llm,
    get_video_llm_for_platform,
    get_video_llm_for_batch_processing,
    get_video_llm_for_real_time,
    create_video_llm,
    get_available_providers,
    validate_video_llm_config,
    get_video_llm_manager
)

# Main LLM implementation
from .llm.enhanced_viral_video_llm import (
    EnhancedViralVideoLLM,
    VideoUnderstandingTransformer
)

# Video processing tools
from .tools.video_processing_tools import (
    VideoProcessingTool,
    ToolResult,
    YouTubeDownloaderTool,
    VideoAnalyzerTool,
    ClipGeneratorTool,
    CaptionGeneratorTool,
    EffectsApplicatorTool,
    VideoToolRegistry,
    download_youtube_video,
    analyze_video,
    generate_clips,
    add_captions,
    apply_effects
)

# Intelligent agent system
from .agents.viral_video_agent import (
    ViralVideoAgent,
    AgentState,
    AgentTask,
    AgentContext,
    create_viral_video_agent,
    quick_youtube_to_clips,
    quick_video_to_clips
)

# Convenience imports for common use cases
from .llm.video_llm_factory import get_default_video_llm as get_llm
from .agents.viral_video_agent import create_viral_video_agent as create_agent

# Package-level convenience functions
def create_viral_clips_from_youtube(
    url: str,
    platforms: list = None,
    output_dir: str = "./output",
    model_variant: str = "medium"
) -> dict:
    """
    High-level function to create viral clips from YouTube URL
    
    Args:
        url: YouTube video URL
        platforms: List of target platforms (default: TikTok, Instagram, YouTube)
        output_dir: Output directory for generated clips
        model_variant: Model variant to use (small, medium, large)
    
    Returns:
        Dictionary with processing results and generated clips
    """
    import asyncio
    
    if platforms is None:
        platforms = [PlatformType.TIKTOK, PlatformType.INSTAGRAM_REELS, PlatformType.YOUTUBE_SHORTS]
    
    # Create agent with specified model variant
    llm = get_default_video_llm(model_variant=model_variant)
    agent = create_viral_video_agent(output_dir=output_dir, video_llm=llm)
    
    # Process the request
    return asyncio.run(agent.process_request(url, platforms))


def create_viral_clips_from_video(
    video_path: str,
    platforms: list = None,
    output_dir: str = "./output",
    model_variant: str = "medium"
) -> dict:
    """
    High-level function to create viral clips from local video file
    
    Args:
        video_path: Path to local video file
        platforms: List of target platforms (default: TikTok, Instagram, YouTube)
        output_dir: Output directory for generated clips
        model_variant: Model variant to use (small, medium, large)
    
    Returns:
        Dictionary with processing results and generated clips
    """
    import asyncio
    
    if platforms is None:
        platforms = [PlatformType.TIKTOK, PlatformType.INSTAGRAM_REELS, PlatformType.YOUTUBE_SHORTS]
    
    # Create agent with specified model variant
    llm = get_default_video_llm(model_variant=model_variant)
    agent = create_viral_video_agent(output_dir=output_dir, video_llm=llm)
    
    # Process the request
    return asyncio.run(agent.process_request(video_path, platforms))


def analyze_video_viral_potential(
    video_path: str,
    platforms: list = None,
    model_variant: str = "medium"
) -> dict:
    """
    Analyze viral potential of a video for different platforms
    
    Args:
        video_path: Path to video file
        platforms: List of platforms to analyze for
        model_variant: Model variant to use
    
    Returns:
        Dictionary with viral analysis results
    """
    import asyncio
    
    if platforms is None:
        platforms = [PlatformType.TIKTOK, PlatformType.INSTAGRAM_REELS, PlatformType.YOUTUBE_SHORTS]
    
    # Create LLM for analysis
    llm = get_default_video_llm(model_variant=model_variant)
    
    # Extract features and analyze
    features = llm.extract_video_features(video_path)
    viral_scores = llm.analyze_viral_potential(features, platforms)
    highlights = llm.detect_highlights(features)
    
    return {
        "video_path": video_path,
        "viral_scores": viral_scores,
        "highlights": len(highlights),
        "highlight_segments": [
            {
                "start_time": h.start_time,
                "end_time": h.end_time,
                "viral_score": h.viral_score,
                "emotions": h.emotions
            }
            for h in highlights
        ],
        "recommendations": _generate_viral_recommendations(viral_scores, highlights)
    }


def _generate_viral_recommendations(viral_scores: dict, highlights: list) -> list:
    """Generate recommendations based on viral analysis"""
    recommendations = []
    
    # Platform-specific recommendations
    for platform, score in viral_scores.items():
        if score < 0.5:
            recommendations.append(f"Consider optimizing content for {platform} (current score: {score:.2f})")
        elif score > 0.8:
            recommendations.append(f"Excellent viral potential for {platform} (score: {score:.2f})")
    
    # Highlight recommendations
    if len(highlights) == 0:
        recommendations.append("No clear highlights detected - consider adding more dynamic content")
    elif len(highlights) > 10:
        recommendations.append("Many highlights detected - focus on the top 5-10 for best results")
    
    return recommendations


def get_supported_platforms() -> list:
    """Get list of supported social media platforms"""
    return [platform.value for platform in PlatformType]


def get_available_model_variants() -> list:
    """Get list of available model variants"""
    return ["small", "medium", "large"]


def get_processing_modes() -> list:
    """Get list of available processing modes"""
    return [mode.value for mode in VideoProcessingMode]


# Package metadata
__all__ = [
    # Core interfaces
    "VideoLLM",
    "VideoLLMConfig", 
    "VideoProcessingMode",
    "PlatformType",
    "VideoMetadata",
    "ClipSegment",
    "ViralClipOutput",
    
    # Exceptions
    "VideoLLMException",
    "VideoProcessingError",
    "ModelNotLoadedError",
    "UnsupportedFormatError",
    "InsufficientResourcesError",
    
    # Configuration
    "get_model_config",
    "get_platform_config",
    "get_processing_config",
    "validate_config",
    "load_custom_config",
    "save_config",
    
    # LLM factory
    "VideoLLMRegistry",
    "VideoLLMManager",
    "create_video_llm_config",
    "get_default_video_llm",
    "get_video_llm_for_platform",
    "get_video_llm_for_batch_processing",
    "get_video_llm_for_real_time",
    "create_video_llm",
    "get_available_providers",
    "validate_video_llm_config",
    "get_video_llm_manager",
    
    # Main implementation
    "EnhancedViralVideoLLM",
    "VideoUnderstandingTransformer",
    
    # Tools
    "VideoProcessingTool",
    "ToolResult",
    "YouTubeDownloaderTool",
    "VideoAnalyzerTool", 
    "ClipGeneratorTool",
    "CaptionGeneratorTool",
    "EffectsApplicatorTool",
    "VideoToolRegistry",
    "download_youtube_video",
    "analyze_video",
    "generate_clips",
    "add_captions",
    "apply_effects",
    
    # Agent system
    "ViralVideoAgent",
    "AgentState",
    "AgentTask",
    "AgentContext",
    "create_viral_video_agent",
    "quick_youtube_to_clips",
    "quick_video_to_clips",
    
    # Convenience functions
    "create_viral_clips_from_youtube",
    "create_viral_clips_from_video",
    "analyze_video_viral_potential",
    "get_supported_platforms",
    "get_available_model_variants",
    "get_processing_modes",
    
    # Shortcuts
    "get_llm",
    "create_agent"
]

# Package information
PACKAGE_INFO = {
    "name": "enhanced_viral_video_clips_onyx",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "Multi-modal video understanding with transformers",
        "Viral potential prediction for multiple platforms",
        "Intelligent highlight detection and clip generation",
        "Automated caption generation with styling",
        "Platform-specific optimization (TikTok, Instagram, YouTube, etc.)",
        "Enterprise-grade architecture with factory patterns",
        "Comprehensive configuration management",
        "Modular tool system for extensibility",
        "Intelligent agent framework for workflow automation",
        "Real-time processing capabilities",
        "Batch processing for multiple videos",
        "Advanced effects and transitions",
        "YouTube video extraction and processing",
        "LangChain integration for chat interfaces"
    ],
    "supported_platforms": get_supported_platforms(),
    "model_variants": get_available_model_variants(),
    "processing_modes": get_processing_modes()
}


def print_package_info():
    """Print package information and capabilities"""
    print(f"\n🎬 {PACKAGE_INFO['name']} v{PACKAGE_INFO['version']}")
    print(f"📝 {PACKAGE_INFO['description']}")
    print(f"👨‍💻 {PACKAGE_INFO['author']}")
    
    print("\n🚀 Key Features:")
    for feature in PACKAGE_INFO['features']:
        print(f"  • {feature}")
    
    print(f"\n📱 Supported Platforms: {', '.join(PACKAGE_INFO['supported_platforms'])}")
    print(f"🤖 Model Variants: {', '.join(PACKAGE_INFO['model_variants'])}")
    print(f"⚙️ Processing Modes: {', '.join(PACKAGE_INFO['processing_modes'])}")
    
    print("\n💡 Quick Start:")
    print("  from enhanced_viral_video_clips_onyx import create_viral_clips_from_youtube")
    print("  result = create_viral_clips_from_youtube('https://youtube.com/watch?v=...')")
    print("  print(f'Generated {result[\"total_clips\"]} viral clips!')")


# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version check and compatibility
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("Enhanced Viral Video Clips requires Python 3.8 or higher")

# Optional: Print package info on import (can be disabled with environment variable)
import os
if os.environ.get("ENHANCED_VIRAL_VIDEO_CLIPS_SHOW_INFO", "false").lower() == "true":
    print_package_info()