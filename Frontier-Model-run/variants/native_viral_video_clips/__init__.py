"""
Native Viral Video Clips Model
Streamlined AI-powered video processing for viral content creation

A simplified, native implementation focused on essential viral video processing 
capabilities without complex enterprise patterns. Optimized for speed, simplicity, 
and effectiveness.
"""

__version__ = "1.0.0"
__author__ = "Native Viral Video Clips Team"
__description__ = "Streamlined AI-powered video processing for viral content creation"

# Core model components
from .model import (
    NativeViralVideoModel,
    ViralVideoTransformer,
    VideoClip,
    ProcessingResult,
    create_viral_clips_from_youtube,
    create_viral_clips_from_video,
    analyze_video_viral_potential
)

# Convenience functions
def quick_youtube_to_clips(url: str, platforms: list = None) -> dict:
    """
    Quick function to convert YouTube video to viral clips
    
    Args:
        url: YouTube video URL
        platforms: List of target platforms (default: tiktok, instagram, youtube)
    
    Returns:
        Dictionary with processing results
    """
    if platforms is None:
        platforms = ["tiktok", "instagram", "youtube"]
    
    result = create_viral_clips_from_youtube(url, platforms)
    
    return {
        "success": result.success,
        "total_clips": result.total_clips,
        "processing_time": result.processing_time,
        "clips": [
            {
                "platform": clip.platform,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "viral_score": clip.viral_score,
                "caption": clip.caption,
                "output_path": clip.output_path
            }
            for clip in result.clips
        ],
        "error": result.error_message
    }


def quick_video_analysis(video_path: str) -> dict:
    """
    Quick function to analyze video viral potential
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with analysis results
    """
    analysis = analyze_video_viral_potential(video_path)
    
    return {
        "viral_scores": analysis["viral_scores"],
        "highlights_count": analysis["highlights"],
        "top_highlights": analysis["highlight_segments"][:3],
        "metadata": analysis["metadata"]
    }


def get_supported_platforms() -> list:
    """Get list of supported platforms"""
    return ["tiktok", "instagram", "youtube", "facebook", "twitter"]


def get_model_sizes() -> list:
    """Get list of available model sizes"""
    return ["small", "medium", "large"]


# Package metadata
PACKAGE_INFO = {
    "name": "native_viral_video_clips",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "Multi-modal video understanding",
        "Viral potential prediction",
        "Intelligent highlight detection", 
        "Automated clip generation",
        "Platform-specific optimization",
        "Basic effects and transitions",
        "Fast processing pipeline",
        "Simple configuration"
    ],
    "supported_platforms": get_supported_platforms(),
    "model_sizes": get_model_sizes()
}


def print_info():
    """Print package information"""
    print(f"\n🎬 {PACKAGE_INFO['name']} v{PACKAGE_INFO['version']}")
    print(f"📝 {PACKAGE_INFO['description']}")
    print(f"👨‍💻 {PACKAGE_INFO['author']}")
    
    print("\n✨ Features:")
    for feature in PACKAGE_INFO['features']:
        print(f"  • {feature}")
    
    print(f"\n📱 Supported Platforms: {', '.join(PACKAGE_INFO['supported_platforms'])}")
    print(f"🤖 Model Sizes: {', '.join(PACKAGE_INFO['model_sizes'])}")
    
    print("\n🚀 Quick Start:")
    print("  from native_viral_video_clips import quick_youtube_to_clips")
    print("  result = quick_youtube_to_clips('https://youtube.com/watch?v=...')")
    print("  print(f'Generated {result[\"total_clips\"]} clips!')")


# Export main components
__all__ = [
    # Core classes
    "NativeViralVideoModel",
    "ViralVideoTransformer", 
    "VideoClip",
    "ProcessingResult",
    
    # Main functions
    "create_viral_clips_from_youtube",
    "create_viral_clips_from_video",
    "analyze_video_viral_potential",
    
    # Convenience functions
    "quick_youtube_to_clips",
    "quick_video_analysis",
    "get_supported_platforms",
    "get_model_sizes",
    "print_info",
    
    # Package info
    "PACKAGE_INFO"
]

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("Native Viral Video Clips requires Python 3.8 or higher")

# Optional: Print info on import
import os
if os.environ.get("NATIVE_VIRAL_VIDEO_CLIPS_SHOW_INFO", "false").lower() == "true":
    print_info()