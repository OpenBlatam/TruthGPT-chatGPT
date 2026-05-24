#!/usr/bin/env python3
"""
Test script for Enhanced Viral Video Clips Model imports
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_basic_imports():
    """Test basic module imports"""
    print("🧪 Testing Enhanced Viral Video Clips Model imports...")
    
    try:
        # Test interface imports
        from interfaces.video_llm_interface import (
            VideoLLM, VideoLLMConfig, VideoProcessingMode, PlatformType
        )
        print("✅ Interface imports successful")
        
        # Test config imports
        from configs.video_model_configs import (
            get_model_config, get_platform_config, get_processing_config
        )
        print("✅ Configuration imports successful")
        
        # Test factory imports
        from llm.video_llm_factory import (
            VideoLLMRegistry, create_video_llm_config
        )
        print("✅ Factory imports successful")
        
        # Test tools imports
        from tools.video_processing_tools import (
            VideoProcessingTool, VideoToolRegistry
        )
        print("✅ Tools imports successful")
        
        # Test agents imports
        from agents.viral_video_agent import (
            ViralVideoAgent, AgentState, AgentTask
        )
        print("✅ Agent imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_enums_and_configs():
    """Test enum values and configurations"""
    print("\n🔧 Testing enums and configurations...")
    
    try:
        from interfaces.video_llm_interface import VideoProcessingMode, PlatformType
        
        # Test processing modes
        modes = [mode.value for mode in VideoProcessingMode]
        print(f"📋 Processing modes: {modes}")
        
        # Test platforms
        platforms = [platform.value for platform in PlatformType]
        print(f"📱 Supported platforms: {platforms}")
        
        # Test model variants
        from configs.video_model_configs import VIDEO_MODEL_VARIANTS
        variants = list(VIDEO_MODEL_VARIANTS.keys())
        print(f"🤖 Model variants: {variants}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_model_creation():
    """Test model configuration creation"""
    print("\n🏭 Testing model configuration creation...")
    
    try:
        from llm.video_llm_factory import create_video_llm_config
        from interfaces.video_llm_interface import VideoProcessingMode, PlatformType
        
        # Create a test configuration
        config = create_video_llm_config(
            model_variant="medium",
            processing_mode=VideoProcessingMode.VIRAL_CLIPS,
            target_platforms=[PlatformType.TIKTOK, PlatformType.INSTAGRAM_REELS]
        )
        
        print(f"✅ Created config for {config.model_variant} variant")
        print(f"📱 Target platforms: {[p.value for p in config.target_platforms]}")
        print(f"🎯 Processing mode: {config.video_processing_mode.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_tool_registry():
    """Test tool registry functionality"""
    print("\n🔧 Testing tool registry...")
    
    try:
        from tools.video_processing_tools import VideoToolRegistry
        
        # Get available tools
        tools = VideoToolRegistry.list_tools()
        print(f"🛠️ Available tools: {tools}")
        
        # Test getting a specific tool
        if "youtube_downloader" in tools:
            tool = VideoToolRegistry.get_tool("youtube_downloader")
            print(f"✅ Retrieved tool: {tool.name} - {tool.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool registry error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎬 Enhanced Viral Video Clips Model - Test Suite")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_enums_and_configs,
        test_model_creation,
        test_tool_registry
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Enhanced Viral Video Clips Model is ready!")
        print("\n🚀 Quick Start:")
        print("   from enhanced_viral_video_clips_onyx import create_viral_clips_from_youtube")
        print("   result = create_viral_clips_from_youtube('https://youtube.com/watch?v=...')")
        print("   print(f'Generated {result[\"total_clips\"]} viral clips!')")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)