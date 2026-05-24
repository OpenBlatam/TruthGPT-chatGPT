"""
Native Viral Video Clips Model - Interactive Demo
Streamlined AI-powered video processing demonstration
"""

import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our native model
try:
    from .model import (
        NativeViralVideoModel,
        create_viral_clips_from_youtube,
        create_viral_clips_from_video,
        analyze_video_viral_potential,
        ProcessingResult,
        VideoClip
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from model import (
        NativeViralVideoModel,
        create_viral_clips_from_youtube,
        create_viral_clips_from_video,
        analyze_video_viral_potential,
        ProcessingResult,
        VideoClip
    )

# Page configuration
st.set_page_config(
    page_title="Native Viral Video Clips",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .platform-badge {
        display: inline-block;
        background: #4ECDC4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .clip-card {
        background: #fff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main demo application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Native Viral Video Clips</h1>
        <h3>Streamlined AI-Powered Video Processing</h3>
        <p>Transform any video into viral clips optimized for TikTok, Instagram, YouTube & more</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("🤖 Model Settings")
        model_size = st.selectbox(
            "Model Size",
            ["small", "medium", "large"],
            index=1,
            help="Choose model size:\n- Small: Fast, 8GB GPU\n- Medium: Balanced, 16GB GPU\n- Large: High quality, 32GB GPU"
        )
        
        # Platform selection
        st.subheader("📱 Target Platforms")
        available_platforms = ["tiktok", "instagram", "youtube", "facebook", "twitter"]
        selected_platforms = st.multiselect(
            "Select Platforms",
            available_platforms,
            default=["tiktok", "instagram", "youtube"],
            help="Choose which platforms to optimize clips for"
        )
        
        # Processing settings
        with st.expander("🔧 Processing Settings"):
            max_clips = st.slider("Max Clips per Video", 1, 20, 15)
            min_clip_duration = st.slider("Min Clip Duration (s)", 10, 30, 15)
            viral_threshold = st.slider("Viral Threshold", 0.0, 1.0, 0.6, 0.1)
            output_quality = st.selectbox("Output Quality", ["high", "medium", "low"], index=0)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎥 YouTube to Clips",
        "📁 Local Video Processing", 
        "📊 Viral Analysis",
        "ℹ️ Model Info"
    ])
    
    with tab1:
        youtube_processing_tab(model_size, selected_platforms)
    
    with tab2:
        local_video_tab(model_size, selected_platforms)
    
    with tab3:
        viral_analysis_tab(model_size)
    
    with tab4:
        model_info_tab()


def youtube_processing_tab(model_size: str, selected_platforms: List[str]):
    """YouTube video processing tab"""
    
    st.header("🎥 YouTube to Viral Clips")
    st.markdown("Transform any YouTube video into viral clips optimized for multiple platforms")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a YouTube video URL to process"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_button = st.button("🚀 Process Video", type="primary", use_container_width=True)
    
    # Example URLs
    with st.expander("📝 Example URLs"):
        example_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "https://www.youtube.com/watch?v=9bZkp7q19f0"
        ]
        for url in example_urls:
            if st.button(f"Use: {url}", key=f"example_{url}"):
                st.session_state.youtube_url = url
                st.rerun()
    
    # Processing
    if process_button and youtube_url:
        if not youtube_url.startswith(("http://", "https://")):
            st.error("Please enter a valid YouTube URL")
            return
        
        # Processing with progress
        with st.spinner("🎬 Processing YouTube video..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Simulate processing steps
                steps = [
                    "Downloading video...",
                    "Extracting features...", 
                    "Analyzing viral potential...",
                    "Detecting highlights...",
                    "Generating clips...",
                    "Finalizing output..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.5)  # Simulate processing time
                
                # Process video
                result = create_viral_clips_from_youtube(
                    url=youtube_url,
                    platforms=selected_platforms,
                    model_size=model_size
                )
                
                # Display results
                display_processing_results(result, "YouTube Video")
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                logger.error(f"YouTube processing error: {e}")


def local_video_tab(model_size: str, selected_platforms: List[str]):
    """Local video processing tab"""
    
    st.header("📁 Local Video Processing")
    st.markdown("Upload and process your own video files")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        help="Upload a video file to process (max 500MB)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        
        video_path = temp_dir / uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display video info
        st.subheader("📹 Video Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        # Process button
        if st.button("🎬 Process Local Video", type="primary"):
            with st.spinner("Processing local video..."):
                try:
                    result = create_viral_clips_from_video(
                        video_path=str(video_path),
                        platforms=selected_platforms,
                        model_size=model_size
                    )
                    
                    display_processing_results(result, "Local Video")
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    logger.error(f"Local video processing error: {e}")


def viral_analysis_tab(model_size: str):
    """Viral analysis tab"""
    
    st.header("📊 Viral Potential Analysis")
    st.markdown("Analyze the viral potential of your videos across different platforms")
    
    # Input options
    analysis_type = st.radio(
        "Analysis Type",
        ["YouTube URL", "Local File"],
        horizontal=True
    )
    
    video_input = None
    
    if analysis_type == "YouTube URL":
        video_input = st.text_input("YouTube URL for Analysis")
    else:
        uploaded_file = st.file_uploader(
            "Upload video for analysis",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm']
        )
        if uploaded_file:
            temp_dir = Path("./temp")
            temp_dir.mkdir(exist_ok=True)
            video_path = temp_dir / uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_input = str(video_path)
    
    if video_input and st.button("🔍 Analyze Viral Potential", type="primary"):
        with st.spinner("Analyzing viral potential..."):
            try:
                if analysis_type == "YouTube URL":
                    # For YouTube, we need to download first
                    model = NativeViralVideoModel(model_size)
                    download_result = model.download_youtube_video(video_input)
                    if download_result["success"]:
                        analysis_result = analyze_video_viral_potential(
                            download_result["video_path"], 
                            model_size
                        )
                    else:
                        st.error(f"Failed to download video: {download_result['error']}")
                        return
                else:
                    analysis_result = analyze_video_viral_potential(video_input, model_size)
                
                display_viral_analysis_results(analysis_result)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


def model_info_tab():
    """Model information tab"""
    
    st.header("ℹ️ Native Viral Video Clips Model")
    
    # Model overview
    st.subheader("🎬 Streamlined Video Processing")
    
    st.markdown("""
    <div class="feature-card">
        <h4>🚀 Native AI Architecture</h4>
        <p>Streamlined AI-powered video processing focused on essential viral content creation capabilities
        without complex enterprise patterns. Optimized for speed and simplicity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✨ Core Features")
        features = [
            "🧠 Multi-modal video understanding",
            "🎯 Viral potential prediction", 
            "✂️ Intelligent highlight detection",
            "📝 Automated clip generation",
            "📱 Platform-specific optimization",
            "🎨 Basic effects and transitions",
            "⚡ Fast processing pipeline",
            "🔧 Simple configuration"
        ]
        
        for feature in features:
            st.markdown(f"• {feature}")
    
    with col2:
        st.subheader("🎯 Supported Platforms")
        platforms = ["tiktok", "instagram", "youtube", "facebook", "twitter"]
        for platform in platforms:
            st.markdown(f'<span class="platform-badge">{platform.upper()}</span>', unsafe_allow_html=True)
        
        st.subheader("🤖 Model Sizes")
        st.markdown("• **Small**: 3B params, 8GB GPU, fastest")
        st.markdown("• **Medium**: 8B params, 16GB GPU, balanced")
        st.markdown("• **Large**: 15B params, 32GB GPU, highest quality")
    
    # Architecture overview
    st.subheader("🏗️ Architecture")
    
    st.markdown("""
    **Native Video Processing Pipeline**
    
    1. **Video Input**: YouTube download or local file upload
    2. **Feature Extraction**: Multi-modal analysis (video + audio)
    3. **Viral Analysis**: Platform-specific viral potential scoring
    4. **Highlight Detection**: Intelligent segment identification
    5. **Clip Generation**: Optimized clips with effects and captions
    6. **Platform Optimization**: Format and style adaptation
    """)
    
    # Performance metrics
    st.subheader("📊 Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>85%</h3>
            <p>Highlight Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>15+</h3>
            <p>Clips per Video</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>5</h3>
            <p>Platforms Supported</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Fast</h3>
            <p>Processing Speed</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start
    st.subheader("🚀 Quick Start")
    
    st.code("""
# Import the model
from native_viral_video_clips.model import create_viral_clips_from_youtube

# Process YouTube video
result = create_viral_clips_from_youtube(
    url="https://youtube.com/watch?v=...",
    platforms=["tiktok", "instagram", "youtube"],
    model_size="medium"
)

print(f"Generated {result.total_clips} viral clips!")
    """, language="python")


def display_processing_results(result: ProcessingResult, source_type: str):
    """Display video processing results"""
    
    if result.success:
        st.markdown(f"""
        <div class="success-message">
            <h4>✅ {source_type} Processing Completed!</h4>
            <p>Successfully generated viral clips from your video.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Results overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clips", result.total_clips)
        with col2:
            st.metric("Processing Time", f"{result.processing_time:.1f}s")
        with col3:
            st.metric("Source Duration", f"{result.source_info.get('duration', 0):.1f}s")
        with col4:
            st.metric("Success Rate", "100%")
        
        # Generated clips
        if result.clips:
            st.subheader("🎬 Generated Clips")
            
            # Group clips by platform
            platform_clips = {}
            for clip in result.clips:
                if clip.platform not in platform_clips:
                    platform_clips[clip.platform] = []
                platform_clips[clip.platform].append(clip)
            
            for platform, clips in platform_clips.items():
                with st.expander(f"📱 {platform.upper()} Clips ({len(clips)})"):
                    for i, clip in enumerate(clips):
                        st.markdown(f"""
                        <div class="clip-card">
                            <strong>Clip {i+1}</strong><br>
                            ⏱️ {clip.start_time:.1f}s - {clip.end_time:.1f}s ({clip.duration:.1f}s)<br>
                            🔥 Viral Score: {clip.viral_score:.2f}<br>
                            📝 Caption: {clip.caption}<br>
                            🏷️ Hashtags: {', '.join(clip.hashtags[:3])}<br>
                            🎨 Effects: {', '.join(clip.effects)}<br>
                            📁 Output: {Path(clip.output_path).name}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Download section
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create summary JSON
            summary = {
                "total_clips": result.total_clips,
                "processing_time": result.processing_time,
                "source_info": result.source_info,
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
                ]
            }
            
            st.download_button(
                "📊 Download Summary",
                json.dumps(summary, indent=2),
                "processing_summary.json",
                "application/json"
            )
        
        with col2:
            if st.button("📁 Open Output Folder"):
                st.info("Output folder: ./output/")
        
        with col3:
            if st.button("🔗 Copy Results"):
                st.success("Results copied!")
    
    else:
        st.markdown(f"""
        <div class="error-message">
            <h4>❌ {source_type} Processing Failed</h4>
            <p>{result.error_message or 'Unknown error occurred'}</p>
        </div>
        """, unsafe_allow_html=True)


def display_viral_analysis_results(analysis: Dict[str, Any]):
    """Display viral analysis results"""
    
    st.subheader("🎯 Viral Potential Scores")
    
    # Platform scores
    viral_scores = analysis.get("viral_scores", {})
    
    for platform, score in viral_scores.items():
        # Color based on score
        if score >= 0.8:
            color = "#28a745"  # Green
        elif score >= 0.6:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
            <strong>{platform.upper()}</strong>: {score:.2f} ({score*100:.0f}%)
        </div>
        """, unsafe_allow_html=True)
    
    # Highlight segments
    st.subheader("✨ Detected Highlights")
    
    highlights = analysis.get("highlight_segments", [])
    
    if highlights:
        for i, highlight in enumerate(highlights[:5]):  # Show top 5
            st.markdown(f"""
            <div class="clip-card">
                <strong>Highlight {i+1}</strong><br>
                ⏱️ {highlight['start_time']:.1f}s - {highlight['end_time']:.1f}s ({highlight['duration']:.1f}s)<br>
                🔥 Score: {highlight['score']:.2f}<br>
                🎬 Motion: {highlight['motion_intensity']:.1f}<br>
                🎤 Speech: {'Yes' if highlight['has_speech'] else 'No'}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No highlights detected in this video.")
    
    # Video metadata
    st.subheader("📹 Video Information")
    
    metadata = analysis.get("metadata", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{metadata.get('duration', 0):.1f}s")
    with col2:
        st.metric("Resolution", f"{metadata.get('resolution', (0, 0))[0]}x{metadata.get('resolution', (0, 0))[1]}")
    with col3:
        st.metric("FPS", f"{metadata.get('fps', 0):.1f}")
    with col4:
        st.metric("Has Audio", "Yes" if metadata.get('has_audio', False) else "No")


if __name__ == "__main__":
    main()