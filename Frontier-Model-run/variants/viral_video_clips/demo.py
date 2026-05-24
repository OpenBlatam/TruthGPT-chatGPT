"""
Viral Video Clips Model - Interactive Demo

This demo showcases the Viral Video Clips model capabilities:
1. YouTube video extraction and analysis
2. Intelligent highlight detection and viral moment identification
3. Automatic clip generation with optimal duration
4. Dynamic caption generation with animations
5. Visual effects and transitions application
6. Logo and branding integration
7. Platform-specific optimization (TikTok, Instagram, YouTube Shorts)
8. Viral potential prediction and engagement scoring
9. Real-time processing and batch generation
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import moviepy.editor as mp
from PIL import Image
import json
import yaml
import os
import time
import tempfile
import requests
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import base64
from io import BytesIO
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import subprocess
import hashlib

from model import ViralVideoClipsModel, ViralVideoClipsConfig, VideoClip, VideoAnalysis


class ViralVideoClipsDemo:
    """Interactive demo for Viral Video Clips model"""
    
    def __init__(self, config_path: str, model_size: str = "medium"):
        self.config_path = config_path
        self.model_size = model_size
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = None
        self._initialize_model()
        
        # Demo state
        self.current_video_analysis = None
        self.current_clips = []
        self.processing_history = []
        
        # Viral content templates
        self.viral_templates = self._load_viral_templates()
        
        # Platform specifications
        self.platform_specs = {
            'tiktok': {
                'name': 'TikTok',
                'aspect_ratio': '9:16',
                'max_duration': 60,
                'optimal_duration': [15, 30],
                'features': ['Quick cuts', 'Trending audio', 'Text overlays', 'Effects'],
                'hashtags': ['#fyp', '#viral', '#trending', '#foryou']
            },
            'instagram': {
                'name': 'Instagram Reels',
                'aspect_ratio': '9:16',
                'max_duration': 90,
                'optimal_duration': [15, 60],
                'features': ['Stories integration', 'Music sync', 'AR effects', 'Shopping tags'],
                'hashtags': ['#reels', '#viral', '#explore', '#instagram']
            },
            'youtube_shorts': {
                'name': 'YouTube Shorts',
                'aspect_ratio': '9:16',
                'max_duration': 60,
                'optimal_duration': [15, 45],
                'features': ['Thumbnails', 'End screens', 'Chapters', 'Analytics'],
                'hashtags': ['#shorts', '#viral', '#youtube']
            }
        }
    
    def _load_config(self) -> ViralVideoClipsConfig:
        """Load model configuration"""
        try:
            config = ViralVideoClipsConfig.from_yaml(self.config_path)
            
            # Apply model size variant
            if self.model_size in ['small', 'medium', 'large']:
                with open(self.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                variant_config = yaml_config.get('model_variants', {}).get(self.model_size, {})
                for key, value in variant_config.items():
                    if hasattr(config, key) and key != 'description':
                        setattr(config, key, value)
            
            return config
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return ViralVideoClipsConfig()
    
    def _initialize_model(self):
        """Initialize model"""
        try:
            # Create model with configuration
            self.model = ViralVideoClipsModel(self.config)
            
            # Set to evaluation mode
            self.model.eval()
            
            st.success(f"Model initialized successfully! ({self.model_size} variant)")
            
        except Exception as e:
            st.error(f"Error initializing model: {e}")
            self.model = None
    
    def _load_viral_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load viral content templates"""
        return {
            'comedy': {
                'description': 'Funny moments and comedic content',
                'typical_duration': [15, 30],
                'key_elements': ['Punchline', 'Reaction', 'Timing'],
                'effects': ['Quick cuts', 'Zoom ins', 'Sound effects'],
                'caption_style': 'humorous'
            },
            'tutorial': {
                'description': 'Educational and how-to content',
                'typical_duration': [30, 60],
                'key_elements': ['Hook', 'Steps', 'Result'],
                'effects': ['Text overlays', 'Arrows', 'Highlights'],
                'caption_style': 'informative'
            },
            'transformation': {
                'description': 'Before and after content',
                'typical_duration': [15, 45],
                'key_elements': ['Before', 'Process', 'After'],
                'effects': ['Split screen', 'Transitions', 'Reveals'],
                'caption_style': 'dramatic'
            },
            'reaction': {
                'description': 'Reaction and response videos',
                'typical_duration': [15, 30],
                'key_elements': ['Original content', 'Reaction', 'Commentary'],
                'effects': ['Picture in picture', 'Overlays', 'Highlights'],
                'caption_style': 'emotional'
            },
            'challenge': {
                'description': 'Trending challenges and dances',
                'typical_duration': [15, 30],
                'key_elements': ['Setup', 'Attempt', 'Result'],
                'effects': ['Music sync', 'Slow motion', 'Replays'],
                'caption_style': 'energetic'
            }
        }
    
    def process_youtube_video(self, url: str) -> Tuple[Optional[VideoAnalysis], List[VideoClip]]:
        """Process YouTube video and generate clips"""
        try:
            if not self.model:
                st.error("Model not initialized")
                return None, []
            
            with st.spinner(f"Processing YouTube video: {url}"):
                # Simulate processing time
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract video
                status_text.text("📥 Extracting video from YouTube...")
                progress_bar.progress(10)
                time.sleep(1)
                
                # Step 2: Analyze video
                status_text.text("🔍 Analyzing video content...")
                progress_bar.progress(30)
                time.sleep(2)
                
                # Step 3: Detect highlights
                status_text.text("⭐ Detecting viral highlights...")
                progress_bar.progress(50)
                time.sleep(1)
                
                # Step 4: Generate clips
                status_text.text("✂️ Generating viral clips...")
                progress_bar.progress(70)
                time.sleep(2)
                
                # Step 5: Apply effects
                status_text.text("🎨 Applying effects and captions...")
                progress_bar.progress(90)
                time.sleep(1)
                
                # Simulate processing (in real implementation, would call model)
                video_analysis, clips = self._simulate_video_processing(url)
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Store results
                self.current_video_analysis = video_analysis
                self.current_clips = clips
                
                # Add to history
                self.processing_history.append({
                    'timestamp': datetime.now(),
                    'url': url,
                    'video_analysis': video_analysis,
                    'clips_generated': len(clips)
                })
                
                return video_analysis, clips
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return None, []
    
    def _simulate_video_processing(self, url: str) -> Tuple[VideoAnalysis, List[VideoClip]]:
        """Simulate video processing for demo purposes"""
        
        # Extract video ID from URL
        video_id = self._extract_video_id(url)
        
        # Simulate video analysis
        video_analysis = VideoAnalysis(
            video_id=video_id,
            title=f"Amazing Video Content - {video_id}",
            duration=np.random.uniform(120, 600),  # 2-10 minutes
            resolution=(1920, 1080),
            fps=30,
            audio_features={
                'has_speech': True,
                'has_music': np.random.choice([True, False]),
                'speech_clarity': np.random.uniform(0.7, 1.0),
                'music_energy': np.random.uniform(0.3, 0.9),
                'audio_quality': np.random.uniform(0.8, 1.0)
            },
            visual_features={
                'brightness': np.random.uniform(0.4, 0.8),
                'contrast': np.random.uniform(0.5, 0.9),
                'saturation': np.random.uniform(0.6, 1.0),
                'motion_intensity': np.random.uniform(0.3, 0.8),
                'face_presence': np.random.uniform(0.2, 0.9)
            },
            scene_changes=[],
            highlight_moments=[],
            speech_segments=[],
            face_detections=[],
            object_detections=[],
            motion_analysis={},
            viral_potential=np.random.uniform(0.4, 0.95),
            trending_topics=[],
            recommended_clips=[]
        )
        
        # Generate scene changes
        num_scenes = int(video_analysis.duration / 15)  # Scene every 15 seconds on average
        scene_changes = sorted(np.random.uniform(0, video_analysis.duration, num_scenes))
        video_analysis.scene_changes = scene_changes
        
        # Generate highlight moments
        num_highlights = np.random.randint(3, 8)
        highlights = []
        for i in range(num_highlights):
            start = np.random.uniform(0, video_analysis.duration - 30)
            duration = np.random.uniform(10, 30)
            end = min(video_analysis.duration, start + duration)
            
            highlights.append({
                'start': start,
                'end': end,
                'peak_time': start + duration / 2,
                'score': np.random.uniform(0.5, 1.0),
                'type': np.random.choice(['comedy', 'surprise', 'emotional', 'action', 'reveal']),
                'confidence': np.random.uniform(0.7, 0.95)
            })
        
        video_analysis.highlight_moments = sorted(highlights, key=lambda x: x['score'], reverse=True)
        
        # Generate speech segments
        num_segments = np.random.randint(5, 15)
        speech_segments = []
        for i in range(num_segments):
            start = np.random.uniform(0, video_analysis.duration - 10)
            duration = np.random.uniform(3, 15)
            end = min(video_analysis.duration, start + duration)
            
            speech_segments.append({
                'start': start,
                'end': end,
                'text': f"This is speech segment {i+1} with interesting content about the topic.",
                'confidence': np.random.uniform(0.8, 0.98),
                'speaker_id': np.random.randint(1, 4)
            })
        
        video_analysis.speech_segments = speech_segments
        
        # Generate trending topics
        topics = ['ai', 'technology', 'lifestyle', 'tutorial', 'funny', 'amazing', 'viral', 'trending']
        video_analysis.trending_topics = np.random.choice(topics, size=np.random.randint(2, 5), replace=False).tolist()
        
        # Generate clips based on highlights
        clips = []
        for i, highlight in enumerate(video_analysis.highlight_moments[:self.config.num_clips_to_generate]):
            clip = self._generate_clip_from_highlight(highlight, i, video_analysis)
            clips.append(clip)
        
        return video_analysis, clips
    
    def _generate_clip_from_highlight(
        self,
        highlight: Dict[str, Any],
        clip_index: int,
        video_analysis: VideoAnalysis
    ) -> VideoClip:
        """Generate a clip from a highlight moment"""
        
        # Determine optimal clip duration
        min_duration, max_duration = self.config.clip_duration_range
        highlight_duration = highlight['end'] - highlight['start']
        
        # Extend clip around highlight for context
        clip_duration = min(max_duration, max(min_duration, highlight_duration + 10))
        
        # Center clip around highlight peak
        start_time = max(0, highlight['peak_time'] - clip_duration / 2)
        end_time = min(video_analysis.duration, start_time + clip_duration)
        
        # Adjust start if end exceeds duration
        if end_time >= video_analysis.duration:
            end_time = video_analysis.duration
            start_time = max(0, end_time - clip_duration)
        
        # Generate captions
        captions = self._generate_clip_captions(start_time, end_time, video_analysis, highlight['type'])
        
        # Calculate viral score
        viral_score = self._calculate_clip_viral_score(highlight, video_analysis)
        
        # Generate title and description
        title = self._generate_viral_title(highlight['type'], clip_index)
        description = self._generate_viral_description(highlight, video_analysis)
        
        # Generate effects
        effects_applied = self._select_effects_for_clip(highlight['type'], viral_score)
        
        # Platform optimizations
        platform_optimizations = self._generate_platform_optimizations(highlight['type'], clip_duration)
        
        clip_id = f"{video_analysis.video_id}_clip_{clip_index:02d}"
        
        return VideoClip(
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            title=title,
            description=description,
            captions=captions,
            viral_score=viral_score,
            engagement_prediction=viral_score * np.random.uniform(0.85, 0.95),
            platform_optimizations=platform_optimizations,
            effects_applied=effects_applied,
            audio_features=video_analysis.audio_features,
            visual_features=video_analysis.visual_features,
            file_path=f"/tmp/clips/{clip_id}.mp4",
            thumbnail_path=f"/tmp/clips/{clip_id}_thumb.jpg",
            metadata={
                'highlight_type': highlight['type'],
                'highlight_score': highlight['score'],
                'generation_time': datetime.now().isoformat(),
                'model_version': self.model_size
            }
        )
    
    def _generate_clip_captions(
        self,
        start_time: float,
        end_time: float,
        video_analysis: VideoAnalysis,
        highlight_type: str
    ) -> List[Dict[str, Any]]:
        """Generate captions for a clip"""
        
        captions = []
        
        # Find speech segments that overlap with clip
        for segment in video_analysis.speech_segments:
            if segment['start'] < end_time and segment['end'] > start_time:
                # Adjust timing relative to clip
                caption_start = max(0, segment['start'] - start_time)
                caption_end = min(end_time - start_time, segment['end'] - start_time)
                
                if caption_end > caption_start:
                    captions.append({
                        'start': caption_start,
                        'end': caption_end,
                        'text': segment['text'],
                        'confidence': segment['confidence'],
                        'style': self._get_caption_style(highlight_type),
                        'animation': self._get_caption_animation(highlight_type),
                        'position': 'bottom_center',
                        'font_size': 48,
                        'color': 'white',
                        'stroke_color': 'black'
                    })
        
        # Add viral-style captions if no speech or to enhance engagement
        if not captions or len(captions) < 2:
            viral_captions = self._generate_viral_captions(end_time - start_time, highlight_type)
            captions.extend(viral_captions)
        
        return captions
    
    def _generate_viral_captions(self, duration: float, highlight_type: str) -> List[Dict[str, Any]]:
        """Generate viral-style captions"""
        
        viral_phrases = {
            'comedy': [
                "😂 This is hilarious!",
                "Wait for it...",
                "I can't stop laughing!",
                "This is too funny!",
                "Comedy gold! 🏆"
            ],
            'surprise': [
                "😱 OMG! Did you see that?",
                "Plot twist!",
                "This is insane!",
                "You won't believe this!",
                "Mind = Blown 🤯"
            ],
            'emotional': [
                "😭 This hits different",
                "Right in the feels",
                "So wholesome ❤️",
                "This is beautiful",
                "Tears of joy! 😊"
            ],
            'action': [
                "🔥 This is epic!",
                "Adrenaline rush!",
                "So intense!",
                "Action packed! ⚡",
                "Heart racing!"
            ],
            'reveal': [
                "The big reveal!",
                "Finally! 🎉",
                "This is it!",
                "The moment we've been waiting for",
                "Reveal time! ✨"
            ]
        }
        
        phrases = viral_phrases.get(highlight_type, viral_phrases['surprise'])
        
        captions = []
        num_captions = min(3, int(duration / 8))  # One caption every 8 seconds
        
        for i in range(num_captions):
            start_time = (i * duration) / num_captions
            end_time = min(duration, start_time + 4)  # 4 second duration
            
            captions.append({
                'start': start_time,
                'end': end_time,
                'text': phrases[i % len(phrases)],
                'confidence': 1.0,
                'style': 'viral',
                'animation': 'zoom_in',
                'position': 'center',
                'font_size': 56,
                'color': 'yellow',
                'stroke_color': 'black'
            })
        
        return captions
    
    def _calculate_clip_viral_score(self, highlight: Dict[str, Any], video_analysis: VideoAnalysis) -> float:
        """Calculate viral potential score for a clip"""
        
        # Base score from highlight
        base_score = highlight['score']
        
        # Boost based on video viral potential
        video_boost = video_analysis.viral_potential * 0.3
        
        # Boost based on highlight type
        type_boosts = {
            'comedy': 0.2,
            'surprise': 0.25,
            'emotional': 0.15,
            'action': 0.2,
            'reveal': 0.18
        }
        type_boost = type_boosts.get(highlight['type'], 0.1)
        
        # Boost based on audio/visual features
        audio_boost = 0.1 if video_analysis.audio_features.get('has_music', False) else 0.0
        visual_boost = 0.1 if video_analysis.visual_features.get('face_presence', 0) > 0.5 else 0.0
        
        # Combine scores
        viral_score = min(1.0, base_score + video_boost + type_boost + audio_boost + visual_boost)
        
        return viral_score
    
    def _generate_viral_title(self, highlight_type: str, clip_index: int) -> str:
        """Generate viral title for clip"""
        
        title_templates = {
            'comedy': [
                "😂 This will make you laugh out loud!",
                "🤣 Funniest moment ever!",
                "😆 Comedy gold right here!",
                "😂 You can't watch this without laughing!",
                "🤣 This is too funny to miss!"
            ],
            'surprise': [
                "😱 You won't believe what happens next!",
                "🤯 This plot twist is insane!",
                "😲 The most unexpected moment!",
                "🤯 Mind-blowing surprise!",
                "😱 This will shock you!"
            ],
            'emotional': [
                "😭 This will make you cry happy tears!",
                "❤️ The most wholesome moment!",
                "😊 This restored my faith in humanity!",
                "💕 Pure emotional gold!",
                "😭 Prepare the tissues!"
            ],
            'action': [
                "🔥 Most epic moment ever!",
                "⚡ Adrenaline-pumping action!",
                "💥 This is absolutely insane!",
                "🔥 Action-packed perfection!",
                "⚡ Heart-racing moment!"
            ],
            'reveal': [
                "🎉 The big reveal is here!",
                "✨ Finally, the moment we've been waiting for!",
                "🎊 The ultimate reveal!",
                "🎉 This reveal changes everything!",
                "✨ The most satisfying reveal!"
            ]
        }
        
        templates = title_templates.get(highlight_type, title_templates['surprise'])
        return templates[clip_index % len(templates)]
    
    def _generate_viral_description(self, highlight: Dict[str, Any], video_analysis: VideoAnalysis) -> str:
        """Generate viral description for clip"""
        
        description = f"🎬 Viral clip featuring {highlight['type']} content!\n\n"
        description += f"⭐ Viral Score: {highlight['score']:.1%}\n"
        description += f"🎯 Peak Moment: {highlight['peak_time']:.1f}s\n\n"
        
        if video_analysis.trending_topics:
            description += f"🔥 Trending: {', '.join(video_analysis.trending_topics[:3])}\n\n"
        
        description += "📱 Optimized for:\n"
        description += "• TikTok & Instagram Reels\n"
        description += "• YouTube Shorts\n"
        description += "• All mobile platforms\n\n"
        
        description += "#viral #trending #shorts #fyp #amazing #mustwatch"
        
        return description
    
    def _select_effects_for_clip(self, highlight_type: str, viral_score: float) -> List[str]:
        """Select appropriate effects for clip"""
        
        base_effects = ["resize_vertical", "captions", "logo"]
        
        type_effects = {
            'comedy': ["zoom_in", "quick_cuts", "sound_effects"],
            'surprise': ["dramatic_zoom", "pause_effect", "highlight_moment"],
            'emotional': ["soft_transitions", "color_grade", "gentle_zoom"],
            'action': ["speed_ramp", "motion_blur", "dynamic_cuts"],
            'reveal': ["build_up", "dramatic_reveal", "celebration_effects"]
        }
        
        effects = base_effects.copy()
        effects.extend(type_effects.get(highlight_type, []))
        
        # Add more effects for higher viral scores
        if viral_score > 0.8:
            effects.extend(["premium_transitions", "advanced_color_grade"])
        
        return effects
    
    def _generate_platform_optimizations(self, highlight_type: str, duration: float) -> Dict[str, Any]:
        """Generate platform-specific optimizations"""
        
        optimizations = {}
        
        for platform, specs in self.platform_specs.items():
            optimization = {
                'aspect_ratio': specs['aspect_ratio'],
                'duration': min(duration, specs['max_duration']),
                'hashtags': specs['hashtags'].copy(),
                'features': specs['features'].copy()
            }
            
            # Add type-specific hashtags
            type_hashtags = {
                'comedy': ['#funny', '#laugh', '#humor'],
                'surprise': ['#shocking', '#unexpected', '#wow'],
                'emotional': ['#wholesome', '#heartwarming', '#feels'],
                'action': ['#epic', '#intense', '#adrenaline'],
                'reveal': ['#reveal', '#surprise', '#amazing']
            }
            
            if highlight_type in type_hashtags:
                optimization['hashtags'].extend(type_hashtags[highlight_type])
            
            optimizations[platform] = optimization
        
        return optimizations
    
    def _get_caption_style(self, highlight_type: str) -> str:
        """Get caption style based on highlight type"""
        style_map = {
            'comedy': 'humorous',
            'surprise': 'dramatic',
            'emotional': 'gentle',
            'action': 'bold',
            'reveal': 'exciting'
        }
        return style_map.get(highlight_type, 'default')
    
    def _get_caption_animation(self, highlight_type: str) -> str:
        """Get caption animation based on highlight type"""
        animation_map = {
            'comedy': 'bounce',
            'surprise': 'zoom_in',
            'emotional': 'fade_in_out',
            'action': 'slide_up',
            'reveal': 'typewriter'
        }
        return animation_map.get(highlight_type, 'fade_in_out')
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        # Simplified extraction (would use proper regex in practice)
        if 'youtube.com/watch?v=' in url:
            return url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        else:
            return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def visualize_video_analysis(self, video_analysis: VideoAnalysis):
        """Create visualizations for video analysis"""
        
        # Video metrics overview
        st.subheader("📊 Video Analysis Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{video_analysis.duration:.1f}s")
        
        with col2:
            st.metric("Viral Potential", f"{video_analysis.viral_potential:.1%}")
        
        with col3:
            st.metric("Highlights Found", len(video_analysis.highlight_moments))
        
        with col4:
            st.metric("Speech Segments", len(video_analysis.speech_segments))
        
        # Viral potential gauge
        st.subheader("🎯 Viral Potential Analysis")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = video_analysis.viral_potential * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Viral Potential Score"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline visualization
        st.subheader("⏱️ Video Timeline Analysis")
        
        # Create timeline data
        timeline_data = []
        
        # Add highlights
        for i, highlight in enumerate(video_analysis.highlight_moments):
            timeline_data.append({
                'start': highlight['start'],
                'end': highlight['end'],
                'type': 'Highlight',
                'score': highlight['score'],
                'label': f"Highlight {i+1} ({highlight['type']})"
            })
        
        # Add speech segments
        for i, segment in enumerate(video_analysis.speech_segments[:5]):  # Show first 5
            timeline_data.append({
                'start': segment['start'],
                'end': segment['end'],
                'type': 'Speech',
                'score': segment['confidence'],
                'label': f"Speech {i+1}"
            })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            
            fig = px.timeline(
                df,
                x_start='start',
                x_end='end',
                y='label',
                color='type',
                hover_data=['score'],
                title="Video Content Timeline"
            )
            
            fig.update_layout(
                xaxis_title="Time (seconds)",
                yaxis_title="Content Type",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Audio/Visual features
        st.subheader("🎵 Audio & Visual Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Audio Features**")
            audio_features = video_analysis.audio_features
            
            for feature, value in audio_features.items():
                if isinstance(value, bool):
                    st.write(f"• {feature.replace('_', ' ').title()}: {'✅' if value else '❌'}")
                elif isinstance(value, (int, float)):
                    st.write(f"• {feature.replace('_', ' ').title()}: {value:.2f}")
        
        with col2:
            st.write("**Visual Features**")
            visual_features = video_analysis.visual_features
            
            for feature, value in visual_features.items():
                if isinstance(value, (int, float)):
                    st.write(f"• {feature.replace('_', ' ').title()}: {value:.2f}")
        
        # Trending topics
        if video_analysis.trending_topics:
            st.subheader("🔥 Trending Topics")
            
            topics_text = " • ".join([f"#{topic}" for topic in video_analysis.trending_topics])
            st.write(topics_text)
    
    def display_generated_clips(self, clips: List[VideoClip]):
        """Display generated clips with details"""
        
        st.subheader(f"✂️ Generated Clips ({len(clips)} clips)")
        
        # Clips overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_viral_score = np.mean([clip.viral_score for clip in clips])
            st.metric("Average Viral Score", f"{avg_viral_score:.1%}")
        
        with col2:
            avg_duration = np.mean([clip.duration for clip in clips])
            st.metric("Average Duration", f"{avg_duration:.1f}s")
        
        with col3:
            total_effects = sum([len(clip.effects_applied) for clip in clips])
            st.metric("Total Effects Applied", total_effects)
        
        # Clips distribution chart
        st.subheader("📈 Clips Performance Distribution")
        
        clip_data = []
        for i, clip in enumerate(clips):
            clip_data.append({
                'Clip': f"Clip {i+1}",
                'Viral Score': clip.viral_score,
                'Engagement Prediction': clip.engagement_prediction,
                'Duration': clip.duration
            })
        
        df = pd.DataFrame(clip_data)
        
        fig = px.scatter(
            df,
            x='Duration',
            y='Viral Score',
            size='Engagement Prediction',
            hover_name='Clip',
            title="Clips Performance Analysis",
            labels={'Duration': 'Duration (seconds)', 'Viral Score': 'Viral Score'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual clip details
        st.subheader("🎬 Individual Clip Details")
        
        # Sort clips by viral score
        sorted_clips = sorted(clips, key=lambda x: x.viral_score, reverse=True)
        
        for i, clip in enumerate(sorted_clips):
            with st.expander(f"🏆 {clip.title} (Viral Score: {clip.viral_score:.1%})", expanded=i == 0):
                
                # Clip metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Duration", f"{clip.duration:.1f}s")
                
                with col2:
                    st.metric("Start Time", f"{clip.start_time:.1f}s")
                
                with col3:
                    st.metric("Viral Score", f"{clip.viral_score:.1%}")
                
                with col4:
                    st.metric("Engagement", f"{clip.engagement_prediction:.1%}")
                
                # Clip details
                st.write("**Description:**")
                st.write(clip.description)
                
                # Captions
                if clip.captions:
                    st.write("**Captions:**")
                    for j, caption in enumerate(clip.captions):
                        st.write(f"{j+1}. [{caption['start']:.1f}s - {caption['end']:.1f}s] {caption['text']}")
                
                # Effects applied
                st.write("**Effects Applied:**")
                effects_text = " • ".join(clip.effects_applied)
                st.write(effects_text)
                
                # Platform optimizations
                st.write("**Platform Optimizations:**")
                
                platform_cols = st.columns(len(self.platform_specs))
                
                for idx, (platform, specs) in enumerate(self.platform_specs.items()):
                    with platform_cols[idx]:
                        st.write(f"**{specs['name']}**")
                        
                        if platform in clip.platform_optimizations:
                            opt = clip.platform_optimizations[platform]
                            st.write(f"Duration: {opt['duration']:.1f}s")
                            st.write(f"Hashtags: {len(opt['hashtags'])}")
                            
                            # Show hashtags
                            hashtags = " ".join(opt['hashtags'][:3])
                            st.write(f"{hashtags}...")
                
                # Download buttons (simulated)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📱 Download for TikTok", key=f"tiktok_{clip.clip_id}"):
                        st.success("TikTok version downloaded!")
                
                with col2:
                    if st.button(f"📸 Download for Instagram", key=f"instagram_{clip.clip_id}"):
                        st.success("Instagram version downloaded!")
                
                with col3:
                    if st.button(f"🎬 Download for YouTube", key=f"youtube_{clip.clip_id}"):
                        st.success("YouTube version downloaded!")
    
    def show_platform_comparison(self, clips: List[VideoClip]):
        """Show platform-specific performance comparison"""
        
        st.subheader("📱 Platform Performance Comparison")
        
        # Create comparison data
        platform_data = []
        
        for clip in clips:
            for platform, optimization in clip.platform_optimizations.items():
                platform_data.append({
                    'Platform': self.platform_specs[platform]['name'],
                    'Clip': clip.clip_id,
                    'Viral Score': clip.viral_score,
                    'Duration': optimization['duration'],
                    'Hashtags': len(optimization['hashtags']),
                    'Features': len(optimization['features'])
                })
        
        if platform_data:
            df = pd.DataFrame(platform_data)
            
            # Average performance by platform
            platform_avg = df.groupby('Platform').agg({
                'Viral Score': 'mean',
                'Duration': 'mean',
                'Hashtags': 'mean',
                'Features': 'mean'
            }).reset_index()
            
            # Create comparison chart
            fig = px.bar(
                platform_avg,
                x='Platform',
                y='Viral Score',
                title="Average Viral Score by Platform",
                color='Platform'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Platform specifications table
            st.write("**Platform Specifications:**")
            
            specs_data = []
            for platform, specs in self.platform_specs.items():
                specs_data.append({
                    'Platform': specs['name'],
                    'Aspect Ratio': specs['aspect_ratio'],
                    'Max Duration': f"{specs['max_duration']}s",
                    'Optimal Duration': f"{specs['optimal_duration'][0]}-{specs['optimal_duration'][1]}s",
                    'Key Features': ", ".join(specs['features'][:2])
                })
            
            specs_df = pd.DataFrame(specs_data)
            st.dataframe(specs_df, use_container_width=True)
    
    def run_demo(self):
        """Run the interactive demo"""
        
        st.set_page_config(
            page_title="Viral Video Clips Model Demo",
            page_icon="🎬",
            layout="wide"
        )
        
        st.title("🎬 Viral Video Clips Model Demo")
        st.markdown("**Extract YouTube videos, detect highlights, and generate viral clips automatically**")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model information
            st.subheader("Model Info")
            st.info(f"**Size**: {self.model_size.title()}")
            if self.model:
                st.info(f"**Status**: ✅ Ready")
            else:
                st.error("**Status**: ❌ Not Ready")
            
            # Processing options
            st.subheader("Processing Options")
            num_clips = st.slider("Number of Clips", 1, 20, self.config.num_clips_to_generate)
            clip_duration = st.slider("Clip Duration Range", 10, 120, (15, 60))
            
            # Platform selection
            st.subheader("Target Platforms")
            selected_platforms = []
            for platform, specs in self.platform_specs.items():
                if st.checkbox(specs['name'], value=True):
                    selected_platforms.append(platform)
            
            # Effects options
            st.subheader("Effects & Enhancements")
            enable_captions = st.checkbox("Auto Captions", value=True)
            enable_effects = st.checkbox("Visual Effects", value=True)
            enable_logo = st.checkbox("Logo/Watermark", value=True)
        
        # Main interface
        tab1, tab2, tab3, tab4 = st.tabs(["🎬 Video Processing", "📊 Analysis Results", "✂️ Generated Clips", "📱 Platform Optimization"])
        
        with tab1:
            st.header("YouTube Video Processing")
            
            # URL input
            col1, col2 = st.columns([4, 1])
            with col1:
                youtube_url = st.text_input(
                    "Enter YouTube URL:",
                    placeholder="https://www.youtube.com/watch?v=...",
                    help="Paste any YouTube video URL to extract and generate viral clips"
                )
            
            with col2:
                process_button = st.button("🚀 Process Video", type="primary")
            
            # Example URLs
            st.write("**Try these example videos:**")
            example_urls = [
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "https://www.youtube.com/watch?v=jNQXAC9IVRw",
                "https://www.youtube.com/watch?v=9bZkp7q19f0",
                "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
                "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
            ]
            
            cols = st.columns(len(example_urls))
            for i, example_url in enumerate(example_urls):
                with cols[i]:
                    if st.button(f"Example {i+1}", key=f"example_{i}"):
                        youtube_url = example_url
                        process_button = True
            
            # Processing
            if process_button and youtube_url:
                video_analysis, clips = self.process_youtube_video(youtube_url)
                
                if video_analysis and clips:
                    st.success(f"✅ Successfully processed video and generated {len(clips)} clips!")
                    
                    # Quick stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Video Duration", f"{video_analysis.duration:.1f}s")
                    
                    with col2:
                        st.metric("Viral Potential", f"{video_analysis.viral_potential:.1%}")
                    
                    with col3:
                        st.metric("Clips Generated", len(clips))
                    
                    with col4:
                        avg_viral_score = np.mean([clip.viral_score for clip in clips])
                        st.metric("Avg Clip Score", f"{avg_viral_score:.1%}")
            
            # Processing history
            if self.processing_history:
                st.subheader("📝 Processing History")
                
                history_data = []
                for entry in self.processing_history[-5:]:  # Show last 5
                    history_data.append({
                        'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'URL': entry['url'][:50] + "..." if len(entry['url']) > 50 else entry['url'],
                        'Duration': f"{entry['video_analysis'].duration:.1f}s",
                        'Viral Potential': f"{entry['video_analysis'].viral_potential:.1%}",
                        'Clips Generated': entry['clips_generated']
                    })
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True)
        
        with tab2:
            st.header("Video Analysis Results")
            
            if self.current_video_analysis:
                self.visualize_video_analysis(self.current_video_analysis)
            else:
                st.info("📊 Process a video to see detailed analysis results here.")
                
                # Show example analysis
                st.subheader("📋 Example Analysis Features")
                
                features = [
                    "🎯 Viral potential scoring with AI prediction",
                    "⭐ Highlight moment detection and ranking",
                    "🎵 Audio analysis (speech, music, quality)",
                    "👁️ Visual feature extraction (faces, motion, quality)",
                    "📈 Timeline visualization with content mapping",
                    "🔥 Trending topic identification",
                    "📊 Engagement prediction modeling",
                    "🎬 Scene change detection and analysis"
                ]
                
                for feature in features:
                    st.write(feature)
        
        with tab3:
            st.header("Generated Viral Clips")
            
            if self.current_clips:
                self.display_generated_clips(self.current_clips)
            else:
                st.info("✂️ Process a video to see generated clips here.")
                
                # Show example clip features
                st.subheader("🎬 Clip Generation Features")
                
                clip_features = [
                    "✂️ Intelligent clip extraction from highlights",
                    "📝 Auto-generated viral titles and descriptions",
                    "💬 Dynamic caption generation with animations",
                    "🎨 Visual effects and transitions application",
                    "🏷️ Logo and watermark integration",
                    "📱 Platform-specific optimization",
                    "🎯 Viral score prediction for each clip",
                    "📊 Engagement prediction and ranking"
                ]
                
                for feature in clip_features:
                    st.write(feature)
        
        with tab4:
            st.header("Platform Optimization")
            
            if self.current_clips:
                self.show_platform_comparison(self.current_clips)
            else:
                st.info("📱 Process a video to see platform optimizations here.")
                
                # Show platform specifications
                st.subheader("📱 Supported Platforms")
                
                for platform, specs in self.platform_specs.items():
                    with st.expander(f"{specs['name']} Specifications"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Technical Specs:**")
                            st.write(f"• Aspect Ratio: {specs['aspect_ratio']}")
                            st.write(f"• Max Duration: {specs['max_duration']}s")
                            st.write(f"• Optimal Duration: {specs['optimal_duration'][0]}-{specs['optimal_duration'][1]}s")
                        
                        with col2:
                            st.write("**Key Features:**")
                            for feature in specs['features']:
                                st.write(f"• {feature}")
                        
                        st.write("**Trending Hashtags:**")
                        hashtags_text = " ".join(specs['hashtags'])
                        st.write(hashtags_text)
        
        # Footer
        st.markdown("---")
        st.markdown("**Viral Video Clips Model** - Powered by TruthGPT AI 🚀")


def main():
    """Main function to run the demo"""
    
    # Configuration
    config_path = "config.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        st.info("Please ensure the config.yaml file is in the same directory as this demo.")
        return
    
    # Model size selection
    model_size = st.sidebar.selectbox(
        "Select Model Size:",
        options=['small', 'medium', 'large'],
        index=1,
        help="Choose model size based on your hardware capabilities"
    )
    
    # Initialize and run demo
    try:
        demo = ViralVideoClipsDemo(config_path, model_size)
        demo.run_demo()
        
    except Exception as e:
        st.error(f"Error running demo: {e}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()