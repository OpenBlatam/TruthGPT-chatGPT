"""
Enhanced Viral Video Clips Model - Video Processing Tools
Inspired by Onyx tool system for modular video processing capabilities
"""

import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging

import yt_dlp
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..interfaces.video_llm_interface import (
    PlatformType, VideoProcessingMode, ClipSegment, ViralClipOutput
)

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    message: str
    execution_time: float
    metadata: Dict[str, Any] = None


class VideoProcessingTool(ABC):
    """Abstract base class for video processing tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        return {
            "name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": (
                self.total_execution_time / self.execution_count 
                if self.execution_count > 0 else 0
            )
        }


class YouTubeDownloaderTool(VideoProcessingTool):
    """Tool for downloading YouTube videos"""
    
    def __init__(self):
        super().__init__(
            name="youtube_downloader",
            description="Download YouTube videos with metadata extraction"
        )
    
    async def execute(
        self,
        url: str,
        output_dir: str = "./downloads",
        quality: str = "best",
        extract_audio: bool = True,
        **kwargs
    ) -> ToolResult:
        """Download YouTube video"""
        
        import time
        start_time = time.time()
        
        try:
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
                'format': quality,
                'extractaudio': extract_audio,
                'audioformat': 'mp3' if extract_audio else None,
                'writeinfojson': True,
                'writedescription': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
            }
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                video_path = ydl.prepare_filename(info)
                
                result_data = {
                    "video_path": video_path,
                    "title": info.get("title", ""),
                    "description": info.get("description", ""),
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "like_count": info.get("like_count", 0),
                    "upload_date": info.get("upload_date", ""),
                    "uploader": info.get("uploader", ""),
                    "tags": info.get("tags", []),
                    "categories": info.get("categories", []),
                    "thumbnail": info.get("thumbnail", ""),
                    "webpage_url": info.get("webpage_url", url)
                }
            
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ToolResult(
                success=True,
                data=result_data,
                message=f"Successfully downloaded: {info.get('title', 'video')}",
                execution_time=execution_time,
                metadata={"url": url, "quality": quality}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"YouTube download failed: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                message=f"Download failed: {str(e)}",
                execution_time=execution_time,
                metadata={"url": url, "error": str(e)}
            )


class VideoAnalyzerTool(VideoProcessingTool):
    """Tool for comprehensive video analysis"""
    
    def __init__(self):
        super().__init__(
            name="video_analyzer",
            description="Analyze video content for viral potential and highlights"
        )
    
    async def execute(
        self,
        video_path: str,
        analysis_type: str = "full",
        frame_interval: float = 1.0,
        **kwargs
    ) -> ToolResult:
        """Analyze video content"""
        
        import time
        start_time = time.time()
        
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Load video
            video_clip = VideoFileClip(video_path)
            
            analysis_results = {
                "basic_info": self._analyze_basic_info(video_clip),
                "visual_analysis": self._analyze_visual_content(video_clip, frame_interval),
                "audio_analysis": self._analyze_audio_content(video_clip),
                "motion_analysis": self._analyze_motion(video_clip),
                "scene_analysis": self._analyze_scenes(video_clip),
                "engagement_factors": self._analyze_engagement_factors(video_clip)
            }
            
            # Calculate overall viral score
            viral_score = self._calculate_viral_score(analysis_results)
            analysis_results["viral_score"] = viral_score
            
            video_clip.close()
            
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ToolResult(
                success=True,
                data=analysis_results,
                message=f"Video analysis completed. Viral score: {viral_score:.2f}",
                execution_time=execution_time,
                metadata={"video_path": video_path, "analysis_type": analysis_type}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Video analysis failed: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                message=f"Analysis failed: {str(e)}",
                execution_time=execution_time,
                metadata={"video_path": video_path, "error": str(e)}
            )
    
    def _analyze_basic_info(self, video_clip) -> Dict[str, Any]:
        """Analyze basic video information"""
        return {
            "duration": video_clip.duration,
            "fps": video_clip.fps,
            "size": (video_clip.w, video_clip.h),
            "aspect_ratio": video_clip.w / video_clip.h,
            "has_audio": video_clip.audio is not None
        }
    
    def _analyze_visual_content(self, video_clip, frame_interval: float) -> Dict[str, Any]:
        """Analyze visual content"""
        frames_analyzed = 0
        brightness_scores = []
        contrast_scores = []
        color_diversity_scores = []
        
        for t in np.arange(0, min(video_clip.duration, 60), frame_interval):  # Limit to 60 seconds
            frame = video_clip.get_frame(t)
            
            # Convert to grayscale for brightness/contrast analysis
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate brightness
            brightness = np.mean(gray_frame)
            brightness_scores.append(brightness)
            
            # Calculate contrast
            contrast = np.std(gray_frame)
            contrast_scores.append(contrast)
            
            # Calculate color diversity
            color_diversity = len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))
            color_diversity_scores.append(color_diversity)
            
            frames_analyzed += 1
        
        return {
            "frames_analyzed": frames_analyzed,
            "average_brightness": np.mean(brightness_scores) if brightness_scores else 0,
            "brightness_variance": np.var(brightness_scores) if brightness_scores else 0,
            "average_contrast": np.mean(contrast_scores) if contrast_scores else 0,
            "contrast_variance": np.var(contrast_scores) if contrast_scores else 0,
            "average_color_diversity": np.mean(color_diversity_scores) if color_diversity_scores else 0
        }
    
    def _analyze_audio_content(self, video_clip) -> Dict[str, Any]:
        """Analyze audio content"""
        if not video_clip.audio:
            return {"has_audio": False}
        
        # Extract audio array
        audio_array = video_clip.audio.to_soundarray()
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Calculate audio features
        volume_levels = np.abs(audio_array)
        
        return {
            "has_audio": True,
            "duration": len(audio_array) / video_clip.audio.fps,
            "sample_rate": video_clip.audio.fps,
            "average_volume": np.mean(volume_levels),
            "max_volume": np.max(volume_levels),
            "volume_variance": np.var(volume_levels),
            "silence_ratio": np.sum(volume_levels < 0.01) / len(volume_levels)
        }
    
    def _analyze_motion(self, video_clip) -> Dict[str, Any]:
        """Analyze motion in video"""
        motion_scores = []
        prev_frame = None
        
        for t in np.arange(0, min(video_clip.duration, 30), 1.0):  # Sample every second, limit to 30s
            frame = video_clip.get_frame(t)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray_frame)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray_frame
        
        return {
            "average_motion": np.mean(motion_scores) if motion_scores else 0,
            "max_motion": np.max(motion_scores) if motion_scores else 0,
            "motion_variance": np.var(motion_scores) if motion_scores else 0,
            "high_motion_ratio": np.sum(np.array(motion_scores) > 20) / len(motion_scores) if motion_scores else 0
        }
    
    def _analyze_scenes(self, video_clip) -> Dict[str, Any]:
        """Analyze scene changes"""
        scene_changes = []
        prev_frame = None
        
        for t in np.arange(0, min(video_clip.duration, 60), 0.5):  # Check every 0.5 seconds
            frame = video_clip.get_frame(t)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray_frame)
                diff_score = np.mean(diff)
                
                if diff_score > 30:  # Scene change threshold
                    scene_changes.append(t)
            
            prev_frame = gray_frame
        
        return {
            "scene_changes": scene_changes,
            "scene_count": len(scene_changes) + 1,
            "average_scene_length": video_clip.duration / (len(scene_changes) + 1) if scene_changes else video_clip.duration
        }
    
    def _analyze_engagement_factors(self, video_clip) -> Dict[str, Any]:
        """Analyze factors that contribute to engagement"""
        # This is a simplified analysis - could be enhanced with ML models
        
        basic_info = self._analyze_basic_info(video_clip)
        visual_info = self._analyze_visual_content(video_clip, 2.0)
        audio_info = self._analyze_audio_content(video_clip)
        motion_info = self._analyze_motion(video_clip)
        
        # Calculate engagement factors
        factors = {
            "optimal_duration": 15 <= basic_info["duration"] <= 60,
            "good_aspect_ratio": 0.5 <= basic_info["aspect_ratio"] <= 2.0,
            "has_audio": audio_info["has_audio"],
            "good_brightness": 50 <= visual_info["average_brightness"] <= 200,
            "good_contrast": visual_info["average_contrast"] > 20,
            "sufficient_motion": motion_info["average_motion"] > 10,
            "not_too_static": motion_info["high_motion_ratio"] > 0.1
        }
        
        engagement_score = sum(factors.values()) / len(factors)
        
        return {
            "factors": factors,
            "engagement_score": engagement_score,
            "recommendations": self._generate_recommendations(factors)
        }
    
    def _generate_recommendations(self, factors: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not factors["optimal_duration"]:
            recommendations.append("Consider trimming to 15-60 seconds for better engagement")
        
        if not factors["good_aspect_ratio"]:
            recommendations.append("Adjust aspect ratio for mobile viewing (9:16 or 16:9)")
        
        if not factors["has_audio"]:
            recommendations.append("Add audio or music to increase engagement")
        
        if not factors["good_brightness"]:
            recommendations.append("Adjust brightness for better visibility")
        
        if not factors["good_contrast"]:
            recommendations.append("Increase contrast to make content more visually appealing")
        
        if not factors["sufficient_motion"]:
            recommendations.append("Add more dynamic content or camera movement")
        
        return recommendations
    
    def _calculate_viral_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall viral potential score"""
        engagement_score = analysis_results["engagement_factors"]["engagement_score"]
        
        # Weight different factors
        weights = {
            "engagement": 0.4,
            "motion": 0.2,
            "audio": 0.2,
            "visual": 0.2
        }
        
        motion_score = min(analysis_results["motion_analysis"]["average_motion"] / 50, 1.0)
        audio_score = 1.0 if analysis_results["audio_analysis"]["has_audio"] else 0.3
        visual_score = min(analysis_results["visual_analysis"]["average_contrast"] / 100, 1.0)
        
        viral_score = (
            weights["engagement"] * engagement_score +
            weights["motion"] * motion_score +
            weights["audio"] * audio_score +
            weights["visual"] * visual_score
        )
        
        return min(viral_score, 1.0)


class ClipGeneratorTool(VideoProcessingTool):
    """Tool for generating viral clips from videos"""
    
    def __init__(self):
        super().__init__(
            name="clip_generator",
            description="Generate viral clips from video highlights"
        )
    
    async def execute(
        self,
        video_path: str,
        highlights: List[ClipSegment],
        output_dir: str = "./clips",
        platforms: List[PlatformType] = None,
        **kwargs
    ) -> ToolResult:
        """Generate clips from highlights"""
        
        import time
        start_time = time.time()
        
        try:
            if platforms is None:
                platforms = [PlatformType.TIKTOK, PlatformType.INSTAGRAM_REELS]
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            generated_clips = []
            
            for i, highlight in enumerate(highlights):
                for platform in platforms:
                    clip_path = await self._generate_clip(
                        video_path, highlight, platform, output_dir, i
                    )
                    
                    if clip_path:
                        generated_clips.append({
                            "clip_path": clip_path,
                            "highlight": highlight,
                            "platform": platform,
                            "clip_index": i
                        })
            
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ToolResult(
                success=True,
                data={"clips": generated_clips, "total_clips": len(generated_clips)},
                message=f"Generated {len(generated_clips)} clips from {len(highlights)} highlights",
                execution_time=execution_time,
                metadata={"video_path": video_path, "platforms": [p.value for p in platforms]}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Clip generation failed: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                message=f"Clip generation failed: {str(e)}",
                execution_time=execution_time,
                metadata={"video_path": video_path, "error": str(e)}
            )
    
    async def _generate_clip(
        self,
        video_path: str,
        highlight: ClipSegment,
        platform: PlatformType,
        output_dir: str,
        clip_index: int
    ) -> Optional[str]:
        """Generate a single clip"""
        
        try:
            # Load video
            video_clip = VideoFileClip(video_path)
            
            # Extract highlight segment
            clip = video_clip.subclip(highlight.start_time, highlight.end_time)
            
            # Apply platform-specific optimizations
            optimized_clip = self._optimize_for_platform(clip, platform)
            
            # Generate output filename
            platform_name = platform.value.lower()
            output_filename = f"clip_{clip_index:03d}_{platform_name}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Write clip
            optimized_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Cleanup
            video_clip.close()
            clip.close()
            optimized_clip.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate clip {clip_index}: {e}")
            return None
    
    def _optimize_for_platform(self, clip, platform: PlatformType):
        """Optimize clip for specific platform"""
        
        if platform == PlatformType.TIKTOK:
            # TikTok: 9:16 aspect ratio, max 60 seconds
            target_ratio = 9/16
            clip = self._resize_to_aspect_ratio(clip, target_ratio)
            clip = clip.subclip(0, min(clip.duration, 60))
            
        elif platform == PlatformType.INSTAGRAM_REELS:
            # Instagram Reels: 9:16 aspect ratio, max 90 seconds
            target_ratio = 9/16
            clip = self._resize_to_aspect_ratio(clip, target_ratio)
            clip = clip.subclip(0, min(clip.duration, 90))
            
        elif platform == PlatformType.YOUTUBE_SHORTS:
            # YouTube Shorts: 9:16 aspect ratio, max 60 seconds
            target_ratio = 9/16
            clip = self._resize_to_aspect_ratio(clip, target_ratio)
            clip = clip.subclip(0, min(clip.duration, 60))
            
        return clip
    
    def _resize_to_aspect_ratio(self, clip, target_ratio: float):
        """Resize clip to target aspect ratio"""
        current_ratio = clip.w / clip.h
        
        if abs(current_ratio - target_ratio) < 0.1:
            return clip  # Already close to target ratio
        
        if current_ratio > target_ratio:
            # Crop width
            new_width = int(clip.h * target_ratio)
            x_center = clip.w // 2
            x1 = x_center - new_width // 2
            x2 = x_center + new_width // 2
            return clip.crop(x1=x1, x2=x2)
        else:
            # Crop height
            new_height = int(clip.w / target_ratio)
            y_center = clip.h // 2
            y1 = y_center - new_height // 2
            y2 = y_center + new_height // 2
            return clip.crop(y1=y1, y2=y2)


class CaptionGeneratorTool(VideoProcessingTool):
    """Tool for generating and styling captions"""
    
    def __init__(self):
        super().__init__(
            name="caption_generator",
            description="Generate styled captions for viral videos"
        )
    
    async def execute(
        self,
        video_path: str,
        transcript_segments: List[Dict[str, Any]],
        style: str = "viral",
        platform: PlatformType = PlatformType.TIKTOK,
        **kwargs
    ) -> ToolResult:
        """Generate styled captions"""
        
        import time
        start_time = time.time()
        
        try:
            # Load video
            video_clip = VideoFileClip(video_path)
            
            # Generate caption clips
            caption_clips = []
            
            for segment in transcript_segments:
                caption_clip = self._create_caption_clip(
                    segment, style, platform, video_clip.size
                )
                if caption_clip:
                    caption_clips.append(caption_clip)
            
            # Composite captions with video
            if caption_clips:
                final_clip = CompositeVideoClip([video_clip] + caption_clips)
            else:
                final_clip = video_clip
            
            # Generate output path
            output_path = video_path.replace('.mp4', '_with_captions.mp4')
            
            # Write video with captions
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Cleanup
            video_clip.close()
            final_clip.close()
            for clip in caption_clips:
                clip.close()
            
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ToolResult(
                success=True,
                data={"output_path": output_path, "caption_count": len(caption_clips)},
                message=f"Generated {len(caption_clips)} caption segments",
                execution_time=execution_time,
                metadata={"style": style, "platform": platform.value}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Caption generation failed: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                message=f"Caption generation failed: {str(e)}",
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    def _create_caption_clip(
        self,
        segment: Dict[str, Any],
        style: str,
        platform: PlatformType,
        video_size: tuple
    ):
        """Create a caption clip for a segment"""
        
        try:
            text = segment.get("text", "").strip()
            if not text:
                return None
            
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 2)
            duration = end_time - start_time
            
            # Style configuration
            style_config = self._get_style_config(style, platform)
            
            # Create text clip
            caption_clip = TextClip(
                text,
                fontsize=style_config["font_size"],
                color=style_config["color"],
                font=style_config["font"],
                stroke_color=style_config["stroke_color"],
                stroke_width=style_config["stroke_width"]
            ).set_duration(duration).set_start(start_time)
            
            # Position caption
            caption_clip = caption_clip.set_position(style_config["position"])
            
            # Add animation if specified
            if style_config["animation"]:
                caption_clip = self._add_animation(caption_clip, style_config["animation"])
            
            return caption_clip
            
        except Exception as e:
            logger.error(f"Failed to create caption clip: {e}")
            return None
    
    def _get_style_config(self, style: str, platform: PlatformType) -> Dict[str, Any]:
        """Get style configuration for captions"""
        
        base_configs = {
            "viral": {
                "font_size": 48,
                "color": "yellow",
                "font": "Arial-Bold",
                "stroke_color": "black",
                "stroke_width": 2,
                "position": ("center", "bottom"),
                "animation": "bounce"
            },
            "clean": {
                "font_size": 36,
                "color": "white",
                "font": "Arial",
                "stroke_color": "black",
                "stroke_width": 1,
                "position": ("center", "bottom"),
                "animation": "fade"
            },
            "professional": {
                "font_size": 32,
                "color": "white",
                "font": "Arial",
                "stroke_color": None,
                "stroke_width": 0,
                "position": ("center", "bottom"),
                "animation": None
            }
        }
        
        # Platform-specific adjustments
        config = base_configs.get(style, base_configs["clean"]).copy()
        
        if platform == PlatformType.TIKTOK:
            config["font_size"] = max(config["font_size"], 44)
            config["animation"] = "bounce"
        elif platform == PlatformType.INSTAGRAM_REELS:
            config["color"] = "white"
            config["animation"] = "fade"
        
        return config
    
    def _add_animation(self, clip, animation_type: str):
        """Add animation to caption clip"""
        
        if animation_type == "bounce":
            return clip.resize(lambda t: 1 + 0.1 * abs(np.sin(4 * np.pi * t)))
        elif animation_type == "fade":
            return clip.crossfadein(0.3).crossfadeout(0.3)
        elif animation_type == "slide":
            return clip.set_position(lambda t: ("center", "bottom" if t > 0.5 else "top"))
        
        return clip


class EffectsApplicatorTool(VideoProcessingTool):
    """Tool for applying viral effects to videos"""
    
    def __init__(self):
        super().__init__(
            name="effects_applicator",
            description="Apply viral effects and transitions to videos"
        )
    
    async def execute(
        self,
        video_path: str,
        effects: List[str],
        output_path: str = None,
        **kwargs
    ) -> ToolResult:
        """Apply effects to video"""
        
        import time
        start_time = time.time()
        
        try:
            # Load video
            video_clip = VideoFileClip(video_path)
            processed_clip = video_clip
            
            # Apply each effect
            for effect in effects:
                processed_clip = self._apply_effect(processed_clip, effect)
            
            # Generate output path if not provided
            if output_path is None:
                output_path = video_path.replace('.mp4', '_with_effects.mp4')
            
            # Write processed video
            processed_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Cleanup
            video_clip.close()
            processed_clip.close()
            
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            
            return ToolResult(
                success=True,
                data={"output_path": output_path, "effects_applied": effects},
                message=f"Applied {len(effects)} effects to video",
                execution_time=execution_time,
                metadata={"effects": effects}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Effects application failed: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                message=f"Effects application failed: {str(e)}",
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    def _apply_effect(self, clip, effect: str):
        """Apply a specific effect to the clip"""
        
        if effect == "speed_ramp":
            return self._apply_speed_ramp(clip)
        elif effect == "zoom_effect":
            return self._apply_zoom_effect(clip)
        elif effect == "color_boost":
            return self._apply_color_boost(clip)
        elif effect == "slow_motion":
            return self._apply_slow_motion(clip)
        elif effect == "reverse":
            return self._apply_reverse(clip)
        else:
            logger.warning(f"Unknown effect: {effect}")
            return clip
    
    def _apply_speed_ramp(self, clip):
        """Apply speed ramp effect"""
        duration = clip.duration
        if duration > 10:
            # Speed up middle section
            part1 = clip.subclip(0, duration * 0.3)
            part2 = clip.subclip(duration * 0.3, duration * 0.7).fx(lambda c: c.speedx(1.5))
            part3 = clip.subclip(duration * 0.7, duration)
            return CompositeVideoClip([part1, part2, part3])
        return clip
    
    def _apply_zoom_effect(self, clip):
        """Apply zoom effect"""
        return clip.resize(lambda t: 1 + 0.1 * np.sin(2 * np.pi * t / clip.duration))
    
    def _apply_color_boost(self, clip):
        """Apply color boost effect"""
        return clip.fx(lambda c: c.colorx(1.2))
    
    def _apply_slow_motion(self, clip):
        """Apply slow motion effect"""
        return clip.fx(lambda c: c.speedx(0.5))
    
    def _apply_reverse(self, clip):
        """Apply reverse effect"""
        return clip.fx(lambda c: c.time_mirror())


class VideoToolRegistry:
    """Registry for video processing tools"""
    
    _tools: Dict[str, VideoProcessingTool] = {}
    
    @classmethod
    def register_tool(cls, tool: VideoProcessingTool) -> None:
        """Register a video processing tool"""
        cls._tools[tool.name] = tool
        logger.info(f"Registered video tool: {tool.name}")
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[VideoProcessingTool]:
        """Get tool by name"""
        return cls._tools.get(name)
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tools"""
        return list(cls._tools.keys())
    
    @classmethod
    def get_all_tools(cls) -> Dict[str, VideoProcessingTool]:
        """Get all registered tools"""
        return cls._tools.copy()


# Auto-register default tools
def _register_default_tools():
    """Register default video processing tools"""
    VideoToolRegistry.register_tool(YouTubeDownloaderTool())
    VideoToolRegistry.register_tool(VideoAnalyzerTool())
    VideoToolRegistry.register_tool(ClipGeneratorTool())
    VideoToolRegistry.register_tool(CaptionGeneratorTool())
    VideoToolRegistry.register_tool(EffectsApplicatorTool())


# Register tools on import
_register_default_tools()


# Convenience functions
async def download_youtube_video(url: str, **kwargs) -> ToolResult:
    """Download YouTube video"""
    tool = VideoToolRegistry.get_tool("youtube_downloader")
    return await tool.execute(url=url, **kwargs)


async def analyze_video(video_path: str, **kwargs) -> ToolResult:
    """Analyze video content"""
    tool = VideoToolRegistry.get_tool("video_analyzer")
    return await tool.execute(video_path=video_path, **kwargs)


async def generate_clips(video_path: str, highlights: List[ClipSegment], **kwargs) -> ToolResult:
    """Generate clips from highlights"""
    tool = VideoToolRegistry.get_tool("clip_generator")
    return await tool.execute(video_path=video_path, highlights=highlights, **kwargs)


async def add_captions(video_path: str, transcript_segments: List[Dict[str, Any]], **kwargs) -> ToolResult:
    """Add captions to video"""
    tool = VideoToolRegistry.get_tool("caption_generator")
    return await tool.execute(video_path=video_path, transcript_segments=transcript_segments, **kwargs)


async def apply_effects(video_path: str, effects: List[str], **kwargs) -> ToolResult:
    """Apply effects to video"""
    tool = VideoToolRegistry.get_tool("effects_applicator")
    return await tool.execute(video_path=video_path, effects=effects, **kwargs)