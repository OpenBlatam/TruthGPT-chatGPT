"""
Native Video Processing Tools
Pure native video processing tools without external API dependencies

This module provides tools for video processing, editing, and optimization
using only native libraries and models.
"""

import asyncio
import logging
import os
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
from moviepy.video.fx import resize, speedx
from moviepy.audio.fx import audio_normalize
import librosa
import yt_dlp
from dataclasses import dataclass

from ..configs.native_model_configs import PlatformConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoClip:
    """Processed video clip"""
    start_time: float
    end_time: float
    platform: str
    output_path: str
    viral_score: float
    caption: str
    effects_applied: List[str]
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Video processing result"""
    success: bool
    clips: List[VideoClip]
    total_clips: int
    processing_time: float
    error_message: Optional[str] = None


class NativeVideoProcessor:
    """Native video processor for viral clips"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Video processing settings
        self.temp_dir = Path(tempfile.mkdtemp())
        
        logger.info(f"Native video processor initialized with output dir: {output_dir}")
    
    async def process_video_for_platforms(
        self,
        video_path: str,
        highlights: List[Dict[str, Any]],
        platforms: List[str],
        platform_configs: Dict[str, PlatformConfig]
    ) -> ProcessingResult:
        """Process video for multiple platforms"""
        
        start_time = asyncio.get_event_loop().time()
        clips = []
        
        try:
            # Load video
            video = mp.VideoFileClip(video_path)
            
            for platform in platforms:
                platform_config = platform_configs.get(platform)
                if not platform_config:
                    logger.warning(f"No config found for platform: {platform}")
                    continue
                
                # Process highlights for this platform
                platform_clips = await self._process_platform_clips(
                    video, highlights, platform, platform_config
                )
                clips.extend(platform_clips)
            
            video.close()
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ProcessingResult(
                success=True,
                clips=clips,
                total_clips=len(clips),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ProcessingResult(
                success=False,
                clips=[],
                total_clips=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _process_platform_clips(
        self,
        video: mp.VideoFileClip,
        highlights: List[Dict[str, Any]],
        platform: str,
        config: PlatformConfig
    ) -> List[VideoClip]:
        """Process clips for specific platform"""
        
        clips = []
        
        for i, highlight in enumerate(highlights):
            try:
                # Extract clip
                start_time = highlight["start_time"]
                end_time = highlight["end_time"]
                
                # Ensure clip duration is within platform limits
                duration = end_time - start_time
                if duration < config.min_duration:
                    # Extend clip
                    extension = (config.min_duration - duration) / 2
                    start_time = max(0, start_time - extension)
                    end_time = min(video.duration, end_time + extension)
                elif duration > config.max_duration:
                    # Trim clip
                    end_time = start_time + config.max_duration
                
                # Create clip
                clip = video.subclip(start_time, end_time)
                
                # Apply platform-specific processing
                processed_clip = await self._apply_platform_effects(
                    clip, platform, config, highlight
                )
                
                # Generate output filename
                output_filename = f"{platform}_clip_{i+1}_{start_time:.1f}s-{end_time:.1f}s.mp4"
                output_path = self.output_dir / output_filename
                
                # Write video
                processed_clip.write_videofile(
                    str(output_path),
                    fps=config.fps,
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                
                processed_clip.close()
                
                # Create clip metadata
                video_clip = VideoClip(
                    start_time=start_time,
                    end_time=end_time,
                    platform=platform,
                    output_path=str(output_path),
                    viral_score=highlight.get("viral_score", 0.0),
                    caption=highlight.get("caption", ""),
                    effects_applied=self._get_applied_effects(platform, config),
                    metadata={
                        "duration": end_time - start_time,
                        "resolution": config.resolution,
                        "fps": config.fps,
                        "aspect_ratio": config.aspect_ratio
                    }
                )
                
                clips.append(video_clip)
                
                logger.info(f"Created {platform} clip: {output_filename}")
                
            except Exception as e:
                logger.error(f"Error processing clip {i} for {platform}: {e}")
                continue
        
        return clips
    
    async def _apply_platform_effects(
        self,
        clip: mp.VideoFileClip,
        platform: str,
        config: PlatformConfig,
        highlight: Dict[str, Any]
    ) -> mp.VideoFileClip:
        """Apply platform-specific effects"""
        
        processed_clip = clip
        
        # Resize to platform resolution
        target_width, target_height = config.resolution
        processed_clip = resize(processed_clip, (target_width, target_height))
        
        # Apply platform-specific effects
        if platform == "tiktok":
            processed_clip = await self._apply_tiktok_effects(processed_clip, config, highlight)
        elif platform == "instagram":
            processed_clip = await self._apply_instagram_effects(processed_clip, config, highlight)
        elif platform == "youtube":
            processed_clip = await self._apply_youtube_effects(processed_clip, config, highlight)
        elif platform == "facebook":
            processed_clip = await self._apply_facebook_effects(processed_clip, config, highlight)
        elif platform == "twitter":
            processed_clip = await self._apply_twitter_effects(processed_clip, config, highlight)
        
        # Add captions if required
        if config.captions_required and highlight.get("caption"):
            processed_clip = await self._add_captions(processed_clip, highlight["caption"])
        
        # Normalize audio
        if processed_clip.audio:
            processed_clip = processed_clip.set_audio(
                audio_normalize(processed_clip.audio)
            )
        
        return processed_clip
    
    async def _apply_tiktok_effects(
        self,
        clip: mp.VideoFileClip,
        config: PlatformConfig,
        highlight: Dict[str, Any]
    ) -> mp.VideoFileClip:
        """Apply TikTok-specific effects"""
        
        processed_clip = clip
        
        # Quick cuts effect (speed up certain parts)
        if config.quick_cuts and highlight.get("motion_intensity", 0) > 0.5:
            processed_clip = speedx(processed_clip, factor=1.2)
        
        # Zoom effects
        if config.zoom_effects:
            processed_clip = await self._add_zoom_effect(processed_clip)
        
        # Speed ramps
        if config.speed_ramps:
            processed_clip = await self._add_speed_ramps(processed_clip)
        
        return processed_clip
    
    async def _apply_instagram_effects(
        self,
        clip: mp.VideoFileClip,
        config: PlatformConfig,
        highlight: Dict[str, Any]
    ) -> mp.VideoFileClip:
        """Apply Instagram-specific effects"""
        
        processed_clip = clip
        
        # Color grading
        if config.color_grading:
            processed_clip = await self._apply_color_grading(processed_clip, "instagram")
        
        # Smooth transitions
        if config.transitions:
            processed_clip = await self._add_smooth_transitions(processed_clip)
        
        return processed_clip
    
    async def _apply_youtube_effects(
        self,
        clip: mp.VideoFileClip,
        config: PlatformConfig,
        highlight: Dict[str, Any]
    ) -> mp.VideoFileClip:
        """Apply YouTube-specific effects"""
        
        processed_clip = clip
        
        # Add thumbnails (as overlay at the beginning)
        if config.thumbnails:
            processed_clip = await self._add_thumbnail_overlay(processed_clip, highlight)
        
        # End screens
        if config.end_screens:
            processed_clip = await self._add_end_screen(processed_clip)
        
        return processed_clip
    
    async def _apply_facebook_effects(
        self,
        clip: mp.VideoFileClip,
        config: PlatformConfig,
        highlight: Dict[str, Any]
    ) -> mp.VideoFileClip:
        """Apply Facebook-specific effects"""
        
        processed_clip = clip
        
        # Community-focused effects
        if config.transitions:
            processed_clip = await self._add_smooth_transitions(processed_clip)
        
        return processed_clip
    
    async def _apply_twitter_effects(
        self,
        clip: mp.VideoFileClip,
        config: PlatformConfig,
        highlight: Dict[str, Any]
    ) -> mp.VideoFileClip:
        """Apply Twitter-specific effects"""
        
        processed_clip = clip
        
        # Minimal effects for Twitter
        # Just ensure good quality and proper aspect ratio
        
        return processed_clip
    
    async def _add_captions(self, clip: mp.VideoFileClip, caption: str) -> mp.VideoFileClip:
        """Add captions to video"""
        
        try:
            # Create text clip
            text_clip = mp.TextClip(
                caption,
                fontsize=50,
                color='white',
                stroke_color='black',
                stroke_width=2,
                font='Arial-Bold'
            ).set_duration(clip.duration).set_position(('center', 'bottom'))
            
            # Composite with video
            return mp.CompositeVideoClip([clip, text_clip])
            
        except Exception as e:
            logger.warning(f"Failed to add captions: {e}")
            return clip
    
    async def _add_zoom_effect(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Add zoom effect"""
        
        try:
            # Simple zoom effect - scale from 1.0 to 1.1
            def zoom_function(get_frame, t):
                frame = get_frame(t)
                zoom_factor = 1.0 + 0.1 * (t / clip.duration)
                
                # Resize frame
                h, w = frame.shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                
                # Resize and crop to original size
                resized = cv2.resize(frame, (new_w, new_h))
                
                # Crop from center
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                cropped = resized[start_y:start_y+h, start_x:start_x+w]
                
                return cropped
            
            return clip.fl(zoom_function)
            
        except Exception as e:
            logger.warning(f"Failed to add zoom effect: {e}")
            return clip
    
    async def _add_speed_ramps(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Add speed ramps"""
        
        try:
            # Simple speed ramp - slow start, fast middle, slow end
            duration = clip.duration
            
            if duration > 3:
                # Split into three parts
                part1 = clip.subclip(0, duration * 0.3)
                part2 = clip.subclip(duration * 0.3, duration * 0.7)
                part3 = clip.subclip(duration * 0.7, duration)
                
                # Apply different speeds
                part1_slow = speedx(part1, factor=0.8)
                part2_fast = speedx(part2, factor=1.3)
                part3_slow = speedx(part3, factor=0.8)
                
                # Concatenate
                return mp.concatenate_videoclips([part1_slow, part2_fast, part3_slow])
            
            return clip
            
        except Exception as e:
            logger.warning(f"Failed to add speed ramps: {e}")
            return clip
    
    async def _apply_color_grading(self, clip: mp.VideoFileClip, style: str) -> mp.VideoFileClip:
        """Apply color grading"""
        
        try:
            def color_grade(get_frame, t):
                frame = get_frame(t)
                
                if style == "instagram":
                    # Instagram-style color grading
                    # Increase saturation and warmth
                    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    
                    # Add warm tone
                    frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.1, 0, 255)  # Increase red
                    frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.9, 0, 255)  # Decrease blue
                
                return frame
            
            return clip.fl(color_grade)
            
        except Exception as e:
            logger.warning(f"Failed to apply color grading: {e}")
            return clip
    
    async def _add_smooth_transitions(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Add smooth transitions"""
        
        try:
            # Add fade in and fade out
            fade_duration = min(0.5, clip.duration / 4)
            
            return clip.fadein(fade_duration).fadeout(fade_duration)
            
        except Exception as e:
            logger.warning(f"Failed to add transitions: {e}")
            return clip
    
    async def _add_thumbnail_overlay(
        self,
        clip: mp.VideoFileClip,
        highlight: Dict[str, Any]
    ) -> mp.VideoFileClip:
        """Add thumbnail overlay"""
        
        try:
            # Create thumbnail text
            thumbnail_text = highlight.get("caption", "Viral Clip")[:30]
            
            text_clip = mp.TextClip(
                thumbnail_text,
                fontsize=60,
                color='yellow',
                stroke_color='black',
                stroke_width=3,
                font='Arial-Bold'
            ).set_duration(2).set_position(('center', 'top'))
            
            # Show only for first 2 seconds
            return mp.CompositeVideoClip([clip, text_clip])
            
        except Exception as e:
            logger.warning(f"Failed to add thumbnail overlay: {e}")
            return clip
    
    async def _add_end_screen(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Add end screen"""
        
        try:
            # Create end screen text
            end_text = mp.TextClip(
                "Subscribe for more!",
                fontsize=40,
                color='white',
                stroke_color='black',
                stroke_width=2,
                font='Arial-Bold'
            ).set_duration(2).set_position(('center', 'center'))
            
            # Add to last 2 seconds
            end_screen = end_text.set_start(clip.duration - 2)
            
            return mp.CompositeVideoClip([clip, end_screen])
            
        except Exception as e:
            logger.warning(f"Failed to add end screen: {e}")
            return clip
    
    def _get_applied_effects(self, platform: str, config: PlatformConfig) -> List[str]:
        """Get list of applied effects"""
        
        effects = ["resize", "audio_normalize"]
        
        if config.captions_required:
            effects.append("captions")
        
        if platform == "tiktok":
            if config.quick_cuts:
                effects.append("quick_cuts")
            if config.zoom_effects:
                effects.append("zoom_effects")
            if config.speed_ramps:
                effects.append("speed_ramps")
        
        elif platform == "instagram":
            if config.color_grading:
                effects.append("color_grading")
            if config.transitions:
                effects.append("smooth_transitions")
        
        elif platform == "youtube":
            if config.thumbnails:
                effects.append("thumbnail_overlay")
            if config.end_screens:
                effects.append("end_screen")
        
        return effects


class YouTubeDownloader:
    """Native YouTube video downloader"""
    
    def __init__(self, output_dir: str = "./downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    async def download_video(self, url: str, quality: str = "720p") -> str:
        """Download YouTube video"""
        
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': f'best[height<={quality[:-1]}]',
                'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
                'noplaylist': True,
            }
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Get downloaded file path
                filename = ydl.prepare_filename(info)
                
                logger.info(f"Downloaded video: {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Error downloading video from {url}: {e}")
            raise


class VideoAnalyzer:
    """Native video analyzer"""
    
    @staticmethod
    def analyze_video_properties(video_path: str) -> Dict[str, Any]:
        """Analyze video properties"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            properties = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": 0,
                "aspect_ratio": (16, 9),
                "file_size": os.path.getsize(video_path)
            }
            
            # Calculate duration
            if properties["fps"] > 0:
                properties["duration"] = properties["frame_count"] / properties["fps"]
            
            # Calculate aspect ratio
            if properties["height"] > 0:
                ratio = properties["width"] / properties["height"]
                if abs(ratio - 16/9) < 0.1:
                    properties["aspect_ratio"] = (16, 9)
                elif abs(ratio - 9/16) < 0.1:
                    properties["aspect_ratio"] = (9, 16)
                elif abs(ratio - 4/3) < 0.1:
                    properties["aspect_ratio"] = (4, 3)
                elif abs(ratio - 1) < 0.1:
                    properties["aspect_ratio"] = (1, 1)
                else:
                    properties["aspect_ratio"] = (properties["width"], properties["height"])
            
            cap.release()
            
            return properties
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            return {}
    
    @staticmethod
    def extract_audio_features(video_path: str) -> Dict[str, Any]:
        """Extract audio features from video"""
        
        try:
            # Load audio
            y, sr = librosa.load(video_path, sr=22050)
            
            # Extract features
            features = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "rms_energy": float(np.mean(librosa.feature.rms(y=y))),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
                "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0])
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features from {video_path}: {e}")
            return {}


# Convenience functions
async def process_video_to_viral_clips(
    video_path: str,
    highlights: List[Dict[str, Any]],
    platforms: List[str],
    platform_configs: Dict[str, PlatformConfig],
    output_dir: str = "./output"
) -> ProcessingResult:
    """Process video to viral clips"""
    
    processor = NativeVideoProcessor(output_dir)
    return await processor.process_video_for_platforms(
        video_path, highlights, platforms, platform_configs
    )


async def download_youtube_video(url: str, output_dir: str = "./downloads") -> str:
    """Download YouTube video"""
    
    downloader = YouTubeDownloader(output_dir)
    return await downloader.download_video(url)


def analyze_video(video_path: str) -> Dict[str, Any]:
    """Analyze video properties and features"""
    
    properties = VideoAnalyzer.analyze_video_properties(video_path)
    audio_features = VideoAnalyzer.extract_audio_features(video_path)
    
    return {
        "properties": properties,
        "audio_features": audio_features
    }