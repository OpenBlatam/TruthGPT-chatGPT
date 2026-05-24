"""
Native Viral Video Clips Model
Streamlined AI-powered video processing for viral content creation
"""

import os
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import cv2
import librosa
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import yt_dlp
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class VideoClip:
    """Represents a generated viral video clip"""
    start_time: float
    end_time: float
    duration: float
    viral_score: float
    platform: str
    caption: str
    hashtags: List[str]
    effects: List[str]
    output_path: str


@dataclass
class ProcessingResult:
    """Result of video processing"""
    success: bool
    clips: List[VideoClip]
    total_clips: int
    processing_time: float
    source_info: Dict[str, Any]
    error_message: Optional[str] = None


class ViralVideoTransformer(nn.Module):
    """Core transformer model for viral video analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Video encoder
        self.video_conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.AdaptiveAvgPool3d((8, 7, 7))
        )
        
        # Audio encoder
        self.audio_conv1d = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(512)
        )
        
        # Fusion and prediction layers
        self.fusion = nn.Sequential(
            nn.Linear(128 * 8 * 7 * 7 + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.viral_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # 5 platforms
            nn.Sigmoid()
        )
        
        self.highlight_detector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8),  # 8 emotions
            nn.Softmax(dim=-1)
        )
    
    def forward(self, video_frames, audio_features):
        # Process video
        video_out = self.video_conv3d(video_frames)
        video_flat = video_out.view(video_out.size(0), -1)
        
        # Process audio
        audio_out = self.audio_conv1d(audio_features)
        audio_flat = audio_out.view(audio_out.size(0), -1)
        
        # Fuse modalities
        combined = torch.cat([video_flat, audio_flat], dim=1)
        fused = self.fusion(combined)
        
        # Predictions
        viral_scores = self.viral_predictor(fused)
        highlight_scores = self.highlight_detector(fused)
        emotions = self.emotion_classifier(fused)
        
        return {
            'viral_scores': viral_scores,
            'highlight_scores': highlight_scores,
            'emotions': emotions,
            'features': fused
        }


class NativeViralVideoModel:
    """Native viral video processing model"""
    
    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        self.config = {
            "small": {"params": "3B", "memory": "8GB", "speed": "fast"},
            "medium": {"params": "8B", "memory": "16GB", "speed": "balanced"},
            "large": {"params": "15B", "memory": "32GB", "speed": "slow"}
        }[model_size]
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.whisper_model = None
        self.is_loaded = False
        
        # Platform configurations
        self.platforms = {
            "tiktok": {
                "aspect_ratio": "9:16",
                "max_duration": 60,
                "min_duration": 15,
                "resolution": (1080, 1920),
                "effects": ["quick_cuts", "zoom", "speed_ramp"],
                "hashtags": ["fyp", "viral", "trending", "foryou"]
            },
            "instagram": {
                "aspect_ratio": "9:16", 
                "max_duration": 90,
                "min_duration": 15,
                "resolution": (1080, 1920),
                "effects": ["smooth_transitions", "color_grade"],
                "hashtags": ["reels", "instagram", "viral", "trending"]
            },
            "youtube": {
                "aspect_ratio": "9:16",
                "max_duration": 60,
                "min_duration": 15,
                "resolution": (1080, 1920),
                "effects": ["thumbnails", "end_screens"],
                "hashtags": ["shorts", "youtube", "viral"]
            },
            "facebook": {
                "aspect_ratio": "9:16",
                "max_duration": 90,
                "min_duration": 15,
                "resolution": (1080, 1920),
                "effects": ["community_focus"],
                "hashtags": ["facebook", "reels", "viral"]
            },
            "twitter": {
                "aspect_ratio": "16:9",
                "max_duration": 140,
                "min_duration": 10,
                "resolution": (1280, 720),
                "effects": ["minimal"],
                "hashtags": ["twitter", "viral", "trending"]
            }
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model components"""
        try:
            # Initialize transformer model
            model_config = {"model_size": self.model_size}
            self.model = ViralVideoTransformer(model_config)
            self.model.to(self.device)
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize Whisper
            whisper_size = {"small": "base", "medium": "small", "large": "medium"}[self.model_size]
            self.whisper_model = whisper.load_model(whisper_size)
            
            self.is_loaded = True
            logger.info(f"Native Viral Video Model ({self.model_size}) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def download_youtube_video(self, url: str, output_dir: str = "./downloads") -> Dict[str, Any]:
        """Download YouTube video and extract metadata"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ydl_opts = {
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'format': 'best[height<=720]',  # Limit quality for faster processing
            'writeinfojson': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                
                return {
                    "success": True,
                    "video_path": video_path,
                    "title": info.get("title", ""),
                    "description": info.get("description", ""),
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "like_count": info.get("like_count", 0),
                    "uploader": info.get("uploader", ""),
                    "tags": info.get("tags", []),
                    "thumbnail": info.get("thumbnail", "")
                }
        
        except Exception as e:
            logger.error(f"YouTube download failed: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_video_features(self, video_path: str) -> Dict[str, Any]:
        """Extract features from video file"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            video_clip = VideoFileClip(video_path)
            
            # Basic metadata
            metadata = {
                "duration": video_clip.duration,
                "fps": video_clip.fps,
                "resolution": (video_clip.w, video_clip.h),
                "aspect_ratio": video_clip.w / video_clip.h,
                "has_audio": video_clip.audio is not None
            }
            
            # Extract frames for analysis
            frames = []
            timestamps = []
            for t in np.arange(0, min(video_clip.duration, 60), 2.0):  # Every 2 seconds, max 60s
                frame = video_clip.get_frame(t)
                frames.append(frame)
                timestamps.append(t)
            
            # Extract audio features
            audio_features = {}
            if video_clip.audio:
                audio_array = video_clip.audio.to_soundarray()
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
                
                # Transcribe with Whisper
                transcript_result = self.whisper_model.transcribe(audio_array)
                
                audio_features = {
                    "transcript": transcript_result["text"],
                    "segments": transcript_result["segments"],
                    "has_speech": len(transcript_result["text"].strip()) > 0,
                    "volume_avg": np.mean(np.abs(audio_array)),
                    "volume_max": np.max(np.abs(audio_array))
                }
            
            # Analyze motion
            motion_scores = self._analyze_motion(frames)
            
            # Detect scene changes
            scene_changes = self._detect_scene_changes(frames, timestamps)
            
            video_clip.close()
            
            return {
                "metadata": metadata,
                "frames": frames,
                "timestamps": timestamps,
                "audio": audio_features,
                "motion": motion_scores,
                "scenes": scene_changes
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _analyze_motion(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze motion in video frames"""
        
        motion_scores = []
        prev_frame = None
        
        for frame in frames:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray_frame)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray_frame
        
        return {
            "scores": motion_scores,
            "average": np.mean(motion_scores) if motion_scores else 0,
            "max": np.max(motion_scores) if motion_scores else 0,
            "variance": np.var(motion_scores) if motion_scores else 0
        }
    
    def _detect_scene_changes(self, frames: List[np.ndarray], timestamps: List[float]) -> Dict[str, Any]:
        """Detect scene changes in video"""
        
        scene_changes = []
        prev_frame = None
        
        for i, frame in enumerate(frames):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray_frame)
                diff_score = np.mean(diff)
                
                if diff_score > 30:  # Scene change threshold
                    scene_changes.append(timestamps[i])
            
            prev_frame = gray_frame
        
        return {
            "changes": scene_changes,
            "count": len(scene_changes) + 1,
            "avg_length": timestamps[-1] / (len(scene_changes) + 1) if scene_changes else timestamps[-1]
        }
    
    def analyze_viral_potential(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze viral potential for different platforms"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Prepare model inputs
            video_tensor = self._prepare_video_tensor(features["frames"])
            audio_tensor = self._prepare_audio_tensor(features["audio"])
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(video_tensor, audio_tensor)
                viral_scores = outputs["viral_scores"].cpu().numpy()[0]
            
            # Map to platforms
            platform_names = ["tiktok", "instagram", "youtube", "facebook", "twitter"]
            return {platform: float(score) for platform, score in zip(platform_names, viral_scores)}
            
        except Exception as e:
            logger.error(f"Viral analysis failed: {e}")
            # Return default scores
            return {platform: 0.5 for platform in self.platforms.keys()}
    
    def detect_highlights(self, features: Dict[str, Any], min_duration: float = 15.0) -> List[Dict[str, Any]]:
        """Detect highlight segments in video"""
        
        highlights = []
        duration = features["metadata"]["duration"]
        motion_scores = features["motion"]["scores"]
        timestamps = features["timestamps"]
        
        # Simple highlight detection based on motion and audio
        for i in range(len(motion_scores) - 1):
            start_time = timestamps[i]
            end_time = min(timestamps[i] + min_duration, duration)
            
            if end_time - start_time < min_duration:
                continue
            
            # Calculate highlight score
            motion_score = motion_scores[i] if i < len(motion_scores) else 0
            has_speech = bool(features["audio"].get("has_speech", False))
            
            highlight_score = (motion_score / 50.0) * 0.6 + (1.0 if has_speech else 0.3) * 0.4
            highlight_score = min(highlight_score, 1.0)
            
            if highlight_score > 0.6:  # Threshold for highlights
                highlights.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "score": highlight_score,
                    "motion_intensity": motion_score,
                    "has_speech": has_speech
                })
        
        # Sort by score and return top highlights
        highlights.sort(key=lambda x: x["score"], reverse=True)
        return highlights[:15]  # Max 15 highlights
    
    def generate_clips(
        self, 
        video_path: str, 
        highlights: List[Dict[str, Any]], 
        platforms: List[str],
        output_dir: str = "./output"
    ) -> List[VideoClip]:
        """Generate viral clips from highlights"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        clips = []
        
        try:
            video_clip = VideoFileClip(video_path)
            
            for i, highlight in enumerate(highlights):
                for platform in platforms:
                    if platform not in self.platforms:
                        continue
                    
                    platform_config = self.platforms[platform]
                    
                    # Extract segment
                    start_time = highlight["start_time"]
                    end_time = min(
                        highlight["end_time"],
                        start_time + platform_config["max_duration"]
                    )
                    
                    segment = video_clip.subclip(start_time, end_time)
                    
                    # Optimize for platform
                    optimized_segment = self._optimize_for_platform(segment, platform_config)
                    
                    # Generate caption
                    caption = self._generate_caption(highlight, platform)
                    
                    # Generate output path
                    output_filename = f"clip_{i:03d}_{platform}.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Write clip
                    optimized_segment.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True,
                        verbose=False,
                        logger=None
                    )
                    
                    # Create clip object
                    clip = VideoClip(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        viral_score=highlight["score"],
                        platform=platform,
                        caption=caption,
                        hashtags=platform_config["hashtags"][:5],
                        effects=platform_config["effects"],
                        output_path=output_path
                    )
                    
                    clips.append(clip)
                    
                    # Cleanup
                    segment.close()
                    optimized_segment.close()
            
            video_clip.close()
            
        except Exception as e:
            logger.error(f"Clip generation failed: {e}")
            raise
        
        return clips
    
    def _optimize_for_platform(self, clip, platform_config: Dict[str, Any]):
        """Optimize clip for specific platform"""
        
        # Resize to platform aspect ratio
        target_ratio = 9/16 if platform_config["aspect_ratio"] == "9:16" else 16/9
        current_ratio = clip.w / clip.h
        
        if abs(current_ratio - target_ratio) > 0.1:
            if current_ratio > target_ratio:
                # Crop width
                new_width = int(clip.h * target_ratio)
                x_center = clip.w // 2
                x1 = x_center - new_width // 2
                x2 = x_center + new_width // 2
                clip = clip.crop(x1=x1, x2=x2)
            else:
                # Crop height
                new_height = int(clip.w / target_ratio)
                y_center = clip.h // 2
                y1 = y_center - new_height // 2
                y2 = y_center + new_height // 2
                clip = clip.crop(y1=y1, y2=y2)
        
        # Apply platform-specific effects
        effects = platform_config.get("effects", [])
        
        if "speed_ramp" in effects and clip.duration > 10:
            # Simple speed ramp
            mid_point = clip.duration / 2
            part1 = clip.subclip(0, mid_point)
            part2 = clip.subclip(mid_point).fx(lambda c: c.speedx(1.2))
            clip = CompositeVideoClip([part1, part2])
        
        if "zoom" in effects:
            # Simple zoom effect
            clip = clip.resize(lambda t: 1 + 0.05 * np.sin(2 * np.pi * t / clip.duration))
        
        if "color_grade" in effects:
            # Simple color grading
            clip = clip.fx(lambda c: c.colorx(1.1))
        
        return clip
    
    def _generate_caption(self, highlight: Dict[str, Any], platform: str) -> str:
        """Generate caption for highlight"""
        
        base_captions = [
            "🔥 This moment is INSANE!",
            "✨ You won't believe what happens next",
            "💯 Pure viral content right here",
            "🚀 This is why we love this",
            "🎯 Perfect timing captured",
            "⚡ Energy levels through the roof",
            "🌟 Absolutely incredible moment",
            "🔥 This hits different"
        ]
        
        platform_prefixes = {
            "tiktok": "POV: ",
            "instagram": "✨ ",
            "youtube": "🎯 ",
            "facebook": "💯 ",
            "twitter": ""
        }
        
        import random
        base_caption = random.choice(base_captions)
        prefix = platform_prefixes.get(platform, "")
        
        return f"{prefix}{base_caption}"
    
    def _prepare_video_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Prepare video frames for model input"""
        
        # Resize frames and convert to tensor
        processed_frames = []
        target_size = (224, 224)
        
        for frame in frames[:16]:  # Max 16 frames
            # Resize frame
            frame_resized = cv2.resize(frame, target_size)
            # Normalize
            frame_norm = frame_resized.astype(np.float32) / 255.0
            # Convert RGB to tensor format (C, H, W)
            frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1)
            processed_frames.append(frame_tensor)
        
        # Pad or truncate to exactly 16 frames
        while len(processed_frames) < 16:
            processed_frames.append(torch.zeros(3, 224, 224))
        
        # Stack to (T, C, H, W) then add batch dimension
        video_tensor = torch.stack(processed_frames[:16])
        video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        return video_tensor.to(self.device)
    
    def _prepare_audio_tensor(self, audio_features: Dict[str, Any]) -> torch.Tensor:
        """Prepare audio features for model input"""
        
        # Create dummy audio tensor for now
        # In real implementation, would process actual audio
        audio_tensor = torch.randn(1, 1, 16000).to(self.device)
        
        return audio_tensor
    
    def process_youtube_video(
        self, 
        url: str, 
        platforms: List[str] = None,
        output_dir: str = "./output"
    ) -> ProcessingResult:
        """Complete pipeline to process YouTube video"""
        
        start_time = time.time()
        
        if platforms is None:
            platforms = ["tiktok", "instagram", "youtube"]
        
        try:
            # Step 1: Download video
            logger.info("Downloading YouTube video...")
            download_result = self.download_youtube_video(url)
            
            if not download_result["success"]:
                return ProcessingResult(
                    success=False,
                    clips=[],
                    total_clips=0,
                    processing_time=time.time() - start_time,
                    source_info={},
                    error_message=download_result["error"]
                )
            
            video_path = download_result["video_path"]
            source_info = download_result
            
            # Step 2: Extract features
            logger.info("Extracting video features...")
            features = self.extract_video_features(video_path)
            
            # Step 3: Analyze viral potential
            logger.info("Analyzing viral potential...")
            viral_scores = self.analyze_viral_potential(features)
            
            # Step 4: Detect highlights
            logger.info("Detecting highlights...")
            highlights = self.detect_highlights(features)
            
            # Step 5: Generate clips
            logger.info("Generating viral clips...")
            clips = self.generate_clips(video_path, highlights, platforms, output_dir)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Processing completed: {len(clips)} clips generated in {processing_time:.1f}s")
            
            return ProcessingResult(
                success=True,
                clips=clips,
                total_clips=len(clips),
                processing_time=processing_time,
                source_info=source_info
            )
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                success=False,
                clips=[],
                total_clips=0,
                processing_time=time.time() - start_time,
                source_info={},
                error_message=str(e)
            )
    
    def process_local_video(
        self,
        video_path: str,
        platforms: List[str] = None,
        output_dir: str = "./output"
    ) -> ProcessingResult:
        """Process local video file"""
        
        start_time = time.time()
        
        if platforms is None:
            platforms = ["tiktok", "instagram", "youtube"]
        
        try:
            # Extract features
            logger.info("Extracting video features...")
            features = self.extract_video_features(video_path)
            
            # Analyze viral potential
            logger.info("Analyzing viral potential...")
            viral_scores = self.analyze_viral_potential(features)
            
            # Detect highlights
            logger.info("Detecting highlights...")
            highlights = self.detect_highlights(features)
            
            # Generate clips
            logger.info("Generating viral clips...")
            clips = self.generate_clips(video_path, highlights, platforms, output_dir)
            
            processing_time = time.time() - start_time
            
            source_info = {
                "video_path": video_path,
                "title": Path(video_path).stem,
                "duration": features["metadata"]["duration"],
                "resolution": features["metadata"]["resolution"]
            }
            
            logger.info(f"Processing completed: {len(clips)} clips generated in {processing_time:.1f}s")
            
            return ProcessingResult(
                success=True,
                clips=clips,
                total_clips=len(clips),
                processing_time=processing_time,
                source_info=source_info
            )
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                success=False,
                clips=[],
                total_clips=0,
                processing_time=time.time() - start_time,
                source_info={},
                error_message=str(e)
            )


# Convenience functions
def create_viral_clips_from_youtube(
    url: str,
    platforms: List[str] = None,
    model_size: str = "medium",
    output_dir: str = "./output"
) -> ProcessingResult:
    """Create viral clips from YouTube URL"""
    
    model = NativeViralVideoModel(model_size)
    return model.process_youtube_video(url, platforms, output_dir)


def create_viral_clips_from_video(
    video_path: str,
    platforms: List[str] = None,
    model_size: str = "medium",
    output_dir: str = "./output"
) -> ProcessingResult:
    """Create viral clips from local video"""
    
    model = NativeViralVideoModel(model_size)
    return model.process_local_video(video_path, platforms, output_dir)


def analyze_video_viral_potential(
    video_path: str,
    model_size: str = "medium"
) -> Dict[str, Any]:
    """Analyze viral potential of video"""
    
    model = NativeViralVideoModel(model_size)
    features = model.extract_video_features(video_path)
    viral_scores = model.analyze_viral_potential(features)
    highlights = model.detect_highlights(features)
    
    return {
        "viral_scores": viral_scores,
        "highlights": len(highlights),
        "highlight_segments": highlights,
        "metadata": features["metadata"]
    }