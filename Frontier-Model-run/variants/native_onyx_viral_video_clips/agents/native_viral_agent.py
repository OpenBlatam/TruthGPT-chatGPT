"""
Native Viral Video Agent
Intelligent agent for viral video processing using native AI models

This module provides an agent that orchestrates the entire viral video
processing pipeline using only native models and tools.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import time

from ..interfaces.native_video_interface import VideoAnalysisResult, VideoSegment
from ..llm.native_llm_factory import create_native_llm, NativeVideoLLMInterface
from ..llm.enhanced_native_viral_llm import NativeViralVideoModel
from ..tools.native_video_tools import (
    NativeVideoProcessor, YouTubeDownloader, VideoAnalyzer,
    ProcessingResult, VideoClip
)
from ..configs.native_model_configs import NativeModelConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Task for the viral video agent"""
    task_id: str
    task_type: str  # "analyze", "process", "download_and_process"
    input_data: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error_message: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class AgentCapabilities:
    """Agent capabilities"""
    video_analysis: bool = True
    caption_generation: bool = True
    highlight_detection: bool = True
    viral_prediction: bool = True
    multi_platform_processing: bool = True
    youtube_download: bool = True
    real_time_processing: bool = True
    batch_processing: bool = True


class NativeViralVideoAgent:
    """Intelligent agent for viral video processing"""
    
    def __init__(
        self,
        config: Optional[NativeModelConfig] = None,
        output_dir: str = "./output"
    ):
        self.config = config or get_config("default")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.llm: Optional[NativeVideoLLMInterface] = None
        self.video_processor: Optional[NativeVideoProcessor] = None
        self.youtube_downloader: Optional[YouTubeDownloader] = None
        
        # Task management
        self.tasks: Dict[str, AgentTask] = {}
        self.task_counter = 0
        
        # Agent state
        self.is_initialized = False
        self.capabilities = AgentCapabilities()
        
        logger.info("Native viral video agent created")
    
    async def initialize(self):
        """Initialize the agent"""
        try:
            logger.info("Initializing native viral video agent...")
            
            # Initialize LLM
            self.llm = NativeViralVideoModel(self.config)
            
            # Initialize video processor
            self.video_processor = NativeVideoProcessor(str(self.output_dir))
            
            # Initialize YouTube downloader
            self.youtube_downloader = YouTubeDownloader(str(self.output_dir / "downloads"))
            
            self.is_initialized = True
            logger.info("Native viral video agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def process_youtube_video(
        self,
        url: str,
        platforms: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process YouTube video to viral clips"""
        
        if not self.is_initialized:
            await self.initialize()
        
        if platforms is None:
            platforms = ["tiktok", "instagram", "youtube"]
        
        task_id = self._create_task_id()
        task = AgentTask(
            task_id=task_id,
            task_type="download_and_process",
            input_data={
                "url": url,
                "platforms": platforms,
                **kwargs
            },
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        
        try:
            task.status = "running"
            task.started_at = time.time()
            
            logger.info(f"Processing YouTube video: {url}")
            
            # Step 1: Download video
            logger.info("Downloading video...")
            video_path = await self.youtube_downloader.download_video(url)
            
            # Step 2: Analyze video
            logger.info("Analyzing video...")
            analysis_result = await self.llm.analyze_video(video_path)
            
            # Step 3: Process clips
            logger.info("Processing viral clips...")
            processing_result = await self.video_processor.process_video_for_platforms(
                video_path,
                analysis_result.highlights,
                platforms,
                {platform: self.config.get_platform_config(platform) for platform in platforms}
            )
            
            # Compile results
            result = {
                "task_id": task_id,
                "success": True,
                "video_path": video_path,
                "analysis": {
                    "viral_scores": analysis_result.viral_scores,
                    "highlights_count": len(analysis_result.highlights),
                    "emotions": analysis_result.emotions,
                    "objects": analysis_result.objects,
                    "scenes": analysis_result.scenes,
                    "confidence": analysis_result.confidence,
                    "processing_time": analysis_result.processing_time
                },
                "clips": [
                    {
                        "platform": clip.platform,
                        "output_path": clip.output_path,
                        "viral_score": clip.viral_score,
                        "caption": clip.caption,
                        "effects_applied": clip.effects_applied,
                        "duration": clip.metadata.get("duration", 0),
                        "start_time": clip.start_time,
                        "end_time": clip.end_time
                    }
                    for clip in processing_result.clips
                ],
                "total_clips": processing_result.total_clips,
                "platforms": platforms,
                "total_processing_time": processing_result.processing_time
            }
            
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
            
            logger.info(f"Successfully processed YouTube video. Generated {processing_result.total_clips} clips.")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing YouTube video: {e}")
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = time.time()
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e)
            }
    
    async def process_local_video(
        self,
        video_path: str,
        platforms: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process local video to viral clips"""
        
        if not self.is_initialized:
            await self.initialize()
        
        if platforms is None:
            platforms = ["tiktok", "instagram", "youtube"]
        
        task_id = self._create_task_id()
        task = AgentTask(
            task_id=task_id,
            task_type="process",
            input_data={
                "video_path": video_path,
                "platforms": platforms,
                **kwargs
            },
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        
        try:
            task.status = "running"
            task.started_at = time.time()
            
            logger.info(f"Processing local video: {video_path}")
            
            # Step 1: Analyze video
            logger.info("Analyzing video...")
            analysis_result = await self.llm.analyze_video(video_path)
            
            # Step 2: Process clips
            logger.info("Processing viral clips...")
            processing_result = await self.video_processor.process_video_for_platforms(
                video_path,
                analysis_result.highlights,
                platforms,
                {platform: self.config.get_platform_config(platform) for platform in platforms}
            )
            
            # Compile results
            result = {
                "task_id": task_id,
                "success": True,
                "video_path": video_path,
                "analysis": {
                    "viral_scores": analysis_result.viral_scores,
                    "highlights_count": len(analysis_result.highlights),
                    "emotions": analysis_result.emotions,
                    "objects": analysis_result.objects,
                    "scenes": analysis_result.scenes,
                    "confidence": analysis_result.confidence,
                    "processing_time": analysis_result.processing_time
                },
                "clips": [
                    {
                        "platform": clip.platform,
                        "output_path": clip.output_path,
                        "viral_score": clip.viral_score,
                        "caption": clip.caption,
                        "effects_applied": clip.effects_applied,
                        "duration": clip.metadata.get("duration", 0),
                        "start_time": clip.start_time,
                        "end_time": clip.end_time
                    }
                    for clip in processing_result.clips
                ],
                "total_clips": processing_result.total_clips,
                "platforms": platforms,
                "total_processing_time": processing_result.processing_time
            }
            
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
            
            logger.info(f"Successfully processed local video. Generated {processing_result.total_clips} clips.")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing local video: {e}")
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = time.time()
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e)
            }
    
    async def analyze_video_only(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """Analyze video without processing clips"""
        
        if not self.is_initialized:
            await self.initialize()
        
        task_id = self._create_task_id()
        task = AgentTask(
            task_id=task_id,
            task_type="analyze",
            input_data={
                "video_path": video_path,
                **kwargs
            },
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        
        try:
            task.status = "running"
            task.started_at = time.time()
            
            logger.info(f"Analyzing video: {video_path}")
            
            # Analyze video
            analysis_result = await self.llm.analyze_video(video_path)
            
            # Get video properties
            video_properties = VideoAnalyzer.analyze_video_properties(video_path)
            audio_features = VideoAnalyzer.extract_audio_features(video_path)
            
            result = {
                "task_id": task_id,
                "success": True,
                "video_path": video_path,
                "viral_scores": analysis_result.viral_scores,
                "highlights": [
                    {
                        "start_time": h["start_time"],
                        "end_time": h["end_time"],
                        "viral_score": h["viral_score"],
                        "caption": h["caption"],
                        "emotions": h["emotions"],
                        "objects": h["objects"]
                    }
                    for h in analysis_result.highlights
                ],
                "captions": analysis_result.captions,
                "emotions": analysis_result.emotions,
                "objects": analysis_result.objects,
                "scenes": analysis_result.scenes,
                "confidence": analysis_result.confidence,
                "processing_time": analysis_result.processing_time,
                "video_properties": video_properties,
                "audio_features": audio_features,
                "metadata": analysis_result.metadata
            }
            
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
            
            logger.info("Video analysis completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = time.time()
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e)
            }
    
    async def batch_process_videos(
        self,
        video_inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple videos in batch"""
        
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"Starting batch processing of {len(video_inputs)} videos")
        
        results = []
        
        # Process videos concurrently (with limit)
        max_concurrent = min(self.config.processing.max_workers, len(video_inputs))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_video(video_input):
            async with semaphore:
                if "url" in video_input:
                    return await self.process_youtube_video(
                        video_input["url"],
                        video_input.get("platforms", ["tiktok", "instagram", "youtube"]),
                        **kwargs
                    )
                elif "video_path" in video_input:
                    return await self.process_local_video(
                        video_input["video_path"],
                        video_input.get("platforms", ["tiktok", "instagram", "youtube"]),
                        **kwargs
                    )
                else:
                    return {
                        "success": False,
                        "error": "Invalid input: must contain 'url' or 'video_path'"
                    }
        
        # Execute batch processing
        tasks = [process_single_video(video_input) for video_input in video_inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "input": video_inputs[i]
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processing completed. {len(processed_results)} results.")
        
        return processed_results
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error_message": task.error_message,
            "has_result": task.result is not None
        }
    
    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tasks with optional status filter"""
        
        tasks = []
        for task in self.tasks.values():
            if status_filter is None or task.status == status_filter:
                tasks.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "error_message": task.error_message
                })
        
        return tasks
    
    def get_capabilities(self) -> AgentCapabilities:
        """Get agent capabilities"""
        return self.capabilities
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for task in self.tasks.values() if task.status == "completed")
        failed_tasks = sum(1 for task in self.tasks.values() if task.status == "failed")
        running_tasks = sum(1 for task in self.tasks.values() if task.status == "running")
        
        # Calculate average processing time for completed tasks
        completed_task_times = [
            task.completed_at - task.started_at
            for task in self.tasks.values()
            if task.status == "completed" and task.started_at and task.completed_at
        ]
        
        avg_processing_time = sum(completed_task_times) / len(completed_task_times) if completed_task_times else 0
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "average_processing_time": avg_processing_time,
            "is_initialized": self.is_initialized,
            "capabilities": self.capabilities.__dict__
        }
    
    def clear_completed_tasks(self):
        """Clear completed and failed tasks"""
        
        self.tasks = {
            task_id: task for task_id, task in self.tasks.items()
            if task.status in ["pending", "running"]
        }
        
        logger.info("Cleared completed and failed tasks")
    
    def _create_task_id(self) -> str:
        """Create unique task ID"""
        self.task_counter += 1
        return f"task_{self.task_counter}_{int(time.time())}"


# Convenience functions
async def create_viral_agent(
    config: Optional[NativeModelConfig] = None,
    output_dir: str = "./output"
) -> NativeViralVideoAgent:
    """Create and initialize viral video agent"""
    
    agent = NativeViralVideoAgent(config, output_dir)
    await agent.initialize()
    return agent


async def quick_youtube_to_clips(
    url: str,
    platforms: List[str] = None,
    output_dir: str = "./output"
) -> Dict[str, Any]:
    """Quick function to convert YouTube video to viral clips"""
    
    agent = await create_viral_agent(output_dir=output_dir)
    return await agent.process_youtube_video(url, platforms)


async def quick_video_to_clips(
    video_path: str,
    platforms: List[str] = None,
    output_dir: str = "./output"
) -> Dict[str, Any]:
    """Quick function to convert local video to viral clips"""
    
    agent = await create_viral_agent(output_dir=output_dir)
    return await agent.process_local_video(video_path, platforms)


async def quick_video_analysis(
    video_path: str,
    output_dir: str = "./output"
) -> Dict[str, Any]:
    """Quick function to analyze video"""
    
    agent = await create_viral_agent(output_dir=output_dir)
    return await agent.analyze_video_only(video_path)