"""
Enhanced Viral Video Clips Model - Viral Video Agent
Inspired by Onyx agent system for intelligent video processing workflows
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema.language_model import LanguageModelInput

from ..interfaces.video_llm_interface import (
    VideoLLM, VideoProcessingMode, PlatformType, ClipSegment, ViralClipOutput
)
from ..llm.video_llm_factory import get_default_video_llm, get_video_llm_for_platform
from ..tools.video_processing_tools import (
    VideoToolRegistry, download_youtube_video, analyze_video,
    generate_clips, add_captions, apply_effects, ToolResult
)

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    OPTIMIZING = "optimizing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentTask:
    """Task for the viral video agent"""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    target_platforms: List[PlatformType]
    processing_mode: VideoProcessingMode
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: AgentState = AgentState.IDLE
    progress: float = 0.0
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class AgentContext:
    """Context for agent execution"""
    current_task: Optional[AgentTask] = None
    video_llm: Optional[VideoLLM] = None
    conversation_history: List[BaseMessage] = field(default_factory=list)
    workspace_dir: str = "./workspace"
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    max_concurrent_tasks: int = 3
    enable_caching: bool = True
    cache_dir: str = "./cache"


class ViralVideoAgent:
    """
    Intelligent agent for viral video processing
    Inspired by Onyx agent architecture with video-specific capabilities
    """
    
    def __init__(
        self,
        context: Optional[AgentContext] = None,
        video_llm: Optional[VideoLLM] = None
    ):
        self.context = context or AgentContext()
        self.video_llm = video_llm or get_default_video_llm()
        
        # Task management
        self.task_queue: List[AgentTask] = []
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentTask] = {}
        
        # Agent state
        self.state = AgentState.IDLE
        self.is_running = False
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.success_rate = 0.0
        
        # Initialize directories
        self._initialize_directories()
        
        logger.info("Viral Video Agent initialized")
    
    def _initialize_directories(self) -> None:
        """Initialize required directories"""
        for dir_path in [
            self.context.workspace_dir,
            self.context.output_dir,
            self.context.temp_dir,
            self.context.cache_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    async def process_request(
        self,
        request: Union[str, Dict[str, Any], BaseMessage],
        platforms: List[PlatformType] = None,
        processing_mode: VideoProcessingMode = VideoProcessingMode.VIRAL_CLIPS
    ) -> Dict[str, Any]:
        """Process a user request for viral video creation"""
        
        # Parse request
        task = self._parse_request(request, platforms, processing_mode)
        
        # Add to queue
        self.task_queue.append(task)
        
        # Process task
        result = await self._process_task(task)
        
        return result
    
    def _parse_request(
        self,
        request: Union[str, Dict[str, Any], BaseMessage],
        platforms: List[PlatformType],
        processing_mode: VideoProcessingMode
    ) -> AgentTask:
        """Parse user request into agent task"""
        
        task_id = f"task_{int(time.time())}_{len(self.task_queue)}"
        
        if isinstance(request, str):
            # Simple string request
            if request.startswith("http"):
                # YouTube URL
                task_type = "youtube_to_clips"
                input_data = {"url": request}
            else:
                # File path or text instruction
                task_type = "process_video" if Path(request).exists() else "text_instruction"
                input_data = {"input": request}
        
        elif isinstance(request, dict):
            # Structured request
            task_type = request.get("type", "process_video")
            input_data = request
        
        elif isinstance(request, BaseMessage):
            # LangChain message
            task_type = "chat_request"
            input_data = {"message": request}
        
        else:
            task_type = "unknown"
            input_data = {"raw_request": str(request)}
        
        # Default platforms if not specified
        if platforms is None:
            platforms = [PlatformType.TIKTOK, PlatformType.INSTAGRAM_REELS, PlatformType.YOUTUBE_SHORTS]
        
        return AgentTask(
            task_id=task_id,
            task_type=task_type,
            input_data=input_data,
            target_platforms=platforms,
            processing_mode=processing_mode
        )
    
    async def _process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a single task"""
        
        task.started_at = time.time()
        task.status = AgentState.PROCESSING
        self.active_tasks[task.task_id] = task
        
        try:
            logger.info(f"Processing task {task.task_id}: {task.task_type}")
            
            # Route to appropriate handler
            if task.task_type == "youtube_to_clips":
                result = await self._handle_youtube_to_clips(task)
            elif task.task_type == "process_video":
                result = await self._handle_process_video(task)
            elif task.task_type == "chat_request":
                result = await self._handle_chat_request(task)
            elif task.task_type == "text_instruction":
                result = await self._handle_text_instruction(task)
            else:
                result = await self._handle_unknown_task(task)
            
            # Mark as completed
            task.status = AgentState.COMPLETED
            task.completed_at = time.time()
            task.progress = 1.0
            task.results = result
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            del self.active_tasks[task.task_id]
            
            # Update statistics
            self.total_tasks_processed += 1
            self.total_processing_time += task.completed_at - task.started_at
            self._update_success_rate()
            
            logger.info(f"Task {task.task_id} completed successfully")
            
            return result
            
        except Exception as e:
            # Handle error
            task.status = AgentState.ERROR
            task.error_message = str(e)
            task.completed_at = time.time()
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Move to completed tasks (with error)
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            return {
                "success": False,
                "error": str(e),
                "task_id": task.task_id
            }
    
    async def _handle_youtube_to_clips(self, task: AgentTask) -> Dict[str, Any]:
        """Handle YouTube URL to viral clips conversion"""
        
        url = task.input_data["url"]
        
        # Step 1: Download YouTube video
        task.progress = 0.1
        task.status = AgentState.PROCESSING
        
        download_result = await download_youtube_video(
            url=url,
            output_dir=self.context.workspace_dir
        )
        
        if not download_result.success:
            raise Exception(f"Failed to download video: {download_result.message}")
        
        video_path = download_result.data["video_path"]
        video_metadata = download_result.data
        
        # Step 2: Analyze video
        task.progress = 0.3
        task.status = AgentState.ANALYZING
        
        analysis_result = await analyze_video(video_path=video_path)
        
        if not analysis_result.success:
            raise Exception(f"Failed to analyze video: {analysis_result.message}")
        
        # Step 3: Generate highlights using LLM
        task.progress = 0.5
        task.status = AgentState.GENERATING
        
        video_features = analysis_result.data
        highlights = self.video_llm.detect_highlights(video_features)
        
        # Step 4: Generate clips
        task.progress = 0.7
        
        clips_result = await generate_clips(
            video_path=video_path,
            highlights=highlights,
            platforms=task.target_platforms,
            output_dir=self.context.output_dir
        )
        
        if not clips_result.success:
            raise Exception(f"Failed to generate clips: {clips_result.message}")
        
        # Step 5: Add captions and effects
        task.progress = 0.9
        task.status = AgentState.OPTIMIZING
        
        enhanced_clips = []
        for clip_data in clips_result.data["clips"]:
            # Add captions if transcript available
            if "transcript" in video_metadata:
                caption_result = await add_captions(
                    video_path=clip_data["clip_path"],
                    transcript_segments=video_metadata.get("transcript_segments", [])
                )
                if caption_result.success:
                    clip_data["clip_path"] = caption_result.data["output_path"]
            
            # Apply viral effects
            effects = self._get_platform_effects(clip_data["platform"])
            if effects:
                effects_result = await apply_effects(
                    video_path=clip_data["clip_path"],
                    effects=effects
                )
                if effects_result.success:
                    clip_data["clip_path"] = effects_result.data["output_path"]
            
            enhanced_clips.append(clip_data)
        
        task.progress = 1.0
        task.status = AgentState.FINALIZING
        
        return {
            "success": True,
            "task_id": task.task_id,
            "source_url": url,
            "source_metadata": video_metadata,
            "analysis": video_features,
            "highlights": [self._serialize_clip_segment(h) for h in highlights],
            "clips": enhanced_clips,
            "total_clips": len(enhanced_clips),
            "platforms": [p.value for p in task.target_platforms],
            "processing_time": time.time() - task.started_at
        }
    
    async def _handle_process_video(self, task: AgentTask) -> Dict[str, Any]:
        """Handle local video file processing"""
        
        video_path = task.input_data["input"]
        
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Similar to YouTube processing but skip download step
        task.progress = 0.2
        task.status = AgentState.ANALYZING
        
        analysis_result = await analyze_video(video_path=video_path)
        
        if not analysis_result.success:
            raise Exception(f"Failed to analyze video: {analysis_result.message}")
        
        task.progress = 0.4
        task.status = AgentState.GENERATING
        
        video_features = analysis_result.data
        highlights = self.video_llm.detect_highlights(video_features)
        
        task.progress = 0.6
        
        clips_result = await generate_clips(
            video_path=video_path,
            highlights=highlights,
            platforms=task.target_platforms,
            output_dir=self.context.output_dir
        )
        
        if not clips_result.success:
            raise Exception(f"Failed to generate clips: {clips_result.message}")
        
        task.progress = 0.8
        task.status = AgentState.OPTIMIZING
        
        # Apply enhancements
        enhanced_clips = []
        for clip_data in clips_result.data["clips"]:
            effects = self._get_platform_effects(clip_data["platform"])
            if effects:
                effects_result = await apply_effects(
                    video_path=clip_data["clip_path"],
                    effects=effects
                )
                if effects_result.success:
                    clip_data["clip_path"] = effects_result.data["output_path"]
            
            enhanced_clips.append(clip_data)
        
        task.progress = 1.0
        
        return {
            "success": True,
            "task_id": task.task_id,
            "source_path": video_path,
            "analysis": video_features,
            "highlights": [self._serialize_clip_segment(h) for h in highlights],
            "clips": enhanced_clips,
            "total_clips": len(enhanced_clips),
            "platforms": [p.value for p in task.target_platforms],
            "processing_time": time.time() - task.started_at
        }
    
    async def _handle_chat_request(self, task: AgentTask) -> Dict[str, Any]:
        """Handle chat-based requests"""
        
        message = task.input_data["message"]
        
        # Add to conversation history
        self.context.conversation_history.append(message)
        
        # Process with video LLM
        response = self.video_llm.invoke(
            prompt=self.context.conversation_history,
            processing_mode=task.processing_mode,
            target_platforms=task.target_platforms
        )
        
        # Add response to history
        self.context.conversation_history.append(response)
        
        return {
            "success": True,
            "task_id": task.task_id,
            "response": response.content,
            "conversation_length": len(self.context.conversation_history),
            "processing_time": time.time() - task.started_at
        }
    
    async def _handle_text_instruction(self, task: AgentTask) -> Dict[str, Any]:
        """Handle text-based instructions"""
        
        instruction = task.input_data["input"]
        
        # Create human message
        human_message = HumanMessage(content=instruction)
        
        # Process with video LLM
        response = self.video_llm.invoke(
            prompt=[human_message],
            processing_mode=task.processing_mode,
            target_platforms=task.target_platforms
        )
        
        return {
            "success": True,
            "task_id": task.task_id,
            "instruction": instruction,
            "response": response.content,
            "processing_time": time.time() - task.started_at
        }
    
    async def _handle_unknown_task(self, task: AgentTask) -> Dict[str, Any]:
        """Handle unknown task types"""
        
        return {
            "success": False,
            "task_id": task.task_id,
            "error": f"Unknown task type: {task.task_type}",
            "input_data": task.input_data,
            "processing_time": time.time() - task.started_at
        }
    
    def _get_platform_effects(self, platform: PlatformType) -> List[str]:
        """Get effects to apply for specific platform"""
        
        platform_effects = {
            PlatformType.TIKTOK: ["speed_ramp", "zoom_effect", "color_boost"],
            PlatformType.INSTAGRAM_REELS: ["color_boost", "zoom_effect"],
            PlatformType.YOUTUBE_SHORTS: ["color_boost"],
            PlatformType.FACEBOOK_REELS: ["color_boost"],
            PlatformType.TWITTER_X: []
        }
        
        return platform_effects.get(platform, [])
    
    def _serialize_clip_segment(self, segment: ClipSegment) -> Dict[str, Any]:
        """Serialize clip segment for JSON output"""
        
        return {
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "duration": segment.duration,
            "viral_score": segment.viral_score,
            "engagement_prediction": segment.engagement_prediction,
            "content_type": segment.content_type,
            "emotions": segment.emotions,
            "objects_detected": segment.objects_detected,
            "faces_count": segment.faces_count,
            "motion_intensity": segment.motion_intensity,
            "transcript": segment.transcript
        }
    
    def _update_success_rate(self) -> None:
        """Update success rate based on completed tasks"""
        
        if not self.completed_tasks:
            self.success_rate = 0.0
            return
        
        successful_tasks = sum(
            1 for task in self.completed_tasks.values()
            if task.status == AgentState.COMPLETED
        )
        
        self.success_rate = successful_tasks / len(self.completed_tasks)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "progress": task.progress,
                "started_at": task.started_at,
                "processing_time": time.time() - task.started_at if task.started_at else 0
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "progress": task.progress,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "processing_time": (task.completed_at - task.started_at) if task.started_at and task.completed_at else 0,
                "error_message": task.error_message,
                "results": task.results
            }
        
        # Check queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "progress": task.progress,
                    "queue_position": self.task_queue.index(task)
                }
        
        return None
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "total_tasks_processed": self.total_tasks_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": (
                self.total_processing_time / self.total_tasks_processed
                if self.total_tasks_processed > 0 else 0
            ),
            "success_rate": self.success_rate,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "conversation_length": len(self.context.conversation_history)
        }
    
    def clear_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clear old completed tasks"""
        
        cutoff_time = time.time() - (older_than_hours * 3600)
        tasks_to_remove = []
        
        for task_id, task in self.completed_tasks.items():
            if task.completed_at and task.completed_at < cutoff_time:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]
        
        logger.info(f"Cleared {len(tasks_to_remove)} old completed tasks")
        return len(tasks_to_remove)
    
    async def batch_process_urls(
        self,
        urls: List[str],
        platforms: List[PlatformType] = None,
        processing_mode: VideoProcessingMode = VideoProcessingMode.VIRAL_CLIPS
    ) -> List[Dict[str, Any]]:
        """Process multiple YouTube URLs in batch"""
        
        tasks = []
        for url in urls:
            task = self._parse_request(url, platforms, processing_mode)
            tasks.append(task)
        
        # Process tasks concurrently
        results = await asyncio.gather(
            *[self._process_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "url": urls[i],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def stream_process_request(
        self,
        request: Union[str, Dict[str, Any], BaseMessage],
        platforms: List[PlatformType] = None,
        processing_mode: VideoProcessingMode = VideoProcessingMode.VIRAL_CLIPS
    ):
        """Stream processing updates for a request"""
        
        task = self._parse_request(request, platforms, processing_mode)
        task.started_at = time.time()
        task.status = AgentState.PROCESSING
        self.active_tasks[task.task_id] = task
        
        try:
            # Yield initial status
            yield {
                "task_id": task.task_id,
                "status": "started",
                "progress": 0.0,
                "message": "Processing request..."
            }
            
            # Process based on task type
            if task.task_type == "youtube_to_clips":
                async for update in self._stream_youtube_to_clips(task):
                    yield update
            elif task.task_type == "process_video":
                async for update in self._stream_process_video(task):
                    yield update
            else:
                # For other types, process normally and yield final result
                result = await self._process_task(task)
                yield {
                    "task_id": task.task_id,
                    "status": "completed",
                    "progress": 1.0,
                    "result": result
                }
        
        except Exception as e:
            yield {
                "task_id": task.task_id,
                "status": "error",
                "progress": task.progress,
                "error": str(e)
            }
    
    async def _stream_youtube_to_clips(self, task: AgentTask):
        """Stream YouTube to clips processing"""
        
        url = task.input_data["url"]
        
        # Download
        yield {
            "task_id": task.task_id,
            "status": "downloading",
            "progress": 0.1,
            "message": f"Downloading video from {url}..."
        }
        
        download_result = await download_youtube_video(
            url=url,
            output_dir=self.context.workspace_dir
        )
        
        if not download_result.success:
            raise Exception(f"Download failed: {download_result.message}")
        
        video_path = download_result.data["video_path"]
        
        # Analyze
        yield {
            "task_id": task.task_id,
            "status": "analyzing",
            "progress": 0.3,
            "message": "Analyzing video content..."
        }
        
        analysis_result = await analyze_video(video_path=video_path)
        
        if not analysis_result.success:
            raise Exception(f"Analysis failed: {analysis_result.message}")
        
        # Generate highlights
        yield {
            "task_id": task.task_id,
            "status": "generating",
            "progress": 0.5,
            "message": "Detecting viral highlights..."
        }
        
        video_features = analysis_result.data
        highlights = self.video_llm.detect_highlights(video_features)
        
        yield {
            "task_id": task.task_id,
            "status": "generating",
            "progress": 0.6,
            "message": f"Found {len(highlights)} highlight segments"
        }
        
        # Generate clips
        yield {
            "task_id": task.task_id,
            "status": "generating",
            "progress": 0.7,
            "message": "Generating viral clips..."
        }
        
        clips_result = await generate_clips(
            video_path=video_path,
            highlights=highlights,
            platforms=task.target_platforms,
            output_dir=self.context.output_dir
        )
        
        if not clips_result.success:
            raise Exception(f"Clip generation failed: {clips_result.message}")
        
        # Optimize
        yield {
            "task_id": task.task_id,
            "status": "optimizing",
            "progress": 0.9,
            "message": "Applying viral effects and optimizations..."
        }
        
        # Final result
        yield {
            "task_id": task.task_id,
            "status": "completed",
            "progress": 1.0,
            "message": f"Generated {len(clips_result.data['clips'])} viral clips!",
            "result": {
                "clips": clips_result.data["clips"],
                "highlights": len(highlights),
                "platforms": [p.value for p in task.target_platforms]
            }
        }
    
    async def _stream_process_video(self, task: AgentTask):
        """Stream local video processing"""
        
        video_path = task.input_data["input"]
        
        # Similar streaming pattern for local video processing
        yield {
            "task_id": task.task_id,
            "status": "analyzing",
            "progress": 0.2,
            "message": f"Analyzing video: {Path(video_path).name}..."
        }
        
        # Continue with similar pattern as YouTube processing
        # ... (implementation similar to _stream_youtube_to_clips but without download step)


# Convenience functions for agent usage
def create_viral_video_agent(
    workspace_dir: str = "./workspace",
    output_dir: str = "./output",
    video_llm: Optional[VideoLLM] = None
) -> ViralVideoAgent:
    """Create a viral video agent with custom configuration"""
    
    context = AgentContext(
        workspace_dir=workspace_dir,
        output_dir=output_dir
    )
    
    return ViralVideoAgent(context=context, video_llm=video_llm)


async def quick_youtube_to_clips(
    url: str,
    platforms: List[PlatformType] = None,
    output_dir: str = "./output"
) -> Dict[str, Any]:
    """Quick function to convert YouTube video to viral clips"""
    
    agent = create_viral_video_agent(output_dir=output_dir)
    return await agent.process_request(url, platforms)


async def quick_video_to_clips(
    video_path: str,
    platforms: List[PlatformType] = None,
    output_dir: str = "./output"
) -> Dict[str, Any]:
    """Quick function to convert local video to viral clips"""
    
    agent = create_viral_video_agent(output_dir=output_dir)
    return await agent.process_request(video_path, platforms)