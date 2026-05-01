"""
TruthGPT Real-Time Computing Features
Advanced real-time computing, stream processing, and adaptive batching for TruthGPT
"""

import asyncio
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from pydantic import BaseModel, ConfigDict, Field
from ..common.base_advanced_system import BaseAdvancedSystem
from enum import Enum
import logging
from pathlib import Path
import pickle
import threading
from datetime import datetime, timedelta
import uuid
import math
import random
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import heapq
import queue
import weakref
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import psutil
import signal
import sys

# Import TruthGPT modules
from .models import TruthGPTModel, TruthGPTModelConfig
from .distributed_computing import DistributedCoordinator, DistributedWorker
from .ai_enhancement import TruthGPTAIEnhancementManager


class RealTimeMode(Enum):
    """Real-time processing modes"""
    STREAMING = "streaming"
    BATCH_STREAMING = "batch_streaming"
    MICRO_BATCHING = "micro_batching"
    CONTINUOUS = "continuous"
    EVENT_DRIVEN = "event_driven"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"


class LatencyRequirement(Enum):
    """Latency requirements"""
    ULTRA_LOW = "ultra_low"      # < 1ms
    VERY_LOW = "very_low"        # < 10ms
    LOW = "low"                  # < 100ms
    MEDIUM = "medium"            # < 1s
    HIGH = "high"                # < 10s
    VERY_HIGH = "very_high"      # > 10s


class ProcessingPriority(Enum):
    """Processing priorities"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class StreamType(Enum):
    """Stream types"""
    TEXT_STREAM = "text_stream"
    AUDIO_STREAM = "audio_stream"
    VIDEO_STREAM = "video_stream"
    SENSOR_STREAM = "sensor_stream"
    LOG_STREAM = "log_stream"
    METRIC_STREAM = "metric_stream"
    EVENT_STREAM = "event_stream"
    DATA_STREAM = "data_stream"


class RealTimeConfig(BaseModel):
    """Configuration for real-time computing"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    mode: RealTimeMode = RealTimeMode.STREAMING
    latency_requirement: LatencyRequirement = LatencyRequirement.LOW
    max_latency_ms: float = 100.0
    batch_size: int = 32
    max_batch_size: int = 128
    min_batch_size: int = 1
    batch_timeout_ms: float = 50.0
    enable_adaptive_batching: bool = True
    enable_backpressure: bool = True
    enable_flow_control: bool = True
    buffer_size: int = 10000
    enable_compression: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    enable_monitoring: bool = True
    enable_optimization: bool = True
    enable_fault_tolerance: bool = True
    checkpoint_interval: float = 1.0


class StreamEvent(BaseModel):
    """Stream event"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    event_id: str
    stream_type: StreamType
    data: Any
    timestamp: float = Field(default_factory=time.time)
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    sequence_number: int = 0


class ProcessingBatch(BaseModel):
    """Processing batch"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    batch_id: str
    events: List[StreamEvent]
    created_at: float = Field(default_factory=time.time)
    processing_started: Optional[float] = None
    processing_completed: Optional[float] = None
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    estimated_processing_time: float = 0.0


class RealTimeBuffer:
    """Real-time buffer for stream processing"""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(f"RealTimeBuffer_{id(self)}")
        
        # Buffer management
        self.buffer: deque = deque(maxlen=config.buffer_size)
        self.priority_queues: Dict[ProcessingPriority, deque] = {
            priority: deque() for priority in ProcessingPriority
        }
        
        # Statistics
        self.stats = {
            "events_received": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "buffer_overflows": 0,
            "average_latency": 0.0,
            "max_latency": 0.0,
            "min_latency": float('inf')
        }
        
        # Performance tracking
        self.latency_history: deque = deque(maxlen=1000)
        self.throughput_history: deque = deque(maxlen=100)
        
        # Flow control
        self.backpressure_threshold = config.buffer_size * 0.8
        self.flow_control_active = False
    
    def add_event(self, event: StreamEvent) -> bool:
        """Add event to buffer"""
        try:
            # Check backpressure
            if self._should_apply_backpressure():
                self._apply_backpressure()
                return False
            
            # Add to priority queue
            self.priority_queues[event.priority].append(event)
            
            # Update statistics
            self.stats["events_received"] += 1
            
            return True
    
        except Exception as e:
            self.logger.error(f"Failed to add event: {e}")
            self.stats["events_dropped"] += 1
            return False
    
    def get_next_batch(self, batch_size: int = None) -> Optional[ProcessingBatch]:
        """Get next batch for processing"""
        batch_size = batch_size or self.config.batch_size
        
        # Collect events from priority queues
        events = []
        for priority in ProcessingPriority:
            queue = self.priority_queues[priority]
            while queue and len(events) < batch_size:
                events.append(queue.popleft())
        
        if not events:
                return None
            
        # Create processing batch
        batch = ProcessingBatch(
            batch_id=str(uuid.uuid4()),
            events=events,
            priority=self._determine_batch_priority(events)
        )
        
        return batch
    
    def _should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied"""
        if not self.config.enable_backpressure:
            return False
        
        total_events = sum(len(queue) for queue in self.priority_queues.values())
        return total_events > self.backpressure_threshold
    
    def _apply_backpressure(self):
        """Apply backpressure"""
        self.flow_control_active = True
        self.stats["buffer_overflows"] += 1
        
        # Drop low priority events
        for priority in [ProcessingPriority.BACKGROUND, ProcessingPriority.LOW]:
            queue = self.priority_queues[priority]
            while len(queue) > queue.maxlen * 0.5:
                queue.popleft()
                self.stats["events_dropped"] += 1
    
    def _determine_batch_priority(self, events: List[StreamEvent]) -> ProcessingPriority:
        """Determine batch priority based on events"""
        if not events:
            return ProcessingPriority.NORMAL
        
        # Use highest priority event's priority
        return min(event.priority for event in events)
    
    def update_latency_stats(self, latency: float):
        """Update latency statistics"""
        self.latency_history.append(latency)
        
        # Update stats
        self.stats["average_latency"] = np.mean(self.latency_history)
        self.stats["max_latency"] = max(self.stats["max_latency"], latency)
        self.stats["min_latency"] = min(self.stats["min_latency"], latency)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        total_events = sum(len(queue) for queue in self.priority_queues.values())
        
        return {
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.__dict__,
            "total_events_in_buffer": total_events,
            "events_by_priority": {
                priority.name: len(queue) 
                for priority, queue in self.priority_queues.items()
            },
            "stats": self.stats,
            "flow_control_active": self.flow_control_active,
            "buffer_utilization": total_events / self.config.buffer_size
        }


class AdaptiveBatcher:
    """Adaptive batching for real-time processing"""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(f"AdaptiveBatcher_{id(self)}")
        
        # Batching parameters
        self.current_batch_size = config.batch_size
        self.target_latency = config.max_latency_ms
        self.batch_timeout = config.batch_timeout_ms / 1000.0
        
        # Adaptive parameters
        self.adaptation_rate = 0.1
        self.min_batch_size = config.min_batch_size
        self.max_batch_size = config.max_batch_size
        
        # Performance tracking
        self.latency_history: deque = deque(maxlen=100)
        self.throughput_history: deque = deque(maxlen=100)
        self.batch_size_history: deque = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            "total_batches": 0,
            "adaptive_adjustments": 0,
            "latency_violations": 0,
            "throughput_improvements": 0
        }
    
    def should_create_batch(self, buffer: RealTimeBuffer, 
                          time_since_last_batch: float) -> bool:
        """Determine if a batch should be created"""
        # Time-based batching
        if time_since_last_batch >= self.batch_timeout:
            return True
        
        # Size-based batching
        total_events = sum(len(queue) for queue in buffer.priority_queues.values())
        if total_events >= self.current_batch_size:
            return True
        
        # Priority-based batching (immediate processing for high priority)
        high_priority_events = len(buffer.priority_queues[ProcessingPriority.CRITICAL])
        if high_priority_events > 0:
            return True
        
        return False
    
    def get_optimal_batch_size(self, buffer: RealTimeBuffer) -> int:
        """Get optimal batch size based on current conditions"""
        if not self.config.enable_adaptive_batching:
            return self.current_batch_size
        
        # Calculate current performance metrics
        current_latency = np.mean(buffer.latency_history) if buffer.latency_history else 0
        current_throughput = len(buffer.latency_history) / max(len(buffer.latency_history), 1)
        
        # Adaptive adjustment
        if current_latency > self.target_latency:
            # Reduce batch size to improve latency
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adaptation_rate))
            )
            self.stats["latency_violations"] += 1
        elif current_latency < self.target_latency * 0.5 and current_throughput < 100:
            # Increase batch size to improve throughput
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adaptation_rate))
            )
            self.stats["throughput_improvements"] += 1
        
        # Record batch size
        self.batch_size_history.append(self.current_batch_size)
        
        return self.current_batch_size
    
    def update_performance(self, batch: ProcessingBatch, processing_time: float):
        """Update performance metrics after batch processing"""
        # Calculate latency
        latency = (batch.processing_completed - batch.created_at) * 1000  # Convert to ms
        
        # Update histories
        self.latency_history.append(latency)
        self.throughput_history.append(len(batch.events) / processing_time)
        
        # Update statistics
        self.stats["total_batches"] += 1
        
        # Check for adaptation
        if len(self.latency_history) >= 10:
            avg_latency = np.mean(list(self.latency_history)[-10:])
            if abs(avg_latency - self.target_latency) > self.target_latency * 0.1:
                self.stats["adaptive_adjustments"] += 1
    
    def get_batcher_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        return {
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.__dict__,
            "current_batch_size": self.current_batch_size,
            "target_latency_ms": self.target_latency,
            "batch_timeout_ms": self.batch_timeout * 1000,
            "stats": self.stats,
            "average_latency": np.mean(self.latency_history) if self.latency_history else 0,
            "average_throughput": np.mean(self.throughput_history) if self.throughput_history else 0,
            "batch_size_trend": list(self.batch_size_history)[-10:] if self.batch_size_history else []
        }


class StreamProcessor:
    """Stream processor for real-time computing"""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(f"StreamProcessor_{id(self)}")
        
        # Processing components
        self.buffer = RealTimeBuffer(config)
        self.batcher = AdaptiveBatcher(config)
        
        # Model management
        self.model: Optional[TruthGPTModel] = None
        self.model_config: Optional[TruthGPTModelConfig] = None
        
        # Processing state
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.processing_stats = {
            "events_processed": 0,
            "batches_processed": 0,
            "processing_time": 0.0,
            "average_processing_time": 0.0,
            "throughput_events_per_second": 0.0
        }
        
        # Event handlers
        self.event_handlers: Dict[StreamType, Callable] = {}
        self._init_default_handlers()
    
    def _init_default_handlers(self):
        """Initialize default event handlers"""
        self.event_handlers = {
            StreamType.TEXT_STREAM: self._handle_text_stream,
            StreamType.AUDIO_STREAM: self._handle_audio_stream,
            StreamType.VIDEO_STREAM: self._handle_video_stream,
            StreamType.SENSOR_STREAM: self._handle_sensor_stream,
            StreamType.LOG_STREAM: self._handle_log_stream,
            StreamType.METRIC_STREAM: self._handle_metric_stream,
            StreamType.EVENT_STREAM: self._handle_event_stream,
            StreamType.DATA_STREAM: self._handle_data_stream
        }
    
    def set_model(self, model: TruthGPTModel, config: TruthGPTModelConfig):
        """Set model for processing"""
        self.model = model
        self.model_config = config
        self.logger.info("Model set for stream processing")
    
    def add_event_handler(self, stream_type: StreamType, handler: Callable):
        """Add custom event handler"""
        self.event_handlers[stream_type] = handler
        self.logger.info(f"Added custom handler for {stream_type.value}")
    
    async def start_processing(self):
        """Start stream processing"""
        if self.is_processing:
            self.logger.warning("Processing already started")
            return
        
        self.is_processing = True
        self.logger.info("Starting stream processing")
        
        # Start processing loop
        await self._processing_loop()
    
    async def stop_processing(self):
        """Stop stream processing"""
        self.is_processing = False
        self.logger.info("Stopping stream processing")
        
    async def _processing_loop(self):
        """Main processing loop"""
        last_batch_time = time.time()
        
        while self.is_processing:
            try:
                current_time = time.time()
                time_since_last_batch = current_time - last_batch_time
                
                # Check if we should create a batch
                if self.batcher.should_create_batch(self.buffer, time_since_last_batch):
                    # Get optimal batch size
                    batch_size = self.batcher.get_optimal_batch_size(self.buffer)
                    
                    # Get batch from buffer
                    batch = self.buffer.get_next_batch(batch_size)
                    
                    if batch:
                        # Process batch
                        await self._process_batch(batch)
                        last_batch_time = current_time
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: ProcessingBatch):
        """Process a batch of events"""
        batch.processing_started = time.time()
        
        try:
            # Group events by type
            events_by_type = defaultdict(list)
            for event in batch.events:
                events_by_type[event.stream_type].append(event)
            
            # Process each stream type
            results = {}
            for stream_type, events in events_by_type.items():
                handler = self.event_handlers.get(stream_type)
                if handler:
                    result = await handler(events)
                    results[stream_type] = result
                else:
                    self.logger.warning(f"No handler for stream type {stream_type.value}")
            
            # Update statistics
            batch.processing_completed = time.time()
            processing_time = batch.processing_completed - batch.processing_started
            
            self._update_processing_stats(batch, processing_time)
            self.batcher.update_performance(batch, processing_time)
            
            # Update buffer latency stats
            for event in batch.events:
                latency = (batch.processing_completed - event.timestamp) * 1000
                self.buffer.update_latency_stats(latency)
            
            self.logger.debug(f"Processed batch {batch.batch_id} with {len(batch.events)} events")
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            batch.processing_completed = time.time()
    
    def _update_processing_stats(self, batch: ProcessingBatch, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["events_processed"] += len(batch.events)
        self.processing_stats["batches_processed"] += 1
        self.processing_stats["processing_time"] += processing_time
        
        # Update averages
        self.processing_stats["average_processing_time"] = \
            self.processing_stats["processing_time"] / self.processing_stats["batches_processed"]
        
        # Calculate throughput
        if processing_time > 0:
            throughput = len(batch.events) / processing_time
            self.processing_stats["throughput_events_per_second"] = throughput
    
    async def _handle_text_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle text stream events with actual model inference if available."""
        if not self.model:
            return {"error": "No model available"}
        
        # Extract text data
        texts = [event.data for event in events if isinstance(event.data, str)]
        
        if not texts:
            return {"processed": 0}
        
        # Process with model
        results = []
        for text in texts:
            try:
                # Simulate embedding + forward pass
                # In a real system, we'd tokenize here
                mock_input = torch.randn(1, 512) # Simulated input vector
                with torch.no_grad():
                    output = self.model(mock_input)
                results.append({"text": text[:50], "inference": output.tolist()})
            except Exception as e:
                self.logger.error(f"Inference failed for text: {e}")
                results.append({"text": text[:50], "error": str(e)})
        
        return {
            "stream_type": "text",
            "events_processed": len(events),
            "results": results
        }
    
    async def _handle_audio_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle audio stream events"""
        # Simulate audio processing
        return {
            "stream_type": "audio",
            "events_processed": len(events),
            "audio_features_extracted": len(events)
        }
    
    async def _handle_video_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle video stream events"""
        # Simulate video processing
        return {
            "stream_type": "video",
            "events_processed": len(events),
            "frames_processed": len(events)
        }
    
    async def _handle_sensor_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle sensor stream events"""
        # Simulate sensor data processing
        return {
            "stream_type": "sensor",
            "events_processed": len(events),
            "sensor_readings": len(events)
        }
    
    async def _handle_log_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle log stream events"""
        # Simulate log processing
        return {
            "stream_type": "log",
            "events_processed": len(events),
            "log_entries": len(events)
        }
    
    async def _handle_metric_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle metric stream events"""
        # Simulate metric processing
        return {
            "stream_type": "metric",
            "events_processed": len(events),
            "metrics_processed": len(events)
        }
    
    async def _handle_event_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle event stream events"""
        # Simulate event processing
        return {
            "stream_type": "event",
            "events_processed": len(events),
            "events_analyzed": len(events)
        }
    
    async def _handle_data_stream(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Handle data stream events"""
        # Simulate data processing
        return {
            "stream_type": "data",
            "events_processed": len(events),
            "data_points": len(events)
        }
    
    def add_event(self, event: StreamEvent) -> bool:
        """Add event to stream processor"""
        return self.buffer.add_event(event)
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.__dict__,
            "processing_stats": self.processing_stats,
            "buffer_stats": self.buffer.get_buffer_stats(),
            "batcher_stats": self.batcher.get_batcher_stats(),
            "is_processing": self.is_processing
        }


class RealTimeManager(BaseAdvancedSystem):
    """Real-time manager for TruthGPT"""
    
    def __init__(self, config: RealTimeConfig):
        super().__init__(config, "RealTimeManager")
        self.logger = logging.getLogger(f"RealTimeManager_{id(self)}")
        
        # Stream processors
        self.processors: Dict[str, StreamProcessor] = {}
        
        # Event routing
        self.event_routers: Dict[StreamType, List[str]] = defaultdict(list)
        
        # Performance monitoring
        self.monitoring_enabled = config.enable_monitoring
        self.performance_monitor = PerformanceMonitor() if self.monitoring_enabled else None
        
        # Integration components
        self.distributed_coordinator: Optional[DistributedCoordinator] = None
        self.ai_enhancement: Optional[TruthGPTAIEnhancementManager] = None
    
    def set_distributed_coordinator(self, coordinator: DistributedCoordinator):
        """Set distributed coordinator for integration"""
        self.distributed_coordinator = coordinator
    
    def set_ai_enhancement(self, ai_enhancement: TruthGPTAIEnhancementManager):
        """Set AI enhancement manager"""
        self.ai_enhancement = ai_enhancement
    
    def create_processor(self, processor_id: str, 
                        model: TruthGPTModel = None,
                        model_config: TruthGPTModelConfig = None) -> StreamProcessor:
        """Create stream processor"""
        processor = StreamProcessor(self.config)
        
        if model and model_config:
            processor.set_model(model, model_config)
        
        self.processors[processor_id] = processor
        
        # Register for event routing
        for stream_type in StreamType:
            self.event_routers[stream_type].append(processor_id)
        
        self.logger.info(f"Created processor {processor_id}")
        return processor
    
    def remove_processor(self, processor_id: str):
        """Remove stream processor"""
        if processor_id in self.processors:
            del self.processors[processor_id]
            
            # Remove from event routing
            for stream_type in self.event_routers:
                if processor_id in self.event_routers[stream_type]:
                    self.event_routers[stream_type].remove(processor_id)
            
            self.logger.info(f"Removed processor {processor_id}")
    
    async def start_all_processors(self):
        """Start all stream processors"""
        self.start_monitoring()
        tasks = []
        for processor_id, processor in self.processors.items():
            task = asyncio.create_task(processor.start_processing())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def stop_all_processors(self):
        """Stop all stream processors"""
        self.stop_monitoring()
        tasks = []
        for processor_id, processor in self.processors.items():
            task = asyncio.create_task(processor.stop_processing())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    def route_event(self, event: StreamEvent) -> bool:
        """Route event to appropriate processors"""
        processor_ids = self.event_routers.get(event.stream_type, [])
        
        if not processor_ids:
            self.logger.warning(f"No processors for stream type {event.stream_type.value}")
            return False
        
        # Route to all processors for this stream type
        success_count = 0
        for processor_id in processor_ids:
            if processor_id in self.processors:
                processor = self.processors[processor_id]
                if processor.add_event(event):
                    success_count += 1
        
        return success_count > 0
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        processor_stats = {}
        for processor_id, processor in self.processors.items():
            processor_stats[processor_id] = processor.get_processor_stats()
        
        base_stats = self.get_stats()
        
        return {
            **base_stats,
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.__dict__,
            "total_processors": len(self.processors),
            "processor_stats": processor_stats,
            "event_routing": dict(self.event_routers),
            "monitoring_enabled": self.monitoring_enabled
        }
        
    def update_metrics(self):
        """Update metrics for BaseAdvancedSystem."""
        total_events_processed = sum(
            p.processing_stats.get('events_processed', 0) for p in self.processors.values()
        )
        self.metrics.throughput = total_events_processed
        self.metrics.efficiency = 1.0 if len(self.processors) > 0 else 0.0
        self.metrics.stability = 1.0


class PerformanceMonitor:
    """Performance monitor for real-time computing"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"PerformanceMonitor_{id(self)}")
        
        # Monitoring data
        self.metrics_history: deque = deque(maxlen=1000)
        self.alerts: List[Dict[str, Any]] = []
        
        # Thresholds
        self.latency_threshold = 100.0  # ms
        self.throughput_threshold = 1000.0  # events/second
        self.error_rate_threshold = 0.05  # 5%
    
    def record_metrics(self, metrics: Dict[str, Any]):
        """Record performance metrics"""
        self.metrics_history.append({
            "timestamp": time.time(),
            "metrics": metrics
        })
        
        # Check for alerts
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts"""
        # Latency alert
        if metrics.get("average_latency", 0) > self.latency_threshold:
            self._create_alert("HIGH_LATENCY", metrics)
        
        # Throughput alert
        if metrics.get("throughput_events_per_second", 0) < self.throughput_threshold:
            self._create_alert("LOW_THROUGHPUT", metrics)
        
        # Error rate alert
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.error_rate_threshold:
            self._create_alert("HIGH_ERROR_RATE", metrics)
    
    def _create_alert(self, alert_type: str, metrics: Dict[str, Any]):
        """Create performance alert"""
        alert = {
            "alert_type": alert_type,
            "timestamp": time.time(),
            "metrics": metrics,
            "severity": "HIGH" if alert_type in ["HIGH_LATENCY", "HIGH_ERROR_RATE"] else "MEDIUM"
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"Performance alert: {alert_type}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "total_metrics_recorded": len(self.metrics_history),
            "total_alerts": len(self.alerts),
            "recent_alerts": self.alerts[-10:] if self.alerts else [],
            "thresholds": {
                "latency_threshold": self.latency_threshold,
                "throughput_threshold": self.throughput_threshold,
                "error_rate_threshold": self.error_rate_threshold
            }
        }


def create_real_time_manager(config: RealTimeConfig) -> RealTimeManager:
    """Create real-time manager"""
    return RealTimeManager(config)


def create_stream_processor(config: RealTimeConfig) -> StreamProcessor:
    """Create stream processor"""
    return StreamProcessor(config)


def create_real_time_buffer(config: RealTimeConfig) -> RealTimeBuffer:
    """Create real-time buffer"""
    return RealTimeBuffer(config)


def create_adaptive_batcher(config: RealTimeConfig) -> AdaptiveBatcher:
    """Create adaptive batcher"""
    return AdaptiveBatcher(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create real-time config
        config = RealTimeConfig(
            mode=RealTimeMode.STREAMING,
            latency_requirement=LatencyRequirement.LOW,
            max_latency_ms=100.0,
            batch_size=32,
            enable_adaptive_batching=True
        )
    
        # Create real-time manager
        manager = create_real_time_manager(config)
        
        # Create processor
        processor = manager.create_processor("processor_1")
        
        # Add events
        for i in range(100):
            event = StreamEvent(
                event_id=str(uuid.uuid4()),
                stream_type=StreamType.TEXT_STREAM,
                data=f"Event {i}",
                priority=ProcessingPriority.NORMAL
            )
            manager.route_event(event)
    
        # Start processing
        await manager.start_all_processors()
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Stop processing
        await manager.stop_all_processors()
        
        # Get stats
        stats = manager.get_manager_stats()
        print(f"Real-time manager stats: {stats}")
    
    # Run example
    asyncio.run(main())
