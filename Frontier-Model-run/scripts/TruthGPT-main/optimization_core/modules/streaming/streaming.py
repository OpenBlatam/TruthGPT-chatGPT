"""
Real-time Streaming Module for TruthGPT Optimization Core
WebSocket support, Server-Sent Events, and live inference streaming
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import logging
from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict
from abc import ABC, abstractmethod

from ..common.base_advanced_system import BaseAdvancedSystem
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import json
import pickle
import hashlib
from collections import defaultdict, deque
import math
import random
from pathlib import Path
import asyncio
from contextlib import contextmanager
import uuid
import base64
import websockets
import sseclient
import requests

logger = logging.getLogger(__name__)

class StreamType(Enum):
    """Types of streaming"""
    WEBSOCKET = "websocket"
    SERVER_SENT_EVENTS = "sse"
    HTTP_STREAMING = "http_streaming"
    GRPC_STREAMING = "grpc_streaming"
    KAFKA_STREAMING = "kafka_streaming"

class ConnectionState(Enum):
    """Connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class MessageType(Enum):
    """Message types"""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    CONTROL = "control"

class StreamConfig(BaseModel):
    """Configuration for streaming"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    websocket_port: int = 8081
    sse_port: int = 8082
    
    # Connection settings
    max_connections: int = 1000
    connection_timeout: int = 30
    heartbeat_interval: int = 30
    reconnect_interval: int = 5
    
    # Message settings
    max_message_size: int = 1024 * 1024  # 1MB
    message_queue_size: int = 1000
    compression_enabled: bool = True
    
    # Security
    enable_auth: bool = True
    enable_ssl: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    
    # Performance
    enable_batching: bool = True
    batch_size: int = 100
    batch_timeout: float = 0.1
    
    @model_validator(mode='after')
    def validate_config(self) -> 'StreamConfig':
        """Validate configuration"""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        return self

class StreamMessage(BaseModel):
    """Stream message"""
    message_id: str
    message_type: MessageType
    data: Any
    timestamp: float
    sender_id: str = ""
    recipient_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConnectionInfo(BaseModel):
    """Connection information"""
    connection_id: str
    stream_type: StreamType
    state: ConnectionState
    connected_at: float
    last_activity: float
    ip_address: str = ""
    user_agent: str = ""
    user_id: str = ""

class TruthGPTStreamManager(BaseAdvancedSystem):
    """Main stream manager for real-time communication"""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config, "TruthGPTStreamManager")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Stream state
        self.connections = {}
        self.message_queue = queue.Queue(maxsize=config.message_queue_size)
        self.subscribers = defaultdict(set)
        
        # Stream handlers
        self.websocket_handler = None
        self.sse_handler = None
        
        # Performance tracking
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        }
        
        logger.info("✅ TruthGPT Stream Manager initialized")
    
    def start(self):
        """Start stream manager"""
        self.start_monitoring()
        try:
            # Start WebSocket handler
            self._start_websocket_handler()
            
            # Start SSE handler
            self._start_sse_handler()
            
            # Start message processor
            self._start_message_processor()
            
            logger.info("✅ Stream Manager started")
            
        except Exception as e:
            logger.error(f"Failed to start stream manager: {e}")
    
    def stop(self):
        """Stop stream manager"""
        self.stop_monitoring()
        try:
            # Close all connections
            for connection_id in list(self.connections.keys()):
                self.disconnect(connection_id)
            
            logger.info("✅ Stream Manager stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop stream manager: {e}")
    
    def connect(self, connection_id: str, stream_type: StreamType, 
                ip_address: str = "", user_agent: str = "", user_id: str = "") -> bool:
        """Connect new client"""
        try:
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                stream_type=stream_type,
                state=ConnectionState.CONNECTED,
                connected_at=time.time(),
                last_activity=time.time(),
                ip_address=ip_address,
                user_agent=user_agent,
                user_id=user_id
            )
            
            self.connections[connection_id] = connection_info
            self.stats['total_connections'] += 1
            self.stats['active_connections'] += 1
            
            logger.info(f"✅ Client connected: {connection_id} ({stream_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect client: {e}")
            return False
    
    def disconnect(self, connection_id: str) -> bool:
        """Disconnect client"""
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                connection.state = ConnectionState.DISCONNECTED
                
                # Remove from subscribers
                for topic, subscribers in self.subscribers.items():
                    subscribers.discard(connection_id)
                
                del self.connections[connection_id]
                self.stats['active_connections'] -= 1
                
                logger.info(f"✅ Client disconnected: {connection_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to disconnect client: {e}")
            return False
    
    def send_message(self, connection_id: str, message: StreamMessage) -> bool:
        """Send message to specific connection"""
        try:
            if connection_id not in self.connections:
                logger.warning(f"Connection {connection_id} not found")
                return False
            
            connection = self.connections[connection_id]
            
            # Update activity
            connection.last_activity = time.time()
            
            # Send message based on stream type
            if connection.stream_type == StreamType.WEBSOCKET:
                return self._send_websocket_message(connection_id, message)
            elif connection.stream_type == StreamType.SERVER_SENT_EVENTS:
                return self._send_sse_message(connection_id, message)
            else:
                logger.warning(f"Unsupported stream type: {connection.stream_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def broadcast_message(self, message: StreamMessage, topic: str = None) -> int:
        """Broadcast message to all connections or topic subscribers"""
        try:
            sent_count = 0
            
            if topic:
                # Send to topic subscribers
                subscribers = self.subscribers.get(topic, set())
                for connection_id in subscribers:
                    if self.send_message(connection_id, message):
                        sent_count += 1
            else:
                # Send to all connections
                for connection_id in self.connections:
                    if self.send_message(connection_id, message):
                        sent_count += 1
            
            self.stats['messages_sent'] += sent_count
            logger.info(f"✅ Message broadcasted to {sent_count} connections")
            return sent_count
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return 0
    
    def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to topic"""
        try:
            if connection_id in self.connections:
                self.subscribers[topic].add(connection_id)
                logger.info(f"✅ Subscribed {connection_id} to {topic}")
                return True
            
            return False
                
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            return False
    
    def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe connection from topic"""
        try:
            if topic in self.subscribers:
                self.subscribers[topic].discard(connection_id)
                logger.info(f"✅ Unsubscribed {connection_id} from {topic}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
            return False
    
    def _start_websocket_handler(self):
        """Start WebSocket handler with a simulated event loop."""
        def handle_ws():
            while True:
                # Simulated WebSocket connection monitoring
                for conn_id, conn in list(self.connections.items()):
                    if conn.stream_type == StreamType.WEBSOCKET:
                        if time.time() - conn.last_activity > 60: # 1 min timeout
                            logger.info(f"WebSocket timeout for {conn_id}")
                            self.disconnect(connection_id=conn_id)
                time.sleep(10)
        
        ws_thread = threading.Thread(target=handle_ws, daemon=True)
        ws_thread.start()
        logger.info("✅ WebSocket handler started (Background Monitoring)")
    
    def _start_sse_handler(self):
        """Start Server-Sent Events handler with monitoring."""
        def handle_sse():
            while True:
                for conn_id, conn in list(self.connections.items()):
                    if conn.stream_type == StreamType.SERVER_SENT_EVENTS:
                        # Simulated keep-alive
                        msg = StreamMessage(
                            message_id=str(uuid.uuid4()),
                            message_type=MessageType.HEARTBEAT,
                            sender_id="system",
                            data="keep-alive"
                        )
                        self.send_message(conn_id, msg)
                time.sleep(30)
        
        sse_thread = threading.Thread(target=handle_sse, daemon=True)
        sse_thread.start()
        logger.info("✅ SSE handler started (Auto-Heartbeat)")
    
    def _start_message_processor(self):
        """Start message processor thread"""
        def process_messages():
            while True:
                try:
                    message = self.message_queue.get(timeout=1)
                    self._process_message(message)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Message processing failed: {e}")
        
        processor_thread = threading.Thread(target=process_messages, daemon=True)
        processor_thread.start()
        logger.info("✅ Message processor started")
    
    def _process_message(self, message: StreamMessage):
        """Process incoming message"""
        try:
            # Update stats
            self.stats['messages_received'] += 1
            
            # Process based on message type
            if message.message_type == MessageType.CONTROL:
                self._handle_control_message(message)
            elif message.message_type == MessageType.HEARTBEAT:
                self._handle_heartbeat_message(message)
            else:
                self._handle_data_message(message)
                
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
    
    def _handle_control_message(self, message: StreamMessage):
        """Handle control message"""
        control_type = message.metadata.get('control_type')
        
        if control_type == 'subscribe':
            topic = message.metadata.get('topic')
            if topic:
                self.subscribe(message.sender_id, topic)
        elif control_type == 'unsubscribe':
            topic = message.metadata.get('topic')
            if topic:
                self.unsubscribe(message.sender_id, topic)
    
    def _handle_heartbeat_message(self, message: StreamMessage):
        """Handle heartbeat message"""
        if message.sender_id in self.connections:
            self.connections[message.sender_id].last_activity = time.time()
    
    def _handle_data_message(self, message: StreamMessage):
        """Handle data message"""
        # Process data message
        logger.debug(f"Processing data message: {message.message_id}")
    
    def _send_websocket_message(self, connection_id: str, message: StreamMessage) -> bool:
        """Send WebSocket message"""
        # Simplified implementation
        logger.debug(f"Sending WebSocket message to {connection_id}")
        return True
    
    def _send_sse_message(self, connection_id: str, message: StreamMessage) -> bool:
        """Send SSE message"""
        # Simplified implementation
        logger.debug(f"Sending SSE message to {connection_id}")
        return True
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        base_stats = self.get_stats()
        return {
            **base_stats,
            **self.stats,
            'active_topics': len(self.subscribers),
            'topic_subscribers': {topic: len(subscribers) for topic, subscribers in self.subscribers.items()}
        }
        
    def update_metrics(self):
        """Update metrics for BaseAdvancedSystem."""
        self.metrics.throughput = self.stats['messages_sent'] + self.stats['messages_received']
        self.metrics.efficiency = 1.0 if self.stats['active_connections'] > 0 else 0.0
        self.metrics.stability = 1.0

class TruthGPTServerSentEvents:
    """Server-Sent Events handler"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # SSE state
        self.connections = {}
        self.event_queue = queue.Queue()
        
        logger.info("✅ Server-Sent Events handler initialized")
    
    def add_connection(self, connection_id: str, response):
        """Add SSE connection"""
        try:
            self.connections[connection_id] = {
                'response': response,
                'connected_at': time.time(),
                'last_event': time.time()
            }
            
            logger.info(f"✅ SSE connection added: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add SSE connection: {e}")
            return False
    
    def remove_connection(self, connection_id: str):
        """Remove SSE connection"""
        try:
            if connection_id in self.connections:
                del self.connections[connection_id]
                logger.info(f"✅ SSE connection removed: {connection_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove SSE connection: {e}")
            return False
    
    def send_event(self, connection_id: str, event_type: str, data: Any) -> bool:
        """Send SSE event"""
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            response = connection['response']
            
            # Format SSE event
            event_data = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            
            # Send event
            response.write(event_data.encode())
            response.flush()
            
            connection['last_event'] = time.time()
            
            logger.debug(f"✅ SSE event sent to {connection_id}: {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SSE event: {e}")
            return False
    
    def broadcast_event(self, event_type: str, data: Any) -> int:
        """Broadcast SSE event to all connections"""
        sent_count = 0
        
        for connection_id in list(self.connections.keys()):
            if self.send_event(connection_id, event_type, data):
                sent_count += 1
        
        logger.info(f"✅ SSE event broadcasted to {sent_count} connections")
        return sent_count

class TruthGPTRealTimeManager:
    """Real-time manager for live inference and streaming"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.stream_manager = TruthGPTStreamManager(config)
        self.sse_handler = TruthGPTServerSentEvents(config)
        
        # Real-time state
        self.live_inference_queue = queue.Queue()
        self.inference_results = {}
        
        # Performance tracking
        self.inference_stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0,
            'throughput': 0.0
        }
        
        logger.info("✅ TruthGPT Real-Time Manager initialized")
    
    def start(self):
        """Start real-time manager"""
        try:
            # Start stream manager
            self.stream_manager.start()
            
            # Start live inference processor
            self._start_live_inference_processor()
            
            logger.info("✅ Real-Time Manager started")
            
        except Exception as e:
            logger.error(f"Failed to start real-time manager: {e}")
    
    def stop(self):
        """Stop real-time manager"""
        try:
            # Stop stream manager
            self.stream_manager.stop()
            
            logger.info("✅ Real-Time Manager stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop real-time manager: {e}")
    
    def process_live_inference(self, model: nn.Module, input_data: torch.Tensor, 
                              connection_id: str, request_id: str = None) -> str:
        """Process live inference request"""
        try:
            if request_id is None:
                request_id = str(uuid.uuid4())
            
            # Create inference request
            request = {
                'request_id': request_id,
                'model': model,
                'input_data': input_data,
                'connection_id': connection_id,
                'timestamp': time.time(),
                'status': 'queued'
            }
            
            # Add to queue
            self.live_inference_queue.put(request)
            
            # Update stats
            self.inference_stats['total_requests'] += 1
            
            logger.info(f"✅ Live inference request queued: {request_id}")
            return request_id
                
        except Exception as e:
            logger.error(f"Failed to queue live inference: {e}")
            return ""
    
    def _start_live_inference_processor(self):
        """Start live inference processor thread"""
        def process_inference():
            while True:
                try:
                    request = self.live_inference_queue.get(timeout=1)
                    self._process_inference_request(request)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Inference processing failed: {e}")
        
        processor_thread = threading.Thread(target=process_inference, daemon=True)
        processor_thread.start()
        logger.info("✅ Live inference processor started")
    
    def _process_inference_request(self, request: Dict[str, Any]):
        """Process inference request"""
        try:
            request_id = request['request_id']
            model = request['model']
            input_data = request['input_data']
            connection_id = request['connection_id']
            
            # Update request status
            request['status'] = 'processing'
            
            # Perform inference
            start_time = time.time()
            
            with torch.no_grad():
                output = model(input_data)
            
            inference_time = time.time() - start_time
            
            # Create result
            result = {
                'request_id': request_id,
                'output': output.tolist() if isinstance(output, torch.Tensor) else output,
                'inference_time': inference_time,
                'timestamp': time.time(),
                'status': 'completed'
            }
            
            # Store result
            self.inference_results[request_id] = result
            
            # Send result to client
            message = StreamMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.JSON,
                data=result,
                timestamp=time.time(),
                sender_id="server",
                recipient_id=connection_id
            )
            
            self.stream_manager.send_message(connection_id, message)
            
            # Update stats
            self.inference_stats['completed_requests'] += 1
            self.inference_stats['average_latency'] = (
                (self.inference_stats['average_latency'] * (self.inference_stats['completed_requests'] - 1) + inference_time) /
                self.inference_stats['completed_requests']
            )
            
            logger.info(f"✅ Inference completed: {request_id} ({inference_time:.4f}s)")
            
        except Exception as e:
            logger.error(f"Inference request failed: {e}")
            
            # Update stats
            self.inference_stats['failed_requests'] += 1
            
            # Send error to client
            error_result = {
                'request_id': request['request_id'],
                'error': str(e),
                'timestamp': time.time(),
                'status': 'failed'
            }
            
            message = StreamMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR,
                data=error_result,
                timestamp=time.time(),
                sender_id="server",
                recipient_id=request['connection_id']
            )
            
            self.stream_manager.send_message(request['connection_id'], message)
    
    def get_inference_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get inference result by request ID"""
        return self.inference_results.get(request_id)
    
    def stream_model_training(self, model: nn.Module, training_data: DataLoader, 
                             connection_id: str) -> str:
        """Stream model training progress"""
        try:
            training_id = str(uuid.uuid4())
            
            # Start training in separate thread
            def train_model():
                try:
                    model.train()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.MSELoss()
                    
                    for epoch in range(10):
                        epoch_loss = 0.0
                        batch_count = 0
                        
                        for batch_idx, (data, target) in enumerate(training_data):
                            optimizer.zero_grad()
                            output = model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            optimizer.step()
                            
                            epoch_loss += loss.item()
                            batch_count += 1
                            
                            # Send progress update
                            progress = {
                                'training_id': training_id,
                                'epoch': epoch + 1,
                                'batch': batch_idx + 1,
                                'loss': loss.item(),
                                'timestamp': time.time()
                            }
                            
                            message = StreamMessage(
                                message_id=str(uuid.uuid4()),
                                message_type=MessageType.JSON,
                                data=progress,
                                timestamp=time.time(),
                                sender_id="server",
                                recipient_id=connection_id
                            )
                            
                            self.stream_manager.send_message(connection_id, message)
                        
                        # Send epoch summary
                        epoch_summary = {
                            'training_id': training_id,
                            'epoch': epoch + 1,
                            'average_loss': epoch_loss / batch_count,
                            'timestamp': time.time(),
                            'status': 'epoch_completed'
                        }
                        
                        message = StreamMessage(
                            message_id=str(uuid.uuid4()),
                            message_type=MessageType.JSON,
                            data=epoch_summary,
                            timestamp=time.time(),
                            sender_id="server",
                            recipient_id=connection_id
                        )
                        
                        self.stream_manager.send_message(connection_id, message)
                    
                    # Send training completion
                    completion = {
                        'training_id': training_id,
                        'status': 'completed',
                        'timestamp': time.time()
                    }
                    
                    message = StreamMessage(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.JSON,
                        data=completion,
                        timestamp=time.time(),
                        sender_id="server",
                        recipient_id=connection_id
                    )
                    
                    self.stream_manager.send_message(connection_id, message)
                    
                except Exception as e:
                    logger.error(f"Training failed: {e}")
                    
                    error = {
                        'training_id': training_id,
                        'error': str(e),
                        'status': 'failed',
                        'timestamp': time.time()
                    }
                    
                    message = StreamMessage(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.ERROR,
                        data=error,
                        timestamp=time.time(),
                        sender_id="server",
                        recipient_id=connection_id
                    )
                    
                    self.stream_manager.send_message(connection_id, message)
            
            # Start training thread
            training_thread = threading.Thread(target=train_model, daemon=True)
            training_thread.start()
            
            logger.info(f"✅ Model training started: {training_id}")
            return training_id
            
        except Exception as e:
            logger.error(f"Failed to start model training: {e}")
            return ""
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics"""
        return {
            'stream_stats': self.stream_manager.get_stream_stats(),
            'inference_stats': self.inference_stats,
            'active_inference_requests': self.live_inference_queue.qsize(),
            'completed_inference_results': len(self.inference_results)
        }

# Factory functions
def create_stream_config(**kwargs) -> StreamConfig:
    """Create stream configuration"""
    return StreamConfig(**kwargs)

def create_stream_message(message_type: MessageType, data: Any, sender_id: str = "", recipient_id: str = "") -> StreamMessage:
    """Create stream message"""
    return StreamMessage(
        message_id=str(uuid.uuid4()),
        message_type=message_type,
        data=data,
        timestamp=time.time(),
        sender_id=sender_id,
        recipient_id=recipient_id
    )

def create_stream_manager(config: StreamConfig) -> TruthGPTStreamManager:
    """Create stream manager"""
    return TruthGPTStreamManager(config)

def create_sse_handler(config: StreamConfig) -> TruthGPTServerSentEvents:
    """Create Server-Sent Events handler"""
    return TruthGPTServerSentEvents(config)

def create_real_time_manager(config: StreamConfig) -> TruthGPTRealTimeManager:
    """Create real-time manager"""
    return TruthGPTRealTimeManager(config)

def quick_streaming_setup() -> TruthGPTRealTimeManager:
    """Quick streaming setup for testing"""
    config = create_stream_config(
        host="localhost",
        port=8080,
        websocket_port=8081,
        sse_port=8082,
        max_connections=100
    )
    
    return create_real_time_manager(config)

# Example usage
def example_streaming():
    """Example of streaming features"""
    # Create real-time manager
    manager = quick_streaming_setup()
    
    # Start manager
    manager.start()
    
    print("✅ Real-time streaming started!")
    print(f"📡 WebSocket: ws://{manager.config.host}:{manager.config.websocket_port}")
    print(f"📡 SSE: http://{manager.config.host}:{manager.config.sse_port}")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # Simulate client connection
    connection_id = str(uuid.uuid4())
    manager.stream_manager.connect(connection_id, StreamType.WEBSOCKET)
    
    # Subscribe to inference topic
    manager.stream_manager.subscribe(connection_id, "inference")
    
    # Process live inference
    input_data = torch.randn(1, 10)
    request_id = manager.process_live_inference(model, input_data, connection_id)
    
    print(f"✅ Live inference request: {request_id}")
    
    # Wait for result
    time.sleep(1)
    
    result = manager.get_inference_result(request_id)
    if result:
        print(f"📊 Inference result: {result['status']}")
        print(f"⏱️ Inference time: {result['inference_time']:.4f}s")
    
    # Get real-time stats
    stats = manager.get_real_time_stats()
    print(f"📈 Real-time stats: {stats}")
    
    # Stop manager
    manager.stop()
    
    print("✅ Streaming example completed!")
    
    return manager

# Export utilities
__all__ = [
    'StreamType',
    'ConnectionState',
    'MessageType',
    'StreamConfig',
    'StreamMessage',
    'ConnectionInfo',
    'TruthGPTStreamManager',
    'TruthGPTServerSentEvents',
    'TruthGPTRealTimeManager',
    'create_stream_config',
    'create_stream_message',
    'create_stream_manager',
    'create_sse_handler',
    'create_real_time_manager',
    'quick_streaming_setup',
    'example_streaming'
]

if __name__ == "__main__":
    import threading
    example_streaming()
    print("✅ Streaming module complete!")
