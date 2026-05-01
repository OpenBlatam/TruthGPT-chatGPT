"""
TruthGPT Distributed Computing Features
Advanced distributed computing, load balancing, and coordination for TruthGPT
"""

import asyncio
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
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
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
import socket
import psutil
import subprocess
import queue
import zmq
import redis
import consul
import etcd3

# Import TruthGPT modules
from .models import TruthGPTModel, TruthGPTModelConfig
from .orchestration import AIOrchestrator, AIAgent, AgentType
from .federation import DecentralizedAINetwork, FederatedNode


class DistributionStrategy(Enum):
    """Distribution strategies for computing"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    FEDERATED_PARALLEL = "federated_parallel"
    EDGE_COMPUTING = "edge_computing"
    CLOUD_COMPUTING = "cloud_computing"


class CommunicationBackend(Enum):
    """Communication backends for distributed computing"""
    TCP = "tcp"
    UDP = "udp"
    ZMQ = "zmq"
    GRPC = "grpc"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    NATS = "nats"
    CONSUL = "consul"
    ETCD = "etcd"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE_LOAD_BALANCING = "adaptive"
    MACHINE_LEARNING_BASED = "ml_based"


class WorkerStatus(Enum):
    """Worker status"""
    IDLE = "idle"
    BUSY = "busy"
    TRAINING = "training"
    INFERENCE = "inference"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class DistributedConfig:
    """Configuration for distributed computing"""
    distribution_strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL
    communication_backend: CommunicationBackend = CommunicationBackend.ZMQ
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    num_workers: int = 4
    master_port: int = 5555
    worker_ports: List[int] = field(default_factory=lambda: [5556, 5557, 5558, 5559])
    enable_auto_scaling: bool = True
    min_workers: int = 2
    max_workers: int = 16
    scaling_threshold: float = 0.8
    heartbeat_interval: float = 5.0
    timeout: float = 30.0
    enable_fault_tolerance: bool = True
    checkpoint_interval: float = 60.0
    enable_load_balancing: bool = True
    enable_monitoring: bool = True
    enable_optimization: bool = True


@dataclass
class WorkerInfo:
    """Worker information"""
    worker_id: str
    host: str
    port: int
    status: WorkerStatus = WorkerStatus.IDLE
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0


@dataclass
class TaskAssignment:
    """Task assignment for distributed computing"""
    task_id: str
    worker_id: str
    task_type: str
    priority: int = 1
    estimated_duration: float = 60.0
    data_size: int = 0
    assigned_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "assigned"


class DistributedWorker:
    """Distributed worker for TruthGPT"""
    
    def __init__(self, worker_id: str, config: DistributedConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = logging.getLogger(f"DistributedWorker_{worker_id}")
        
        # Worker state
        self.info = WorkerInfo(
            worker_id=worker_id,
            host=socket.gethostname(),
            port=config.worker_ports[0] if config.worker_ports else 5556
        )
        
        # Communication
        self.context = None
        self.socket = None
        self._init_communication()
        
        # Task management
        self.current_task: Optional[TaskAssignment] = None
        self.task_queue: queue.Queue = queue.Queue()
        self.completed_tasks: List[TaskAssignment] = []
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.resource_monitor = ResourceMonitor()
        
        # Model management
        self.local_model: Optional[TruthGPTModel] = None
        self.model_shards: Dict[str, torch.Tensor] = {}
    
    def _init_communication(self):
        """Initialize communication backend"""
        if self.config.communication_backend == CommunicationBackend.ZMQ:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.info.port}")
        elif self.config.communication_backend == CommunicationBackend.REDIS:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        elif self.config.communication_backend == CommunicationBackend.TCP:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind(('localhost', self.info.port))
            self.socket.listen(1)
    
    async def start_worker(self):
        """Start distributed worker"""
        self.logger.info(f"Starting worker {self.worker_id}")
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start task processing
        task_task = asyncio.create_task(self._task_processing_loop())
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(self._resource_monitoring_loop())
        
        # Wait for tasks
        await asyncio.gather(heartbeat_task, task_task, monitor_task)
    
    async def _heartbeat_loop(self):
        """Heartbeat loop for worker"""
        while True:
            try:
                # Update heartbeat
                self.info.last_heartbeat = time.time()
                
                # Send heartbeat to master
                await self._send_heartbeat()
                
            await asyncio.sleep(self.config.heartbeat_interval)
            
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _send_heartbeat(self):
        """Send heartbeat to master"""
        heartbeat_data = {
            "worker_id": self.worker_id,
            "status": self.info.status.value,
            "timestamp": time.time(),
            "resource_usage": self.resource_monitor.get_current_usage(),
            "performance_metrics": self.info.performance_metrics
        }
        
        if self.config.communication_backend == CommunicationBackend.ZMQ:
            # Send heartbeat via ZMQ
            master_socket = self.context.socket(zmq.REQ)
            master_socket.connect(f"tcp://localhost:{self.config.master_port}")
            master_socket.send_json(heartbeat_data)
            master_socket.close()
    
    async def _task_processing_loop(self):
        """Task processing loop"""
        while True:
            try:
                # Check for new tasks
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    await self._process_task(task)
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_task(self, task: TaskAssignment):
        """Process assigned task"""
        self.logger.info(f"Processing task {task.task_id}")
        
        self.info.status = WorkerStatus.BUSY
        task.started_at = time.time()
        
        try:
            # Process task based on type
            if task.task_type == "training":
                result = await self._process_training_task(task)
            elif task.task_type == "inference":
                result = await self._process_inference_task(task)
            elif task.task_type == "optimization":
                result = await self._process_optimization_task(task)
            else:
                result = await self._process_generic_task(task)
            
            # Update task status
            task.completed_at = time.time()
            task.status = "completed"
            self.completed_tasks.append(task)
            
            # Update performance metrics
            execution_time = task.completed_at - task.started_at
            self._update_performance_metrics(task, execution_time, True)
            
            # Send result to master
            await self._send_task_result(task, result)
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            task.status = "failed"
            self.info.tasks_failed += 1
            self._update_performance_metrics(task, 0, False)
        
        finally:
            self.info.status = WorkerStatus.IDLE
            self.current_task = None
    
    async def _process_training_task(self, task: TaskAssignment) -> Dict[str, Any]:
        """Process training task"""
        # Simulate training
        epochs = task.data.get("epochs", 10)
        training_losses = []
        
        for epoch in range(epochs):
            # Simulate training step
            loss = random.uniform(0.1, 1.0) * (1 - epoch / epochs)
            training_losses.append(loss)
        
        return {
            "task_type": "training",
            "epochs": epochs,
            "final_loss": training_losses[-1],
            "training_history": training_losses
        }
    
    async def _process_inference_task(self, task: TaskAssignment) -> Dict[str, Any]:
        """Process inference task"""
        input_data = task.data.get("input_data", "")
        
        # Simulate inference
        output = f"Inferred result for: {input_data}"
        confidence = random.uniform(0.7, 0.95)
        
        return {
            "task_type": "inference",
            "input": input_data,
            "output": output,
            "confidence": confidence
        }
    
    async def _process_optimization_task(self, task: TaskAssignment) -> Dict[str, Any]:
        """Process optimization task"""
        # Simulate optimization
        iterations = task.data.get("iterations", 100)
        best_value = random.uniform(0.5, 1.0)
        
        return {
            "task_type": "optimization",
            "iterations": iterations,
            "best_value": best_value,
            "improvement": random.uniform(0.1, 0.3)
        }
    
    async def _process_generic_task(self, task: TaskAssignment) -> Dict[str, Any]:
        """Process generic task"""
        return {
            "task_type": task.task_type,
            "status": "completed",
            "result": f"Generic task {task.task_id} completed"
        }
    
    def _update_performance_metrics(self, task: TaskAssignment, execution_time: float, success: bool):
        """Update performance metrics"""
        self.info.tasks_completed += 1
        
        # Update average response time
        if self.info.average_response_time == 0:
            self.info.average_response_time = execution_time
        else:
            self.info.average_response_time = (self.info.average_response_time + execution_time) / 2
        
        # Update performance metrics
        self.info.performance_metrics.update({
            "tasks_completed": self.info.tasks_completed,
            "tasks_failed": self.info.tasks_failed,
            "average_response_time": self.info.average_response_time,
            "success_rate": self.info.tasks_completed / (self.info.tasks_completed + self.info.tasks_failed)
        })
    
    async def _send_task_result(self, task: TaskAssignment, result: Dict[str, Any]):
        """Send task result to master"""
        result_data = {
            "task_id": task.task_id,
            "worker_id": self.worker_id,
            "result": result,
            "execution_time": task.completed_at - task.started_at,
            "timestamp": time.time()
        }
        
        if self.config.communication_backend == CommunicationBackend.ZMQ:
            master_socket = self.context.socket(zmq.REQ)
            master_socket.connect(f"tcp://localhost:{self.config.master_port}")
            master_socket.send_json(result_data)
            master_socket.close()
    
    async def _resource_monitoring_loop(self):
        """Resource monitoring loop"""
        while True:
            try:
                # Update resource usage
                self.info.resource_usage = self.resource_monitor.get_current_usage()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    def assign_task(self, task: TaskAssignment):
        """Assign task to worker"""
        self.task_queue.put(task)
        self.current_task = task
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "status": self.info.status.value,
            "tasks_completed": self.info.tasks_completed,
            "tasks_failed": self.info.tasks_failed,
            "average_response_time": self.info.average_response_time,
            "resource_usage": self.info.resource_usage,
            "performance_metrics": self.info.performance_metrics
        }


class ResourceMonitor:
    """Resource monitor for workers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"ResourceMonitor_{id(self)}")
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU usage (if available)
            gpu_usage = self._get_gpu_usage()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "gpu_percent": gpu_usage,
                "disk_percent": disk_percent,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Resource monitoring error: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "gpu_percent": 0.0,
                "disk_percent": 0.0,
                "timestamp": time.time()
            }
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage"""
        try:
            # Try to get GPU usage using nvidia-ml-py or similar
            # This is a simplified implementation
            return 0.0
        except Exception:
            return 0.0


class LoadBalancer:
    """Load balancer for distributed computing"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(f"LoadBalancer_{id(self)}")
        
        # Worker management
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_weights: Dict[str, float] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.consistent_hash_ring = {}
        
        # Performance tracking
        self.load_balancing_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
    
    def add_worker(self, worker_info: WorkerInfo):
        """Add worker to load balancer"""
        self.workers[worker_info.worker_id] = worker_info
        self.worker_weights[worker_info.worker_id] = 1.0
        
        # Update consistent hash ring
        if self.config.load_balancing_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            self._update_consistent_hash_ring()
        
        self.logger.info(f"Added worker {worker_info.worker_id} to load balancer")
    
    def remove_worker(self, worker_id: str):
        """Remove worker from load balancer"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.worker_weights[worker_id]
            
            # Update consistent hash ring
            if self.config.load_balancing_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self._update_consistent_hash_ring()
            
            self.logger.info(f"Removed worker {worker_id} from load balancer")
    
    def select_worker(self, task_data: Dict[str, Any] = None) -> Optional[str]:
        """Select worker based on load balancing strategy"""
        if not self.workers:
            return None
        
        # Filter available workers
        available_workers = {
            worker_id: worker for worker_id, worker in self.workers.items()
            if worker.status == WorkerStatus.IDLE
        }
        
        if not available_workers:
            return None
        
        # Select worker based on strategy
        if self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_workers)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_workers)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_workers)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_workers)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_selection(available_workers, task_data)
        else:
            return self._round_robin_selection(available_workers)
    
    def _round_robin_selection(self, available_workers: Dict[str, WorkerInfo]) -> str:
        """Round robin worker selection"""
        worker_ids = list(available_workers.keys())
        selected_worker = worker_ids[self.round_robin_index % len(worker_ids)]
        self.round_robin_index += 1
        return selected_worker
    
    def _least_connections_selection(self, available_workers: Dict[str, WorkerInfo]) -> str:
        """Least connections worker selection"""
        return min(available_workers.keys(), 
                  key=lambda w: available_workers[w].tasks_completed)
    
    def _least_response_time_selection(self, available_workers: Dict[str, WorkerInfo]) -> str:
        """Least response time worker selection"""
        return min(available_workers.keys(),
                  key=lambda w: available_workers[w].average_response_time)
    
    def _weighted_round_robin_selection(self, available_workers: Dict[str, WorkerInfo]) -> str:
        """Weighted round robin worker selection"""
        # Calculate weighted selection
        total_weight = sum(self.worker_weights[w] for w in available_workers.keys())
        if total_weight == 0:
            return self._round_robin_selection(available_workers)
        
        # Weighted random selection
        rand = random.uniform(0, total_weight)
        current_weight = 0
        
        for worker_id in available_workers.keys():
            current_weight += self.worker_weights[worker_id]
            if rand <= current_weight:
                return worker_id
        
        return list(available_workers.keys())[0]
    
    def _consistent_hash_selection(self, available_workers: Dict[str, WorkerInfo], 
                                task_data: Dict[str, Any] = None) -> str:
        """Consistent hash worker selection"""
        if not self.consistent_hash_ring:
            return self._round_robin_selection(available_workers)
        
        # Use task data for consistent hashing
        hash_key = str(task_data) if task_data else str(time.time())
        hash_value = hash(hash_key)
        
        # Find the closest worker in the ring
        sorted_keys = sorted(self.consistent_hash_ring.keys())
        for key in sorted_keys:
            if hash_value <= key:
                return self.consistent_hash_ring[key]
        
        # Wrap around to the first worker
        return self.consistent_hash_ring[sorted_keys[0]]
    
    def _update_consistent_hash_ring(self):
        """Update consistent hash ring"""
        self.consistent_hash_ring = {}
        
        for worker_id in self.workers.keys():
            # Create multiple virtual nodes for better distribution
            for i in range(100):
                virtual_key = hash(f"{worker_id}_{i}")
                self.consistent_hash_ring[virtual_key] = worker_id
    
    def update_worker_performance(self, worker_id: str, performance_data: Dict[str, Any]):
        """Update worker performance data"""
        if worker_id in self.workers:
            self.workers[worker_id].performance_metrics.update(performance_data)
            
            # Update worker weight based on performance
            success_rate = performance_data.get("success_rate", 1.0)
            response_time = performance_data.get("average_response_time", 1.0)
            
            # Weight inversely proportional to response time, proportional to success rate
            self.worker_weights[worker_id] = success_rate / max(response_time, 0.1)
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        return {
            "strategy": self.config.load_balancing_strategy.value,
            "total_workers": len(self.workers),
            "available_workers": len([w for w in self.workers.values() if w.status == WorkerStatus.IDLE]),
            "worker_weights": self.worker_weights,
            "load_balancing_metrics": self.load_balancing_metrics
        }


class DistributedCoordinator:
    """Distributed coordinator for TruthGPT"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(f"DistributedCoordinator_{id(self)}")
        
        # Worker management
        self.workers: Dict[str, DistributedWorker] = {}
        self.worker_info: Dict[str, WorkerInfo] = {}
        self.load_balancer = LoadBalancer(config)
        
        # Task management
        self.task_queue: queue.Queue = queue.Queue()
        self.completed_tasks: List[TaskAssignment] = []
        self.failed_tasks: List[TaskAssignment] = []
        
        # Communication
        self.context = None
        self.socket = None
        self._init_communication()
        
        # Performance tracking
        self.coordinator_metrics: Dict[str, Any] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "worker_utilization": {}
        }
    
    def _init_communication(self):
        """Initialize communication backend"""
        if self.config.communication_backend == CommunicationBackend.ZMQ:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.config.master_port}")
        elif self.config.communication_backend == CommunicationBackend.REDIS:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    async def start_coordinator(self):
        """Start distributed coordinator"""
        self.logger.info("Starting distributed coordinator")
        
        # Start worker management
        worker_task = asyncio.create_task(self._worker_management_loop())
        
        # Start task distribution
        distribution_task = asyncio.create_task(self._task_distribution_loop())
        
        # Start communication handling
        communication_task = asyncio.create_task(self._communication_loop())
        
        # Start monitoring
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Wait for tasks
        await asyncio.gather(worker_task, distribution_task, communication_task, monitoring_task)
    
    async def _worker_management_loop(self):
        """Worker management loop"""
        while True:
            try:
                # Check worker health
                await self._check_worker_health()
                
                # Auto-scaling
                if self.config.enable_auto_scaling:
                    await self._auto_scaling()
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Worker management error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _check_worker_health(self):
        """Check worker health"""
        current_time = time.time()
        unhealthy_workers = []
        
        for worker_id, worker_info in self.worker_info.items():
            if current_time - worker_info.last_heartbeat > self.config.timeout:
                unhealthy_workers.append(worker_id)
        
        # Remove unhealthy workers
        for worker_id in unhealthy_workers:
            self.logger.warning(f"Worker {worker_id} is unhealthy, removing")
            await self._remove_worker(worker_id)
    
    async def _auto_scaling(self):
        """Auto-scaling based on load"""
        current_load = self._calculate_current_load()
        
        if current_load > self.config.scaling_threshold and len(self.workers) < self.config.max_workers:
            # Scale up
            await self._scale_up()
        elif current_load < 0.3 and len(self.workers) > self.config.min_workers:
            # Scale down
            await self._scale_down()
    
    def _calculate_current_load(self) -> float:
        """Calculate current system load"""
        if not self.workers:
            return 0.0
        
        total_cpu = sum(w.resource_usage.get("cpu_percent", 0) for w in self.worker_info.values())
        total_memory = sum(w.resource_usage.get("memory_percent", 0) for w in self.worker_info.values())
        
        avg_cpu = total_cpu / len(self.worker_info)
        avg_memory = total_memory / len(self.worker_info)
        
        return (avg_cpu + avg_memory) / 200.0  # Normalize to 0-1
    
    async def _scale_up(self):
        """Scale up by adding workers"""
        if len(self.workers) >= self.config.max_workers:
            return
        
        # Create new worker
        new_worker_id = f"worker_{len(self.workers)}"
        new_worker = DistributedWorker(new_worker_id, self.config)
        
        # Add to coordinator
        self.workers[new_worker_id] = new_worker
        self.worker_info[new_worker_id] = new_worker.info
        self.load_balancer.add_worker(new_worker.info)
        
        self.logger.info(f"Scaled up: added worker {new_worker_id}")
    
    async def _scale_down(self):
        """Scale down by removing workers"""
        if len(self.workers) <= self.config.min_workers:
            return
        
        # Find least utilized worker
        least_utilized = min(self.workers.keys(),
                           key=lambda w: self.worker_info[w].tasks_completed)
        
        await self._remove_worker(least_utilized)
        self.logger.info(f"Scaled down: removed worker {least_utilized}")
    
    async def _remove_worker(self, worker_id: str):
        """Remove worker from coordinator"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.worker_info[worker_id]
            self.load_balancer.remove_worker(worker_id)
    
    async def _task_distribution_loop(self):
        """Task distribution loop"""
        while True:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    await self._distribute_task(task)
                else:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Task distribution error: {e}")
                await asyncio.sleep(1.0)
    
    async def _distribute_task(self, task: TaskAssignment):
        """Distribute task to worker"""
        # Select worker using load balancer
        selected_worker_id = self.load_balancer.select_worker(task.data)
        
        if selected_worker_id and selected_worker_id in self.workers:
            # Assign task to worker
            task.worker_id = selected_worker_id
            self.workers[selected_worker_id].assign_task(task)
            
            self.logger.info(f"Distributed task {task.task_id} to worker {selected_worker_id}")
        else:
            # No available workers, queue task
            self.task_queue.put(task)
            self.logger.warning(f"No available workers for task {task.task_id}")
    
    async def _communication_loop(self):
        """Communication handling loop"""
        while True:
            try:
                if self.config.communication_backend == CommunicationBackend.ZMQ:
                    await self._handle_zmq_communication()
                elif self.config.communication_backend == CommunicationBackend.REDIS:
                    await self._handle_redis_communication()
                
                await asyncio.sleep(0.01)
            
        except Exception as e:
                self.logger.error(f"Communication error: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_zmq_communication(self):
        """Handle ZMQ communication"""
        try:
            # Check for messages
            if self.socket.poll(timeout=0):
                message = self.socket.recv_json(zmq.NOBLOCK)
                await self._process_message(message)
        except zmq.Again:
            pass
    
    async def _handle_redis_communication(self):
        """Handle Redis communication"""
        try:
            # Check for messages in Redis queue
            message = self.redis_client.lpop("coordinator_queue")
            if message:
                message_data = json.loads(message)
                await self._process_message(message_data)
        except Exception as e:
            self.logger.error(f"Redis communication error: {e}")
    
    async def _process_message(self, message: Dict[str, Any]):
        """Process incoming message"""
        message_type = message.get("type")
        
        if message_type == "heartbeat":
            await self._handle_heartbeat(message)
        elif message_type == "task_result":
            await self._handle_task_result(message)
        elif message_type == "worker_registration":
            await self._handle_worker_registration(message)
    
    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle worker heartbeat"""
        worker_id = message.get("worker_id")
        if worker_id in self.worker_info:
            self.worker_info[worker_id].last_heartbeat = time.time()
            self.worker_info[worker_id].resource_usage = message.get("resource_usage", {})
            self.worker_info[worker_id].performance_metrics = message.get("performance_metrics", {})
    
    async def _handle_task_result(self, message: Dict[str, Any]):
        """Handle task result"""
        task_id = message.get("task_id")
        worker_id = message.get("worker_id")
        result = message.get("result")
        execution_time = message.get("execution_time", 0)
        
        # Update metrics
        self.coordinator_metrics["completed_tasks"] += 1
        self.coordinator_metrics["average_execution_time"] = \
            (self.coordinator_metrics["average_execution_time"] + execution_time) / 2
        
        # Update worker performance
        self.load_balancer.update_worker_performance(worker_id, {
            "success_rate": 1.0,
            "average_response_time": execution_time
        })
        
        self.logger.info(f"Received result for task {task_id} from worker {worker_id}")
    
    async def _handle_worker_registration(self, message: Dict[str, Any]):
        """Handle worker registration"""
        worker_id = message.get("worker_id")
        worker_info = WorkerInfo(**message.get("worker_info", {}))
        
        self.worker_info[worker_id] = worker_info
        self.load_balancer.add_worker(worker_info)
        
        self.logger.info(f"Registered worker {worker_id}")
    
    async def _monitoring_loop(self):
        """Monitoring loop"""
        while True:
            try:
                # Update coordinator metrics
                self._update_coordinator_metrics()
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    def _update_coordinator_metrics(self):
        """Update coordinator metrics"""
        self.coordinator_metrics["total_tasks"] = len(self.completed_tasks) + len(self.failed_tasks)
        
        # Calculate worker utilization
        for worker_id, worker_info in self.worker_info.items():
            utilization = worker_info.tasks_completed / max(worker_info.tasks_completed + worker_info.tasks_failed, 1)
            self.coordinator_metrics["worker_utilization"][worker_id] = utilization
    
    def submit_task(self, task_type: str, task_data: Dict[str, Any], priority: int = 1) -> str:
        """Submit task to coordinator"""
        task_id = str(uuid.uuid4())
        
        task = TaskAssignment(
            task_id=task_id,
            worker_id="",  # Will be assigned by load balancer
            task_type=task_type,
            priority=priority,
            data=task_data
        )
        
        self.task_queue.put(task)
        self.coordinator_metrics["total_tasks"] += 1
        
        self.logger.info(f"Submitted task {task_id} of type {task_type}")
        return task_id
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "config": self.config.__dict__,
            "total_workers": len(self.workers),
            "active_workers": len([w for w in self.worker_info.values() if w.status != WorkerStatus.OFFLINE]),
            "pending_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "coordinator_metrics": self.coordinator_metrics,
            "load_balancer_stats": self.load_balancer.get_load_balancing_stats()
        }


def create_distributed_coordinator(config: DistributedConfig) -> DistributedCoordinator:
    """Create distributed coordinator"""
    return DistributedCoordinator(config)


def create_distributed_worker(worker_id: str, config: DistributedConfig) -> DistributedWorker:
    """Create distributed worker"""
    return DistributedWorker(worker_id, config)


def create_load_balancer(config: DistributedConfig) -> LoadBalancer:
    """Create load balancer"""
    return LoadBalancer(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create distributed config
    config = DistributedConfig(
            distribution_strategy=DistributionStrategy.DATA_PARALLEL,
            communication_backend=CommunicationBackend.ZMQ,
            load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
            num_workers=4,
            enable_auto_scaling=True
        )
        
        # Create coordinator
    coordinator = create_distributed_coordinator(config)
    
        # Submit tasks
        for i in range(10):
            task_id = coordinator.submit_task("training", {"epochs": 5}, priority=1)
            print(f"Submitted task {task_id}")
        
        # Get stats
    stats = coordinator.get_coordinator_stats()
    print(f"Coordinator stats: {stats}")

    # Run example
    asyncio.run(main())
