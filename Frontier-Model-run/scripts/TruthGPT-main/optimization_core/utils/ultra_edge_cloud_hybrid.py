"""
Ultra-Advanced Edge-Cloud Hybrid Computing Module
================================================

This module provides edge-cloud hybrid computing capabilities for TruthGPT models,
including edge inference, cloud training, adaptive offloading, and distributed processing.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics
import warnings
import asyncio
import aiohttp
import threading
import queue

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class EdgeDeviceType(Enum):
    """Types of edge devices."""
    MOBILE = "mobile"
    IOT = "iot"
    EMBEDDED = "embedded"
    GATEWAY = "gateway"
    EDGE_SERVER = "edge_server"
    FOG_NODE = "fog_node"

class ComputingMode(Enum):
    """Computing modes for edge-cloud hybrid."""
    EDGE_ONLY = "edge_only"
    CLOUD_ONLY = "cloud_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    OFFLOADING = "offloading"

class OffloadingStrategy(Enum):
    """Offloading strategies."""
    LATENCY_OPTIMIZED = "latency_optimized"
    ENERGY_OPTIMIZED = "energy_optimized"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    DYNAMIC = "dynamic"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASHING = "consistent_hashing"

class EdgeConfig(BaseModel):
    """Configuration for edge computing."""
    device_type: EdgeDeviceType = EdgeDeviceType.MOBILE
    compute_capacity: float = 1.0
    memory_capacity: float = 1.0
    bandwidth: float = 10.0
    battery_level: float = 1.0
    latency_threshold: float = 100.0
    energy_efficiency: float = 1.0
    device_id: str = "edge_device_1"
    location: Tuple[float, float] = (0.0, 0.0)

class CloudConfig(BaseModel):
    """Configuration for cloud computing."""
    server_capacity: float = 1000.0
    memory_capacity: float = 100.0
    bandwidth: float = 1000.0
    latency: float = 50.0
    cost_per_hour: float = 0.1
    server_id: str = "cloud_server_1"
    region: str = "us-east-1"

class HybridConfig(BaseModel):
    """Configuration for edge-cloud hybrid computing."""
    edge_config: EdgeConfig = Field(default_factory=EdgeConfig)
    cloud_config: CloudConfig = Field(default_factory=CloudConfig)
    computing_mode: ComputingMode = ComputingMode.HYBRID
    offloading_strategy: OffloadingStrategy = OffloadingStrategy.BALANCED
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    adaptive_threshold: float = 0.5
    max_offload_ratio: float = 0.8
    min_edge_ratio: float = 0.2
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./edge_cloud_results"

class EdgeDevice:
    """Edge device implementation."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.current_load = 0.0
        self.active_tasks = []
        self.performance_history = deque(maxlen=100)
        self.energy_consumption = 0.0
        self.is_online = True
        
    def get_compute_capacity(self) -> float:
        """Get current compute capacity."""
        return self.config.compute_capacity * (1.0 - self.current_load)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage."""
        return sum(task.memory_usage for task in self.active_tasks)
    
    def get_available_memory(self) -> float:
        """Get available memory."""
        return self.config.memory_capacity - self.get_memory_usage()
    
    def can_handle_task(self, task_complexity: float, memory_required: float) -> bool:
        """Check if device can handle a task."""
        return (self.get_compute_capacity() >= task_complexity and 
                self.get_available_memory() >= memory_required and
                self.is_online)
    
    def execute_task(self, task: 'ComputingTask') -> Dict[str, Any]:
        """Execute a task on the edge device."""
        start_time = time.time()
        
        # Simulate task execution
        execution_time = task.complexity / self.get_compute_capacity()
        time.sleep(min(execution_time, 0.1))  # Cap simulation time
        
        # Update device state
        self.current_load += task.complexity / self.config.compute_capacity
        self.active_tasks.append(task)
        self.energy_consumption += task.complexity * self.config.energy_efficiency
        
        execution_time = time.time() - start_time
        
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'load': self.current_load,
            'energy': self.energy_consumption
        })
        
        return {
            'device_id': self.config.device_id,
            'execution_time': execution_time,
            'energy_consumption': task.complexity * self.config.energy_efficiency,
            'success': True
        }
    
    def complete_task(self, task: 'ComputingTask'):
        """Complete a task and update device state."""
        if task in self.active_tasks:
            self.active_tasks.remove(task)
            self.current_load -= task.complexity / self.config.compute_capacity
            self.current_load = max(0.0, self.current_load)

class CloudServer:
    """Cloud server implementation."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.current_load = 0.0
        self.active_tasks = []
        self.performance_history = deque(maxlen=1000)
        self.total_cost = 0.0
        self.is_online = True
        
    def get_compute_capacity(self) -> float:
        """Get current compute capacity."""
        return self.config.server_capacity * (1.0 - self.current_load)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage."""
        return sum(task.memory_usage for task in self.active_tasks)
    
    def get_available_memory(self) -> float:
        """Get available memory."""
        return self.config.memory_capacity - self.get_memory_usage()
    
    def can_handle_task(self, task_complexity: float, memory_required: float) -> bool:
        """Check if server can handle a task."""
        return (self.get_compute_capacity() >= task_complexity and 
                self.get_available_memory() >= memory_required and
                self.is_online)
    
    def execute_task(self, task: 'ComputingTask') -> Dict[str, Any]:
        """Execute a task on the cloud server."""
        start_time = time.time()
        
        # Simulate task execution
        execution_time = task.complexity / self.get_compute_capacity()
        time.sleep(min(execution_time, 0.05))  # Cap simulation time
        
        # Update server state
        self.current_load += task.complexity / self.config.server_capacity
        self.active_tasks.append(task)
        
        # Calculate cost
        task_cost = (execution_time / 3600) * self.config.cost_per_hour
        self.total_cost += task_cost
        
        execution_time = time.time() - start_time
        
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'load': self.current_load,
            'cost': task_cost
        })
        
        return {
            'server_id': self.config.server_id,
            'execution_time': execution_time,
            'cost': task_cost,
            'success': True
        }
    
    def complete_task(self, task: 'ComputingTask'):
        """Complete a task and update server state."""
        if task in self.active_tasks:
            self.active_tasks.remove(task)
            self.current_load -= task.complexity / self.config.server_capacity
            self.current_load = max(0.0, self.current_load)

class ComputingTask:
    """Computing task representation."""
    
    def __init__(self, 
                 task_id: str,
                 complexity: float,
                 memory_usage: float,
                 latency_requirement: float,
                 priority: int = 1):
        self.task_id = task_id
        self.complexity = complexity
        self.memory_usage = memory_usage
        self.latency_requirement = latency_requirement
        self.priority = priority
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.execution_location = None

class AdaptiveOffloader:
    """Adaptive offloading decision maker."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.offloading_history = deque(maxlen=1000)
        self.performance_metrics = {
            'edge_success_rate': 0.0,
            'cloud_success_rate': 0.0,
            'average_latency': 0.0,
            'energy_efficiency': 0.0,
            'cost_efficiency': 0.0
        }
        
    def should_offload(self, 
                      task: ComputingTask,
                      edge_device: EdgeDevice,
                      cloud_server: CloudServer) -> bool:
        """Decide whether to offload a task to cloud."""
        
        # Calculate metrics
        edge_capacity = edge_device.get_compute_capacity()
        cloud_capacity = cloud_server.get_compute_capacity()
        
        edge_latency = self._estimate_edge_latency(task, edge_device)
        cloud_latency = self._estimate_cloud_latency(task, cloud_server)
        
        edge_energy = self._estimate_edge_energy(task, edge_device)
        cloud_cost = self._estimate_cloud_cost(task, cloud_server)
        
        # Decision logic based on strategy
        if self.config.offloading_strategy == OffloadingStrategy.LATENCY_OPTIMIZED:
            return cloud_latency < edge_latency
        elif self.config.offloading_strategy == OffloadingStrategy.ENERGY_OPTIMIZED:
            return edge_energy > self.config.adaptive_threshold
        elif self.config.offloading_strategy == OffloadingStrategy.COST_OPTIMIZED:
            return cloud_cost < self.config.adaptive_threshold
        elif self.config.offloading_strategy == OffloadingStrategy.BALANCED:
            return self._balanced_decision(task, edge_device, cloud_server)
        else:  # DYNAMIC
            return self._dynamic_decision(task, edge_device, cloud_server)
    
    def _estimate_edge_latency(self, task: ComputingTask, edge_device: EdgeDevice) -> float:
        """Estimate latency for edge execution."""
        execution_time = task.complexity / edge_device.get_compute_capacity()
        return execution_time * 1000  # Convert to ms
    
    def _estimate_cloud_latency(self, task: ComputingTask, cloud_server: CloudServer) -> float:
        """Estimate latency for cloud execution."""
        execution_time = task.complexity / cloud_server.get_compute_capacity()
        network_latency = self.config.cloud_config.latency
        return execution_time * 1000 + network_latency  # Convert to ms
    
    def _estimate_edge_energy(self, task: ComputingTask, edge_device: EdgeDevice) -> float:
        """Estimate energy consumption for edge execution."""
        return task.complexity * edge_device.config.energy_efficiency
    
    def _estimate_cloud_cost(self, task: ComputingTask, cloud_server: CloudServer) -> float:
        """Estimate cost for cloud execution."""
        execution_time = task.complexity / cloud_server.get_compute_capacity()
        return (execution_time / 3600) * cloud_server.config.cost_per_hour
    
    def _balanced_decision(self, 
                          task: ComputingTask,
                          edge_device: EdgeDevice,
                          cloud_server: CloudServer) -> bool:
        """Make balanced offloading decision."""
        edge_score = self._calculate_edge_score(task, edge_device)
        cloud_score = self._calculate_cloud_score(task, cloud_server)
        
        return cloud_score > edge_score
    
    def _dynamic_decision(self, 
                         task: ComputingTask,
                         edge_device: EdgeDevice,
                         cloud_server: CloudServer) -> bool:
        """Make dynamic offloading decision based on history."""
        if len(self.offloading_history) < 10:
            return self._balanced_decision(task, edge_device, cloud_server)
        
        # Analyze recent performance
        recent_history = list(self.offloading_history)[-10:]
        edge_performance = [h['edge_latency'] for h in recent_history if h['offloaded'] == False]
        cloud_performance = [h['cloud_latency'] for h in recent_history if h['offloaded'] == True]
        
        if edge_performance and cloud_performance:
            avg_edge_latency = statistics.mean(edge_performance)
            avg_cloud_latency = statistics.mean(cloud_performance)
            return avg_cloud_latency < avg_edge_latency
        
        return self._balanced_decision(task, edge_device, cloud_server)
    
    def _calculate_edge_score(self, task: ComputingTask, edge_device: EdgeDevice) -> float:
        """Calculate edge execution score."""
        latency_score = 1.0 / (1.0 + self._estimate_edge_latency(task, edge_device) / 1000)
        energy_score = 1.0 / (1.0 + self._estimate_edge_energy(task, edge_device))
        capacity_score = edge_device.get_compute_capacity() / edge_device.config.compute_capacity
        
        return (latency_score + energy_score + capacity_score) / 3.0
    
    def _calculate_cloud_score(self, task: ComputingTask, cloud_server: CloudServer) -> float:
        """Calculate cloud execution score."""
        latency_score = 1.0 / (1.0 + self._estimate_cloud_latency(task, cloud_server) / 1000)
        cost_score = 1.0 / (1.0 + self._estimate_cloud_cost(task, cloud_server))
        capacity_score = cloud_server.get_compute_capacity() / cloud_server.config.server_capacity
        
        return (latency_score + cost_score + capacity_score) / 3.0
    
    def record_offloading_decision(self, 
                                  task: ComputingTask,
                                  offloaded: bool,
                                  edge_latency: float,
                                  cloud_latency: float):
        """Record offloading decision for learning."""
        self.offloading_history.append({
            'task_id': task.task_id,
            'offloaded': offloaded,
            'edge_latency': edge_latency,
            'cloud_latency': cloud_latency,
            'timestamp': time.time()
        })

class LoadBalancer:
    """Load balancer for edge-cloud hybrid system."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.edge_devices = []
        self.cloud_servers = []
        self.task_queue = queue.Queue()
        self.load_balancing_history = deque(maxlen=1000)
        
    def add_edge_device(self, device: EdgeDevice):
        """Add edge device to load balancer."""
        self.edge_devices.append(device)
        
    def add_cloud_server(self, server: CloudServer):
        """Add cloud server to load balancer."""
        self.cloud_servers.append(server)
    
    def select_executor(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Select executor for task based on load balancing strategy."""
        
        if self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(task)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(task)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(task)
        elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(task)
        else:  # CONSISTENT_HASHING
            return self._consistent_hashing_selection(task)
    
    def _round_robin_selection(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Round robin selection."""
        all_executors = self.edge_devices + self.cloud_servers
        if not all_executors:
            return None
        
        # Simple round robin based on task ID hash
        index = hash(task.task_id) % len(all_executors)
        return all_executors[index]
    
    def _least_connections_selection(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Least connections selection."""
        all_executors = self.edge_devices + self.cloud_servers
        if not all_executors:
            return None
        
        # Find executor with least active tasks
        min_tasks = min(len(executor.active_tasks) for executor in all_executors)
        candidates = [executor for executor in all_executors 
                     if len(executor.active_tasks) == min_tasks]
        
        return random.choice(candidates)
    
    def _weighted_round_robin_selection(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Weighted round robin selection."""
        all_executors = self.edge_devices + self.cloud_servers
        if not all_executors:
            return None
        
        # Weight based on compute capacity
        weights = [executor.get_compute_capacity() for executor in all_executors]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(all_executors)
        
        # Weighted random selection
        rand = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for executor, weight in zip(all_executors, weights):
            cumulative_weight += weight
            if rand <= cumulative_weight:
                return executor
        
        return all_executors[-1]
    
    def _least_response_time_selection(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Least response time selection."""
        all_executors = self.edge_devices + self.cloud_servers
        if not all_executors:
            return None
        
        # Estimate response time for each executor
        response_times = []
        for executor in all_executors:
            if isinstance(executor, EdgeDevice):
                response_time = task.complexity / executor.get_compute_capacity()
            else:  # CloudServer
                response_time = task.complexity / executor.get_compute_capacity() + executor.config.latency / 1000
            
            response_times.append(response_time)
        
        # Select executor with minimum response time
        min_response_time = min(response_times)
        candidates = [executor for executor, rt in zip(all_executors, response_times) 
                     if rt == min_response_time]
        
        return random.choice(candidates)
    
    def _consistent_hashing_selection(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Consistent hashing selection."""
        all_executors = self.edge_devices + self.cloud_servers
        if not all_executors:
            return None
        
        # Simple consistent hashing based on task ID
        task_hash = hash(task.task_id)
        executor_hash = task_hash % len(all_executors)
        
        return all_executors[executor_hash]

class EdgeCloudHybridManager:
    """Main manager for edge-cloud hybrid computing."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.edge_devices = []
        self.cloud_servers = []
        self.offloader = AdaptiveOffloader(config)
        self.load_balancer = LoadBalancer(config)
        self.execution_history = []
        self.performance_metrics = {
            'total_tasks': 0,
            'edge_tasks': 0,
            'cloud_tasks': 0,
            'average_latency': 0.0,
            'total_energy': 0.0,
            'total_cost': 0.0,
            'success_rate': 0.0
        }
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def add_edge_device(self, device: EdgeDevice):
        """Add edge device to the system."""
        self.edge_devices.append(device)
        self.load_balancer.add_edge_device(device)
        logger.info(f"Added edge device: {device.config.device_id}")
        
    def add_cloud_server(self, server: CloudServer):
        """Add cloud server to the system."""
        self.cloud_servers.append(server)
        self.load_balancer.add_cloud_server(server)
        logger.info(f"Added cloud server: {server.config.server_id}")
        
    def execute_task(self, task: ComputingTask) -> Dict[str, Any]:
        """Execute a task using edge-cloud hybrid computing."""
        logger.info(f"Executing task: {task.task_id}")
        
        start_time = time.time()
        
        # Select executor based on computing mode
        if self.config.computing_mode == ComputingMode.EDGE_ONLY:
            executor = self._select_edge_executor(task)
        elif self.config.computing_mode == ComputingMode.CLOUD_ONLY:
            executor = self._select_cloud_executor(task)
        elif self.config.computing_mode == ComputingMode.HYBRID:
            executor = self._select_hybrid_executor(task)
        elif self.config.computing_mode == ComputingMode.ADAPTIVE:
            executor = self._select_adaptive_executor(task)
        else:  # OFFLOADING
            executor = self._select_offloading_executor(task)
        
        if executor is None:
            return {
                'task_id': task.task_id,
                'success': False,
                'error': 'No available executor',
                'execution_time': 0.0
            }
        
        # Execute task
        task.started_at = time.time()
        task.execution_location = 'edge' if isinstance(executor, EdgeDevice) else 'cloud'
        
        result = executor.execute_task(task)
        
        # Complete task
        executor.complete_task(task)
        task.completed_at = time.time()
        
        # Update metrics
        execution_time = time.time() - start_time
        self._update_metrics(task, result, execution_time)
        
        # Record execution
        self.execution_history.append({
            'task_id': task.task_id,
            'executor': executor.config.device_id if isinstance(executor, EdgeDevice) else executor.config.server_id,
            'location': task.execution_location,
            'execution_time': execution_time,
            'success': result['success'],
            'timestamp': time.time()
        })
        
        logger.info(f"Task {task.task_id} completed in {execution_time:.4f}s on {task.execution_location}")
        
        return {
            'task_id': task.task_id,
            'executor': executor.config.device_id if isinstance(executor, EdgeDevice) else executor.config.server_id,
            'location': task.execution_location,
            'execution_time': execution_time,
            'success': result['success'],
            'result': result
        }
    
    def _select_edge_executor(self, task: ComputingTask) -> Optional[EdgeDevice]:
        """Select edge executor for task."""
        available_devices = [device for device in self.edge_devices 
                           if device.can_handle_task(task.complexity, task.memory_usage)]
        
        if not available_devices:
            return None
        
        # Select device with highest capacity
        return max(available_devices, key=lambda d: d.get_compute_capacity())
    
    def _select_cloud_executor(self, task: ComputingTask) -> Optional[CloudServer]:
        """Select cloud executor for task."""
        available_servers = [server for server in self.cloud_servers 
                           if server.can_handle_task(task.complexity, task.memory_usage)]
        
        if not available_servers:
            return None
        
        # Select server with highest capacity
        return max(available_servers, key=lambda s: s.get_compute_capacity())
    
    def _select_hybrid_executor(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Select executor using hybrid strategy."""
        # Use load balancer to select executor
        return self.load_balancer.select_executor(task)
    
    def _select_adaptive_executor(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Select executor using adaptive strategy."""
        if not self.edge_devices or not self.cloud_servers:
            return self._select_hybrid_executor(task)
        
        # Use adaptive offloader to decide
        edge_device = self.edge_devices[0]  # Simplified: use first device
        cloud_server = self.cloud_servers[0]  # Simplified: use first server
        
        if self.offloader.should_offload(task, edge_device, cloud_server):
            return cloud_server
        else:
            return edge_device
    
    def _select_offloading_executor(self, task: ComputingTask) -> Union[EdgeDevice, CloudServer]:
        """Select executor using offloading strategy."""
        return self._select_adaptive_executor(task)
    
    def _update_metrics(self, task: ComputingTask, result: Dict[str, Any], execution_time: float):
        """Update performance metrics."""
        self.performance_metrics['total_tasks'] += 1
        
        if task.execution_location == 'edge':
            self.performance_metrics['edge_tasks'] += 1
            if 'energy_consumption' in result:
                self.performance_metrics['total_energy'] += result['energy_consumption']
        else:
            self.performance_metrics['cloud_tasks'] += 1
            if 'cost' in result:
                self.performance_metrics['total_cost'] += result['cost']
        
        # Update average latency
        total_latency = self.performance_metrics['average_latency'] * (self.performance_metrics['total_tasks'] - 1)
        self.performance_metrics['average_latency'] = (total_latency + execution_time * 1000) / self.performance_metrics['total_tasks']
        
        # Update success rate
        if result['success']:
            success_count = self.performance_metrics['success_rate'] * (self.performance_metrics['total_tasks'] - 1)
            self.performance_metrics['success_rate'] = (success_count + 1) / self.performance_metrics['total_tasks']
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'edge_devices': len(self.edge_devices),
            'cloud_servers': len(self.cloud_servers),
            'performance_metrics': self.performance_metrics.copy(),
            'offloading_history_size': len(self.offloader.offloading_history),
            'execution_history_size': len(self.execution_history),
            'system_load': {
                'edge_load': statistics.mean([device.current_load for device in self.edge_devices]) if self.edge_devices else 0.0,
                'cloud_load': statistics.mean([server.current_load for server in self.cloud_servers]) if self.cloud_servers else 0.0
            }
        }
    
    def optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance."""
        logger.info("Optimizing edge-cloud hybrid system")
        
        optimization_results = {
            'load_balancing_optimization': self._optimize_load_balancing(),
            'offloading_optimization': self._optimize_offloading(),
            'resource_allocation_optimization': self._optimize_resource_allocation()
        }
        
        return optimization_results
    
    def _optimize_load_balancing(self) -> Dict[str, Any]:
        """Optimize load balancing strategy."""
        # Analyze load distribution
        edge_loads = [device.current_load for device in self.edge_devices]
        cloud_loads = [server.current_load for server in self.cloud_servers]
        
        edge_load_variance = statistics.variance(edge_loads) if len(edge_loads) > 1 else 0.0
        cloud_load_variance = statistics.variance(cloud_loads) if len(cloud_loads) > 1 else 0.0
        
        # Suggest optimization
        if edge_load_variance > 0.1:
            suggestion = "Consider using weighted round robin for better load distribution"
        else:
            suggestion = "Current load balancing strategy is performing well"
        
        return {
            'edge_load_variance': edge_load_variance,
            'cloud_load_variance': cloud_load_variance,
            'suggestion': suggestion
        }
    
    def _optimize_offloading(self) -> Dict[str, Any]:
        """Optimize offloading strategy."""
        if len(self.offloader.offloading_history) < 10:
            return {'suggestion': 'Insufficient data for offloading optimization'}
        
        # Analyze offloading performance
        recent_history = list(self.offloader.offloading_history)[-50:]
        offloaded_tasks = [h for h in recent_history if h['offloaded']]
        local_tasks = [h for h in recent_history if not h['offloaded']]
        
        if offloaded_tasks and local_tasks:
            avg_offloaded_latency = statistics.mean([h['cloud_latency'] for h in offloaded_tasks])
            avg_local_latency = statistics.mean([h['edge_latency'] for h in local_tasks])
            
            if avg_offloaded_latency < avg_local_latency:
                suggestion = "Consider increasing offloading ratio"
            else:
                suggestion = "Consider decreasing offloading ratio"
        else:
            suggestion = "Maintain current offloading strategy"
        
        return {
            'avg_offloaded_latency': avg_offloaded_latency if offloaded_tasks else 0.0,
            'avg_local_latency': avg_local_latency if local_tasks else 0.0,
            'suggestion': suggestion
        }
    
    def _optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation."""
        # Analyze resource utilization
        edge_utilization = statistics.mean([device.current_load for device in self.edge_devices]) if self.edge_devices else 0.0
        cloud_utilization = statistics.mean([server.current_load for server in self.cloud_servers]) if self.cloud_servers else 0.0
        
        # Suggest resource allocation
        if edge_utilization > 0.8:
            suggestion = "Consider adding more edge devices or offloading more tasks"
        elif cloud_utilization > 0.8:
            suggestion = "Consider adding more cloud servers"
        else:
            suggestion = "Resource allocation is optimal"
        
        return {
            'edge_utilization': edge_utilization,
            'cloud_utilization': cloud_utilization,
            'suggestion': suggestion
        }

# Factory functions
def create_edge_config(device_type: EdgeDeviceType = EdgeDeviceType.MOBILE,
                      compute_capacity: float = 1.0,
                      memory_capacity: float = 1.0,
                      **kwargs) -> EdgeConfig:
    """Create edge device configuration."""
    return EdgeConfig(
        device_type=device_type,
        compute_capacity=compute_capacity,
        memory_capacity=memory_capacity,
        **kwargs
    )

def create_cloud_config(server_capacity: float = 1000.0,
                       memory_capacity: float = 100.0,
                       **kwargs) -> CloudConfig:
    """Create cloud server configuration."""
    return CloudConfig(
        server_capacity=server_capacity,
        memory_capacity=memory_capacity,
        **kwargs
    )

def create_hybrid_config(edge_config: Optional[EdgeConfig] = None,
                        cloud_config: Optional[CloudConfig] = None,
                        computing_mode: ComputingMode = ComputingMode.HYBRID,
                        **kwargs) -> HybridConfig:
    """Create hybrid configuration."""
    if edge_config is None:
        edge_config = create_edge_config()
    if cloud_config is None:
        cloud_config = create_cloud_config()
    
    return HybridConfig(
        edge_config=edge_config,
        cloud_config=cloud_config,
        computing_mode=computing_mode,
        **kwargs
    )

def create_edge_device(config: EdgeConfig) -> EdgeDevice:
    """Create edge device."""
    return EdgeDevice(config)

def create_cloud_server(config: CloudConfig) -> CloudServer:
    """Create cloud server."""
    return CloudServer(config)

def create_computing_task(task_id: str,
                         complexity: float,
                         memory_usage: float,
                         latency_requirement: float = 100.0) -> ComputingTask:
    """Create computing task."""
    return ComputingTask(task_id, complexity, memory_usage, latency_requirement)

def create_edge_cloud_hybrid_manager(config: Optional[HybridConfig] = None) -> EdgeCloudHybridManager:
    """Create edge-cloud hybrid manager."""
    if config is None:
        config = create_hybrid_config()
    return EdgeCloudHybridManager(config)

# Example usage
def example_edge_cloud_hybrid():
    """Example of edge-cloud hybrid computing."""
    # Create configurations
    edge_config = create_edge_config(
        device_type=EdgeDeviceType.MOBILE,
        compute_capacity=2.0,
        memory_capacity=4.0
    )
    
    cloud_config = create_cloud_config(
        server_capacity=2000.0,
        memory_capacity=200.0
    )
    
    hybrid_config = create_hybrid_config(
        edge_config=edge_config,
        cloud_config=cloud_config,
        computing_mode=ComputingMode.ADAPTIVE,
        offloading_strategy=OffloadingStrategy.BALANCED
    )
    
    # Create manager
    manager = create_edge_cloud_hybrid_manager(hybrid_config)
    
    # Add devices
    edge_device = create_edge_device(edge_config)
    cloud_server = create_cloud_server(cloud_config)
    
    manager.add_edge_device(edge_device)
    manager.add_cloud_server(cloud_server)
    
    # Create and execute tasks
    tasks = [
        create_computing_task(f"task_{i}", random.uniform(0.1, 2.0), random.uniform(0.1, 1.0))
        for i in range(10)
    ]
    
    results = []
    for task in tasks:
        result = manager.execute_task(task)
        results.append(result)
    
    # Get system statistics
    stats = manager.get_system_statistics()
    print(f"System statistics: {stats}")
    
    # Optimize system
    optimization = manager.optimize_system()
    print(f"Optimization results: {optimization}")
    
    return results

if __name__ == "__main__":
    # Run example
    example_edge_cloud_hybrid()

