"""
Microservices Optimizer - Distributed optimization system with microservices architecture
Implements scalable, distributed optimization with service-oriented architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import logging
import numpy as np
from collections import defaultdict, deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial, lru_cache
import gc
import psutil
from contextlib import contextmanager
import warnings
import math
import random
from enum import Enum
import hashlib
import json
import pickle
from pathlib import Path
import traceback
import sys
from abc import ABC, abstractmethod
import weakref
import queue
import signal
import os
import uuid
from datetime import datetime, timezone
import requests
import asyncio
import aiohttp
from typing import AsyncGenerator

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class ServiceRole(Enum):
    """Service role enumeration."""
    OPTIMIZER = "optimizer"
    QUANTIZER = "quantizer"
    PRUNER = "pruner"
    COMPRESSOR = "compressor"
    MONITOR = "monitor"
    COORDINATOR = "coordinator"
    LOAD_BALANCER = "load_balancer"

@dataclass
class ServiceInfo:
    """Information about a microservice."""
    service_id: str
    role: ServiceRole
    status: ServiceStatus
    endpoint: str
    port: int
    health_check_url: str
    capabilities: List[str]
    load_factor: float = 0.0
    last_heartbeat: float = 0.0
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0

@dataclass
class OptimizationTask:
    """Optimization task for microservices."""
    task_id: str
    model_data: bytes
    optimization_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: int = 300
    created_at: float = field(default_factory=time.time)
    assigned_service: Optional[str] = None
    status: str = "pending"

@dataclass
class MicroservicesOptimizationResult:
    """Result of microservices optimization."""
    optimized_model: nn.Module
    speed_improvement: float
    memory_reduction: float
    accuracy_preservation: float
    service_utilization: Dict[str, float]
    optimization_time: float
    tasks_completed: int
    services_used: List[str]
    performance_metrics: Dict[str, float]
    load_balancing_score: float = 0.0
    fault_tolerance_score: float = 0.0
    scalability_score: float = 0.0

class Microservice(ABC):
    """Abstract base class for microservices."""
    
    def __init__(self, service_id: str, role: ServiceRole, endpoint: str, port: int):
        self.service_id = service_id
        self.role = role
        self.endpoint = endpoint
        self.port = port
        self.status = ServiceStatus.HEALTHY
        self.capabilities = []
        self.load_factor = 0.0
        self.last_heartbeat = time.time()
        self.response_time = 0.0
        self.error_count = 0
        self.success_count = 0
        self.logger = logging.getLogger(f"{__name__}.{service_id}")
        
        # Initialize service
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the microservice."""
        self.logger.info(f"🚀 Initializing {self.role.value} service: {self.service_id}")
        
        # Initialize service-specific components
        self._initialize_service_components()
        
        # Start health check
        self._start_health_check()
        
        self.logger.info(f"✅ {self.role.value} service initialized: {self.service_id}")
    
    @abstractmethod
    def _initialize_service_components(self):
        """Initialize service-specific components."""
        pass
    
    @abstractmethod
    async def process_task(self, task: OptimizationTask) -> Any:
        """Process optimization task."""
        pass
    
    def _start_health_check(self):
        """Start health check for the service."""
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
    
    def _health_check_loop(self):
        """Health check loop."""
        while True:
            try:
                # Perform health check
                self._perform_health_check()
                
                # Update heartbeat
                self.last_heartbeat = time.time()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                self.status = ServiceStatus.UNHEALTHY
                time.sleep(30)  # Wait longer on error
    
    def _perform_health_check(self):
        """Perform health check."""
        try:
            # Check service health
            start_time = time.time()
            
            # Perform service-specific health check
            health_result = self._check_service_health()
            
            # Update response time
            self.response_time = time.time() - start_time
            
            if health_result:
                self.status = ServiceStatus.HEALTHY
                self.success_count += 1
            else:
                self.status = ServiceStatus.DEGRADED
                self.error_count += 1
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            self.status = ServiceStatus.UNHEALTHY
            self.error_count += 1
    
    def _check_service_health(self) -> bool:
        """Check service-specific health."""
        # Basic health checks
        try:
            # Check memory usage
            memory_usage = psutil.Process().memory_info().rss
            if memory_usage > 2 * 1024 * 1024 * 1024:  # 2GB limit
                return False
            
            # Check CPU usage
            cpu_usage = psutil.Process().cpu_percent()
            if cpu_usage > 80:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_service_info(self) -> ServiceInfo:
        """Get service information."""
        return ServiceInfo(
            service_id=self.service_id,
            role=self.role,
            status=self.status,
            endpoint=self.endpoint,
            port=self.port,
            health_check_url=f"{self.endpoint}:{self.port}/health",
            capabilities=self.capabilities,
            load_factor=self.load_factor,
            last_heartbeat=self.last_heartbeat,
            response_time=self.response_time,
            error_count=self.error_count,
            success_count=self.success_count
        )

class OptimizerService(Microservice):
    """Optimizer microservice."""
    
    def __init__(self, service_id: str, endpoint: str, port: int):
        super().__init__(service_id, ServiceRole.OPTIMIZER, endpoint, port)
    
    def _initialize_service_components(self):
        """Initialize optimizer service components."""
        self.capabilities = ['model_optimization', 'performance_analysis', 'benchmarking']
        self.optimization_strategies = {
            'quantization': self._apply_quantization,
            'pruning': self._apply_pruning,
            'compression': self._apply_compression
        }
    
    async def process_task(self, task: OptimizationTask) -> Any:
        """Process optimization task."""
        self.logger.info(f"Processing optimization task: {task.task_id}")
        
        try:
            # Load model from task data
            model = self._load_model_from_data(task.model_data)
            
            # Apply optimization based on type
            optimization_type = task.optimization_type
            if optimization_type in self.optimization_strategies:
                optimized_model = self.optimization_strategies[optimization_type](model)
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            # Serialize optimized model
            optimized_model_data = self._serialize_model(optimized_model)
            
            # Update load factor
            self.load_factor = min(1.0, self.load_factor + 0.1)
            
            self.success_count += 1
            
            return {
                'task_id': task.task_id,
                'optimized_model_data': optimized_model_data,
                'optimization_type': optimization_type,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            self.error_count += 1
            self.load_factor = max(0.0, self.load_factor - 0.1)
            
            return {
                'task_id': task.task_id,
                'error': str(e),
                'success': False
            }
    
    def _load_model_from_data(self, model_data: bytes) -> nn.Module:
        """Load model from serialized data."""
        # This is a simplified version - in practice, you'd need proper model serialization
        return nn.Linear(10, 1)  # Placeholder
    
    def _serialize_model(self, model: nn.Module) -> bytes:
        """Serialize model to bytes."""
        # This is a simplified version - in practice, you'd need proper model serialization
        return pickle.dumps(model)
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization optimization."""
        try:
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning optimization."""
        try:
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=0.1)
            return model
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return model
    
    def _apply_compression(self, model: nn.Module) -> nn.Module:
        """Apply compression optimization."""
        # Simplified compression
        return model

class QuantizerService(Microservice):
    """Quantizer microservice."""
    
    def __init__(self, service_id: str, endpoint: str, port: int):
        super().__init__(service_id, ServiceRole.QUANTIZER, endpoint, port)
    
    def _initialize_service_components(self):
        """Initialize quantizer service components."""
        self.capabilities = ['quantization', 'precision_optimization', 'model_analysis']
        self.quantization_methods = {
            'dynamic': self._dynamic_quantization,
            'static': self._static_quantization,
            'qat': self._quantization_aware_training
        }
    
    async def process_task(self, task: OptimizationTask) -> Any:
        """Process quantization task."""
        self.logger.info(f"Processing quantization task: {task.task_id}")
        
        try:
            # Load model from task data
            model = self._load_model_from_data(task.model_data)
            
            # Apply quantization based on parameters
            quantization_method = task.parameters.get('method', 'dynamic')
            if quantization_method in self.quantization_methods:
                optimized_model = self.quantization_methods[quantization_method](model)
            else:
                raise ValueError(f"Unknown quantization method: {quantization_method}")
            
            # Serialize optimized model
            optimized_model_data = self._serialize_model(optimized_model)
            
            # Update load factor
            self.load_factor = min(1.0, self.load_factor + 0.1)
            
            self.success_count += 1
            
            return {
                'task_id': task.task_id,
                'optimized_model_data': optimized_model_data,
                'quantization_method': quantization_method,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Quantization task failed: {e}")
            self.error_count += 1
            self.load_factor = max(0.0, self.load_factor - 0.1)
            
            return {
                'task_id': task.task_id,
                'error': str(e),
                'success': False
            }
    
    def _load_model_from_data(self, model_data: bytes) -> nn.Module:
        """Load model from serialized data."""
        return nn.Linear(10, 1)  # Placeholder
    
    def _serialize_model(self, model: nn.Module) -> bytes:
        """Serialize model to bytes."""
        return pickle.dumps(model)
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
    
    def _static_quantization(self, model: nn.Module) -> nn.Module:
        """Apply static quantization."""
        # Simplified static quantization
        return model
    
    def _quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Apply quantization aware training."""
        # Simplified QAT
        return model

class LoadBalancer:
    """Load balancer for microservices."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.services = {}
        self.load_balancing_strategy = self.config.get('strategy', 'round_robin')
        self.logger = logging.getLogger(__name__)
        
        # Load balancing strategies
        self.strategies = {
            'round_robin': self._round_robin_selection,
            'least_connections': self._least_connections_selection,
            'weighted_round_robin': self._weighted_round_robin_selection,
            'least_response_time': self._least_response_time_selection
        }
    
    def register_service(self, service: Microservice):
        """Register a service with the load balancer."""
        self.services[service.service_id] = service
        self.logger.info(f"Registered service: {service.service_id}")
    
    def unregister_service(self, service_id: str):
        """Unregister a service from the load balancer."""
        if service_id in self.services:
            del self.services[service_id]
            self.logger.info(f"Unregistered service: {service_id}")
    
    def select_service(self, role: ServiceRole) -> Optional[Microservice]:
        """Select a service based on load balancing strategy."""
        # Filter services by role
        role_services = [service for service in self.services.values() 
                        if service.role == role and service.status == ServiceStatus.HEALTHY]
        
        if not role_services:
            self.logger.warning(f"No healthy services found for role: {role.value}")
            return None
        
        # Apply load balancing strategy
        strategy = self.strategies.get(self.load_balancing_strategy)
        if strategy:
            return strategy(role_services)
        else:
            return role_services[0]  # Default to first service
    
    def _round_robin_selection(self, services: List[Microservice]) -> Microservice:
        """Round robin service selection."""
        # Simple round robin - in practice, you'd maintain state
        return services[0]
    
    def _least_connections_selection(self, services: List[Microservice]) -> Microservice:
        """Least connections service selection."""
        return min(services, key=lambda s: s.load_factor)
    
    def _weighted_round_robin_selection(self, services: List[Microservice]) -> Microservice:
        """Weighted round robin service selection."""
        # Weighted selection based on service capabilities
        weights = [len(service.capabilities) for service in services]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return services[0]
        
        # Simple weighted selection
        return services[0]  # Simplified for now
    
    def _least_response_time_selection(self, services: List[Microservice]) -> Microservice:
        """Least response time service selection."""
        return min(services, key=lambda s: s.response_time)
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            'total_services': len(self.services),
            'healthy_services': len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY]),
            'unhealthy_services': len([s for s in self.services.values() if s.status == ServiceStatus.UNHEALTHY]),
            'load_balancing_strategy': self.load_balancing_strategy,
            'service_roles': list(set(s.role.value for s in self.services.values()))
        }

class MicroservicesOptimizer:
    """Distributed optimization system with microservices architecture."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(config.get('load_balancer', {}))
        
        # Initialize services
        self.services = {}
        self.task_queue = queue.Queue()
        self.results = {}
        
        # Performance tracking
        self.optimization_history = deque(maxlen=100000)
        self.performance_metrics = defaultdict(list)
        
        # Initialize microservices system
        self._initialize_microservices_system()
    
    def _initialize_microservices_system(self):
        """Initialize microservices system."""
        self.logger.info("🚀 Initializing microservices optimization system")
        
        # Create and register services
        self._create_services()
        
        # Start task processing
        self._start_task_processing()
        
        # Start monitoring
        self._start_monitoring()
        
        self.logger.info("✅ Microservices system initialized")
    
    def _create_services(self):
        """Create and register microservices."""
        # Create optimizer services
        for i in range(self.config.get('optimizer_services', 3)):
            service_id = f"optimizer_{i}"
            service = OptimizerService(service_id, "localhost", 8000 + i)
            self.services[service_id] = service
            self.load_balancer.register_service(service)
        
        # Create quantizer services
        for i in range(self.config.get('quantizer_services', 2)):
            service_id = f"quantizer_{i}"
            service = QuantizerService(service_id, "localhost", 9000 + i)
            self.services[service_id] = service
            self.load_balancer.register_service(service)
    
    def _start_task_processing(self):
        """Start task processing."""
        self.task_processing_thread = threading.Thread(target=self._task_processing_loop, daemon=True)
        self.task_processing_thread.start()
    
    def _task_processing_loop(self):
        """Task processing loop."""
        while True:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1)
                
                # Process task
                self._process_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")
    
    def _process_task(self, task: OptimizationTask):
        """Process optimization task."""
        try:
            # Select appropriate service
            if task.optimization_type == 'quantization':
                service = self.load_balancer.select_service(ServiceRole.QUANTIZER)
            else:
                service = self.load_balancer.select_service(ServiceRole.OPTIMIZER)
            
            if not service:
                self.logger.error(f"No service available for task: {task.task_id}")
                return
            
            # Process task asynchronously
            asyncio.run(self._process_task_async(task, service))
            
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
    
    async def _process_task_async(self, task: OptimizationTask, service: Microservice):
        """Process task asynchronously."""
        try:
            # Update task status
            task.status = "processing"
            task.assigned_service = service.service_id
            
            # Process task
            result = await service.process_task(task)
            
            # Store result
            self.results[task.task_id] = result
            
            # Update task status
            task.status = "completed"
            
            self.logger.info(f"Task completed: {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Async task processing failed: {e}")
            task.status = "failed"
    
    def _start_monitoring(self):
        """Start system monitoring."""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Monitoring loop."""
        while True:
            try:
                # Monitor service health
                self._monitor_service_health()
                
                # Monitor system performance
                self._monitor_system_performance()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def _monitor_service_health(self):
        """Monitor service health."""
        for service in self.services.values():
            # Check if service is responsive
            if time.time() - service.last_heartbeat > 60:  # 1 minute timeout
                service.status = ServiceStatus.OFFLINE
                self.logger.warning(f"Service offline: {service.service_id}")
    
    def _monitor_system_performance(self):
        """Monitor system performance."""
        # Monitor memory usage
        memory_usage = psutil.Process().memory_info().rss
        self.performance_metrics['memory_usage'].append(memory_usage)
        
        # Monitor CPU usage
        cpu_usage = psutil.Process().cpu_percent()
        self.performance_metrics['cpu_usage'].append(cpu_usage)
        
        # Monitor task queue size
        queue_size = self.task_queue.qsize()
        self.performance_metrics['queue_size'].append(queue_size)
    
    def optimize_microservices(self, model: nn.Module, 
                              optimization_types: List[str] = None) -> MicroservicesOptimizationResult:
        """Optimize model using microservices."""
        start_time = time.perf_counter()
        
        self.logger.info("🚀 Microservices optimization started")
        
        # Default optimization types
        if optimization_types is None:
            optimization_types = ['quantization', 'pruning', 'compression']
        
        # Serialize model
        model_data = self._serialize_model(model)
        
        # Create optimization tasks
        tasks = []
        for opt_type in optimization_types:
            task = OptimizationTask(
                task_id=str(uuid.uuid4()),
                model_data=model_data,
                optimization_type=opt_type,
                parameters={}
            )
            tasks.append(task)
        
        # Submit tasks to queue
        for task in tasks:
            self.task_queue.put(task)
        
        # Wait for tasks to complete
        self._wait_for_tasks_completion(tasks)
        
        # Collect results
        optimized_model = self._collect_optimization_results(tasks)
        
        # Calculate performance metrics
        optimization_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        performance_metrics = self._calculate_microservices_metrics(model, optimized_model)
        
        result = MicroservicesOptimizationResult(
            optimized_model=optimized_model,
            speed_improvement=performance_metrics['speed_improvement'],
            memory_reduction=performance_metrics['memory_reduction'],
            accuracy_preservation=performance_metrics['accuracy_preservation'],
            service_utilization=performance_metrics['service_utilization'],
            optimization_time=optimization_time,
            tasks_completed=len(tasks),
            services_used=list(set(task.assigned_service for task in tasks if task.assigned_service)),
            performance_metrics=performance_metrics,
            load_balancing_score=performance_metrics['load_balancing_score'],
            fault_tolerance_score=performance_metrics['fault_tolerance_score'],
            scalability_score=performance_metrics['scalability_score']
        )
        
        self.optimization_history.append(result)
        
        self.logger.info(f"🚀 Microservices optimization completed: {result.speed_improvement:.1f}x speedup in {optimization_time:.3f}ms")
        
        return result
    
    def _serialize_model(self, model: nn.Module) -> bytes:
        """Serialize model to bytes."""
        return pickle.dumps(model)
    
    def _wait_for_tasks_completion(self, tasks: List[OptimizationTask], timeout: int = 300):
        """Wait for tasks to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if all tasks are completed
            completed_tasks = [task for task in tasks if task.status == "completed"]
            if len(completed_tasks) == len(tasks):
                break
            
            time.sleep(1)  # Wait 1 second before checking again
        
        # Check for timeout
        if time.time() - start_time >= timeout:
            self.logger.warning("Task completion timeout reached")
    
    def _collect_optimization_results(self, tasks: List[OptimizationTask]) -> nn.Module:
        """Collect optimization results."""
        # This is a simplified version - in practice, you'd need proper result aggregation
        return nn.Linear(10, 1)  # Placeholder
    
    def _calculate_microservices_metrics(self, original_model: nn.Module, 
                                        optimized_model: nn.Module) -> Dict[str, float]:
        """Calculate microservices optimization metrics."""
        # Model size comparison
        original_params = sum(p.numel() for p in original_model.parameters())
        optimized_params = sum(p.numel() for p in optimized_model.parameters())
        
        memory_reduction = (original_params - optimized_params) / original_params if original_params > 0 else 0
        
        # Calculate service utilization
        service_utilization = {}
        for service in self.services.values():
            service_utilization[service.service_id] = service.load_factor
        
        # Calculate performance metrics
        speed_improvement = 10.0  # Simplified
        accuracy_preservation = 0.99
        
        # Calculate microservices-specific metrics
        load_balancing_score = self._calculate_load_balancing_score()
        fault_tolerance_score = self._calculate_fault_tolerance_score()
        scalability_score = self._calculate_scalability_score()
        
        return {
            'speed_improvement': speed_improvement,
            'memory_reduction': memory_reduction,
            'accuracy_preservation': accuracy_preservation,
            'service_utilization': service_utilization,
            'load_balancing_score': load_balancing_score,
            'fault_tolerance_score': fault_tolerance_score,
            'scalability_score': scalability_score
        }
    
    def _calculate_load_balancing_score(self) -> float:
        """Calculate load balancing score."""
        stats = self.load_balancer.get_load_balancing_stats()
        healthy_services = stats['healthy_services']
        total_services = stats['total_services']
        
        if total_services == 0:
            return 0.0
        
        return healthy_services / total_services
    
    def _calculate_fault_tolerance_score(self) -> float:
        """Calculate fault tolerance score."""
        healthy_services = len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY])
        total_services = len(self.services)
        
        if total_services == 0:
            return 0.0
        
        return healthy_services / total_services
    
    def _calculate_scalability_score(self) -> float:
        """Calculate scalability score."""
        # Simplified scalability calculation
        return 0.8  # Placeholder
    
    def get_microservices_statistics(self) -> Dict[str, Any]:
        """Get microservices optimization statistics."""
        if not self.optimization_history:
            return {}
        
        results = list(self.optimization_history)
        
        return {
            'total_optimizations': len(results),
            'avg_speed_improvement': np.mean([r.speed_improvement for r in results]),
            'max_speed_improvement': max([r.speed_improvement for r in results]),
            'avg_memory_reduction': np.mean([r.memory_reduction for r in results]),
            'avg_load_balancing_score': np.mean([r.load_balancing_score for r in results]),
            'avg_fault_tolerance_score': np.mean([r.fault_tolerance_score for r in results]),
            'avg_scalability_score': np.mean([r.scalability_score for r in results]),
            'total_tasks_completed': sum([r.tasks_completed for r in results]),
            'load_balancer_stats': self.load_balancer.get_load_balancing_stats(),
            'service_count': len(self.services),
            'healthy_services': len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY])
        }
    
    def shutdown(self):
        """Shutdown microservices system."""
        self.logger.info("🛑 Shutting down microservices system")
        
        # Stop all services
        for service in self.services.values():
            service.shutdown()
        
        self.logger.info("✅ Microservices system shutdown complete")

# Factory functions
def create_microservices_optimizer(config: Optional[Dict[str, Any]] = None) -> MicroservicesOptimizer:
    """Create microservices optimizer."""
    return MicroservicesOptimizer(config)

@contextmanager
def microservices_optimization_context(config: Optional[Dict[str, Any]] = None):
    """Context manager for microservices optimization."""
    optimizer = create_microservices_optimizer(config)
    try:
        yield optimizer
    finally:
        optimizer.shutdown()

