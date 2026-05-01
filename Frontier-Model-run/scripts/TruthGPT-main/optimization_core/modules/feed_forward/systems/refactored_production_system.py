"""
Refactored Production PiMoE System
Clean architecture with separation of concerns, dependency injection, and interfaces.
"""

import torch
import torch.nn as nn
import time
import threading
import queue
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
import signal
import sys

from ..core.refactored_pimoe_base import (
    BaseService, BaseConfig, ProductionConfig, SystemConfig,
    LoggerProtocol, MonitorProtocol, ErrorHandlerProtocol, RequestQueueProtocol,
    ServiceFactory, DIContainer, EventBus, ResourceManager,
    MetricsCollector, HealthChecker, BasePiMoESystem,
    RequestData, ResponseData, ProductionMode, LogLevel, OptimizationLevel
)
from ..routing.advanced_pimoe_routing import RoutingStrategy

# Concrete Implementations

class ProductionLogger(BaseService):
    """Production logger implementation."""
    
    def __init__(self, config: ProductionConfig):
        super().__init__(config)
        self.logger = self._setup_logger()
        self.metrics_logger = self._setup_metrics_logger()
        self.error_logger = self._setup_error_logger()
    
    def initialize(self) -> None:
        """Initialize the logger."""
        self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the logger."""
        self._initialized = False
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main application logger."""
        logger = logging.getLogger('pimoe_production')
        logger.setLevel(getattr(logging, self.config.log_level.value.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('pimoe_production.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_metrics_logger(self) -> logging.Logger:
        """Setup metrics logger."""
        logger = logging.getLogger('pimoe_metrics')
        logger.setLevel(logging.INFO)
        
        # Metrics file handler
        metrics_handler = logging.FileHandler('pimoe_metrics.log')
        metrics_handler.setLevel(logging.INFO)
        
        # JSON formatter for metrics
        json_formatter = logging.Formatter('%(message)s')
        metrics_handler.setFormatter(json_formatter)
        
        logger.addHandler(metrics_handler)
        logger.propagate = False
        
        return logger
    
    def _setup_error_logger(self) -> logging.Logger:
        """Setup error logger."""
        logger = logging.getLogger('pimoe_errors')
        logger.setLevel(logging.ERROR)
        
        # Error file handler
        error_handler = logging.FileHandler('pimoe_errors.log')
        error_handler.setLevel(logging.ERROR)
        
        # Detailed formatter for errors
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        
        logger.addHandler(error_handler)
        logger.propagate = False
        
        return logger
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(f"{message} | {kwargs}")
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(f"{message} | {kwargs}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        if exception:
            self.error_logger.error(f"{message} | {kwargs}", exc_info=exception)
        else:
            self.error_logger.error(f"{message} | {kwargs}")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics in JSON format."""
        import json
        metrics['timestamp'] = time.time()
        metrics['deployment_id'] = self.config.deployment_id
        self.metrics_logger.info(json.dumps(metrics))

class ProductionMonitor(BaseService):
    """Production monitoring system."""
    
    def __init__(self, config: ProductionConfig, logger: LoggerProtocol):
        super().__init__(config, logger)
        self.metrics = {}
        self.health_status = "healthy"
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.circuit_breaker_count = 0
        self.monitoring_thread = None
        self._stop_monitoring = False
    
    def initialize(self) -> None:
        """Initialize the monitor."""
        self._start_monitoring()
        self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the monitor."""
        self._stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self._initialized = False
    
    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                self._collect_metrics()
                self._check_health()
                time.sleep(self.config.metrics_interval)
            except Exception as e:
                self.logger.log_error("Monitoring loop error", e)
                time.sleep(1.0)
    
    def _collect_metrics(self) -> None:
        """Collect system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Custom metrics
        uptime = time.time() - self.start_time
        request_rate = self.request_count / uptime if uptime > 0 else 0
        error_rate = self.error_count / max(self.request_count, 1)
        
        metrics = {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk.percent,
                'disk_free_mb': disk.free / (1024 * 1024)
            },
            'process': {
                'memory_rss_mb': process_memory.rss / (1024 * 1024),
                'memory_vms_mb': process_memory.vms / (1024 * 1024),
                'cpu_percent': process.cpu_percent()
            },
            'application': {
                'uptime_seconds': uptime,
                'request_count': self.request_count,
                'error_count': self.error_count,
                'request_rate': request_rate,
                'error_rate': error_rate,
                'health_status': self.health_status
            }
        }
        
        # Log metrics
        self.logger.log_metrics(metrics)
        
        # Store metrics
        self.metrics = metrics
    
    def _check_health(self) -> None:
        """Check system health."""
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > self.config.memory_threshold_mb / 100 * 100:
            self.health_status = "unhealthy"
            self.logger.log_warning(f"High memory usage: {memory.percent}%")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.config.cpu_threshold_percent:
            self.health_status = "unhealthy"
            self.logger.log_warning(f"High CPU usage: {cpu_percent}%")
        
        # Check circuit breaker
        if self.circuit_breaker_count > self.config.circuit_breaker_threshold:
            self.health_status = "circuit_breaker_open"
            self.logger.log_error("Circuit breaker opened due to high error rate")
    
    def record_request(self, success: bool = True) -> None:
        """Record a request."""
        self.request_count += 1
        if not success:
            self.error_count += 1
            self.circuit_breaker_count += 1
        else:
            self.circuit_breaker_count = max(0, self.circuit_breaker_count - 1)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            'status': self.health_status,
            'uptime': time.time() - self.start_time,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'circuit_breaker_count': self.circuit_breaker_count
        }

class ProductionErrorHandler(BaseService):
    """Production error handling system."""
    
    def __init__(self, config: ProductionConfig, logger: LoggerProtocol):
        super().__init__(config, logger)
        self.error_counts = {}
        self.last_error_time = {}
    
    def initialize(self) -> None:
        """Initialize the error handler."""
        self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the error handler."""
        self._initialized = False
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """Handle production errors."""
        error_type = type(error).__name__
        current_time = time.time()
        
        # Check if this is a new error or repeated
        if current_time - self.last_error_time.get(error_type, 0) > 60:  # 1 minute window
            self.error_counts[error_type] = 0
        
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_time[error_type] = current_time
        
        # Log error
        self.logger.log_error(
            f"Error in {context}",
            error,
            error_type=error_type,
            error_count=self.error_counts[error_type]
        )
        
        # Determine if we should retry
        if self.error_counts[error_type] <= self.config.max_retries:
            return True  # Retry
        
        return False  # Don't retry
    
    def should_circuit_break(self) -> bool:
        """Check if circuit breaker should be opened."""
        total_errors = sum(self.error_counts.values())
        return total_errors > self.config.circuit_breaker_threshold

class ProductionRequestQueue(BaseService):
    """Production request queue for handling concurrent requests."""
    
    def __init__(self, config: ProductionConfig, logger: LoggerProtocol):
        super().__init__(config, logger)
        self.request_queue = queue.Queue(maxsize=config.max_concurrent_requests)
        self.processing_pool = None
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
    
    def initialize(self) -> None:
        """Initialize the request queue."""
        self.processing_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        self._start_request_processing()
        self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the request queue."""
        if self.processing_pool:
            self.processing_pool.shutdown(wait=True)
        self._initialized = False
    
    def _start_request_processing(self) -> None:
        """Start request processing thread."""
        processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        processing_thread.start()
    
    def _process_requests(self) -> None:
        """Process requests from the queue."""
        while self._initialized:
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=1.0)
                
                # Process request
                self.processing_pool.submit(self._process_single_request, request)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error("Error processing request", e)
    
    def _process_single_request(self, request: Dict[str, Any]) -> None:
        """Process a single request."""
        request_id = request['request_id']
        start_time = time.time()
        
        try:
            self.active_requests += 1
            
            # Process request
            result = request['callback'](request['data'])
            
            # Record success
            self.completed_requests += 1
            processing_time = time.time() - start_time
            
            self.logger.log_info(
                f"Request completed: {request_id}",
                processing_time=processing_time,
                active_requests=self.active_requests
            )
            
        except Exception as e:
            # Record failure
            self.failed_requests += 1
            processing_time = time.time() - start_time
            
            self.logger.log_error(
                f"Request failed: {request_id}",
                e,
                processing_time=processing_time,
                active_requests=self.active_requests
            )
            
        finally:
            self.active_requests -= 1
    
    def submit_request(self, request_data: RequestData, callback: Callable) -> str:
        """Submit a request for processing."""
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Add to queue
            self.request_queue.put({
                'request_id': request_id,
                'data': request_data,
                'callback': callback,
                'timestamp': time.time()
            }, timeout=1.0)
            
            self.logger.log_info(f"Request submitted: {request_id}")
            return request_id
            
        except queue.Full:
            self.logger.log_error(f"Request queue full, rejecting request: {request_id}")
            raise RuntimeError("Request queue is full")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'queue_size': self.request_queue.qsize(),
            'active_requests': self.active_requests,
            'completed_requests': self.completed_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.completed_requests / max(self.completed_requests + self.failed_requests, 1)
        }

class RefactoredProductionPiMoESystem(BasePiMoESystem):
    """
    Refactored production PiMoE system with clean architecture.
    """
    
    def __init__(self, config: ProductionConfig):
        super().__init__(config)
        
        # Dependency injection container
        self.di_container = DIContainer()
        self._setup_dependencies()
        
        # Core components
        self.logger = self.di_container.get(LoggerProtocol)
        self.monitor = self.di_container.get(MonitorProtocol)
        self.error_handler = self.di_container.get(ErrorHandlerProtocol)
        self.request_queue = self.di_container.get(RequestQueueProtocol)
        
        # PiMoE system
        self.pimoe_system = None
        
        # Event bus for decoupled communication
        self.event_bus = EventBus()
        self._setup_event_handlers()
        
        # Resource manager
        self.resource_manager = ResourceManager(config)
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Health checker
        self.health_checker = HealthChecker()
        self._setup_health_checks()
    
    def _setup_dependencies(self) -> None:
        """Setup dependency injection container."""
        # Register logger
        self.di_container.register_instance(LoggerProtocol, ProductionLogger(self.config))
        
        # Register monitor
        logger = self.di_container.get(LoggerProtocol)
        self.di_container.register_instance(MonitorProtocol, ProductionMonitor(self.config, logger))
        
        # Register error handler
        self.di_container.register_instance(ErrorHandlerProtocol, ProductionErrorHandler(self.config, logger))
        
        # Register request queue
        self.di_container.register_instance(RequestQueueProtocol, ProductionRequestQueue(self.config, logger))
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers."""
        self.event_bus.subscribe('request_started', self._on_request_started)
        self.event_bus.subscribe('request_completed', self._on_request_completed)
        self.event_bus.subscribe('request_failed', self._on_request_failed)
        self.event_bus.subscribe('system_error', self._on_system_error)
    
    def _setup_health_checks(self) -> None:
        """Setup health checks."""
        self.health_checker.register_check('system_health', self._check_system_health)
        self.health_checker.register_check('memory_usage', self._check_memory_usage)
        self.health_checker.register_check('cpu_usage', self._check_cpu_usage)
        self.health_checker.register_check('pimoe_system', self._check_pimoe_system)
    
    def initialize(self) -> None:
        """Initialize the system."""
        try:
            # Initialize core components
            self.logger.initialize()
            self.monitor.initialize()
            self.error_handler.initialize()
            self.request_queue.initialize()
            
            # Initialize PiMoE system
            self._initialize_pimoe_system()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self._initialized = True
            self.logger.log_info("Refactored production PiMoE system initialized")
            
        except Exception as e:
            self.logger.log_error("Failed to initialize system", e)
            raise
    
    def _initialize_pimoe_system(self) -> None:
        """Initialize the PiMoE system."""
        from .advanced_pimoe_routing import AdvancedPiMoESystem, AdvancedRoutingConfig
        # Convert production config to advanced config
        advanced_config = AdvancedRoutingConfig(
            hidden_size=self.config.system_config.hidden_size,
            num_experts=self.config.system_config.num_experts,
            routing_strategy=RoutingStrategy.ATTENTION_BASED,
            optimization_level=OptimizationLevel.EXTREME
        )
        
        self.pimoe_system = AdvancedPiMoESystem(advanced_config)
        
        # Apply production optimizations
        self._apply_production_optimizations()
        
        # Register with resource manager
        self.resource_manager.register_resource(
            'pimoe_system',
            self.pimoe_system,
            lambda: self.pimoe_system.shutdown() if self.pimoe_system else None
        )
    
    def _apply_production_optimizations(self) -> None:
        """Apply production-specific optimizations."""
        # Optimize for inference
        self.pimoe_system.optimize_for_inference()
        
        # Apply quantization if enabled
        if self.config.enable_quantization:
            self._apply_quantization()
        
        # Apply pruning if enabled
        if self.config.enable_pruning:
            self._apply_pruning()
        
        # Set to evaluation mode
        self.pimoe_system.eval()
        
        # Disable gradients
        for param in self.pimoe_system.parameters():
            param.requires_grad = False
    
    def _apply_quantization(self) -> None:
        """Apply quantization for production."""
        try:
            # Apply dynamic quantization
            self.pimoe_system = torch.quantization.quantize_dynamic(
                self.pimoe_system,
                {nn.Linear, nn.MultiheadAttention},
                dtype=torch.qint8
            )
            self.logger.log_info("Quantization applied successfully")
        except Exception as e:
            self.logger.log_error("Failed to apply quantization", e)
    
    def _apply_pruning(self) -> None:
        """Apply pruning for production."""
        try:
            # Apply structured pruning
            for module in self.pimoe_system.modules():
                if isinstance(module, nn.Linear):
                    # Prune 10% of weights
                    weight = module.weight.data
                    threshold = torch.quantile(torch.abs(weight), 0.1)
                    mask = torch.abs(weight) > threshold
                    module.weight.data *= mask.float()
            
            self.logger.log_info("Pruning applied successfully")
        except Exception as e:
            self.logger.log_error("Failed to apply pruning", e)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.log_info(f"Received signal {signum}, shutting down gracefully")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def process_request(self, request_data: RequestData) -> ResponseData:
        """Process a production request."""
        start_time = time.time()
        request_id = request_data.get('request_id', 'unknown')
        
        # Publish request started event
        self.event_bus.publish(Event('request_started', {
            'request_id': request_id,
            'timestamp': start_time
        }))
        
        try:
            # Check circuit breaker
            if self.error_handler.should_circuit_break():
                raise RuntimeError("Circuit breaker is open")
            
            # Extract input tensor
            input_tensor = request_data['input_tensor']
            attention_mask = request_data.get('attention_mask', None)
            return_comprehensive_info = request_data.get('return_comprehensive_info', False)
            
            # Validate input
            self._validate_input(input_tensor)
            
            # Process request
            with torch.no_grad():
                if return_comprehensive_info:
                    output, comprehensive_info = self.pimoe_system.forward(
                        input_tensor, attention_mask, return_comprehensive_info=True
                    )
                else:
                    output = self.pimoe_system.forward(input_tensor, attention_mask)
                    comprehensive_info = None
            
            # Record success
            processing_time = time.time() - start_time
            self.monitor.record_request(success=True)
            self.metrics_collector.record_histogram('request_duration', processing_time)
            self.metrics_collector.increment_counter('requests_successful')
            
            # Publish request completed event
            self.event_bus.publish(Event('request_completed', {
                'request_id': request_id,
                'processing_time': processing_time,
                'success': True
            }))
            
            # Prepare response
            response = {
                'request_id': request_id,
                'output': output.cpu().numpy().tolist(),
                'processing_time': processing_time,
                'success': True
            }
            
            if comprehensive_info:
                response['comprehensive_info'] = comprehensive_info
            
            return response
            
        except Exception as e:
            # Record failure
            processing_time = time.time() - start_time
            self.monitor.record_request(success=False)
            self.metrics_collector.record_histogram('request_duration', processing_time)
            self.metrics_collector.increment_counter('requests_failed')
            
            # Handle error
            should_retry = self.error_handler.handle_error(e, f"Request {request_id}")
            
            # Publish request failed event
            self.event_bus.publish(Event('request_failed', {
                'request_id': request_id,
                'error': str(e),
                'processing_time': processing_time,
                'should_retry': should_retry
            }))
            
            # Return error response
            return {
                'request_id': request_id,
                'error': str(e),
                'processing_time': processing_time,
                'success': False,
                'should_retry': should_retry
            }
    
    def _validate_input(self, input_tensor: torch.Tensor) -> None:
        """Validate input tensor."""
        if not isinstance(input_tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if input_tensor.dim() != 3:
            raise ValueError("Input must be 3D tensor [batch, seq, hidden]")
        
        # Check batch size
        if input_tensor.size(0) > self.config.system_config.max_batch_size:
            raise ValueError(f"Batch size {input_tensor.size(0)} exceeds maximum {self.config.system_config.max_batch_size}")
        
        # Check sequence length
        if input_tensor.size(1) > self.config.system_config.max_sequence_length:
            raise ValueError(f"Sequence length {input_tensor.size(1)} exceeds maximum {self.config.system_config.max_sequence_length}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'system': self.pimoe_system.get_system_stats() if self.pimoe_system else {},
            'monitoring': self.monitor.get_health_status(),
            'queue': self.request_queue.get_queue_stats(),
            'metrics': self.metrics_collector.get_metrics(),
            'production_config': self.config.to_dict()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return self.health_checker.run_checks()
    
    def shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.log_info("Shutting down refactored production PiMoE system")
        
        # Shutdown components
        self.request_queue.shutdown()
        self.monitor.shutdown()
        self.error_handler.shutdown()
        self.logger.shutdown()
        
        # Cleanup resources
        self.resource_manager.cleanup_all()
        
        # Log final statistics
        final_stats = self.get_system_stats()
        self.logger.log_info("Final statistics", **final_stats)
        
        self.logger.log_info("Refactored production PiMoE system shutdown complete")
    
    # Event handlers
    def _on_request_started(self, event: Event) -> None:
        """Handle request started event."""
        self.metrics_collector.increment_counter('requests_started')
    
    def _on_request_completed(self, event: Event) -> None:
        """Handle request completed event."""
        self.logger.log_info(f"Request completed: {event.data['request_id']}")
    
    def _on_request_failed(self, event: Event) -> None:
        """Handle request failed event."""
        self.logger.log_error(f"Request failed: {event.data['request_id']}", 
                             error=event.data.get('error'))
    
    def _on_system_error(self, event: Event) -> None:
        """Handle system error event."""
        self.logger.log_error(f"System error: {event.data.get('error')}")
    
    # Health check methods
    def _check_system_health(self) -> bool:
        """Check overall system health."""
        return self.monitor.get_health_status()['status'] == 'healthy'
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        return memory.percent < self.config.memory_threshold_mb / 100 * 100
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent()
        return cpu_percent < self.config.cpu_threshold_percent
    
    def _check_pimoe_system(self) -> bool:
        """Check PiMoE system health."""
        if not self.pimoe_system:
            return False
        
        try:
            # Test model forward pass
            test_input = torch.randn(1, 10, self.config.system_config.hidden_size)
            with torch.no_grad():
                _ = self.pimoe_system.forward(test_input)
            return True
        except Exception:
            return False

# Factory Functions
def create_refactored_production_system(
    hidden_size: int = 512,
    num_experts: int = 8,
    production_mode: ProductionMode = ProductionMode.PRODUCTION,
    **kwargs
) -> RefactoredProductionPiMoESystem:
    """
    Factory function to create a refactored production PiMoE system.
    """
    # Create configuration
    system_config = SystemConfig(
        hidden_size=hidden_size,
        num_experts=num_experts
    )
    
    config = ProductionConfig(
        system_config=system_config,
        production_mode=production_mode,
        **kwargs
    )
    
    # Create system
    system = RefactoredProductionPiMoESystem(config)
    system.initialize()
    
    return system

def run_refactored_production_demo():
    """Run refactored production system demonstration."""
    print("🚀 Refactored Production PiMoE System Demo")
    print("=" * 60)
    
    # Create refactored production system
    system = create_refactored_production_system(
        hidden_size=512,
        num_experts=8,
        production_mode=ProductionMode.PRODUCTION,
        max_batch_size=16,
        max_sequence_length=1024,
        enable_monitoring=True,
        enable_metrics=True
    )
    
    print(f"📊 System Configuration:")
    print(f"  Hidden Size: {system.config.system_config.hidden_size}")
    print(f"  Number of Experts: {system.config.system_config.num_experts}")
    print(f"  Production Mode: {system.config.production_mode.value}")
    print(f"  Max Batch Size: {system.config.system_config.max_batch_size}")
    print(f"  Max Sequence Length: {system.config.system_config.max_sequence_length}")
    
    # Test request processing
    print(f"\n🔄 Testing Request Processing...")
    
    # Generate test data
    test_input = torch.randn(2, 128, 512)
    
    request_data = {
        'request_id': 'refactored_test_001',
        'input_tensor': test_input,
        'return_comprehensive_info': True
    }
    
    # Process request
    start_time = time.time()
    response = system.process_request(request_data)
    end_time = time.time()
    
    print(f"  Request ID: {response['request_id']}")
    print(f"  Success: {response['success']}")
    print(f"  Processing Time: {response['processing_time']:.4f} s")
    print(f"  Output Shape: {len(response['output'])} x {len(response['output'][0])} x {len(response['output'][0][0])}")
    
    # Test health check
    print(f"\n🏥 Testing Health Check...")
    health_status = system.health_check()
    print(f"  Overall Status: {health_status['overall_status']}")
    for check_name, check_result in health_status['checks'].items():
        print(f"  {check_name}: {check_result['status']}")
    
    # Test system statistics
    print(f"\n📈 System Statistics:")
    stats = system.get_system_stats()
    
    print(f"  System Stats:")
    print(f"    System Type: {stats['system'].get('system_type', 'Unknown')}")
    print(f"    Hidden Size: {stats['system'].get('hidden_size', 'Unknown')}")
    print(f"    Number of Experts: {stats['system'].get('num_experts', 'Unknown')}")
    
    print(f"  Monitoring Stats:")
    print(f"    Health Status: {stats['monitoring']['status']}")
    print(f"    Request Count: {stats['monitoring']['request_count']}")
    print(f"    Error Rate: {stats['monitoring']['error_rate']:.3f}")
    
    print(f"  Queue Stats:")
    print(f"    Queue Size: {stats['queue']['queue_size']}")
    print(f"    Active Requests: {stats['queue']['active_requests']}")
    print(f"    Success Rate: {stats['queue']['success_rate']:.3f}")
    
    print(f"  Metrics:")
    print(f"    Requests Started: {stats['metrics']['counters'].get('requests_started', 0)}")
    print(f"    Requests Successful: {stats['metrics']['counters'].get('requests_successful', 0)}")
    print(f"    Requests Failed: {stats['metrics']['counters'].get('requests_failed', 0)}")
    
    # Test multiple requests
    print(f"\n🔄 Testing Multiple Requests...")
    
    for i in range(5):
        request_data = {
            'request_id': f'refactored_test_{i:03d}',
            'input_tensor': torch.randn(1, 64, 512),
            'return_comprehensive_info': False
        }
        
        response = system.process_request(request_data)
        print(f"  Request {i+1}: {'✅' if response['success'] else '❌'} ({response['processing_time']:.4f}s)")
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    final_stats = system.get_system_stats()
    
    print(f"  Total Requests: {final_stats['monitoring']['request_count']}")
    print(f"  Total Errors: {final_stats['monitoring']['error_count']}")
    print(f"  Error Rate: {final_stats['monitoring']['error_rate']:.3f}")
    print(f"  Success Rate: {final_stats['queue']['success_rate']:.3f}")
    
    # Graceful shutdown
    print(f"\n🛑 Shutting Down...")
    system.shutdown()
    
    print(f"\n✅ Refactored production PiMoE demo completed successfully!")
    
    return system

if __name__ == "__main__":
    # Run refactored production demo
    system = run_refactored_production_demo()





