"""
Main Framework Architecture
===========================

The central orchestrator that coordinates all framework components.
Implements the main interface for the optimization framework.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .factory import OptimizerFactory
from .container import DependencyContainer
from .config import UnifiedConfig
from .monitoring import MetricsCollector
from .caching import CacheManager


class FrameworkState(Enum):
    """Framework operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class OptimizationRequest:
    """Request for optimization task"""
    task_id: str
    model_type: str
    data: Any
    config: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None
    callback: Optional[callable] = None


@dataclass
class OptimizationResult:
    """Result of optimization task"""
    task_id: str
    success: bool
    result: Any
    metrics: Dict[str, float]
    execution_time: float
    error: Optional[str] = None


class OptimizationFramework:
    """
    Main framework orchestrator that coordinates all components.
    
    Features:
    - Unified interface for all optimization tasks
    - Async processing with priority queues
    - Automatic resource management
    - Built-in monitoring and metrics
    - Intelligent caching
    - Plugin system support
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the framework with optional config file"""
        self.logger = logging.getLogger(__name__)
        self.state = FrameworkState.INITIALIZING
        
        # Initialize core components
        self.config = UnifiedConfig(config_path)
        self.container = DependencyContainer()
        self.factory = OptimizerFactory(self.container)
        self.metrics = MetricsCollector()
        self.cache = CacheManager()
        
        # Task management
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, OptimizationResult] = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_processes)
        self.loop = None
        self.loop_thread = None
        
        # Plugin system
        self.plugins: Dict[str, Any] = {}
        
        # Initialize framework
        self._initialize_framework()
    
    def _initialize_framework(self):
        """Initialize framework components"""
        try:
            self.logger.info("Initializing TruthGPT Optimization Framework...")
            
            # Register core services in container
            self.container.register('config', self.config)
            self.container.register('metrics', self.metrics)
            self.container.register('cache', self.cache)
            self.container.register('factory', self.factory)
            
            # Initialize async loop
            self._start_async_loop()
            
            # Load plugins
            self._load_plugins()
            
            self.state = FrameworkState.READY
            self.logger.info("Framework initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize framework: {e}")
            self.state = FrameworkState.ERROR
            raise
    
    def _start_async_loop(self):
        """Start async event loop in separate thread"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
    
    def _load_plugins(self):
        """Load available plugins"""
        plugin_path = self.config.plugin_path
        if plugin_path and plugin_path.exists():
            # Load plugins from directory
            for plugin_file in plugin_path.glob("*.py"):
                try:
                    plugin_name = plugin_file.stem
                    # Import and register plugin
                    self.logger.info(f"Loading plugin: {plugin_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load plugin {plugin_file}: {e}")
    
    async def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Main optimization method - async processing with priority
        
        Args:
            request: Optimization request with task details
            
        Returns:
            OptimizationResult with results and metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing optimization request: {request.task_id}")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for task: {request.task_id}")
                return cached_result
            
            # Get optimizer from factory
            optimizer = await self.factory.create_optimizer(
                request.model_type,
                request.config
            )
            
            # Execute optimization
            result = await self._execute_optimization(optimizer, request)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            metrics = self.metrics.collect_metrics(optimizer, execution_time)
            
            # Create result
            optimization_result = OptimizationResult(
                task_id=request.task_id,
                success=True,
                result=result,
                metrics=metrics,
                execution_time=execution_time
            )
            
            # Cache result
            await self.cache.set(cache_key, optimization_result)
            
            # Store result
            self.task_results[request.task_id] = optimization_result
            
            self.logger.info(f"Optimization completed: {request.task_id}")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Optimization failed for {request.task_id}: {e}")
            
            return OptimizationResult(
                task_id=request.task_id,
                success=False,
                result=None,
                metrics={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _execute_optimization(self, optimizer, request: OptimizationRequest):
        """Execute the actual optimization"""
        # Run in thread pool for CPU-intensive tasks
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            optimizer.optimize,
            request.data
        )
        return result
    
    def _generate_cache_key(self, request: OptimizationRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        key_data = f"{request.model_type}_{request.config}_{request.data}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def submit_task(self, request: OptimizationRequest) -> str:
        """Submit task to queue for async processing"""
        await self.task_queue.put((request.priority, request))
        
        # Start processing if not already running
        if self.state == FrameworkState.READY:
            self.state = FrameworkState.RUNNING
            asyncio.create_task(self._process_task_queue())
        
        return request.task_id
    
    async def _process_task_queue(self):
        """Process tasks from queue"""
        while not self.task_queue.empty():
            try:
                priority, request = await self.task_queue.get()
                
                # Create task
                task = asyncio.create_task(self.optimize(request))
                self.active_tasks[request.task_id] = task
                
                # Wait for completion
                result = await task
                
                # Call callback if provided
                if request.callback:
                    request.callback(result)
                
                # Clean up
                del self.active_tasks[request.task_id]
                
            except Exception as e:
                self.logger.error(f"Task processing error: {e}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of specific task"""
        if task_id in self.active_tasks:
            return {
                'status': 'running',
                'task_id': task_id
            }
        elif task_id in self.task_results:
            result = self.task_results[task_id]
            return {
                'status': 'completed',
                'task_id': task_id,
                'success': result.success,
                'execution_time': result.execution_time,
                'metrics': result.metrics
            }
        else:
            return {
                'status': 'not_found',
                'task_id': task_id
            }
    
    def get_framework_metrics(self) -> Dict[str, Any]:
        """Get overall framework metrics"""
        return {
            'state': self.state.value,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.task_results),
            'queue_size': self.task_queue.qsize(),
            'cache_hit_rate': self.cache.get_hit_rate(),
            'memory_usage': self.metrics.get_memory_usage(),
            'cpu_usage': self.metrics.get_cpu_usage()
        }
    
    async def shutdown(self):
        """Gracefully shutdown framework"""
        self.logger.info("Shutting down framework...")
        self.state = FrameworkState.STOPPING
        
        # Wait for active tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        # Stop event loop
        if self.loop:
            self.loop.stop()
        
        self.state = FrameworkState.READY
        self.logger.info("Framework shutdown complete")
    
    def register_plugin(self, name: str, plugin: Any):
        """Register a plugin"""
        self.plugins[name] = plugin
        self.logger.info(f"Plugin registered: {name}")
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """Get registered plugin"""
        return self.plugins.get(name)



