"""
Base Optimizer Module
Abstract base class and interfaces for all optimization strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import time
import logging
import psutil
import gc
import numpy as np
from contextlib import contextmanager

class OptimizationType(Enum):
    """Optimization type classifications."""
    MEMORY = "memory"
    COMPUTATIONAL = "computational"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    PARALLEL = "parallel"
    CACHE = "cache"
    HARDWARE = "hardware"
    MIXED_PRECISION = "mixed_precision"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"

class OptimizationLevel(Enum):
    """Optimization level classifications."""
    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

@dataclass
class OptimizationResult:
    """Result of optimization operation."""
    optimization_type: str
    success: bool
    performance_gain: float
    memory_saved: float
    processing_time: float
    optimization_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class OptimizerConfig:
    """Base optimizer configuration."""
    optimization_type: OptimizationType
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    target_model: Optional[nn.Module] = None
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    max_optimization_time: float = 300.0  # 5 minutes
    memory_threshold_mb: float = 8000.0
    cpu_threshold_percent: float = 80.0
    gpu_threshold_percent: float = 90.0
    enable_auto_optimization: bool = True
    optimization_schedule: str = "on_demand"  # on_demand, periodic, continuous
    optimization_interval: float = 3600.0  # 1 hour

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization strategies.
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.metrics = {}
        self.cache = {} if config.enable_caching else None
        self._initialized = False
        self._optimization_count = 0
        self._total_optimization_time = 0.0
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the optimizer."""
        pass
    
    @abstractmethod
    def optimize(
        self, 
        model: nn.Module,
        input_data: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize the model.
        
        Args:
            model: Model to optimize
            input_data: Optional input data for optimization
            context: Optional optimization context
            
        Returns:
            OptimizationResult with optimization details
        """
        pass
    
    @abstractmethod
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Get optimizer information and statistics."""
        pass
    
    def validate_model(self, model: nn.Module) -> None:
        """Validate model for optimization."""
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a torch.nn.Module")
        
        if not model.training and not hasattr(model, 'eval'):
            raise ValueError("Model must be in training or evaluation mode")
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        return {
            'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory_mb': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
            'gpu_memory_percent': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.is_available() else 0
        }
    
    def check_optimization_thresholds(self) -> bool:
        """Check if optimization thresholds are met."""
        resources = self.get_system_resources()
        
        # Check memory threshold
        if resources['memory_usage_mb'] > self.config.memory_threshold_mb:
            return True
        
        # Check CPU threshold
        if resources['cpu_percent'] > self.config.cpu_threshold_percent:
            return True
        
        # Check GPU threshold
        if resources['gpu_available'] and resources['gpu_memory_percent'] > self.config.gpu_threshold_percent:
            return True
        
        return False
    
    def get_cache_key(self, model: nn.Module, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for model optimization."""
        if not self.cache:
            return None
        
        # Create hash from model parameters and context
        model_hash = hash(str([p.data.tobytes() for p in model.parameters()]))
        context_hash = hash(str(context)) if context else 0
        return f"{model_hash}_{context_hash}"
    
    def get_cached_result(self, cache_key: str) -> Optional[OptimizationResult]:
        """Get cached optimization result."""
        if not self.cache or cache_key not in self.cache:
            return None
        
        cached_result, timestamp = self.cache[cache_key]
        
        # Check cache expiry (1 hour)
        if time.time() - timestamp > 3600:
            del self.cache[cache_key]
            return None
        
        return cached_result
    
    def cache_result(self, cache_key: str, result: OptimizationResult) -> None:
        """Cache optimization result."""
        if not self.cache:
            return
        
        # Implement LRU cache
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = (result, time.time())
    
    def record_metrics(self, result: OptimizationResult) -> None:
        """Record optimization metrics."""
        if not self.config.enable_metrics:
            return
        
        self.metrics.setdefault('optimization_time', []).append(result.optimization_time)
        self.metrics.setdefault('performance_gain', []).append(result.performance_gain)
        self.metrics.setdefault('memory_saved', []).append(result.memory_saved)
        self.metrics.setdefault('success_rate', []).append(1.0 if result.success else 0.0)
        
        # Update optimization statistics
        self._optimization_count += 1
        self._total_optimization_time += result.optimization_time
        
        # Keep only recent metrics
        max_metrics = 1000
        for key in self.metrics:
            if len(self.metrics[key]) > max_metrics:
                self.metrics[key] = self.metrics[key][-max_metrics:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics."""
        metrics = {
            'optimization_count': self._optimization_count,
            'total_optimization_time': self._total_optimization_time,
            'average_optimization_time': self._total_optimization_time / max(self._optimization_count, 1),
            'cache_size': len(self.cache) if self.cache else 0
        }
        
        if self.metrics:
            for key, values in self.metrics.items():
                if values:
                    metrics[f'{key}_mean'] = np.mean(values)
                    metrics[f'{key}_std'] = np.std(values)
                    metrics[f'{key}_min'] = np.min(values)
                    metrics[f'{key}_max'] = np.max(values)
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset optimization metrics."""
        self.metrics.clear()
        self._optimization_count = 0
        self._total_optimization_time = 0.0
    
    def log_optimization(self, result: OptimizationResult, model_name: str) -> None:
        """Log optimization information."""
        if not self.config.enable_logging:
            return
        
        self.logger.info(
            f"Optimization completed: "
            f"type={result.optimization_type}, "
            f"success={result.success}, "
            f"performance_gain={result.performance_gain:.3f}, "
            f"memory_saved={result.memory_saved:.2f}MB, "
            f"time={result.optimization_time:.4f}s, "
            f"model={model_name}"
        )
    
    @contextmanager
    def optimization_context(self, model: nn.Module):
        """Context manager for optimization operations."""
        original_training = model.training
        
        try:
            # Set model to evaluation mode for optimization
            model.eval()
            
            # Clear cache if needed
            if hasattr(model, 'clear_cache'):
                model.clear_cache()
            
            yield model
            
        finally:
            # Restore original training mode
            if original_training:
                model.train()
    
    def cleanup_memory(self) -> None:
        """Clean up memory after optimization."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @property
    def is_initialized(self) -> bool:
        """Check if optimizer is initialized."""
        return self._initialized
    
    def shutdown(self) -> None:
        """Shutdown the optimizer."""
        self.cache.clear() if self.cache else None
        self.reset_metrics()
        self._initialized = False
        self.logger.info(f"Optimizer {self.__class__.__name__} shutdown")

class OptimizationProfiler:
    """Profiler for optimization operations."""
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
    
    def start_profile(self, name: str) -> None:
        """Start profiling an operation."""
        self.current_profile = {
            'name': name,
            'start_time': time.time(),
            'start_memory': psutil.virtual_memory().used / (1024 * 1024),
            'start_gpu_memory': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        }
    
    def end_profile(self) -> Dict[str, Any]:
        """End profiling and return results."""
        if not self.current_profile:
            return {}
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024 * 1024)
        end_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        
        profile = {
            'name': self.current_profile['name'],
            'duration': end_time - self.current_profile['start_time'],
            'memory_used': end_memory - self.current_profile['start_memory'],
            'gpu_memory_used': end_gpu_memory - self.current_profile['start_gpu_memory'],
            'peak_memory': psutil.virtual_memory().used / (1024 * 1024),
            'peak_gpu_memory': torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        }
        
        self.profiles[self.current_profile['name']] = profile
        self.current_profile = None
        
        return profile
    
    def get_profiles(self) -> Dict[str, Any]:
        """Get all profiling results."""
        return self.profiles.copy()
    
    def clear_profiles(self) -> None:
        """Clear all profiling results."""
        self.profiles.clear()





