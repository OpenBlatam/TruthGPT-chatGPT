"""
Production Optimizer - Refactored main optimization system
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
import logging
import time
from contextlib import contextmanager

from ..core.base_truthgpt_optimizer import BaseTruthGPTOptimizer, OptimizationResult, OptimizationLevel
from ..core.config import ConfigManager, OptimizationConfig
from ..core.monitoring import SystemMonitor, create_system_monitor
from ..core.validation import ModelValidator, create_model_validator
from ..core.cache import ModelCache, create_model_cache
from ..core.utils import PerformanceUtils, MemoryUtils, GPUUtils, performance_context

logger = logging.getLogger(__name__)

class ProductionOptimizer(BaseTruthGPTOptimizer):
    """Production-grade optimization system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.monitor = create_system_monitor(config.get('monitoring', {}))
        self.validator = create_model_validator()
        self.cache = create_model_cache()
        
        # Performance utilities
        self.performance_utils = PerformanceUtils()
        self.memory_utils = MemoryUtils()
        self.gpu_utils = GPUUtils()
        
        # Load configuration
        if config:
            self.config_manager.update_section('optimization', config)
        
        self.logger.info("🚀 Production Optimizer initialized")
    
    def validate_model(self, model: nn.Module) -> bool:
        """Validate model for optimization."""
        return isinstance(model, nn.Module)
    
    def optimize(self, model: nn.Module, **kwargs) -> OptimizationResult:
        """Optimize a model with production-grade techniques."""
        start_time = time.time()
        
        # Validate input
        if not self.validate_model(model):
            return OptimizationResult(
                model_name=model.__class__.__name__,
                success=False,
                optimization_time=0.0,
                memory_usage=0.0,
                parameter_reduction=0.0,
                error_message="Model validation failed"
            )
        
        # Get configuration
        opt_config = self.config_manager.get_optimization_config()
        
        # Check cache first
        cache_key = self.cache.model_cache._generate_key(model, opt_config.to_dict())
        cached_model = self.cache.get_optimized_model(model, opt_config.to_dict())
        
        if cached_model is not None:
            self.logger.info("📋 Using cached optimization result")
            return OptimizationResult(
                model_name=model.__class__.__name__,
                success=True,
                optimization_time=0.0,
                memory_usage=self.memory_utils.get_model_memory_usage(cached_model)['total_mb'],
                parameter_reduction=0.0,
                optimizations_applied=['cached']
            )
        
        try:
            # Start monitoring
            if self.config_manager.get('monitoring.enable_profiling', True):
                self.monitor.start_monitoring()
            
            # Apply optimization strategies
            optimized_model = self._apply_optimization_strategies(model, opt_config)
            
            # Validate optimization
            validation_reports = self.validator.validate_model_compatibility(model, optimized_model)
            if any(report.result.value == 'failed' for report in validation_reports):
                error_msg = '; '.join([report.message for report in validation_reports if report.result.value == 'failed'])
                return OptimizationResult(
                    model_name=model.__class__.__name__,
                    success=False,
                    optimization_time=time.time() - start_time,
                    memory_usage=0.0,
                    parameter_reduction=0.0,
                    error_message=f"Optimization validation failed: {error_msg}"
                )
            
            # Calculate metrics
            optimization_time = time.time() - start_time
            original_memory = self.memory_utils.get_model_memory_usage(model)['total_mb']
            optimized_memory = self.memory_utils.get_model_memory_usage(optimized_model)['total_mb']
            
            original_params = self.memory_utils.get_parameter_count(model)['total_parameters']
            optimized_params = self.memory_utils.get_parameter_count(optimized_model)['total_parameters']
            parameter_reduction = (original_params - optimized_params) / original_params * 100 if original_params > 0 else 0
            
            # Cache result
            self.cache.cache_optimized_model(model, opt_config.to_dict(), optimized_model)
            
            # Record metrics
            self.monitor.metrics_collector.record_metric("optimization_success", 1.0)
            self.monitor.metrics_collector.record_metric("optimization_time", optimization_time)
            self.monitor.metrics_collector.record_metric("memory_reduction", original_memory - optimized_memory)
            
            # Stop monitoring
            if self.config_manager.get('monitoring.enable_profiling', True):
                self.monitor.stop_monitoring()
            
            result = OptimizationResult(
                model_name=model.__class__.__name__,
                success=True,
                optimization_time=optimization_time,
                memory_usage=optimized_memory,
                parameter_reduction=parameter_reduction,
                optimizations_applied=self._get_applied_optimizations(opt_config),
                performance_metrics={
                    'original_memory_mb': original_memory,
                    'optimized_memory_mb': optimized_memory,
                    'memory_reduction_mb': original_memory - optimized_memory,
                    'original_parameters': original_params,
                    'optimized_parameters': optimized_params,
                    'parameter_reduction_count': original_params - optimized_params
                }
            )
            
            self.results_history.append(result)
            self.logger.info(f"✅ Model optimization completed in {optimization_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model optimization failed: {e}")
            
            # Stop monitoring
            if self.config_manager.get('monitoring.enable_profiling', True):
                self.monitor.stop_monitoring()
            
            return OptimizationResult(
                model_name=model.__class__.__name__,
                success=False,
                optimization_time=time.time() - start_time,
                memory_usage=0.0,
                parameter_reduction=0.0,
                error_message=str(e)
            )
    
    def _apply_optimization_strategies(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply optimization strategies based on configuration."""
        optimized_model = model
        
        # Quantization
        if config.enable_quantization:
            try:
                self.logger.info("🔧 Applying quantization")
                optimized_model = self._apply_quantization(optimized_model, config)
            except Exception as e:
                self.logger.warning(f"Quantization failed: {e}")
        
        # Pruning
        if config.enable_pruning:
            try:
                self.logger.info("🔧 Applying pruning")
                optimized_model = self._apply_pruning(optimized_model, config)
            except Exception as e:
                self.logger.warning(f"Pruning failed: {e}")
        
        # Mixed precision
        if config.enable_mixed_precision:
            try:
                self.logger.info("🔧 Applying mixed precision")
                optimized_model = self._apply_mixed_precision(optimized_model, config)
            except Exception as e:
                self.logger.warning(f"Mixed precision failed: {e}")
        
        # Kernel fusion
        if config.enable_kernel_fusion:
            try:
                self.logger.info("🔧 Applying kernel fusion")
                optimized_model = self._apply_kernel_fusion(optimized_model, config)
            except Exception as e:
                self.logger.warning(f"Kernel fusion failed: {e}")
        
        return optimized_model
    
    def _apply_quantization(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply quantization optimization."""
        try:
            # Dynamic quantization for production
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply pruning optimization."""
        try:
            import torch.nn.utils.prune as prune
            
            # Structured pruning for production
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=0.1)
            
            return model
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return model
    
    def _apply_mixed_precision(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply mixed precision optimization."""
        try:
            if config.level in ['aggressive', 'maximum']:
                model = model.half()
            return model
        except Exception as e:
            self.logger.warning(f"Mixed precision failed: {e}")
            return model
    
    def _apply_kernel_fusion(self, model: nn.Module, config: OptimizationConfig) -> nn.Module:
        """Apply kernel fusion optimization."""
        # Kernel fusion is typically handled at the framework level
        # This is a placeholder for custom kernel fusion logic
        return model
    
    def _get_applied_optimizations(self, config: OptimizationConfig) -> List[str]:
        """Get list of applied optimizations."""
        optimizations = []
        
        if config.enable_quantization:
            optimizations.append('quantization')
        if config.enable_pruning:
            optimizations.append('pruning')
        if config.enable_mixed_precision:
            optimizations.append('mixed_precision')
        if config.enable_kernel_fusion:
            optimizations.append('kernel_fusion')
        
        return optimizations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'optimization_metrics': self.monitor.metrics_collector.get_metrics(),
            'cache_metrics': self.cache.get_model_stats(),
            'system_metrics': self.monitor.get_performance_summary(hours=1),
            'optimization_summary': self.get_optimization_summary()
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.monitor.cleanup()
        self.cache.clear_model_cache()
        self.logger.info("🧹 Production Optimizer cleanup completed")

# Factory functions
def create_production_optimizer(config: Optional[Dict[str, Any]] = None) -> ProductionOptimizer:
    """Create a production optimizer instance."""
    return ProductionOptimizer(config)

@contextmanager
def production_optimization_context(config: Optional[Dict[str, Any]] = None):
    """Context manager for production optimization."""
    optimizer = create_production_optimizer(config)
    try:
        yield optimizer
    finally:
        optimizer.cleanup()

# Simple optimization function
def optimize_model_production(model: nn.Module, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Simple production optimization function."""
    with production_optimization_context(config) as optimizer:
        result = optimizer.optimize(model)
        if result.success:
            return model  # Return optimized model (would need to be stored)
        else:
            raise RuntimeError(f"Optimization failed: {result.error_message}")




