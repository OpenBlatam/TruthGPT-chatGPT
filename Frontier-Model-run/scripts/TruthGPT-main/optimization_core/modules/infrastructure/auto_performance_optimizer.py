"""
Enterprise TruthGPT Auto Performance Optimizer
Intelligent performance optimization with ML-driven tuning
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from pydantic import Field, ConfigDict
from datetime import datetime
from enum import Enum
import logging
import json

from ...utils.base import BaseOptimizationModel, system_metrics_collector

class OptimizationTarget(Enum):
    """Optimization target enum."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    ACCURACY = "accuracy"

class OptimizationStrategy(Enum):
    """Optimization strategy enum."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"

class OptimizationConfig(BaseOptimizationModel):
    """Optimization configuration."""
    
    target: OptimizationTarget = OptimizationTarget.THROUGHPUT
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    learning_rate: float = 0.1
    
    # Performance bounds
    min_latency_ms: float = 10.0
    max_latency_ms: float = 1000.0
    min_throughput_rps: float = 1.0
    max_memory_gb: float = 16.0
    min_accuracy: float = 0.8

class PerformanceMetrics(BaseOptimizationModel):
    """Performance metrics dataclass."""
    
    latency_ms: float
    throughput_rps: float
    memory_usage_gb: float
    cpu_usage_percent: float
    accuracy: float
    timestamp: datetime = Field(default_factory=datetime.now)

class OptimizationResult(BaseOptimizationModel):
    """Optimization result dataclass."""
    
    config: Dict[str, Any]
    metrics: PerformanceMetrics
    improvement_percent: float
    timestamp: datetime = Field(default_factory=datetime.now)

class AutoPerformanceOptimizer:
    """Enterprise auto performance optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance history
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        
        # Current configuration
        self.current_config: Dict[str, Any] = {}
        
        # Optimization state
        self.is_optimizing = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Initialize default configuration
        self._init_default_config()
    
    def _init_default_config(self):
        """Initialize default configuration."""
        self.current_config = {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_workers": 4,
            "cache_size": 1000,
            "quantization": False,
            "mixed_precision": True,
            "gradient_checkpointing": False,
            "attention_heads": 16,
            "hidden_size": 512,
            "num_layers": 12
        }
    
    def start_optimization(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start automatic optimization."""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        self._loop = loop or asyncio.get_event_loop()
        
        if self._loop.is_running():
            self._task = self._loop.create_task(self._optimization_loop())
        else:
            # If loop not running, we might need a different strategy or just warn
            self.logger.warning("Event loop is not running. Task created but not started.")
            self._task = self._loop.create_task(self._optimization_loop())
            
        self.logger.info("Auto optimization started (async)")
    
    async def stop_optimization(self):
        """Stop automatic optimization."""
        self.is_optimizing = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Auto optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop (async)."""
        iteration = 0
        
        while self.is_optimizing and iteration < self.config.max_iterations:
            try:
                # Measure current performance
                current_metrics = self._measure_performance()
                
                async with self._lock:
                    self.performance_history.append(current_metrics)
                
                # Check convergence
                if self._check_convergence():
                    self.logger.info("Optimization converged")
                    break
                
                # Generate new configuration
                new_config = self._generate_new_config(current_metrics)
                
                # Test new configuration
                test_metrics = self._test_configuration(new_config)
                
                # Evaluate improvement
                improvement = self._evaluate_improvement(current_metrics, test_metrics)
                
                # Update configuration if improvement is significant
                if improvement > self.config.convergence_threshold:
                    async with self._lock:
                        self.current_config = new_config
                        self.optimization_history.append(OptimizationResult(
                            config=new_config,
                            metrics=test_metrics,
                            improvement_percent=improvement
                        ))
                    self.logger.info(f"Configuration updated with {improvement:.2%} improvement")
                
                iteration += 1
                await asyncio.sleep(1)  # Non-blocking pause
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {str(e)}")
                break
    
    def _measure_performance(self) -> PerformanceMetrics:
        """Measure current performance."""
        # Use helper for system metrics
        sys_metrics = system_metrics_collector()
        
        latency_ms = self._simulate_latency()
        throughput_rps = self._simulate_throughput()
        
        return PerformanceMetrics(
            latency_ms=latency_ms,
            throughput_rps=throughput_rps,
            memory_usage_gb=sys_metrics["memory_used_gb"],
            cpu_usage_percent=sys_metrics["cpu_percent"],
            accuracy=self._simulate_accuracy()
        )
    
    def _simulate_latency(self) -> float:
        """Simulate latency measurement."""
        base_latency = 50.0
        batch_factor = 1.0 / self.current_config.get("batch_size", 32)
        quantization_factor = 0.8 if self.current_config.get("quantization", False) else 1.0
        mixed_precision_factor = 0.9 if self.current_config.get("mixed_precision", True) else 1.0
        
        return base_latency * batch_factor * quantization_factor * mixed_precision_factor
    
    def _simulate_throughput(self) -> float:
        """Simulate throughput measurement."""
        base_throughput = 100.0
        batch_factor = self.current_config.get("batch_size", 32) / 32.0
        workers_factor = self.current_config.get("num_workers", 4) / 4.0
        
        return base_throughput * batch_factor * workers_factor
    
    def _simulate_accuracy(self) -> float:
        """Simulate accuracy."""
        base_accuracy = 0.95
        quantization_penalty = 0.05 if self.current_config.get("quantization", False) else 0.0
        
        return max(base_accuracy - quantization_penalty, 0.0)
    
    # Mapping from OptimizationTarget enum values to PerformanceMetrics field names
    _TARGET_FIELD_MAP = {
        OptimizationTarget.LATENCY: 'latency_ms',
        OptimizationTarget.THROUGHPUT: 'throughput_rps',
        OptimizationTarget.MEMORY: 'memory_usage_gb',
        OptimizationTarget.CPU: 'cpu_usage_percent',
        OptimizationTarget.ACCURACY: 'accuracy',
    }

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.performance_history) < 5:
            return False
        
        # Check if performance has stabilized
        recent_metrics = self.performance_history[-5:]
        field_name = self._TARGET_FIELD_MAP.get(self.config.target, 'throughput_rps')
        target_values = [getattr(m, field_name) for m in recent_metrics]
        
        # Calculate variance
        mean_value = sum(target_values) / len(target_values)
        variance = sum((v - mean_value) ** 2 for v in target_values) / len(target_values)
        
        return variance < self.config.convergence_threshold
    
    def _generate_new_config(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate new configuration based on current metrics."""
        new_config = self.current_config.copy()
        
        # Adjust configuration based on target and strategy
        if self.config.target == OptimizationTarget.LATENCY:
            if current_metrics.latency_ms > self.config.max_latency_ms:
                # Increase batch size to reduce latency
                new_config["batch_size"] = min(new_config["batch_size"] * 2, 128)
                new_config["quantization"] = True
        elif self.config.target == OptimizationTarget.THROUGHPUT:
            if current_metrics.throughput_rps < self.config.min_throughput_rps:
                # Increase workers and batch size
                new_config["num_workers"] = min(new_config["num_workers"] + 1, 8)
                new_config["batch_size"] = min(new_config["batch_size"] + 8, 64)
        elif self.config.target == OptimizationTarget.MEMORY:
            if current_metrics.memory_usage_gb > self.config.max_memory_gb:
                # Reduce batch size and enable gradient checkpointing
                new_config["batch_size"] = max(new_config["batch_size"] // 2, 8)
                new_config["gradient_checkpointing"] = True
        
        # Apply strategy-specific adjustments
        if self.config.strategy == OptimizationStrategy.AGGRESSIVE:
            # More aggressive changes
            pass
        elif self.config.strategy == OptimizationStrategy.CONSERVATIVE:
            # More conservative changes
            pass
        
        return new_config
    
    def _test_configuration(self, config: Dict[str, Any]) -> PerformanceMetrics:
        """Test configuration and return metrics."""
        # Temporarily apply configuration
        old_config = self.current_config.copy()
        self.current_config = config
        
        # Measure performance
        metrics = self._measure_performance()
        
        # Restore old configuration
        self.current_config = old_config
        
        return metrics
    
    def _evaluate_improvement(self, current: PerformanceMetrics, test: PerformanceMetrics) -> float:
        """Evaluate improvement between configurations."""
        try:
            if self.config.target == OptimizationTarget.LATENCY:
                # Lower latency is better
                improvement = (current.latency_ms - test.latency_ms) / max(current.latency_ms, 1.0)
            elif self.config.target == OptimizationTarget.THROUGHPUT:
                # Higher throughput is better
                improvement = (test.throughput_rps - current.throughput_rps) / max(current.throughput_rps, 1.0)
            elif self.config.target == OptimizationTarget.MEMORY:
                # Lower memory usage is better
                improvement = (current.memory_usage_gb - test.memory_usage_gb) / max(current.memory_usage_gb, 0.1)
            elif self.config.target == OptimizationTarget.CPU:
                # Lower CPU usage is better
                improvement = (current.cpu_usage_percent - test.cpu_usage_percent) / max(current.cpu_usage_percent, 1.0)
            elif self.config.target == OptimizationTarget.ACCURACY:
                # Higher accuracy is better
                improvement = (test.accuracy - current.accuracy) / max(current.accuracy, 0.1)
            else:
                improvement = 0.0
            
            return float(improvement)
        except ZeroDivisionError:
            return 0.0
    
    async def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        async with self._lock:
            return self.current_config.copy()
    
    async def get_performance_history(self, limit: int = 100) -> List[PerformanceMetrics]:
        """Get performance history."""
        async with self._lock:
            return self.performance_history[-limit:]
    
    async def get_optimization_history(self, limit: int = 50) -> List[OptimizationResult]:
        """Get optimization history."""
        async with self._lock:
            return self.optimization_history[-limit:]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        async with self._lock:
            if not self.performance_history:
                return {"status": "No performance data available"}
            
            latest_metrics = self.performance_history[-1]
            total_optimizations = len(self.optimization_history)
            current_config = self.current_config.copy()
        
        return {
            "is_optimizing": self.is_optimizing,
            "total_iterations": len(self.performance_history),
            "total_optimizations": total_optimizations,
            "current_config": current_config,
            "latest_metrics": {
                "latency_ms": latest_metrics.latency_ms,
                "throughput_rps": latest_metrics.throughput_rps,
                "memory_usage_gb": latest_metrics.memory_usage_gb,
                "cpu_usage_percent": latest_metrics.cpu_usage_percent,
                "accuracy": latest_metrics.accuracy
            },
            "target": self.config.target.value,
            "strategy": self.config.strategy.value
        }

# Example usage
async def main():
    # Create optimizer
    config = OptimizationConfig(
        target=OptimizationTarget.THROUGHPUT,
        strategy=OptimizationStrategy.BALANCED,
        max_iterations=20
    )
    
    optimizer = AutoPerformanceOptimizer(config)
    
    # Start optimization
    optimizer.start_optimization()
    
    try:
        # Let it run for a bit
        await asyncio.sleep(5)
        
        # Get current stats
        stats = await optimizer.get_stats()
        print("Optimization Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get performance history
        history = await optimizer.get_performance_history(limit=5)
        print(f"\nRecent Performance History ({len(history)} entries):")
        for i, metrics in enumerate(history):
            print(f"  {i+1}. Latency: {metrics.latency_ms:.1f}ms, "
                  f"Throughput: {metrics.throughput_rps:.1f}rps, "
                  f"Memory: {metrics.memory_usage_gb:.1f}GB")
    
    finally:
        await optimizer.stop_optimization()
    
    print("\nOptimization completed")

if __name__ == "__main__":
    asyncio.run(main())


