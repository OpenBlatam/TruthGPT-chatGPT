"""
Performance Analyzer - Advanced performance analysis and profiling
Provides comprehensive performance analysis, profiling, and optimization recommendations
"""

import torch
import torch.nn as nn
import time
import psutil
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pydantic import BaseModel, Field, ConfigDict
from collections import defaultdict, deque
import numpy as np
import json
from pathlib import Path
from contextlib import contextmanager
import gc
import tracemalloc
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ProfilingMode(Enum):
    """Profiling modes."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    COMPREHENSIVE = "comprehensive"

class PerformanceLevel(Enum):
    """Performance levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


class PerformanceProfile(BaseModel):
    """Performance profile of a model or system (Strict Pydantic Architecture)."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(..., description="Name of the profiled model or system")
    timestamp: float = Field(..., description="UNIX timestamp of the profile")
    cpu_usage: float = Field(..., description="CPU utilization percentage")
    memory_usage: float = Field(..., description="System RAM utilization percentage")
    gpu_memory_usage: float = Field(..., description="GPU VRAM utilization percentage")
    gpu_utilization: float = Field(..., description="GPU core utilization percentage")
    inference_time: float = Field(..., description="Time taken for a single forward pass")
    throughput: float = Field(..., description="Inferences per second")
    memory_efficiency: float = Field(..., description="Calculated memory efficiency score")
    energy_efficiency: float = Field(..., description="Estimated energy efficiency score")
    scalability_score: float = Field(..., description="Score calculating system scalability")
    bottlenecks: List[str] = Field(default_factory=list, description="List of identified bottleneck codes")
    recommendations: List[str] = Field(default_factory=list, description="Actionable optimization recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary for arbitrary parameters")


class BottleneckAnalysis(BaseModel):
    """Bottleneck analysis results (Strict Pydantic Architecture)."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    bottleneck_type: str = Field(..., description="Category of bottleneck (cpu, memory, gpu_memory, etc)")
    severity: PerformanceLevel = Field(..., description="Severity classification of the bottleneck")
    location: str = Field(..., description="Hardware component or module bottleneck location")
    impact: float = Field(..., description="Quantified impact metric on overall performance")
    description: str = Field(..., description="Human-readable description of the bottleneck")
    recommendations: List[str] = Field(default_factory=list, description="Actionable optimizations for this specific bottleneck")

class PerformanceProfiler:
    """Advanced performance profiler."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.profiling_data = deque(maxlen=10000)
        self.bottleneck_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.is_profiling = False
        self.profiling_thread = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'gpu_memory_usage': 90.0,
            'inference_time': 1.0,  # seconds
            'throughput': 100.0  # samples/second
        }
    
    def start_profiling(self, mode: ProfilingMode = ProfilingMode.COMPREHENSIVE, 
                       interval: float = 0.1):
        """Start performance profiling."""
        if self.is_profiling:
            self.logger.warning("Profiling already started")
            return
        
        self.is_profiling = True
        self.profiling_mode = mode
        self.profiling_thread = threading.Thread(
            target=self._profiling_loop,
            args=(interval,),
            daemon=True
        )
        self.profiling_thread.start()
        
        self.logger.info(f"🔍 Started {mode.value} profiling with {interval}s interval")
    
    def stop_profiling(self):
        """Stop performance profiling."""
        self.is_profiling = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=5.0)
        
        self.logger.info("🛑 Profiling stopped")
    
    def _profiling_loop(self, interval: float):
        """Main profiling loop."""
        while self.is_profiling:
            try:
                profile = self._collect_performance_profile()
                
                with self.lock:
                    self.profiling_data.append(profile)
                
                # Analyze for bottlenecks
                bottlenecks = self._analyze_bottlenecks(profile)
                if bottlenecks:
                    self.bottleneck_history.extend(bottlenecks)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in profiling loop: {e}")
                time.sleep(interval)
    
    def _collect_performance_profile(self) -> PerformanceProfile:
        """Collect comprehensive performance profile."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics
        gpu_memory_usage = 0.0
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        
        return PerformanceProfile(
            model_name="system",
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            gpu_utilization=gpu_utilization,
            inference_time=0.0,  # Will be updated during model profiling
            throughput=0.0,  # Will be updated during model profiling
            memory_efficiency=0.0,  # Will be calculated
            energy_efficiency=0.0,  # Will be calculated
            scalability_score=0.0  # Will be calculated
        )
    
    def _analyze_bottlenecks(self, profile: PerformanceProfile) -> List[BottleneckAnalysis]:
        """Analyze performance profile for bottlenecks."""
        bottlenecks = []
        
        # CPU bottleneck
        if profile.cpu_usage > self.thresholds['cpu_usage']:
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type="cpu",
                severity=PerformanceLevel.CRITICAL if profile.cpu_usage > 95 else PerformanceLevel.POOR,
                location="system",
                impact=profile.cpu_usage / 100.0,
                description=f"High CPU usage: {profile.cpu_usage:.1f}%",
                recommendations=[
                    "Reduce model complexity",
                    "Use model quantization",
                    "Implement batch processing",
                    "Consider model pruning"
                ]
            ))
        
        # Memory bottleneck
        if profile.memory_usage > self.thresholds['memory_usage']:
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type="memory",
                severity=PerformanceLevel.CRITICAL if profile.memory_usage > 95 else PerformanceLevel.POOR,
                location="system",
                impact=profile.memory_usage / 100.0,
                description=f"High memory usage: {profile.memory_usage:.1f}%",
                recommendations=[
                    "Reduce batch size",
                    "Use gradient checkpointing",
                    "Implement memory pooling",
                    "Consider model compression"
                ]
            ))
        
        # GPU memory bottleneck
        if profile.gpu_memory_usage > self.thresholds['gpu_memory_usage']:
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type="gpu_memory",
                severity=PerformanceLevel.CRITICAL if profile.gpu_memory_usage > 95 else PerformanceLevel.POOR,
                location="gpu",
                impact=profile.gpu_memory_usage / 100.0,
                description=f"High GPU memory usage: {profile.gpu_memory_usage:.1f}%",
                recommendations=[
                    "Reduce model size",
                    "Use mixed precision training",
                    "Implement gradient accumulation",
                    "Consider model sharding"
                ]
            ))
        
        return bottlenecks
    
    def profile_model(self, model: nn.Module, 
                     test_inputs: List[torch.Tensor],
                     warmup_iterations: int = 10,
                     benchmark_iterations: int = 100) -> PerformanceProfile:
        """Profile a specific model."""
        self.logger.info(f"🔍 Profiling model: {model.__class__.__name__}")
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(warmup_iterations):
                for test_input in test_inputs:
                    _ = model(test_input)
        
        # Benchmark
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(benchmark_iterations):
                start_time = time.time()
                start_memory = psutil.virtual_memory().used
                
                for test_input in test_inputs:
                    _ = model(test_input)
                
                inference_time = time.time() - start_time
                end_memory = psutil.virtual_memory().used
                
                inference_times.append(inference_time)
                memory_usage.append((end_memory - start_memory) / (1024 * 1024))  # MB
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        avg_memory_usage = np.mean(memory_usage)
        throughput = len(test_inputs) / avg_inference_time if avg_inference_time > 0 else 0
        
        # Calculate efficiency scores
        memory_efficiency = self._calculate_memory_efficiency(model, avg_memory_usage)
        energy_efficiency = self._calculate_energy_efficiency(avg_inference_time, avg_memory_usage)
        scalability_score = self._calculate_scalability_score(model, throughput)
        
        # Analyze bottlenecks
        bottlenecks = []
        if avg_inference_time > self.thresholds['inference_time']:
            bottlenecks.append("slow_inference")
        if throughput < self.thresholds['throughput']:
            bottlenecks.append("low_throughput")
        if avg_memory_usage > 1000:  # 1GB threshold
            bottlenecks.append("high_memory_usage")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(model, bottlenecks, {
            'inference_time': avg_inference_time,
            'throughput': throughput,
            'memory_usage': avg_memory_usage
        })
        
        profile = PerformanceProfile(
            model_name=model.__class__.__name__,
            timestamp=time.time(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            gpu_memory_usage=torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            gpu_utilization=torch.cuda.utilization() if torch.cuda.is_available() else 0,
            inference_time=avg_inference_time,
            throughput=throughput,
            memory_efficiency=memory_efficiency,
            energy_efficiency=energy_efficiency,
            scalability_score=scalability_score,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            metadata={
                'warmup_iterations': warmup_iterations,
                'benchmark_iterations': benchmark_iterations,
                'test_inputs_count': len(test_inputs),
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            }
        )
        
        with self.lock:
            self.performance_history.append(profile)
        
        return profile
    
    def _calculate_memory_efficiency(self, model: nn.Module, memory_usage: float) -> float:
        """Calculate memory efficiency score."""
        param_count = sum(p.numel() for p in model.parameters())
        expected_memory = param_count * 4 / (1024**2)  # 4 bytes per float32 parameter
        
        if memory_usage == 0:
            return 1.0
        
        efficiency = min(1.0, expected_memory / memory_usage)
        return efficiency
    
    def _calculate_energy_efficiency(self, inference_time: float, memory_usage: float) -> float:
        """Calculate energy efficiency score."""
        # Simplified energy efficiency calculation
        # In practice, you would use more sophisticated energy models
        time_efficiency = 1.0 / (1.0 + inference_time)
        memory_efficiency = 1.0 / (1.0 + memory_usage / 1000)  # Normalize to GB
        
        return (time_efficiency + memory_efficiency) / 2.0
    
    def _calculate_scalability_score(self, model: nn.Module, throughput: float) -> float:
        """Calculate scalability score."""
        param_count = sum(p.numel() for p in model.parameters())
        
        # Models with fewer parameters and higher throughput are more scalable
        param_score = 1.0 / (1.0 + param_count / 1000000)
        throughput_score = min(1.0, throughput / 1000)  # Normalize to 1000 samples/sec
        
        return (param_score + throughput_score) / 2.0
    
    def _generate_recommendations(self, model: nn.Module, bottlenecks: List[str], 
                                 metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if "slow_inference" in bottlenecks:
            recommendations.extend([
                "Consider model quantization",
                "Use model pruning to reduce parameters",
                "Implement batch processing",
                "Use faster activation functions"
            ])
        
        if "low_throughput" in bottlenecks:
            recommendations.extend([
                "Optimize data loading pipeline",
                "Use GPU acceleration",
                "Implement model parallelism",
                "Consider model distillation"
            ])
        
        if "high_memory_usage" in bottlenecks:
            recommendations.extend([
                "Reduce batch size",
                "Use gradient checkpointing",
                "Implement memory pooling",
                "Consider model compression"
            ])
        
        # General recommendations based on model size
        param_count = sum(p.numel() for p in model.parameters())
        if param_count > 1000000:  # 1M parameters
            recommendations.append("Consider model architecture optimization")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_performance_summary(self, time_window: float = 3600) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        cutoff_time = time.time() - time_window
        
        with self.lock:
            recent_profiles = [p for p in self.performance_history if p.timestamp >= cutoff_time]
            recent_bottlenecks = [b for b in self.bottleneck_history if b.timestamp >= cutoff_time]
        
        if not recent_profiles:
            return {}
        
        # Calculate statistics
        inference_times = [p.inference_time for p in recent_profiles if p.inference_time > 0]
        throughputs = [p.throughput for p in recent_profiles if p.throughput > 0]
        memory_efficiencies = [p.memory_efficiency for p in recent_profiles if p.memory_efficiency > 0]
        
        # Bottleneck analysis
        bottleneck_counts = defaultdict(int)
        for bottleneck in recent_bottlenecks:
            bottleneck_counts[bottleneck.bottleneck_type] += 1
        
        return {
            'time_window_hours': time_window / 3600,
            'profiles_analyzed': len(recent_profiles),
            'bottlenecks_detected': len(recent_bottlenecks),
            'performance_metrics': {
                'avg_inference_time': np.mean(inference_times) if inference_times else 0,
                'max_inference_time': np.max(inference_times) if inference_times else 0,
                'avg_throughput': np.mean(throughputs) if throughputs else 0,
                'max_throughput': np.max(throughputs) if throughputs else 0,
                'avg_memory_efficiency': np.mean(memory_efficiencies) if memory_efficiencies else 0
            },
            'bottleneck_analysis': dict(bottleneck_counts),
            'recommendations': self._get_top_recommendations(recent_profiles)
        }
    
    def _get_top_recommendations(self, profiles: List[PerformanceProfile]) -> List[str]:
        """Get top recommendations based on performance profiles."""
        all_recommendations = []
        for profile in profiles:
            all_recommendations.extend(profile.recommendations)
        
        # Count recommendation frequency
        rec_counts = defaultdict(int)
        for rec in all_recommendations:
            rec_counts[rec] += 1
        
        # Return top 5 recommendations
        sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, count in sorted_recs[:5]]
    
    def export_performance_report(self, filepath: str, time_window: float = 3600):
        """Export comprehensive performance report."""
        summary = self.get_performance_summary(time_window)
        
        with self.lock:
            recent_profiles = [p for p in self.performance_history if p.timestamp >= time.time() - time_window]
        
        report = {
            'export_timestamp': time.time(),
            'time_window_hours': time_window / 3600,
            'summary': summary,
            'detailed_profiles': [
                {
                    'model_name': p.model_name,
                    'timestamp': p.timestamp,
                    'cpu_usage': p.cpu_usage,
                    'memory_usage': p.memory_usage,
                    'gpu_memory_usage': p.gpu_memory_usage,
                    'inference_time': p.inference_time,
                    'throughput': p.throughput,
                    'memory_efficiency': p.memory_efficiency,
                    'energy_efficiency': p.energy_efficiency,
                    'scalability_score': p.scalability_score,
                    'bottlenecks': p.bottlenecks,
                    'recommendations': p.recommendations,
                    'metadata': p.metadata
                } for p in recent_profiles
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Performance report exported to {filepath}")
    
    def create_performance_visualization(self, filepath: str, time_window: float = 3600):
        """Create performance visualization charts."""
        with self.lock:
            recent_profiles = [p for p in self.performance_history if p.timestamp >= time.time() - time_window]
        
        if not recent_profiles:
            self.logger.warning("No performance data available for visualization")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Analysis Dashboard', fontsize=16)
        
        # Extract data
        timestamps = [p.timestamp for p in recent_profiles]
        inference_times = [p.inference_time for p in recent_profiles]
        throughputs = [p.throughput for p in recent_profiles]
        memory_efficiencies = [p.memory_efficiency for p in recent_profiles]
        cpu_usage = [p.cpu_usage for p in recent_profiles]
        
        # Plot 1: Inference Time
        axes[0, 0].plot(timestamps, inference_times, 'b-', alpha=0.7)
        axes[0, 0].set_title('Inference Time Over Time')
        axes[0, 0].set_ylabel('Inference Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Throughput
        axes[0, 1].plot(timestamps, throughputs, 'g-', alpha=0.7)
        axes[0, 1].set_title('Throughput Over Time')
        axes[0, 1].set_ylabel('Throughput (samples/s)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Memory Efficiency
        axes[1, 0].plot(timestamps, memory_efficiencies, 'r-', alpha=0.7)
        axes[1, 0].set_title('Memory Efficiency Over Time')
        axes[1, 0].set_ylabel('Memory Efficiency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: CPU Usage
        axes[1, 1].plot(timestamps, cpu_usage, 'm-', alpha=0.7)
        axes[1, 1].set_title('CPU Usage Over Time')
        axes[1, 1].set_ylabel('CPU Usage (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Performance visualization saved to {filepath}")
    
    def cleanup(self):
        """Cleanup profiler resources."""
        self.stop_profiling()
        self.logger.info("🧹 Performance profiler cleanup completed")

# Factory functions
def create_performance_profiler(config: Optional[Dict[str, Any]] = None) -> PerformanceProfiler:
    """Create a performance profiler instance."""
    return PerformanceProfiler(config)

@contextmanager
def performance_profiling_context(config: Optional[Dict[str, Any]] = None,
                                 mode: ProfilingMode = ProfilingMode.COMPREHENSIVE):
    """Context manager for performance profiling."""
    profiler = create_performance_profiler(config)
    try:
        profiler.start_profiling(mode)
        yield profiler
    finally:
        profiler.cleanup()

# Utility functions
def benchmark_model_comprehensive(model: nn.Module, 
                                 test_inputs: List[torch.Tensor],
                                 config: Optional[Dict[str, Any]] = None) -> PerformanceProfile:
    """Comprehensive model benchmarking."""
    with performance_profiling_context(config) as profiler:
        return profiler.profile_model(model, test_inputs)

def analyze_model_bottlenecks(model: nn.Module, 
                              test_inputs: List[torch.Tensor]) -> List[BottleneckAnalysis]:
    """Analyze model for performance bottlenecks."""
    profiler = create_performance_profiler()
    profile = profiler.profile_model(model, test_inputs)
    return profiler._analyze_bottlenecks(profile)

