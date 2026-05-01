"""
Optimization Registry for TruthGPT Optimization Core
Advanced optimization tracking with performance analytics and experiment management
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import hashlib
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import time
import psutil
import GPUtil

logger = logging.getLogger(__name__)

class RegistryStatus(Enum):
    """Registry status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"

class OptimizationCategory(Enum):
    """Optimization category enumeration"""
    MODEL_ARCHITECTURE = "model_architecture"
    TRAINING_OPTIMIZATION = "training_optimization"
    INFERENCE_OPTIMIZATION = "inference_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    FINE_TUNING = "fine_tuning"
    DATA_AUGMENTATION = "data_augmentation"
    LOSS_FUNCTION = "loss_function"

@dataclass
class OptimizationEntry:
    """Optimization entry with comprehensive metadata"""
    entry_id: str
    name: str
    description: str
    category: OptimizationCategory
    status: RegistryStatus
    
    # Implementation details
    implementation: Optional[Callable] = None
    implementation_path: Optional[str] = None
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Usage statistics
    usage_count: int = 0
    success_rate: float = 0.0
    average_performance: float = 0.0
    
    # Metadata
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Experiment tracking
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    parent_entries: List[str] = field(default_factory=list)
    child_entries: List[str] = field(default_factory=list)

class PerformanceProfiler:
    """Advanced performance profiler for optimizations"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.gpu_available = torch.cuda.is_available()
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        self.metrics[f"{name}_duration"].append(duration)
        del self.start_times[name]
        return duration
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        self.metrics[name].append(value)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics.update({
                        'gpu_utilization': gpu.load * 100,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal,
                        'gpu_temperature': gpu.temperature
                    })
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
        
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average performance metrics"""
        return {name: np.mean(values) for name, values in self.metrics.items()}

class OptimizationRegistry:
    """Advanced optimization registry with performance tracking"""
    
    def __init__(self, 
                 registry_path: str = "optimization_registry",
                 use_profiling: bool = True,
                 auto_benchmark: bool = True):
        
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        self.use_profiling = use_profiling
        self.auto_benchmark = auto_benchmark
        
        # Initialize components
        self.entries: Dict[str, OptimizationEntry] = {}
        self.profiler = PerformanceProfiler() if use_profiling else None
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.benchmark_results: Dict[str, Dict[str, Any]] = {}
        
        # Load existing entries
        self._load_entries()
        
        logger.info(f"OptimizationRegistry initialized at {self.registry_path}")
    
    def register_optimization(self, 
                            name: str,
                            description: str,
                            category: OptimizationCategory,
                            implementation: Optional[Callable] = None,
                            implementation_path: Optional[str] = None,
                            hyperparameters: Optional[Dict[str, Any]] = None,
                            requirements: Optional[List[str]] = None,
                            author: str = "",
                            **kwargs) -> str:
        """Register a new optimization"""
        
        # Generate entry ID
        entry_id = self._generate_entry_id(name, category)
        
        # Create optimization entry
        entry = OptimizationEntry(
            entry_id=entry_id,
            name=name,
            description=description,
            category=category,
            status=RegistryStatus.ACTIVE,
            implementation=implementation,
            implementation_path=implementation_path,
            hyperparameters=hyperparameters or {},
            requirements=requirements or [],
            author=author,
            **kwargs
        )
        
        # Store entry
        self.entries[entry_id] = entry
        
        # Save to disk
        self._save_entry(entry)
        
        # Auto-benchmark if enabled
        if self.auto_benchmark and implementation:
            self._run_auto_benchmark(entry)
        
        logger.info(f"Registered optimization: {name} ({entry_id})")
        return entry_id
    
    def get_optimization_entry(self, entry_id: str) -> Optional[OptimizationEntry]:
        """Get optimization entry by ID"""
        return self.entries.get(entry_id)
    
    def get_optimizations_by_category(self, category: OptimizationCategory) -> List[OptimizationEntry]:
        """Get optimizations by category"""
        return [entry for entry in self.entries.values() if entry.category == category]
    
    def get_optimizations_by_status(self, status: RegistryStatus) -> List[OptimizationEntry]:
        """Get optimizations by status"""
        return [entry for entry in self.entries.values() if entry.status == status]
    
    def get_top_optimizations(self, 
                             metric: str = "average_performance",
                             limit: int = 10) -> List[OptimizationEntry]:
        """Get top performing optimizations"""
        
        # Filter entries with performance data
        entries_with_metrics = [
            entry for entry in self.entries.values()
            if metric in entry.performance_metrics
        ]
        
        # Sort by metric value
        entries_with_metrics.sort(
            key=lambda x: x.performance_metrics[metric],
            reverse=True
        )
        
        return entries_with_metrics[:limit]
    
    def benchmark_optimization(self, 
                              entry_id: str,
                              model: nn.Module,
                              input_data: torch.Tensor,
                              iterations: int = 100) -> Dict[str, Any]:
        """Benchmark an optimization"""
        
        if entry_id not in self.entries:
            raise ValueError(f"Optimization {entry_id} not found")
        
        entry = self.entries[entry_id]
        
        if not entry.implementation:
            raise ValueError(f"No implementation found for {entry_id}")
        
        # Prepare benchmarking
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = entry.implementation(model, input_data)
        
        # Benchmark
        if self.profiler:
            self.profiler.start_timer("benchmark")
        
        times = []
        memory_usage = []
        
        for i in range(iterations):
            if self.profiler:
                self.profiler.start_timer(f"iteration_{i}")
            
            # Record memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            # Run optimization
            start_time = time.time()
            with torch.no_grad():
                result = entry.implementation(model, input_data)
            end_time = time.time()
            
            # Record memory after
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_usage.append(memory_after - memory_before)
            
            times.append(end_time - start_time)
            
            if self.profiler:
                self.profiler.end_timer(f"iteration_{i}")
        
        if self.profiler:
            self.profiler.end_timer("benchmark")
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        # Get system metrics
        system_metrics = self.profiler.get_system_metrics() if self.profiler else {}
        
        benchmark_results = {
            'average_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'average_memory': avg_memory,
            'iterations': iterations,
            'system_metrics': system_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update entry
        entry.benchmark_results = benchmark_results
        entry.performance_metrics.update({
            'average_time': avg_time,
            'average_memory': avg_memory,
            'throughput': 1.0 / avg_time if avg_time > 0 else 0
        })
        
        # Store in history
        self.performance_history[entry_id].append(benchmark_results)
        
        # Save updated entry
        self._save_entry(entry)
        
        logger.info(f"Benchmarked {entry.name}: {avg_time:.4f}s average")
        return benchmark_results
    
    def compare_optimizations(self, 
                            entry_ids: List[str],
                            model: nn.Module,
                            input_data: torch.Tensor) -> Dict[str, Any]:
        """Compare multiple optimizations"""
        
        if len(entry_ids) < 2:
            raise ValueError("Need at least 2 optimizations to compare")
        
        comparison_results = {}
        
        for entry_id in entry_ids:
            if entry_id not in self.entries:
                logger.warning(f"Optimization {entry_id} not found, skipping")
                continue
            
            # Benchmark each optimization
            benchmark_results = self.benchmark_optimization(entry_id, model, input_data)
            comparison_results[entry_id] = benchmark_results
        
        # Calculate relative performance
        if len(comparison_results) >= 2:
            baseline_time = min(result['average_time'] for result in comparison_results.values())
            
            for entry_id, results in comparison_results.items():
                speedup = baseline_time / results['average_time']
                results['speedup'] = speedup
                results['relative_performance'] = speedup
        
        return comparison_results
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        
        if not self.entries:
            return {}
        
        # Basic statistics
        total_entries = len(self.entries)
        active_entries = len([e for e in self.entries.values() if e.status == RegistryStatus.ACTIVE])
        
        # Category distribution
        category_counts = Counter(entry.category for entry in self.entries.values())
        
        # Performance statistics
        entries_with_metrics = [
            entry for entry in self.entries.values()
            if entry.performance_metrics
        ]
        
        performance_stats = {}
        if entries_with_metrics:
            for metric in ['average_time', 'average_memory', 'throughput']:
                values = [
                    entry.performance_metrics.get(metric, 0)
                    for entry in entries_with_metrics
                    if metric in entry.performance_metrics
                ]
                if values:
                    performance_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        # Usage statistics
        total_usage = sum(entry.usage_count for entry in self.entries.values())
        average_success_rate = np.mean([
            entry.success_rate for entry in self.entries.values()
            if entry.success_rate > 0
        ]) if any(entry.success_rate > 0 for entry in self.entries.values()) else 0
        
        return {
            'total_entries': total_entries,
            'active_entries': active_entries,
            'category_distribution': dict(category_counts),
            'performance_statistics': performance_stats,
            'usage_statistics': {
                'total_usage': total_usage,
                'average_success_rate': average_success_rate
            },
            'recent_activity': self._get_recent_activity()
        }
    
    def _generate_entry_id(self, name: str, category: OptimizationCategory) -> str:
        """Generate unique entry ID"""
        content = f"{name}_{category.value}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _save_entry(self, entry: OptimizationEntry):
        """Save optimization entry to disk"""
        
        entry_path = self.registry_path / f"{entry.entry_id}.json"
        
        # Convert entry to dictionary
        entry_dict = {
            'entry_id': entry.entry_id,
            'name': entry.name,
            'description': entry.description,
            'category': entry.category.value,
            'status': entry.status.value,
            'implementation_path': entry.implementation_path,
            'class_name': entry.class_name,
            'function_name': entry.function_name,
            'performance_metrics': entry.performance_metrics,
            'benchmark_results': entry.benchmark_results,
            'hyperparameters': entry.hyperparameters,
            'requirements': entry.requirements,
            'dependencies': entry.dependencies,
            'usage_count': entry.usage_count,
            'success_rate': entry.success_rate,
            'average_performance': entry.average_performance,
            'author': entry.author,
            'created_at': entry.created_at.isoformat(),
            'updated_at': entry.updated_at.isoformat(),
            'version': entry.version,
            'tags': entry.tags,
            'notes': entry.notes,
            'experiment_id': entry.experiment_id,
            'run_id': entry.run_id,
            'parent_entries': entry.parent_entries,
            'child_entries': entry.child_entries
        }
        
        with open(entry_path, 'w') as f:
            json.dump(entry_dict, f, indent=2)
    
    def _load_entries(self):
        """Load existing entries from disk"""
        
        for entry_file in self.registry_path.glob("*.json"):
            try:
                with open(entry_file, 'r') as f:
                    entry_dict = json.load(f)
                
                entry = OptimizationEntry(
                    entry_id=entry_dict['entry_id'],
                    name=entry_dict['name'],
                    description=entry_dict['description'],
                    category=OptimizationCategory(entry_dict['category']),
                    status=RegistryStatus(entry_dict['status']),
                    implementation_path=entry_dict.get('implementation_path'),
                    class_name=entry_dict.get('class_name'),
                    function_name=entry_dict.get('function_name'),
                    performance_metrics=entry_dict.get('performance_metrics', {}),
                    benchmark_results=entry_dict.get('benchmark_results', {}),
                    hyperparameters=entry_dict.get('hyperparameters', {}),
                    requirements=entry_dict.get('requirements', []),
                    dependencies=entry_dict.get('dependencies', []),
                    usage_count=entry_dict.get('usage_count', 0),
                    success_rate=entry_dict.get('success_rate', 0.0),
                    average_performance=entry_dict.get('average_performance', 0.0),
                    author=entry_dict.get('author', ''),
                    created_at=datetime.fromisoformat(entry_dict['created_at']),
                    updated_at=datetime.fromisoformat(entry_dict['updated_at']),
                    version=entry_dict.get('version', '1.0.0'),
                    tags=entry_dict.get('tags', []),
                    notes=entry_dict.get('notes'),
                    experiment_id=entry_dict.get('experiment_id'),
                    run_id=entry_dict.get('run_id'),
                    parent_entries=entry_dict.get('parent_entries', []),
                    child_entries=entry_dict.get('child_entries', [])
                )
                
                self.entries[entry.entry_id] = entry
                
            except Exception as e:
                logger.error(f"Failed to load entry {entry_file.name}: {e}")
    
    def _run_auto_benchmark(self, entry: OptimizationEntry):
        """Run automatic benchmarking for new optimization"""
        
        if not entry.implementation:
            return
        
        try:
            # Create dummy model and input for benchmarking
            dummy_model = nn.Linear(100, 10)
            dummy_input = torch.randn(32, 100)
            
            # Run benchmark
            self.benchmark_optimization(entry.entry_id, dummy_model, dummy_input, iterations=10)
            
            logger.info(f"Auto-benchmark completed for {entry.name}")
            
        except Exception as e:
            logger.warning(f"Auto-benchmark failed for {entry.name}: {e}")
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent registry activity"""
        
        recent_entries = sorted(
            self.entries.values(),
            key=lambda x: x.updated_at,
            reverse=True
        )[:10]
        
        return [
            {
                'entry_id': entry.entry_id,
                'name': entry.name,
                'category': entry.category.value,
                'updated_at': entry.updated_at.isoformat(),
                'status': entry.status.value
            }
            for entry in recent_entries
        ]

# Factory functions
def create_optimization_registry(registry_path: str = "optimization_registry",
                                use_profiling: bool = True,
                                auto_benchmark: bool = True) -> OptimizationRegistry:
    """Create a new optimization registry"""
    return OptimizationRegistry(registry_path=registry_path,
                               use_profiling=use_profiling,
                               auto_benchmark=auto_benchmark)

def register_optimization(registry: OptimizationRegistry,
                         name: str,
                         description: str,
                         category: OptimizationCategory,
                         **kwargs) -> str:
    """Register a new optimization"""
    return registry.register_optimization(name, description, category, **kwargs)

def get_optimization_entry(registry: OptimizationRegistry, entry_id: str) -> Optional[OptimizationEntry]:
    """Get optimization entry"""
    return registry.get_optimization_entry(entry_id)

def get_registry_statistics(registry: OptimizationRegistry) -> Dict[str, Any]:
    """Get registry statistics"""
    return registry.get_registry_statistics()
