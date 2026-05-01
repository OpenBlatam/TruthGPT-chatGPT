"""
Refactored Example - Comprehensive example of the refactored optimization system
Demonstrates the new modular architecture and improved functionality
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path

# Import refactored modules
from ..core import (
    OptimizationLevel, OptimizationResult, ConfigManager, Environment,
    SystemMonitor, ModelValidator, CacheManager, PerformanceUtils
)
from ..optimizers import (
    ProductionOptimizer, create_production_optimizer, production_optimization_context
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_example_model() -> nn.Module:
    """Create an example neural network model."""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(25, 10),
        nn.Softmax(dim=-1)
    )

def create_large_model() -> nn.Module:
    """Create a larger model for performance testing."""
    return nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

def example_basic_optimization():
    """Example of basic production optimization with refactored system."""
    print("🚀 Refactored Production Optimization Example")
    print("=" * 60)
    
    # Create model
    model = create_example_model()
    print(f"📝 Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer with refactored configuration
    config = {
        'level': OptimizationLevel.AGGRESSIVE.value,
        'enable_quantization': True,
        'enable_pruning': True,
        'enable_mixed_precision': True,
        'max_memory_gb': 16.0,
        'enable_gpu_acceleration': torch.cuda.is_available()
    }
    
    with production_optimization_context(config) as optimizer:
        print("✅ Refactored production optimizer created")
        
        # Optimize model
        result = optimizer.optimize(model)
        
        if result.success:
            print(f"⚡ Model optimized successfully in {result.optimization_time:.2f}s")
            print(f"📊 Memory usage: {result.memory_usage:.2f} MB")
            print(f"📈 Parameter reduction: {result.parameter_reduction:.1f}%")
            print(f"🔧 Optimizations applied: {', '.join(result.optimizations_applied)}")
        else:
            print(f"❌ Optimization failed: {result.error_message}")

def example_configuration_management():
    """Example of refactored configuration management."""
    print("\n🔧 Refactored Configuration Management Example")
    print("=" * 60)
    
    # Create configuration manager
    config_manager = ConfigManager(Environment.PRODUCTION)
    
    # Load configuration from file
    config_file = Path("example_config.json")
    if config_file.exists():
        config_manager.load_from_file(str(config_file))
        print("📁 Configuration loaded from file")
    
    # Load from environment variables
    config_manager.load_from_environment("OPTIMIZATION_")
    print("🌍 Configuration loaded from environment")
    
    # Get specific configurations
    opt_config = config_manager.get_optimization_config()
    print(f"⚙️ Optimization level: {opt_config.level}")
    print(f"💾 Max memory: {opt_config.max_memory_gb} GB")
    print(f"🔧 Quantization enabled: {opt_config.enable_quantization}")
    
    # Validate configuration
    errors = config_manager.validate_config()
    if errors:
        print(f"❌ Configuration errors: {errors}")
    else:
        print("✅ Configuration is valid")
    
    # Export configuration
    config_manager.export_config("exported_config.json")
    print("📤 Configuration exported")

def example_monitoring_system():
    """Example of refactored monitoring system."""
    print("\n🔍 Refactored Monitoring System Example")
    print("=" * 60)
    
    # Create monitoring configuration
    monitor_config = {
        'thresholds': {
            'cpu_usage': 70.0,
            'memory_usage': 80.0,
            'gpu_memory_usage': 85.0
        }
    }
    
    # Create system monitor
    monitor = SystemMonitor(monitor_config)
    
    # Start monitoring
    monitor.start_monitoring(interval=0.5)
    print("🔍 Monitoring started")
    
    # Record custom metrics
    monitor.metrics_collector.record_metric("optimization_requests", 1)
    monitor.metrics_collector.record_metric("model_size_mb", 15.5)
    monitor.metrics_collector.record_metric("optimization_duration", 2.3)
    
    # Simulate some work
    print("🔄 Simulating optimization work...")
    import time
    time.sleep(2)
    
    # Get health status
    health = monitor.get_health_status()
    print(f"🏥 System health: {health['status']} - {health['message']}")
    
    # Get performance summary
    summary = monitor.get_performance_summary(hours=1)
    print(f"📊 Performance summary: {summary.get('snapshot_count', 0)} snapshots collected")
    
    # Export metrics
    monitor.export_metrics("refactored_metrics.json", hours=1)
    print("📤 Metrics exported")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("🛑 Monitoring stopped")

def example_validation_system():
    """Example of refactored validation system."""
    print("\n✅ Refactored Validation System Example")
    print("=" * 60)
    
    # Create model validator
    validator = ModelValidator()
    
    # Create test models
    model = create_example_model()
    optimized_model = create_example_model()  # In real scenario, this would be optimized
    
    # Validate original model
    print("🔍 Validating original model...")
    reports = validator.validate_model(model)
    for report in reports:
        status_emoji = "✅" if report.result.value == "passed" else "❌"
        print(f"{status_emoji} {report.test_name}: {report.message}")
    
    # Validate model compatibility
    print("\n🔍 Validating model compatibility...")
    compatibility_report = validator.validate_model_compatibility(model, optimized_model)
    status_emoji = "✅" if compatibility_report.result.value == "passed" else "❌"
    print(f"{status_emoji} {compatibility_report.test_name}: {compatibility_report.message}")
    
    # Validate performance
    print("\n🔍 Validating model performance...")
    test_inputs = [torch.randn(1, 100) for _ in range(5)]
    performance_report = validator.validate_model_performance(model, test_inputs)
    status_emoji = "✅" if performance_report.result.value == "passed" else "❌"
    print(f"{status_emoji} {performance_report.test_name}: {performance_report.message}")

def example_caching_system():
    """Example of refactored caching system."""
    print("\n💾 Refactored Caching System Example")
    print("=" * 60)
    
    # Create cache manager
    cache_manager = CacheManager("./refactored_cache")
    
    # Get model cache
    model_cache = cache_manager.get_cache("models", max_size=50, max_memory_mb=512)
    
    # Create test model
    model = create_example_model()
    config = {'level': 'standard', 'enable_quantization': True}
    
    # Cache model
    cache_key = model_cache._generate_key(model, config)
    model_cache.put(cache_key, model, ttl_seconds=3600)
    print(f"💾 Model cached with key: {cache_key[:8]}...")
    
    # Retrieve from cache
    cached_model = model_cache.get(cache_key)
    if cached_model is not None:
        print("✅ Model retrieved from cache")
    else:
        print("❌ Model not found in cache")
    
    # Get cache statistics
    stats = model_cache.get_stats()
    print(f"📊 Cache stats: {stats['size']} entries, {stats['hit_rate']:.2%} hit rate")
    
    # Cleanup
    cache_manager.cleanup_all()
    print("🧹 Cache cleaned up")

def example_performance_utilities():
    """Example of refactored performance utilities."""
    print("\n⚡ Refactored Performance Utilities Example")
    print("=" * 60)
    
    # Create performance utilities
    perf_utils = PerformanceUtils()
    memory_utils = MemoryUtils()
    gpu_utils = GPUUtils()
    
    # Create test model
    model = create_large_model()
    
    # Get system metrics
    metrics = perf_utils.get_system_metrics()
    print(f"🖥️ CPU usage: {metrics.cpu_usage:.1f}%")
    print(f"💾 Memory usage: {metrics.memory_usage:.1f}%")
    print(f"🎮 GPU memory usage: {metrics.gpu_memory_usage:.1f}%")
    
    # Get model memory usage
    memory_info = memory_utils.get_model_memory_usage(model)
    print(f"📊 Model memory: {memory_info['total_mb']:.2f} MB")
    
    # Get parameter count
    param_info = memory_utils.get_parameter_count(model)
    print(f"🔢 Total parameters: {param_info['total_parameters']:,}")
    print(f"🔢 Trainable parameters: {param_info['trainable_parameters']:,}")
    
    # Get GPU information
    if torch.cuda.is_available():
        gpu_info = gpu_utils.get_device_properties()
        print(f"🎮 GPU: {gpu_info['name']}")
        print(f"🎮 Total memory: {gpu_info['total_memory_mb']:.0f} MB")
    else:
        print("🎮 No GPU available")
    
    # Benchmark model
    test_input = torch.randn(32, 1000)
    benchmark_results = perf_utils.benchmark_model_forward(model, test_input, iterations=10)
    print(f"⚡ Model throughput: {benchmark_results['throughput']:.2f} ops/s")

def example_integrated_system():
    """Example of integrated refactored system."""
    print("\n🏭 Integrated Refactored System Example")
    print("=" * 60)
    
    # Create comprehensive configuration
    config_manager = ConfigManager(Environment.PRODUCTION)
    
    # Setup configuration
    optimization_config = {
        'level': OptimizationLevel.AGGRESSIVE.value,
        'enable_quantization': True,
        'enable_pruning': True,
        'enable_mixed_precision': True,
        'max_memory_gb': 16.0,
        'enable_gpu_acceleration': torch.cuda.is_available()
    }
    
    monitoring_config = {
        'enable_profiling': True,
        'profiling_interval': 50,
        'thresholds': {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'gpu_memory_usage': 90.0
        }
    }
    
    config_manager.update_section('optimization', optimization_config)
    config_manager.update_section('monitoring', monitoring_config)
    
    print("🔧 Configuration system ready")
    
    # Create models
    models = [create_example_model() for _ in range(3)]
    print(f"📝 Created {len(models)} models for optimization")
    
    # Setup monitoring
    monitor = SystemMonitor(monitoring_config)
    monitor.start_monitoring()
    print("🔍 Monitoring system started")
    
    # Setup optimization
    with production_optimization_context(optimization_config) as optimizer:
        print("🚀 Optimization system ready")
        
        # Optimize models
        results = []
        for i, model in enumerate(models):
            print(f"⚡ Optimizing model {i+1}/{len(models)}...")
            
            result = optimizer.optimize(model)
            results.append(result)
            
            if result.success:
                print(f"✅ Model {i+1} optimized successfully")
            else:
                print(f"❌ Model {i+1} optimization failed: {result.error_message}")
        
        # Get final metrics
        health = monitor.get_health_status()
        print(f"🏥 Final system health: {health['status']}")
        
        # Get optimization summary
        summary = optimizer.get_optimization_summary()
        print(f"📊 Optimization summary: {summary['successful']}/{summary['total_optimizations']} successful")
        
        # Export all data
        monitor.export_metrics("integrated_refactored_metrics.json")
        config_manager.export_config("integrated_refactored_config.json")
        
        print("📤 All data exported successfully")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("🛑 Monitoring stopped")

def main():
    """Main example function."""
    print("🏭 Refactored Production-Grade Optimization System")
    print("=" * 70)
    print("Demonstrating the new modular architecture with improved functionality")
    print("=" * 70)
    
    try:
        # Run all examples
        example_basic_optimization()
        example_configuration_management()
        example_monitoring_system()
        example_validation_system()
        example_caching_system()
        example_performance_utilities()
        example_integrated_system()
        
        print("\n✅ All refactored examples completed successfully!")
        print("🎉 Refactored system is ready for production deployment!")
        print("\n📈 Key Improvements:")
        print("  • Modular architecture with clear separation of concerns")
        print("  • Improved error handling and validation")
        print("  • Better performance monitoring and metrics")
        print("  • Enhanced configuration management")
        print("  • Robust caching system")
        print("  • Comprehensive testing and validation")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    main()




