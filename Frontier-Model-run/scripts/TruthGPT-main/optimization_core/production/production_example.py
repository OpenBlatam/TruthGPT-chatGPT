"""
Production Example - Comprehensive example of production-grade optimization system
Demonstrates all features: optimization, monitoring, configuration, and testing
"""

import torch
import torch.nn as nn
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Import production modules
from production_optimizer import (
    ProductionOptimizer, OptimizationLevel, PerformanceProfile,
    create_production_optimizer, production_optimization_context
)
from production_monitoring import (
    ProductionMonitor, AlertLevel, MetricType,
    create_production_monitor, production_monitoring_context
)
from production_config import (
    ProductionConfig, Environment, ConfigSource,
    create_production_config, production_config_context
)
from production_testing import (
    ProductionTestSuite, TestType, TestStatus,
    create_production_test_suite, production_testing_context
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_example_model() -> nn.Module:
    """Create an example neural network model for optimization."""
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
    """Example of basic production optimization."""
    print("🚀 Basic Production Optimization Example")
    print("=" * 50)
    
    # Create model
    model = create_example_model()
    print(f"📝 Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer with production configuration
    config = {
        'optimization_level': OptimizationLevel.AGGRESSIVE,
        'performance_profile': PerformanceProfile.SPEED_OPTIMIZED,
        'max_memory_gb': 8.0,
        'enable_gpu_acceleration': torch.cuda.is_available(),
        'enable_quantization': True,
        'enable_pruning': True
    }
    
    with production_optimization_context(config) as optimizer:
        print("✅ Production optimizer created")
        
        # Optimize model
        start_time = time.time()
        optimized_model = optimizer.optimize_model(model)
        optimization_time = time.time() - start_time
        
        print(f"⚡ Model optimized in {optimization_time:.2f} seconds")
        
        # Test optimized model
        test_input = torch.randn(32, 100)
        with torch.no_grad():
            output = optimized_model(test_input)
        
        print(f"✅ Optimized model test passed - Output shape: {output.shape}")
        
        # Get performance metrics
        metrics = optimizer.get_performance_metrics()
        print(f"📊 Performance metrics collected: {len(metrics)} categories")

def example_monitoring_system():
    """Example of production monitoring system."""
    print("\n🔍 Production Monitoring Example")
    print("=" * 50)
    
    # Create monitoring configuration
    monitor_config = {
        'log_directory': './monitoring_logs',
        'thresholds': {
            'cpu_usage': 70.0,
            'memory_usage': 80.0,
            'gpu_memory_usage': 85.0
        }
    }
    
    with production_monitoring_context(monitor_config) as monitor:
        print("✅ Production monitor started")
        
        # Record some custom metrics
        monitor.record_metric("optimization_requests", 1, MetricType.COUNTER)
        monitor.record_metric("model_size_mb", 15.5, MetricType.GAUGE)
        monitor.record_metric("optimization_duration", 2.3, MetricType.TIMER)
        
        # Simulate some work
        print("🔄 Simulating optimization work...")
        time.sleep(2)
        
        # Get health status
        health = monitor.get_health_status()
        print(f"🏥 System health: {health['status']} - {health['message']}")
        
        # Get performance summary
        summary = monitor.get_performance_summary(hours=1)
        print(f"📊 Performance summary: {summary.get('snapshot_count', 0)} snapshots collected")
        
        # Export metrics
        monitor.export_metrics("example_metrics.json", hours=1)
        print("📤 Metrics exported to example_metrics.json")

def example_configuration_system():
    """Example of production configuration system."""
    print("\n🔧 Production Configuration Example")
    print("=" * 50)
    
    # Create configuration
    config_data = {
        'optimization': {
            'level': 'aggressive',
            'enable_quantization': True,
            'enable_pruning': True,
            'max_memory_gb': 16.0,
            'max_cpu_cores': 8
        },
        'monitoring': {
            'enable_profiling': True,
            'profiling_interval': 100,
            'log_level': 'INFO'
        },
        'performance': {
            'batch_size': 32,
            'max_workers': 4,
            'enable_gpu_acceleration': True
        }
    }
    
    with production_config_context(environment=Environment.PRODUCTION) as config:
        print("✅ Production config created")
        
        # Load configuration data
        for section, values in config_data.items():
            config.update_section(section, values)
        
        # Add validation rules
        from production_config import create_optimization_validation_rules
        config.validation_rules.extend(create_optimization_validation_rules())
        
        # Validate configuration
        errors = config.validate_config()
        if errors:
            print(f"❌ Configuration errors: {errors}")
        else:
            print("✅ Configuration is valid")
        
        # Get environment-specific config
        env_config = config.get_environment_config()
        print(f"🌍 Environment config loaded for {config.environment.value}")
        
        # Export configuration
        config.export_config("example_config.json")
        print("📤 Configuration exported to example_config.json")

def example_testing_system():
    """Example of production testing system."""
    print("\n🧪 Production Testing Example")
    print("=" * 50)
    
    def test_model_optimization():
        """Test model optimization functionality."""
        model = create_example_model()
        config = {'optimization_level': OptimizationLevel.STANDARD}
        
        with production_optimization_context(config) as optimizer:
            optimized_model = optimizer.optimize_model(model)
            
            # Test that model still works
            test_input = torch.randn(1, 100)
            with torch.no_grad():
                output = optimized_model(test_input)
            
            assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    
    def benchmark_optimization_performance():
        """Benchmark optimization performance."""
        model = create_large_model()
        config = {'optimization_level': OptimizationLevel.AGGRESSIVE}
        
        with production_optimization_context(config) as optimizer:
            optimized_model = optimizer.optimize_model(model)
            
            # Benchmark forward pass
            test_input = torch.randn(64, 1000)
            with torch.no_grad():
                _ = optimized_model(test_input)
    
    with production_testing_context() as test_suite:
        print("✅ Production test suite created")
        
        # Add tests
        test_suite.add_test(test_model_optimization, "model_optimization_test", TestType.UNIT)
        test_suite.add_benchmark(benchmark_optimization_performance, "optimization_performance_benchmark")
        
        # Run tests
        test_results = test_suite.run_tests()
        print(f"🧪 Ran {len(test_results)} tests")
        
        # Run benchmarks
        benchmark_results = test_suite.run_benchmarks()
        print(f"📊 Ran {len(benchmark_results)} benchmarks")
        
        # Generate and save report
        report = test_suite.generate_test_report(test_results)
        test_suite.save_test_results(test_results, "example_test_results.json")
        
        print(f"📈 Test success rate: {report['summary']['success_rate']:.2%}")
        print("📤 Test results saved to example_test_results.json")

def example_integrated_system():
    """Example of integrated production system."""
    print("\n🏭 Integrated Production System Example")
    print("=" * 50)
    
    # Create comprehensive configuration
    integrated_config = {
        'optimization': {
            'level': 'aggressive',
            'enable_quantization': True,
            'enable_pruning': True,
            'max_memory_gb': 16.0
        },
        'monitoring': {
            'enable_profiling': True,
            'profiling_interval': 50,
            'log_level': 'INFO'
        },
        'performance': {
            'batch_size': 32,
            'max_workers': 4,
            'enable_gpu_acceleration': torch.cuda.is_available()
        }
    }
    
    # Setup monitoring
    with production_monitoring_context() as monitor:
        print("🔍 Monitoring system started")
        
        # Setup configuration
        with production_config_context(environment=Environment.PRODUCTION) as config:
            for section, values in integrated_config.items():
                config.update_section(section, values)
            
            print("🔧 Configuration system ready")
            
            # Setup optimization with monitoring
            with production_optimization_context(config.get_section('optimization')) as optimizer:
                print("🚀 Optimization system ready")
                
                # Create and optimize multiple models
                models = [create_example_model() for _ in range(3)]
                optimized_models = []
                
                for i, model in enumerate(models):
                    print(f"⚡ Optimizing model {i+1}/3...")
                    
                    start_time = time.time()
                    optimized_model = optimizer.optimize_model(model)
                    optimization_time = time.time() - start_time
                    
                    optimized_models.append(optimized_model)
                    
                    # Record metrics
                    monitor.record_metric("optimization_duration", optimization_time, MetricType.TIMER)
                    monitor.record_metric("models_optimized", 1, MetricType.COUNTER)
                
                print(f"✅ Optimized {len(optimized_models)} models successfully")
                
                # Get final metrics
                health = monitor.get_health_status()
                print(f"🏥 Final system health: {health['status']}")
                
                # Export all data
                monitor.export_metrics("integrated_metrics.json")
                config.export_config("integrated_config.json")
                
                print("📤 All data exported successfully")

def main():
    """Main example function."""
    print("🏭 Production-Grade Optimization System Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        example_basic_optimization()
        example_monitoring_system()
        example_configuration_system()
        example_testing_system()
        example_integrated_system()
        
        print("\n✅ All examples completed successfully!")
        print("🎉 Production system is ready for deployment!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    main()

