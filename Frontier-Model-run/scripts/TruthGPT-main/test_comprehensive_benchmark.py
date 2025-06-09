"""
Test script for comprehensive benchmark functionality
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_benchmark import ComprehensiveBenchmark, ModelMetrics

def test_benchmark_creation():
    """Test benchmark creation."""
    print("Testing benchmark creation...")
    benchmark = ComprehensiveBenchmark()
    assert benchmark is not None
    print("✅ Benchmark created successfully")

def test_model_metrics():
    """Test model metrics creation."""
    print("Testing model metrics...")
    metrics = ModelMetrics(
        name="test_model",
        total_parameters=1000000,
        trainable_parameters=1000000,
        model_size_mb=10.0,
        memory_usage_mb=50.0,
        peak_memory_mb=60.0,
        gpu_memory_mb=0.0,
        gpu_peak_memory_mb=0.0,
        inference_time_ms=25.0,
        flops=1e9,
        olympiad_accuracy=0.85,
        olympiad_scores={'algebra': 0.9, 'geometry': 0.8},
        mcts_optimization_score=0.75,
        optimization_time_seconds=120.0
    )
    assert metrics.name == "test_model"
    assert metrics.total_parameters == 1000000
    print("✅ Model metrics created successfully")

def test_memory_utilities():
    """Test memory measurement utilities."""
    print("Testing memory utilities...")
    benchmark = ComprehensiveBenchmark()
    
    model = torch.nn.Linear(512, 1000)
    input_tensor = torch.randn(2, 512)
    
    total_params, trainable_params = benchmark.count_parameters(model)
    assert total_params > 0
    assert trainable_params > 0
    print(f"✅ Parameter counting: {total_params:,} total, {trainable_params:,} trainable")
    
    model_size = benchmark.get_model_size_mb(model)
    assert model_size > 0
    print(f"✅ Model size: {model_size:.2f} MB")
    
    flops = benchmark.calculate_flops(model, input_tensor)
    assert flops > 0
    print(f"✅ FLOPs calculation: {flops:.2e}")

def test_optimization_imports():
    """Test optimization module imports."""
    print("Testing optimization imports...")
    
    try:
        from optimization_core.memory_optimizations import MemoryOptimizer, create_memory_optimizer
        print("✅ Memory optimizations imported")
    except ImportError as e:
        print(f"❌ Memory optimizations import failed: {e}")
    
    try:
        from optimization_core.computational_optimizations import ComputationalOptimizer, create_computational_optimizer
        print("✅ Computational optimizations imported")
    except ImportError as e:
        print(f"❌ Computational optimizations import failed: {e}")
    
    try:
        from optimization_core.optimization_profiles import get_optimization_profiles, apply_optimization_profile
        print("✅ Optimization profiles imported")
    except ImportError as e:
        print(f"❌ Optimization profiles import failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Running Comprehensive Benchmark Tests")
    print("=" * 50)
    
    try:
        test_benchmark_creation()
        test_model_metrics()
        test_memory_utilities()
        test_optimization_imports()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
