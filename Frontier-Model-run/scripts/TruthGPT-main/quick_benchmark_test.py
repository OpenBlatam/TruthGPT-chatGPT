"""
Quick test script for comprehensive benchmark functionality
"""

import torch
import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic benchmark functionality without model imports."""
    print("🧪 Testing Basic Benchmark Functionality")
    print("=" * 50)
    
    try:
        from comprehensive_benchmark import ComprehensiveBenchmark, ModelMetrics
        print("✅ Imports successful")
        
        benchmark = ComprehensiveBenchmark()
        print("✅ Benchmark instance created")
        
        test_model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 100)
        )
        
        total_params, trainable_params = benchmark.count_parameters(test_model)
        print(f"✅ Parameter counting: {total_params:,} total, {trainable_params:,} trainable")
        
        model_size = benchmark.get_model_size_mb(test_model)
        print(f"✅ Model size calculation: {model_size:.2f} MB")
        
        input_tensor = torch.randn(2, 512)
        flops = benchmark.calculate_flops(test_model, input_tensor)
        print(f"✅ FLOPs calculation: {flops:.2e}")
        
        memory_usage, peak_memory, inference_time = benchmark.measure_memory_usage(test_model, input_tensor)
        print(f"✅ Memory measurement: {memory_usage:.2f} MB usage, {inference_time:.2f} ms inference")
        
        gpu_memory, gpu_peak = benchmark.measure_gpu_memory(test_model, input_tensor)
        print(f"✅ GPU memory measurement: {gpu_memory:.2f} MB")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def test_optimization_modules():
    """Test optimization module imports."""
    print("\n🔧 Testing Optimization Modules")
    print("=" * 50)
    
    modules_tested = 0
    modules_passed = 0
    
    try:
        from optimization_core.memory_optimizations import MemoryOptimizer, create_memory_optimizer
        print("✅ Memory optimizations imported")
        modules_tested += 1
        modules_passed += 1
    except ImportError as e:
        print(f"❌ Memory optimizations failed: {e}")
        modules_tested += 1
    
    try:
        from optimization_core.computational_optimizations import ComputationalOptimizer, create_computational_optimizer
        print("✅ Computational optimizations imported")
        modules_tested += 1
        modules_passed += 1
    except ImportError as e:
        print(f"❌ Computational optimizations failed: {e}")
        modules_tested += 1
    
    try:
        from optimization_core.optimization_profiles import get_optimization_profiles, apply_optimization_profile
        print("✅ Optimization profiles imported")
        modules_tested += 1
        modules_passed += 1
    except ImportError as e:
        print(f"❌ Optimization profiles failed: {e}")
        modules_tested += 1
    
    try:
        from optimization_core.enhanced_mcts_optimizer import create_enhanced_mcts_with_benchmarks
        print("✅ Enhanced MCTS imported")
        modules_tested += 1
        modules_passed += 1
    except ImportError as e:
        print(f"❌ Enhanced MCTS failed: {e}")
        modules_tested += 1
    
    try:
        from optimization_core.olympiad_benchmarks import create_olympiad_benchmark_suite
        print("✅ Olympiad benchmarks imported")
        modules_tested += 1
        modules_passed += 1
    except ImportError as e:
        print(f"❌ Olympiad benchmarks failed: {e}")
        modules_tested += 1
    
    print(f"\n📊 Optimization modules: {modules_passed}/{modules_tested} passed")
    return modules_passed == modules_tested

def test_report_generation():
    """Test performance report generation."""
    print("\n📄 Testing Report Generation")
    print("=" * 50)
    
    try:
        from generate_performance_report import PerformanceReportGenerator, ModelMetrics
        
        sample_metrics = [
            ModelMetrics(
                name="Test-Model-1",
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
            ),
            ModelMetrics(
                name="Test-Model-2",
                total_parameters=2000000,
                trainable_parameters=2000000,
                model_size_mb=20.0,
                memory_usage_mb=80.0,
                peak_memory_mb=90.0,
                gpu_memory_mb=0.0,
                gpu_peak_memory_mb=0.0,
                inference_time_ms=35.0,
                flops=2e9,
                olympiad_accuracy=0.90,
                olympiad_scores={'algebra': 0.95, 'geometry': 0.85},
                mcts_optimization_score=0.80,
                optimization_time_seconds=150.0
            )
        ]
        
        generator = PerformanceReportGenerator(sample_metrics)
        print("✅ Report generator created")
        
        report = generator.generate_spanish_report()
        print("✅ Spanish report generated")
        
        report_file = generator.save_report("test_report.md")
        print(f"✅ Report saved to: {report_file}")
        
        csv_file = generator.export_csv("test_metrics.csv")
        print(f"✅ CSV exported to: {csv_file}")
        
        print("\n🎉 Report generation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all quick tests."""
    print("🚀 Quick Benchmark Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_optimization_modules():
        tests_passed += 1
    
    if test_report_generation():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Benchmark system is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
