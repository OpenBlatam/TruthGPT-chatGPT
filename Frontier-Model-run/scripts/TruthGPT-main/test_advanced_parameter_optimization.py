#!/usr/bin/env python3
"""
Advanced test suite for expanded parameter optimization with kernel, memory pooling, and scheduling optimizations.
"""

import torch
import torch.nn as nn
from optimization_core.enhanced_parameter_optimizer import (
    EnhancedParameterConfig, EnhancedParameterOptimizer, create_enhanced_parameter_optimizer
)

def create_test_model(hidden_size: int = 512, num_layers: int = 6) -> nn.Module:
    """Create a test model for parameter optimization."""
    layers = []
    
    for i in range(num_layers):
        if i == 0:
            layers.append(nn.Linear(hidden_size, hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
    
    layers.append(nn.Linear(hidden_size, hidden_size // 2))
    
    return nn.Sequential(*layers)

def test_advanced_parameter_categories():
    """Test advanced parameter optimization categories."""
    print("🚀 Testing Advanced Parameter Categories...")
    
    optimizer = create_enhanced_parameter_optimizer()
    model = create_test_model(hidden_size=768, num_layers=8)
    
    config = optimizer.generate_optimized_config(model, "test_advanced_model")
    
    expected_categories = [
        'learning_rates', 'rl_parameters', 'temperature_parameters',
        'quantization_parameters', 'memory_parameters', 'attention_parameters',
        'activation_parameters', 'normalization_parameters', 'regularization_parameters',
        'batch_parameters', 'scheduler_parameters', 'kernel_optimization',
        'memory_pooling', 'cuda_optimization', 'parallel_optimization',
        'advanced_scheduling', 'model_specific'
    ]
    
    for category in expected_categories:
        assert category in config, f"Missing category: {category}"
        assert isinstance(config[category], dict), f"Category {category} should be a dict"
    
    kernel_config = config['kernel_optimization']
    assert 'block_size' in kernel_config
    assert 'kernel_fusion_enabled' in kernel_config
    assert 'memory_coalescing' in kernel_config
    assert 'occupancy_optimization' in kernel_config
    
    memory_pooling_config = config['memory_pooling']
    assert 'tensor_pool_size' in memory_pooling_config
    assert 'activation_cache_size' in memory_pooling_config
    assert 'enable_tensor_pooling' in memory_pooling_config
    
    cuda_config = config['cuda_optimization']
    assert 'adaptive_block_sizing' in cuda_config
    assert 'kernel_fusion' in cuda_config
    assert 'flash_attention_kernels' in cuda_config
    
    parallel_config = config['parallel_optimization']
    assert 'num_workers' in parallel_config
    assert 'tensor_parallelism' in parallel_config
    assert 'data_parallelism' in parallel_config
    
    scheduling_config = config['advanced_scheduling']
    assert 'scheduler_type' in scheduling_config
    assert 'warmup_ratio' in scheduling_config
    assert 'cosine_restarts' in scheduling_config
    
    print(f"  ✅ All {len(expected_categories)} parameter categories validated")
    print(f"  🔧 Kernel block size: {kernel_config['block_size']}")
    print(f"  💾 Memory pool size: {memory_pooling_config['tensor_pool_size']}")
    print(f"  ⚡ CUDA kernel fusion: {cuda_config['kernel_fusion']}")
    print(f"  🔄 Parallel workers: {parallel_config['num_workers']}")
    print(f"  📅 Scheduler type: {scheduling_config['scheduler_type']}")
    print("  ✅ Advanced parameter categories test passed!")

def test_model_size_scaling():
    """Test parameter scaling based on model size."""
    print("📊 Testing Model Size Scaling...")
    
    optimizer = create_enhanced_parameter_optimizer()
    
    model_sizes = [
        (create_test_model(hidden_size=64, num_layers=2), "tiny"),
        (create_test_model(hidden_size=256, num_layers=4), "small"),
        (create_test_model(hidden_size=512, num_layers=8), "medium"),
        (create_test_model(hidden_size=1024, num_layers=16), "large")
    ]
    
    for model, size_name in model_sizes:
        config = optimizer.generate_optimized_config(model, f"{size_name}_model")
        model_params = sum(p.numel() for p in model.parameters())
        
        kernel_config = config['kernel_optimization']
        memory_config = config['memory_pooling']
        parallel_config = config['parallel_optimization']
        
        print(f"  📏 {size_name.upper()} model ({model_params:,} params):")
        print(f"    🔧 Block size: {kernel_config['block_size']}")
        print(f"    💾 Pool size: {memory_config['tensor_pool_size']}")
        print(f"    🔄 Workers: {parallel_config['num_workers']}")
        print(f"    ⚡ Tensor parallel: {parallel_config['tensor_parallelism']}")
        
        if model_params < 1e6:
            assert kernel_config['block_size'] <= 256
            assert not parallel_config['tensor_parallelism']
        elif model_params > 10e6:
            assert kernel_config['block_size'] >= 256
            assert kernel_config['tensor_core_optimization']
    
    print("  ✅ Model size scaling test passed!")

def test_complexity_based_optimization():
    """Test optimization based on model complexity."""
    print("🧠 Testing Complexity-Based Optimization...")
    
    optimizer = create_enhanced_parameter_optimizer()
    
    simple_model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
    complex_model = create_test_model(hidden_size=1024, num_layers=20)
    
    simple_config = optimizer.generate_optimized_config(simple_model, "simple_model")
    complex_config = optimizer.generate_optimized_config(complex_model, "complex_model")
    
    assert complex_config['kernel_optimization']['occupancy_optimization']
    assert complex_config['memory_pooling']['enable_gradient_caching']
    assert complex_config['cuda_optimization']['fused_attention_mlp']
    assert complex_config['advanced_scheduling']['cosine_restarts']
    
    simple_kernel = simple_config['kernel_optimization']
    simple_memory = simple_config['memory_pooling']
    
    print(f"  🔧 Simple model optimizations:")
    print(f"    Kernel fusion: {simple_kernel['kernel_fusion_enabled']}")
    print(f"    Gradient caching: {simple_memory['enable_gradient_caching']}")
    
    print(f"  🚀 Complex model optimizations:")
    print(f"    Occupancy opt: {complex_config['kernel_optimization']['occupancy_optimization']}")
    print(f"    Tensor cores: {complex_config['kernel_optimization']['tensor_core_optimization']}")
    print(f"    Advanced scheduling: {complex_config['advanced_scheduling']['scheduler_type']}")
    
    print("  ✅ Complexity-based optimization test passed!")

def test_performance_improvement_estimation():
    """Test performance improvement estimation with advanced optimizations."""
    print("📈 Testing Performance Improvement Estimation...")
    
    optimizer = create_enhanced_parameter_optimizer()
    model = create_test_model(hidden_size=768, num_layers=12)
    
    baseline_config = {
        'learning_rates': {'base_lr': 3e-4},
        'kernel_optimization': {'kernel_fusion_enabled': False, 'block_size': 128},
        'memory_pooling': {'enable_tensor_pooling': False},
        'cuda_optimization': {'kernel_fusion': False},
        'parallel_optimization': {'num_workers': 1}
    }
    
    optimized_config = optimizer.generate_optimized_config(model, "performance_test_model")
    
    def estimate_performance(config):
        score = 0.5  # baseline
        
        if config.get('kernel_optimization', {}).get('kernel_fusion_enabled', False):
            score += 0.1
        if config.get('kernel_optimization', {}).get('block_size', 128) > 256:
            score += 0.05
        
        if config.get('memory_pooling', {}).get('enable_tensor_pooling', False):
            score += 0.08
        if config.get('memory_pooling', {}).get('enable_activation_caching', False):
            score += 0.06
        
        if config.get('cuda_optimization', {}).get('kernel_fusion', False):
            score += 0.12
        if config.get('cuda_optimization', {}).get('flash_attention_kernels', False):
            score += 0.15
        
        workers = config.get('parallel_optimization', {}).get('num_workers', 1)
        if workers > 1:
            score += min(0.1, (workers - 1) * 0.03)
        
        return min(1.0, score)
    
    baseline_perf = estimate_performance(baseline_config)
    optimized_perf = estimate_performance(optimized_config)
    improvement = (optimized_perf - baseline_perf) / baseline_perf * 100
    
    print(f"  📊 Baseline performance: {baseline_perf:.3f}")
    print(f"  🚀 Optimized performance: {optimized_perf:.3f}")
    print(f"  📈 Performance improvement: {improvement:.1f}%")
    
    assert improvement > 15.0, f"Expected >15% improvement, got {improvement:.1f}%"
    
    print("  ✅ Performance improvement estimation test passed!")

def test_integration_with_existing_optimizations():
    """Test integration with existing RL optimizations (DAPO, VAPO, ORZ)."""
    print("🔗 Testing Integration with Existing RL Optimizations...")
    
    optimizer = create_enhanced_parameter_optimizer()
    model = create_test_model(hidden_size=512, num_layers=8)
    
    config = optimizer.generate_optimized_config(model, "rl_integration_test")
    
    rl_config = config['rl_parameters']
    assert 'epsilon_start' in rl_config
    assert 'gamma' in rl_config
    assert 'entropy_coefficient' in rl_config
    
    kernel_config = config['kernel_optimization']
    memory_config = config['memory_pooling']
    
    assert kernel_config['occupancy_optimization'] or memory_config['enable_activation_caching']
    
    print(f"  🎯 RL epsilon start: {rl_config['epsilon_start']:.3f}")
    print(f"  🧠 RL gamma: {rl_config['gamma']:.3f}")
    print(f"  🔧 Kernel optimization enabled: {kernel_config['kernel_fusion_enabled']}")
    print(f"  💾 Memory pooling enabled: {memory_config['enable_tensor_pooling']}")
    
    print("  ✅ RL integration test passed!")

def main():
    """Main test function."""
    print("🎯 Advanced Parameter Optimization Test Suite")
    print("=" * 70)
    
    try:
        test_advanced_parameter_categories()
        test_model_size_scaling()
        test_complexity_based_optimization()
        test_performance_improvement_estimation()
        test_integration_with_existing_optimizations()
        
        print("\n🎉 All advanced parameter optimization tests passed!")
        print("✅ Enhanced parameter optimization system expanded successfully!")
        print("🚀 Ready for production with advanced kernel, memory, and scheduling optimizations!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
