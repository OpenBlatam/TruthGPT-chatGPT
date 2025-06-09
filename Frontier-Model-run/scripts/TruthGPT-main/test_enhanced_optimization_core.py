"""
Test suite for Enhanced Optimization Core
Tests the optimization of the optimization_core module itself
"""

import torch
import torch.nn as nn
import logging
from optimization_core import (
    create_enhanced_optimization_core,
    AdaptivePrecisionOptimizer,
    DynamicKernelFusionOptimizer,
    IntelligentMemoryManager,
    EnhancedOptimizedLayerNorm,
    OptimizedLayerNorm
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_adaptive_precision_optimizer():
    """Test adaptive precision optimization."""
    logger.info("🧪 Testing Adaptive Precision Optimizer...")
    
    try:
        from optimization_core.enhanced_optimization_core import EnhancedOptimizationConfig
        
        config = EnhancedOptimizationConfig(enable_adaptive_precision=True)
        precision_optimizer = AdaptivePrecisionOptimizer(config)
        
        test_tensor = torch.randn(10, 512) * 0.001  # Small values
        optimized_tensor = precision_optimizer.optimize_precision(test_tensor, "parameter")
        
        logger.info(f"✅ Original dtype: {test_tensor.dtype}")
        logger.info(f"✅ Optimized dtype: {optimized_tensor.dtype}")
        logger.info(f"✅ Precision history: {len(precision_optimizer.precision_history)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive precision optimizer test failed: {e}")
        return False

def test_dynamic_kernel_fusion():
    """Test dynamic kernel fusion optimization."""
    logger.info("🧪 Testing Dynamic Kernel Fusion...")
    
    try:
        from optimization_core.enhanced_optimization_core import EnhancedOptimizationConfig
        
        config = EnhancedOptimizationConfig(enable_dynamic_kernel_fusion=True)
        fusion_optimizer = DynamicKernelFusionOptimizer(config)
        
        def relu_op(x):
            return torch.relu(x)
        
        def linear_op(x):
            return x * 2.0
        
        operations = [relu_op, linear_op]
        inputs = [torch.randn(5, 10)]
        
        result = fusion_optimizer.fuse_operations(operations, inputs)
        
        logger.info(f"✅ Fusion result shape: {result.shape}")
        logger.info(f"✅ Performance cache size: {len(fusion_optimizer.performance_cache)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dynamic kernel fusion test failed: {e}")
        return False

def test_intelligent_memory_manager():
    """Test intelligent memory management."""
    logger.info("🧪 Testing Intelligent Memory Manager...")
    
    try:
        from optimization_core.enhanced_optimization_core import EnhancedOptimizationConfig
        
        config = EnhancedOptimizationConfig(enable_intelligent_memory_management=True)
        memory_manager = IntelligentMemoryManager(config)
        
        tensor1 = memory_manager.allocate_tensor((10, 20), torch.float32)
        tensor2 = memory_manager.allocate_tensor((10, 20), torch.float32)
        
        memory_manager.deallocate_tensor(tensor1)
        tensor3 = memory_manager.allocate_tensor((10, 20), torch.float32)
        
        logger.info(f"✅ Tensor1 shape: {tensor1.shape}")
        logger.info(f"✅ Memory pools: {len(memory_manager.memory_pools)}")
        logger.info(f"✅ Allocation history: {len(memory_manager.allocation_history)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Intelligent memory manager test failed: {e}")
        return False

def test_enhanced_layer_norm():
    """Test enhanced optimized layer norm."""
    logger.info("🧪 Testing Enhanced Optimized LayerNorm...")
    
    try:
        from optimization_core.enhanced_optimization_core import EnhancedOptimizationConfig
        
        config = EnhancedOptimizationConfig(
            enable_adaptive_precision=True,
            enable_self_optimizing_components=True
        )
        
        enhanced_norm = EnhancedOptimizedLayerNorm(512, config)
        
        x = torch.randn(2, 10, 512)
        output = enhanced_norm(x)
        
        logger.info(f"✅ Input shape: {x.shape}")
        logger.info(f"✅ Output shape: {output.shape}")
        logger.info(f"✅ Optimization counter: {enhanced_norm.optimization_counter}")
        logger.info(f"✅ Performance metrics: {len(enhanced_norm.performance_metrics)}")
        
        for i in range(5):
            _ = enhanced_norm(x)
        
        logger.info(f"✅ After 5 passes - optimization counter: {enhanced_norm.optimization_counter}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced layer norm test failed: {e}")
        return False

def test_enhanced_optimization_core():
    """Test the complete enhanced optimization core."""
    logger.info("🧪 Testing Enhanced Optimization Core...")
    
    try:
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(256)
                self.linear1 = nn.Linear(256, 512)
                self.norm2 = nn.LayerNorm(512)
                self.linear2 = nn.Linear(512, 256)
            
            def forward(self, x):
                x = self.norm1(x)
                x = self.linear1(x)
                x = self.norm2(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        enhanced_optimizer = create_enhanced_optimization_core({
            'enable_adaptive_precision': True,
            'enable_dynamic_kernel_fusion': True,
            'enable_intelligent_memory_management': True,
            'enable_self_optimizing_components': True,
            'optimization_aggressiveness': 0.8
        })
        
        enhanced_model, stats = enhanced_optimizer.enhance_optimization_module(model)
        
        x = torch.randn(2, 10, 256)
        output = enhanced_model(x)
        
        logger.info(f"✅ Enhanced model working: {output.shape}")
        logger.info(f"✅ Optimizations applied: {stats['optimizations_applied']}")
        logger.info(f"✅ Enhanced components: {stats['enhanced_components']}")
        logger.info(f"✅ Enhancement time: {stats['enhancement_time']:.4f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced optimization core test failed: {e}")
        return False

def test_optimization_core_enhancement():
    """Test enhancing existing optimization_core components."""
    logger.info("🧪 Testing Optimization Core Enhancement...")
    
    try:
        original_norm = OptimizedLayerNorm(128)
        
        enhanced_optimizer = create_enhanced_optimization_core({
            'enable_adaptive_precision': True,
            'enable_self_optimizing_components': True
        })
        
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = original_norm
            
            def forward(self, x):
                return self.norm(x)
        
        test_module = TestModule()
        enhanced_module, stats = enhanced_optimizer.enhance_optimization_module(test_module)
        
        x = torch.randn(2, 5, 128)
        output = enhanced_module(x)
        
        logger.info(f"✅ Enhanced optimization module working: {output.shape}")
        logger.info(f"✅ Optimizations applied: {stats['optimizations_applied']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization core enhancement test failed: {e}")
        return False

def main():
    """Run all enhanced optimization core tests."""
    logger.info("🚀 Enhanced Optimization Core Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Adaptive Precision Optimizer", test_adaptive_precision_optimizer),
        ("Dynamic Kernel Fusion", test_dynamic_kernel_fusion),
        ("Intelligent Memory Manager", test_intelligent_memory_manager),
        ("Enhanced LayerNorm", test_enhanced_layer_norm),
        ("Enhanced Optimization Core", test_enhanced_optimization_core),
        ("Optimization Core Enhancement", test_optimization_core_enhancement)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing {test_name}...")
        if test_func():
            logger.info(f"✅ {test_name}: PASS")
            passed += 1
        else:
            logger.info(f"❌ {test_name}: FAIL")
    
    logger.info(f"\n📊 Test Results:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if i < passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All enhanced optimization core tests passed!")
    else:
        logger.info(f"⚠️ {total - passed} tests failed")

if __name__ == "__main__":
    main()
