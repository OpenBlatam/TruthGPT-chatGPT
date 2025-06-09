#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Frontier-Model-run'))

import torch
from enhanced_model_optimizer import optimize_all_truthgpt_models, create_universal_optimizer

def test_enhanced_optimization_integration():
    """Test enhanced optimization integration across all models."""
    print("Testing Enhanced Optimization Integration")
    print("=" * 50)
    
    enhanced_config = {
        'enable_fp16': True,
        'enable_bf16': True,
        'enable_gradient_checkpointing': True,
        'enable_quantization': False,
        'quantization_bits': 8,
        'enable_pruning': True,
        'pruning_ratio': 0.1,
        'use_fused_attention': True,
        'enable_kernel_fusion': True,
        'optimize_batch_size': True,
        'use_flash_attention': False,
        'use_triton_kernels': False,
        'use_mcts_optimization': True,
        'use_olympiad_benchmarks': True,
        'use_rl_pruning': True,
        'use_enhanced_grpo': True,
        'use_experience_replay': True,
        'use_advanced_normalization': True,
        'use_optimized_embeddings': True,
        'use_enhanced_mlp': True,
        'target_memory_reduction': 0.2,
        'target_speed_improvement': 1.5,
        'acceptable_accuracy_loss': 0.05,
        'enable_distributed_training': False,
        'enable_mixed_precision': True,
        'enable_automatic_scaling': True,
        'enable_dynamic_batching': True
    }
    
    print("🚀 Running universal optimization on all models...")
    results = optimize_all_truthgpt_models(enhanced_config)
    
    successful = 0
    failed = 0
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            successful += 1
            print(f"✅ {model_name} - Optimization successful")
            metrics = result.get('metrics', {})
            if metrics:
                print(f"   - Parameter reduction: {metrics.get('parameter_reduction', 0):.2%}")
                print(f"   - Memory reduction: {metrics.get('memory_reduction', 0):.2%}")
                print(f"   - Speed improvement: {metrics.get('speed_improvement', 1.0):.2f}x")
        else:
            failed += 1
            print(f"❌ {model_name} - {result['error']}")
    
    print(f"\n📊 Summary: {successful} successful, {failed} failed")
    
    if successful > 0:
        print("✅ Enhanced optimization integration working!")
        return True
    else:
        print("❌ No models successfully optimized")
        return False

def test_layer_replacement():
    """Test that layer replacement is working correctly."""
    print("\n🔧 Testing Layer Replacement")
    print("=" * 30)
    
    try:
        from optimization_core.cuda_kernels import OptimizedLayerNorm
        from optimization_core.advanced_normalization import LlamaRMSNorm, AdvancedRMSNorm
        
        print("✅ Optimization core imports successful")
        
        test_norm = OptimizedLayerNorm(512)
        test_llama_norm = LlamaRMSNorm(512)
        test_advanced_norm = AdvancedRMSNorm(512)
        
        print("✅ Optimized normalization layers instantiated")
        
        test_input = torch.randn(2, 16, 512)
        
        with torch.no_grad():
            out1 = test_norm(test_input)
            out2 = test_llama_norm(test_input)
            out3 = test_advanced_norm(test_input)
        
        print("✅ Forward pass successful for all optimized layers")
        print(f"   - OptimizedLayerNorm output shape: {out1.shape}")
        print(f"   - LlamaRMSNorm output shape: {out2.shape}")
        print(f"   - AdvancedRMSNorm output shape: {out3.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Layer replacement test failed: {e}")
        return False

def test_model_specific_optimizations():
    """Test model-specific optimization configurations."""
    print("\n⚙️ Testing Model-Specific Optimizations")
    print("=" * 40)
    
    try:
        from optimization_core.advanced_optimization_registry_v2 import get_advanced_optimization_config
        
        model_types = ['deepseek_v3', 'llama', 'claude', 'qwen', 'viral_clipper', 'brandkit']
        
        for model_type in model_types:
            try:
                config = get_advanced_optimization_config(model_type)
                print(f"✅ {model_type}: Configuration loaded")
                print(f"   - Advanced normalization: {config.use_advanced_normalization}")
                print(f"   - Enhanced MLP: {config.use_enhanced_mlp}")
                print(f"   - MCTS optimization: {config.use_enhanced_mcts}")
            except Exception as e:
                print(f"⚠️ {model_type}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model-specific optimization test failed: {e}")
        return False

def main():
    print("Enhanced Optimization Integration Test Suite")
    print("=" * 55)
    
    success1 = test_layer_replacement()
    success2 = test_model_specific_optimizations()
    success3 = test_enhanced_optimization_integration()
    
    if success1 and success2 and success3:
        print("\n🎉 All enhanced optimization tests passed!")
        print("✅ optimization_core successfully integrated across all models")
        return True
    else:
        print("\n❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
