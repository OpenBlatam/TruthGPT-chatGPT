#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Frontier-Model-run'))

import torch
from models.llama_3_1_405b import create_llama_3_1_405b_model
from models.claude_3_5_sonnet import create_claude_3_5_sonnet_model
from models.deepseek_v3 import create_deepseek_v3_model

def test_all_native_models():
    """Test all native model implementations."""
    print("Testing All Native Model Implementations")
    print("=" * 50)
    
    models_to_test = [
        ("Llama-3.1-405B", create_llama_3_1_405b_model, {
            'dim': 256,
            'n_layers': 2,
            'n_heads': 4,
            'n_kv_heads': 2,
            'vocab_size': 1000,
            'max_seq_len': 64,
            'use_flash_attention': False,
            'use_gradient_checkpointing': False,
            'use_quantization': False,
        }),
        ("Claude-3.5-Sonnet", create_claude_3_5_sonnet_model, {
            'dim': 256,
            'n_layers': 2,
            'n_heads': 4,
            'n_kv_heads': 2,
            'vocab_size': 1000,
            'max_seq_len': 64,
            'use_constitutional_ai': True,
            'use_flash_attention': False,
            'use_gradient_checkpointing': False,
            'use_quantization': False,
            'use_mixture_of_depths': False,
        }),
        ("DeepSeek-V3", create_deepseek_v3_model, {
            'hidden_size': 256,
            'num_layers': 2,
            'num_heads': 4,
            'vocab_size': 1000,
            'max_seq_len': 64,
            'use_native_implementation': True,
            'use_fp8': False,
            'q_lora_rank': 128,
            'kv_lora_rank': 64,
            'n_routed_experts': 8,
            'n_shared_experts': 2,
            'n_activated_experts': 2,
        })
    ]
    
    all_passed = True
    
    for model_name, create_func, config in models_to_test:
        print(f"\n🧪 Testing {model_name}...")
        
        try:
            model = create_func(config)
            print(f"✓ {model_name} instantiated successfully")
            
            batch_size = 2
            seq_len = 16
            vocab_size = config.get('vocab_size', 1000)
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
            
            expected_shape = (batch_size, seq_len, vocab_size)
            if output.shape == expected_shape:
                print(f"✓ {model_name} forward pass successful")
                print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            else:
                print(f"✗ {model_name} forward pass failed: expected {expected_shape}, got {output.shape}")
                all_passed = False
                
        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")
            all_passed = False
    
    return all_passed

def test_optimization_integration():
    """Test optimization integration with all models."""
    print("\n🔧 Testing Optimization Integration")
    print("=" * 40)
    
    try:
        from optimization_core.memory_optimizations import MemoryOptimizer, MemoryOptimizationConfig
        from optimization_core.computational_optimizations import ComputationalOptimizer
        from optimization_core.optimization_profiles import apply_optimization_profile
        
        config = {
            'dim': 256,
            'n_layers': 2,
            'n_heads': 4,
            'n_kv_heads': 2,
            'vocab_size': 1000,
            'max_seq_len': 64,
            'use_flash_attention': False,
            'use_gradient_checkpointing': False,
            'use_quantization': False,
        }
        
        model = create_llama_3_1_405b_model(config)
        
        mem_config = MemoryOptimizationConfig(
            enable_fp16=True,
            enable_gradient_checkpointing=True,
            enable_quantization=False
        )
        mem_optimizer = MemoryOptimizer(mem_config)
        optimized_model = mem_optimizer.optimize_model(model)
        
        print("✓ Memory optimization works with native models")
        
        comp_optimizer = ComputationalOptimizer({
            'use_fused_attention': False,
            'optimize_batch_size': True
        })
        comp_optimized_model = comp_optimizer.optimize_model(optimized_model)
        
        print("✓ Computational optimization works with native models")
        
        profile_optimized_model = apply_optimization_profile(model, "balanced")
        
        print("✓ Optimization profiles work with native models")
        return True
        
    except Exception as e:
        print(f"✗ Optimization integration test failed: {e}")
        return False

def test_benchmarking_integration():
    """Test benchmarking framework integration."""
    print("\n📊 Testing Benchmarking Integration")
    print("=" * 40)
    
    try:
        from benchmarking_framework.model_registry import ModelRegistry
        from benchmarking_framework.comparative_benchmark import ComparativeBenchmark
        
        registry = ModelRegistry()
        
        llama_model = registry.get_model("TruthGPT-Llama-3.1-405B")
        claude_model = registry.get_model("TruthGPT-Claude-3.5-Sonnet")
        
        if llama_model and claude_model:
            print("✓ Native models registered in model registry")
            print(f"  - Llama-3.1-405B: {llama_model.parameters:,} parameters")
            print(f"  - Claude-3.5-Sonnet: {claude_model.parameters:,} parameters")
        else:
            print("✗ Native models not found in registry")
            return False
        
        benchmark = ComparativeBenchmark()
        best_models = benchmark.registry.get_best_models_only()
        
        truthgpt_models = best_models["truthgpt_models"]
        model_names = [m.name for m in truthgpt_models if m]
        
        if "TruthGPT-Llama-3.1-405B" in model_names and "TruthGPT-Claude-3.5-Sonnet" in model_names:
            print("✓ Native models included in comparative benchmarking")
            return True
        else:
            print("✗ Native models not included in comparative benchmarking")
            return False
            
    except Exception as e:
        print(f"✗ Benchmarking integration test failed: {e}")
        return False

def main():
    print("Native Models Integration Test Suite")
    print("=" * 50)
    
    success = test_all_native_models()
    if not success:
        print("\n❌ Native models test failed")
        return False
    
    success = test_optimization_integration()
    if not success:
        print("\n❌ Optimization integration test failed")
        return False
    
    success = test_benchmarking_integration()
    if not success:
        print("\n❌ Benchmarking integration test failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All integration tests passed!")
    print("✅ Native Llama-3.1-405B and Claude-3.5-Sonnet models are fully integrated")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
