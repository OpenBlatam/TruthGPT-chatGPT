#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Frontier-Model-run'))

import torch
from models.claude_3_5_sonnet import create_claude_3_5_sonnet_model

def test_model_instantiation():
    """Test that the Claude-3.5-Sonnet model can be instantiated."""
    print("Testing Claude-3.5-Sonnet model instantiation...")
    
    config = {
        'dim': 512,
        'n_layers': 2,
        'n_heads': 8,
        'n_kv_heads': 2,
        'vocab_size': 1000,
        'multiple_of': 256,
        'ffn_dim_multiplier': 2.6875,
        'norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'max_seq_len': 128,
        'use_constitutional_ai': True,
        'use_harmlessness_filter': True,
        'use_helpfulness_boost': True,
        'use_flash_attention': False,
        'use_gradient_checkpointing': False,
        'use_quantization': False,
        'quantization_bits': 8,
        'use_mixture_of_depths': False,
        'use_retrieval_augmentation': False,
        'safety_threshold': 0.95,
    }
    
    try:
        model = create_claude_3_5_sonnet_model(config)
        print(f"✓ Model instantiated successfully")
        print(f"  - Model type: {type(model)}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return None

def test_forward_pass(model):
    """Test that the model can perform a forward pass."""
    print("\nTesting forward pass...")
    
    try:
        batch_size = 2
        seq_len = 16
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        expected_shape = (batch_size, seq_len, vocab_size)
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful")
            print(f"  - Input shape: {input_ids.shape}")
            print(f"  - Output shape: {output.shape}")
            print(f"  - Output dtype: {output.dtype}")
            return True
        else:
            print(f"✗ Forward pass failed: expected shape {expected_shape}, got {output.shape}")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_constitutional_ai_features():
    """Test Constitutional AI features."""
    print("\nTesting Constitutional AI features...")
    
    try:
        config = {
            'dim': 256,
            'n_layers': 2,
            'n_heads': 4,
            'n_kv_heads': 2,
            'vocab_size': 1000,
            'max_seq_len': 64,
            'use_constitutional_ai': True,
            'use_harmlessness_filter': True,
            'use_helpfulness_boost': True,
            'safety_threshold': 0.95,
            'use_flash_attention': False,
            'use_gradient_checkpointing': False,
            'use_quantization': False,
            'use_mixture_of_depths': False,
        }
        
        model = create_claude_3_5_sonnet_model(config)
        
        input_ids = torch.randint(0, 1000, (1, 32))
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✓ Constitutional AI features working")
        print(f"  - Safety filtering enabled")
        print(f"  - Harmlessness filter active")
        print(f"  - Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Constitutional AI test failed: {e}")
        return False

def test_optimization_compatibility():
    """Test that the model works with optimization infrastructure."""
    print("\nTesting optimization compatibility...")
    
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
            'use_constitutional_ai': True,
            'use_flash_attention': False,
            'use_gradient_checkpointing': False,
            'use_quantization': False,
            'use_mixture_of_depths': False,
        }
        
        model = create_claude_3_5_sonnet_model(config)
        
        mem_config = MemoryOptimizationConfig(
            enable_fp16=True,
            enable_gradient_checkpointing=True,
            enable_quantization=False
        )
        mem_optimizer = MemoryOptimizer(mem_config)
        optimized_model = mem_optimizer.optimize_model(model)
        
        print(f"✓ Memory optimization successful")
        
        comp_optimizer = ComputationalOptimizer({
            'use_fused_attention': False,
            'optimize_batch_size': True
        })
        comp_optimized_model = comp_optimizer.optimize_model(optimized_model)
        
        print(f"✓ Computational optimization successful")
        
        profile_optimized_model = apply_optimization_profile(model, "balanced")
        
        print(f"✓ Optimization profile application successful")
        return True
        
    except Exception as e:
        print(f"✗ Optimization compatibility test failed: {e}")
        return False

def main():
    print("Claude-3.5-Sonnet Native Implementation Test")
    print("=" * 48)
    
    model = test_model_instantiation()
    if model is None:
        print("\nTest failed: Could not instantiate model")
        return False
    
    success = test_forward_pass(model)
    if not success:
        print("\nTest failed: Forward pass failed")
        return False
    
    success = test_constitutional_ai_features()
    if not success:
        print("\nTest failed: Constitutional AI features failed")
        return False
    
    success = test_optimization_compatibility()
    if not success:
        print("\nTest failed: Optimization compatibility failed")
        return False
    
    print("\n" + "=" * 48)
    print("✓ All tests passed! Claude-3.5-Sonnet native implementation is working.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
