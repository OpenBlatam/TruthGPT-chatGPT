#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Frontier-Model-run'))

import torch
from models.deepseek_v3 import create_deepseek_v3_model

def test_model_instantiation():
    """Test that the DeepSeek-V3 model can be instantiated."""
    print("Testing DeepSeek-V3 model instantiation...")
    
    config = {
        'hidden_size': 512,
        'num_hidden_layers': 2,
        'num_attention_heads': 8,
        'num_key_value_heads': None,
        'vocab_size': 1000,
        'layer_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'max_position_embeddings': 128,
        'q_lora_rank': 256,
        'kv_lora_rank': 128,
        'qk_rope_head_dim': 32,
        'v_head_dim': 64,
        'qk_nope_head_dim': 32,
        'n_routed_experts': 8,
        'n_shared_experts': 1,
        'n_activated_experts': 2,
        'moe_intermediate_size': 256,
        'shared_intermediate_size': 256,
        'use_fp8': False,
        'original_seq_len': 128,
        'rope_factor': 1.0,
        'beta_fast': 32,
        'beta_slow': 1,
        'mscale': 1.0
    }
    
    try:
        model = create_deepseek_v3_model(config)
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

def main():
    print("DeepSeek-V3 Native Implementation Test")
    print("=" * 40)
    
    model = test_model_instantiation()
    if model is None:
        print("\nTest failed: Could not instantiate model")
        return False
    
    success = test_forward_pass(model)
    if not success:
        print("\nTest failed: Forward pass failed")
        return False
    
    print("\n" + "=" * 40)
    print("✓ All tests passed! DeepSeek-V3 native implementation is working.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
