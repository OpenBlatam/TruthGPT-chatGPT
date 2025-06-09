#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from claude_api import create_claude_api_model, ClaudeAPIConfig, ClaudeAPIOptimizer

def test_claude_api_instantiation():
    """Test that Claude API model can be instantiated."""
    print("Testing Claude API model instantiation...")
    
    config = {
        'model_name': 'claude-3-5-sonnet-20241022',
        'max_tokens': 1000,
        'temperature': 0.7,
        'use_optimization_core': True,
        'enable_caching': True
    }
    
    try:
        model = create_claude_api_model(config)
        print(f"✓ Claude API model instantiated successfully")
        print(f"  - Model type: {type(model)}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Config: {model.config.model_name}")
        return model
    except Exception as e:
        print(f"✗ Claude API model instantiation failed: {e}")
        return None

def test_claude_api_forward_pass(model):
    """Test that the Claude API model can perform a forward pass."""
    print("\nTesting Claude API forward pass...")
    
    try:
        batch_size = 2
        seq_len = 16
        vocab_size = 100000
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
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

def test_claude_api_text_generation(model):
    """Test Claude API text generation."""
    print("\nTesting Claude API text generation...")
    
    try:
        test_prompt = "What is artificial intelligence?"
        
        response = model.generate_text(test_prompt)
        
        if response and len(response) > 0:
            print(f"✓ Text generation successful")
            print(f"  - Prompt: {test_prompt}")
            print(f"  - Response length: {len(response)} characters")
            print(f"  - Response preview: {response[:100]}...")
            return True
        else:
            print(f"✗ Text generation failed: empty response")
            return False
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        return False

def test_claude_api_optimization():
    """Test Claude API optimization features."""
    print("\nTesting Claude API optimization...")
    
    try:
        optimizer = ClaudeAPIOptimizer({
            'enable_request_batching': True,
            'enable_response_caching': True,
            'enable_prompt_optimization': True
        })
        
        test_prompt = "This is a test prompt for optimization"
        optimized_prompt = optimizer.optimize_prompt(test_prompt)
        
        print(f"✓ Claude API optimization successful")
        print(f"  - Original prompt: {test_prompt}")
        print(f"  - Optimized prompt: {optimized_prompt}")
        
        stats = optimizer.get_optimization_stats()
        print(f"  - Optimization stats: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ Claude API optimization failed: {e}")
        return False

def test_optimization_core_integration():
    """Test optimization_core integration with Claude API."""
    print("\nTesting optimization_core integration...")
    
    try:
        from enhanced_model_optimizer import create_universal_optimizer
        
        config = {
            'model_name': 'claude-3-5-sonnet-20241022',
            'use_optimization_core': True
        }
        
        model = create_claude_api_model(config)
        
        optimizer = create_universal_optimizer({
            'enable_fp16': True,
            'enable_gradient_checkpointing': True,
            'use_advanced_normalization': True,
            'use_enhanced_mlp': True
        })
        
        optimized_model = optimizer.optimize_model(model, "claude_api")
        
        print(f"✓ optimization_core integration successful")
        print(f"  - Optimized model type: {type(optimized_model)}")
        
        stats = model.get_stats()
        print(f"  - Model stats: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ optimization_core integration failed: {e}")
        return False

def main():
    print("Claude API Integration Test Suite")
    print("=" * 45)
    
    model = test_claude_api_instantiation()
    if model is None:
        print("\nTest failed: Could not instantiate Claude API model")
        return False
    
    success1 = test_claude_api_forward_pass(model)
    success2 = test_claude_api_text_generation(model)
    success3 = test_claude_api_optimization()
    success4 = test_optimization_core_integration()
    
    if success1 and success2 and success3 and success4:
        print("\n" + "=" * 45)
        print("✓ All Claude API tests passed!")
        print("✅ Claude API integration with optimization_core working correctly")
        return True
    else:
        print("\n" + "=" * 45)
        print("❌ Some Claude API tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
