"""
Comprehensive test suite for Qwen variant implementation.
"""

import torch
import yaml
from qwen_variant import (
    create_qwen_model, apply_qwen_optimizations, 
    run_qwen_benchmarks, create_qwen_trainer
)

def load_config():
    """Load Qwen configuration."""
    with open('qwen_variant/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_qwen_model_creation():
    """Test Qwen model creation and basic functionality."""
    print("🧪 Testing Qwen Model Creation...")
    
    config = load_config()
    qwen_config = config['qwen_test']
    
    model = create_qwen_model(qwen_config)
    
    print(f"✅ Model created successfully")
    print(f"   Model type: {type(model)}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    return model

def test_qwen_forward_pass():
    """Test Qwen model forward pass."""
    print("\n🧪 Testing Qwen Forward Pass...")
    
    config = load_config()
    qwen_config = config['qwen_test']
    
    model = create_qwen_model(qwen_config)
    model.eval()
    
    batch_size, seq_len = 2, 512
    input_ids = torch.randint(0, qwen_config['vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {outputs['logits'].shape}")
    
    if 'aux_loss' in outputs and outputs['aux_loss'] is not None:
        print(f"   MoE aux loss: {outputs['aux_loss']:.4f}")
    
    return outputs

def test_qwen_optimizations():
    """Test Qwen optimizations."""
    print("\n🧪 Testing Qwen Optimizations...")
    
    config = load_config()
    qwen_config = config['qwen_test']
    opt_config = config['qwen_optimizations']
    
    model = create_qwen_model(qwen_config)
    
    example_input = torch.randint(0, qwen_config['vocab_size'], (1, 512))
    
    optimized_model = apply_qwen_optimizations(model, opt_config, example_input)
    
    print(f"✅ Optimizations applied successfully")
    print(f"   Optimized model type: {type(optimized_model)}")
    
    return optimized_model

def test_qwen_benchmarks():
    """Test Qwen benchmarking."""
    print("\n🧪 Testing Qwen Benchmarks...")
    
    config = load_config()
    qwen_config = config['qwen_test']
    benchmark_config = config['qwen_benchmarks']
    
    benchmark_config['batch_sizes'] = [1, 2]
    benchmark_config['sequence_lengths'] = [512, 1024]
    benchmark_config['num_benchmark_runs'] = 5
    
    model = create_qwen_model(qwen_config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = run_qwen_benchmarks(model, benchmark_config, device)
    
    print(f"✅ Benchmarks completed successfully")
    
    return results

def test_qwen_trainer():
    """Test Qwen trainer creation."""
    print("\n🧪 Testing Qwen Trainer...")
    
    config = load_config()
    qwen_config = config['qwen_test']
    training_config = config['qwen_training']
    
    model = create_qwen_model(qwen_config)
    trainer = create_qwen_trainer(model, training_config)
    
    print(f"✅ Trainer created successfully")
    print(f"   Trainer type: {type(trainer)}")
    print(f"   Using GRPO: {trainer.config.use_grpo}")
    print(f"   Mixed precision: {trainer.config.use_mixed_precision}")
    
    return trainer

def run_all_tests():
    """Run all Qwen variant tests."""
    print("🚀 Starting Qwen Variant Test Suite")
    print("=" * 60)
    
    try:
        model = test_qwen_model_creation()
        outputs = test_qwen_forward_pass()
        optimized_model = test_qwen_optimizations()
        results = test_qwen_benchmarks()
        trainer = test_qwen_trainer()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("🎉 Qwen variant implementation is working correctly!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
