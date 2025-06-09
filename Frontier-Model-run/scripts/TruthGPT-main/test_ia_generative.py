"""
Comprehensive test suite for IA-Generative integration.
"""

import torch
import yaml
from ia_generative import (
    create_text_generator, create_image_generator, 
    create_cross_modal_generator, apply_generative_optimizations,
    create_generative_trainer, run_generative_benchmarks
)

def load_config():
    """Load IA-Generative configuration."""
    with open('ia_generative/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_text_generator():
    """Test text generation capabilities."""
    print("🧪 Testing Text Generator...")
    
    config = load_config()
    text_config = config['text_generator']
    
    generator = create_text_generator(text_config)
    
    print(f"✅ Text generator created successfully")
    print(f"   Generator type: {type(generator)}")
    
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    return generator

def test_image_generator():
    """Test image generation capabilities."""
    print("\n🧪 Testing Image Generator...")
    
    config = load_config()
    image_config = config['image_generator']
    
    generator = create_image_generator(image_config)
    
    print(f"✅ Image generator created successfully")
    print(f"   Generator type: {type(generator)}")
    
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    return generator

def test_cross_modal_generator():
    """Test cross-modal generation capabilities."""
    print("\n🧪 Testing Cross-Modal Generator...")
    
    config = load_config()
    cross_modal_config = config['cross_modal_generator']
    
    generator = create_cross_modal_generator(cross_modal_config)
    
    print(f"✅ Cross-modal generator created successfully")
    print(f"   Generator type: {type(generator)}")
    
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    return generator

def test_generative_optimizations():
    """Test generative optimizations."""
    print("\n🧪 Testing Generative Optimizations...")
    
    config = load_config()
    text_config = config['text_generator']
    opt_config = config['generative_optimizations']
    
    generator = create_text_generator(text_config)
    
    example_input = torch.randint(0, text_config['vocab_size'], (1, 128))
    
    optimized_generator = apply_generative_optimizations(generator, opt_config, example_input)
    
    print(f"✅ Optimizations applied successfully")
    print(f"   Optimized generator type: {type(optimized_generator)}")
    
    return optimized_generator

def test_generative_trainer():
    """Test generative trainer creation."""
    print("\n🧪 Testing Generative Trainer...")
    
    config = load_config()
    text_config = config['text_generator']
    training_config = config['generative_training']
    
    generator = create_text_generator(text_config)
    trainer = create_generative_trainer(generator, training_config)
    
    print(f"✅ Trainer created successfully")
    print(f"   Trainer type: {type(trainer)}")
    print(f"   Using GRPO: {trainer.config.use_grpo}")
    print(f"   Adversarial training: {trainer.config.use_adversarial_training}")
    print(f"   Mixed precision: {trainer.config.use_mixed_precision}")
    
    return trainer

def test_generative_benchmarks():
    """Test generative benchmarking."""
    print("\n🧪 Testing Generative Benchmarks...")
    
    config = load_config()
    text_config = config['text_generator']
    benchmark_config = config['generative_benchmarks']
    
    benchmark_config['batch_sizes'] = [1, 2]
    benchmark_config['sequence_lengths'] = [128, 256]
    benchmark_config['generation_lengths'] = [32, 64]
    benchmark_config['num_benchmark_runs'] = 5
    
    generator = create_text_generator(text_config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = run_generative_benchmarks(generator, benchmark_config, device)
    
    print(f"✅ Benchmarks completed successfully")
    
    return results

def test_forward_passes():
    """Test forward passes for all generators."""
    print("\n🧪 Testing Forward Passes...")
    
    config = load_config()
    
    text_generator = create_text_generator(config['text_generator'])
    text_generator.eval()
    
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config['text_generator']['vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        text_outputs = text_generator(input_ids)
    
    print(f"✅ Text generator forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {text_outputs.shape}")
    
    image_generator = create_image_generator(config['image_generator'])
    image_generator.eval()
    
    noise = torch.randn(batch_size, config['image_generator']['latent_dim'])
    
    with torch.no_grad():
        image_outputs = image_generator(noise)
    
    print(f"✅ Image generator forward pass successful")
    print(f"   Input shape: {noise.shape}")
    print(f"   Output shape: {image_outputs.shape}")
    
    cross_modal_generator = create_cross_modal_generator(config['cross_modal_generator'])
    cross_modal_generator.eval()
    
    text_input = torch.randint(0, config['cross_modal_generator']['text_vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        cross_modal_outputs = cross_modal_generator(text_input)
    
    print(f"✅ Cross-modal generator forward pass successful")
    print(f"   Input shape: {text_input.shape}")
    print(f"   Output shape: {cross_modal_outputs.shape}")
    
    return text_outputs, image_outputs, cross_modal_outputs

def run_all_tests():
    """Run all IA-Generative tests."""
    print("🚀 Starting IA-Generative Test Suite")
    print("=" * 60)
    
    try:
        text_generator = test_text_generator()
        image_generator = test_image_generator()
        cross_modal_generator = test_cross_modal_generator()
        optimized_generator = test_generative_optimizations()
        trainer = test_generative_trainer()
        results = test_generative_benchmarks()
        outputs = test_forward_passes()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("🎉 IA-Generative integration is working correctly!")
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
