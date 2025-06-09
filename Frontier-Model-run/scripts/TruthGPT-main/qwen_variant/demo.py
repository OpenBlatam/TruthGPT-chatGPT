"""
Qwen variant demonstration script.
"""

import torch
import yaml
from qwen_model import create_qwen_model
from qwen_optimizations import apply_qwen_optimizations
from qwen_benchmarks import run_qwen_benchmarks

def load_config():
    """Load configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def demo_qwen_capabilities():
    """Demonstrate Qwen model capabilities."""
    print("🚀 Qwen Variant Demonstration")
    print("=" * 50)
    
    config = load_config()
    
    print("📊 Creating Qwen-7B model...")
    model = create_qwen_model(config['qwen_7b'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Model Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    print("\n⚡ Applying optimizations...")
    opt_config = config['qwen_optimizations']
    example_input = torch.randint(0, config['qwen_7b']['vocab_size'], (1, 512))
    optimized_model = apply_qwen_optimizations(model, opt_config, example_input)
    
    print("✅ Model optimized successfully!")
    
    print("\n🧪 Testing forward pass...")
    batch_size, seq_len = 2, 1024
    input_ids = torch.randint(0, config['qwen_7b']['vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = optimized_model(input_ids)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {outputs['logits'].shape}")
    
    if 'aux_loss' in outputs and outputs['aux_loss'] is not None:
        print(f"   MoE aux loss: {outputs['aux_loss']:.4f}")
    
    print("\n🏃 Running quick benchmark...")
    benchmark_config = {
        'batch_sizes': [1, 2],
        'sequence_lengths': [512, 1024],
        'num_benchmark_runs': 5,
        'save_results': False
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = run_qwen_benchmarks(optimized_model, benchmark_config, device)
    
    print("🎉 Qwen variant demonstration completed!")

if __name__ == "__main__":
    demo_qwen_capabilities()
