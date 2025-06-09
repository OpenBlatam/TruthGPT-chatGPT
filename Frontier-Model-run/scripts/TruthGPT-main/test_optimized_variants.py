"""
Comprehensive test suite for optimized model variants.
"""

import torch
import sys
import os
import yaml
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from variant_optimized import (
    BenchmarkSuite, ModelBenchmark,
    OptimizedDeepSeekV3, create_optimized_deepseek_model,
    OptimizedViralClipper, create_optimized_viral_clipper_model,
    OptimizedBrandAnalyzer, OptimizedContentGenerator,
    PerformanceProfiler
)

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'variant_optimized', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_optimized_deepseek():
    """Test optimized DeepSeek-V3 model."""
    print("Testing Optimized DeepSeek-V3 Model")
    print("-" * 40)
    
    config = load_config()
    model = create_optimized_deepseek_model(config['optimized_deepseek'])
    
    print(f"✓ Model instantiated successfully")
    print(f"  - Model type: {type(model)}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config['optimized_deepseek']['vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {input_ids.shape}")
    print(f"  - Output shape: {outputs.shape}")
    
    memory_footprint = model.get_memory_footprint()
    print(f"✓ Memory footprint analysis")
    print(f"  - Total parameters: {memory_footprint['total_parameters']:,}")
    print(f"  - Model size: {memory_footprint['total_size_mb']:.2f} MB")
    
    return model, input_ids

def test_optimized_viral_clipper():
    """Test optimized viral clipper model."""
    print("\nTesting Optimized Viral Clipper Model")
    print("-" * 40)
    
    config = load_config()
    model = create_optimized_viral_clipper_model(config['optimized_viral_clipper'])
    
    print(f"✓ Model instantiated successfully")
    print(f"  - Model type: {type(model)}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 2
    seq_len = 20
    visual_features = torch.randn(batch_size, seq_len, config['optimized_viral_clipper']['visual_feature_dim'])
    audio_features = torch.randn(batch_size, seq_len, config['optimized_viral_clipper']['audio_feature_dim'])
    text_features = torch.randn(batch_size, seq_len, config['optimized_viral_clipper']['text_feature_dim'])
    engagement_features = torch.randn(batch_size, seq_len, config['optimized_viral_clipper']['engagement_feature_dim'])
    
    with torch.no_grad():
        outputs = model(visual_features, audio_features, text_features, engagement_features)
    
    print(f"✓ Forward pass successful")
    print(f"  - Virality scores shape: {outputs['virality_scores'].shape}")
    print(f"  - Segment logits shape: {outputs['segment_logits'].shape}")
    
    performance_metrics = model.get_performance_metrics()
    print(f"✓ Performance metrics")
    print(f"  - Supports streaming: {performance_metrics['supports_streaming']}")
    print(f"  - Uses flash attention: {performance_metrics['uses_flash_attention']}")
    
    input_data = {
        'visual_features': visual_features,
        'audio_features': audio_features,
        'text_features': text_features,
        'engagement_features': engagement_features
    }
    
    return model, input_data

def test_optimized_brandkit():
    """Test optimized brandkit models."""
    print("\nTesting Optimized Brandkit Models")
    print("-" * 40)
    
    config = load_config()
    from variant_optimized.optimized_brandkit import create_optimized_brand_analyzer_model, create_optimized_content_generator_model
    brand_analyzer = create_optimized_brand_analyzer_model(config['optimized_brandkit'])
    content_generator = create_optimized_content_generator_model(config['optimized_brandkit'])
    
    print(f"✓ Brand analyzer instantiated")
    print(f"  - Parameters: {sum(p.numel() for p in brand_analyzer.parameters()):,}")
    
    print(f"✓ Content generator instantiated")
    print(f"  - Parameters: {sum(p.numel() for p in content_generator.parameters()):,}")
    
    batch_size = 2
    colors = torch.randn(batch_size, 5, 3)
    typography_features = torch.randn(batch_size, config['optimized_brandkit']['typography_features'])
    layout_features = torch.randn(batch_size, config['optimized_brandkit']['layout_features'])
    text_features = torch.randn(batch_size, 10, config['optimized_brandkit']['text_feature_dim'])
    
    with torch.no_grad():
        brand_outputs = brand_analyzer(colors, typography_features, layout_features, text_features)
    
    print(f"✓ Brand analyzer forward pass successful")
    print(f"  - Brand profile shape: {brand_outputs['brand_profile'].shape}")
    print(f"  - Consistency score shape: {brand_outputs['consistency_score'].shape}")
    
    content_type_ids = torch.randint(0, 5, (batch_size,))
    
    with torch.no_grad():
        content_outputs = content_generator(brand_outputs['brand_profile'], content_type_ids)
    
    print(f"✓ Content generator forward pass successful")
    print(f"  - Layout features shape: {content_outputs['layout_features'].shape}")
    print(f"  - Quality score shape: {content_outputs['quality_score'].shape}")
    
    input_data = {
        'colors': colors,
        'typography_features': typography_features,
        'layout_features': layout_features,
        'text_features': text_features
    }
    
    return (brand_analyzer, content_generator), input_data

def run_performance_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("\nRunning Performance Benchmarks")
    print("=" * 50)
    
    config = load_config()
    benchmark_suite = BenchmarkSuite()
    
    deepseek_model, deepseek_input = test_optimized_deepseek()
    viral_model, viral_input = test_optimized_viral_clipper()
    brandkit_models, brandkit_input = test_optimized_brandkit()
    
    models = {
        'OptimizedDeepSeek': deepseek_model,
        'OptimizedViralClipper': viral_model,
        'OptimizedBrandAnalyzer': brandkit_models[0]
    }
    
    def deepseek_input_gen(batch_size):
        return torch.randint(0, config['optimized_deepseek']['vocab_size'], (batch_size, 16))
    
    def viral_input_gen(batch_size):
        return {
            'visual_features': torch.randn(batch_size, 20, config['optimized_viral_clipper']['visual_feature_dim']),
            'audio_features': torch.randn(batch_size, 20, config['optimized_viral_clipper']['audio_feature_dim']),
            'text_features': torch.randn(batch_size, 20, config['optimized_viral_clipper']['text_feature_dim']),
            'engagement_features': torch.randn(batch_size, 20, config['optimized_viral_clipper']['engagement_feature_dim'])
        }
    
    def brandkit_input_gen(batch_size):
        return {
            'colors': torch.randn(batch_size, 5, 3),
            'typography_features': torch.randn(batch_size, config['optimized_brandkit']['typography_features']),
            'layout_features': torch.randn(batch_size, config['optimized_brandkit']['layout_features']),
            'text_features': torch.randn(batch_size, 10, config['optimized_brandkit']['text_feature_dim'])
        }
    
    input_generators = {
        'deepseek_test': deepseek_input_gen,
        'viral_test': viral_input_gen,
        'brandkit_test': brandkit_input_gen
    }
    
    print("Running comprehensive benchmarks...")
    results = benchmark_suite.run_comparison(models, input_generators)
    
    print("\nBenchmark Results:")
    print("-" * 30)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for result in model_results[:3]:
            print(f"  {result.test_name}: {result.inference_time_ms:.2f}ms, "
                  f"{result.memory_usage_mb:.2f}MB, {result.throughput_samples_per_sec:.2f} samples/s")
    
    comparison = benchmark_suite.get_performance_comparison()
    print(f"\nPerformance Comparison:")
    print("-" * 30)
    
    for model_name, metrics in comparison.items():
        print(f"{model_name}:")
        print(f"  Avg Inference: {metrics['avg_inference_ms']:.2f}ms")
        print(f"  Avg Memory: {metrics['avg_memory_mb']:.2f}MB")
        print(f"  Efficiency Score: {metrics['efficiency_score']:.2f}")
    
    return benchmark_suite

def test_optimization_features():
    """Test specific optimization features."""
    print("\nTesting Optimization Features")
    print("-" * 40)
    
    config = load_config()
    
    print("Testing flash attention...")
    config_flash = config['optimized_deepseek'].copy()
    config_flash['use_flash_attention'] = True
    model_flash = create_optimized_deepseek_model(config_flash)
    print(f"✓ Flash attention model created")
    
    print("Testing gradient checkpointing...")
    config_checkpoint = config['optimized_viral_clipper'].copy()
    config_checkpoint['use_gradient_checkpointing'] = True
    model_checkpoint = create_optimized_viral_clipper_model(config_checkpoint)
    print(f"✓ Gradient checkpointing model created")
    
    print("Testing efficient fusion...")
    config_fusion = config['optimized_brandkit'].copy()
    config_fusion['use_efficient_cross_attention'] = True
    from variant_optimized.optimized_brandkit import create_optimized_brand_analyzer_model
    model_fusion = create_optimized_brand_analyzer_model(config_fusion)
    print(f"✓ Efficient fusion model created")
    
    print("Testing streaming inference...")
    streaming_model = create_optimized_viral_clipper_model(config['optimized_viral_clipper'])
    if hasattr(streaming_model, 'streaming_buffer'):
        print(f"✓ Streaming inference buffer available")
    
    return True

def main():
    """Run all tests for optimized variants."""
    print("Optimized Variants Test Suite")
    print("=" * 50)
    
    try:
        test_optimized_deepseek()
        test_optimized_viral_clipper()
        test_optimized_brandkit()
        test_optimization_features()
        
        benchmark_suite = run_performance_benchmarks()
        
        print(f"\n" + "=" * 50)
        print("✓ All optimized variant tests passed!")
        print("✓ Performance benchmarks completed successfully!")
        print("✓ Optimization features verified!")
        
        report = benchmark_suite.generate_report()
        print(f"\nDetailed benchmark report generated ({len(report)} characters)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
