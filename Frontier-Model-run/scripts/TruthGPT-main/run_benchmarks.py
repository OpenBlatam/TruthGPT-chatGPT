"""
Comprehensive benchmark runner for comparing original vs optimized models.
"""

import torch
import sys
import os
import yaml
import json
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from variant_optimized import (
    BenchmarkSuite, ModelBenchmark, PerformanceProfiler,
    create_optimized_deepseek_model,
    create_optimized_viral_clipper_model,
    OptimizedBrandAnalyzer, OptimizedContentGenerator
)

try:
    from Frontier_Model_run.models.deepseek_v3 import create_deepseek_model
    from variant.viral_clipper import create_viral_clipper_model
    from brandkit.brand_analyzer import create_brand_analyzer_model
    from brandkit.content_generator import create_content_generator_model
    ORIGINAL_MODELS_AVAILABLE = True
except ImportError:
    ORIGINAL_MODELS_AVAILABLE = False
    print("Warning: Original models not available for comparison")

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'variant_optimized', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_input_generators(config):
    """Create input generators for different models."""
    
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
    
    return {
        'deepseek_test': deepseek_input_gen,
        'viral_test': viral_input_gen,
        'brandkit_test': brandkit_input_gen
    }

def benchmark_optimized_models():
    """Benchmark optimized model variants."""
    print("Benchmarking Optimized Models")
    print("-" * 40)
    
    config = load_config()
    benchmark_suite = BenchmarkSuite()
    
    optimized_deepseek = create_optimized_deepseek_model(config['optimized_deepseek'])
    optimized_viral = create_optimized_viral_clipper_model(config['optimized_viral_clipper'])
    
    from variant_optimized.optimized_brandkit import create_optimized_brand_analyzer_model, create_optimized_content_generator_model
    optimized_brand_analyzer = create_optimized_brand_analyzer_model(config['optimized_brandkit'])
    optimized_content_gen = create_optimized_content_generator_model(config['optimized_brandkit'])
    
    models = {
        'OptimizedDeepSeek': optimized_deepseek,
        'OptimizedViralClipper': optimized_viral,
        'OptimizedBrandAnalyzer': optimized_brand_analyzer,
        'OptimizedContentGenerator': optimized_content_gen
    }
    
    input_generators = create_input_generators(config)
    
    print("Running optimized model benchmarks...")
    results = benchmark_suite.run_comparison(models, input_generators)
    
    return benchmark_suite, results

def benchmark_original_models():
    """Benchmark original model implementations if available."""
    if not ORIGINAL_MODELS_AVAILABLE:
        return None, {}
    
    print("Benchmarking Original Models")
    print("-" * 40)
    
    config = load_config()
    benchmark_suite = BenchmarkSuite()
    
    try:
        original_deepseek_config = {
            'vocab_size': config['optimized_deepseek']['vocab_size'],
            'hidden_size': config['optimized_deepseek']['hidden_size'],
            'num_layers': config['optimized_deepseek']['num_layers'],
            'num_attention_heads': config['optimized_deepseek']['num_attention_heads']
        }
        original_deepseek = create_deepseek_model(original_deepseek_config)
        
        original_viral_config = {
            'hidden_size': config['optimized_viral_clipper']['hidden_size'],
            'num_layers': config['optimized_viral_clipper']['num_layers'],
            'num_attention_heads': config['optimized_viral_clipper']['num_attention_heads']
        }
        original_viral = create_viral_clipper_model(original_viral_config)
        
        original_brand_config = config['optimized_brandkit']
        original_brand_analyzer = create_brand_analyzer_model(original_brand_config)
        original_content_gen = create_content_generator_model(original_brand_config)
        
        models = {
            'OriginalDeepSeek': original_deepseek,
            'OriginalViralClipper': original_viral,
            'OriginalBrandAnalyzer': original_brand_analyzer,
            'OriginalContentGenerator': original_content_gen
        }
        
        input_generators = create_input_generators(config)
        
        print("Running original model benchmarks...")
        results = benchmark_suite.run_comparison(models, input_generators)
        
        return benchmark_suite, results
        
    except Exception as e:
        print(f"Failed to benchmark original models: {e}")
        return None, {}

def compare_performance(optimized_results, original_results=None):
    """Compare performance between optimized and original models."""
    print("\nPerformance Comparison Analysis")
    print("=" * 50)
    
    if original_results:
        print("Optimized vs Original Model Comparison:")
        print("-" * 40)
        
        model_pairs = [
            ('OptimizedDeepSeek', 'OriginalDeepSeek'),
            ('OptimizedViralClipper', 'OriginalViralClipper'),
            ('OptimizedBrandAnalyzer', 'OriginalBrandAnalyzer'),
            ('OptimizedContentGenerator', 'OriginalContentGenerator')
        ]
        
        improvements = {}
        
        for opt_name, orig_name in model_pairs:
            if opt_name in optimized_results and orig_name in original_results:
                opt_results = optimized_results[opt_name]
                orig_results = original_results[orig_name]
                
                if opt_results and orig_results:
                    opt_avg_time = sum(r.inference_time_ms for r in opt_results) / len(opt_results)
                    orig_avg_time = sum(r.inference_time_ms for r in orig_results) / len(orig_results)
                    
                    opt_avg_memory = sum(r.memory_usage_mb for r in opt_results) / len(opt_results)
                    orig_avg_memory = sum(r.memory_usage_mb for r in orig_results) / len(orig_results)
                    
                    time_improvement = ((orig_avg_time - opt_avg_time) / orig_avg_time) * 100
                    memory_improvement = ((orig_avg_memory - opt_avg_memory) / orig_avg_memory) * 100
                    
                    improvements[opt_name] = {
                        'time_improvement_percent': time_improvement,
                        'memory_improvement_percent': memory_improvement,
                        'optimized_time_ms': opt_avg_time,
                        'original_time_ms': orig_avg_time,
                        'optimized_memory_mb': opt_avg_memory,
                        'original_memory_mb': orig_avg_memory
                    }
                    
                    print(f"{opt_name}:")
                    print(f"  Time: {opt_avg_time:.2f}ms vs {orig_avg_time:.2f}ms ({time_improvement:+.1f}%)")
                    print(f"  Memory: {opt_avg_memory:.2f}MB vs {orig_avg_memory:.2f}MB ({memory_improvement:+.1f}%)")
                    print()
        
        return improvements
    
    else:
        print("Optimized Model Performance Summary:")
        print("-" * 40)
        
        for model_name, results in optimized_results.items():
            if results:
                avg_time = sum(r.inference_time_ms for r in results) / len(results)
                avg_memory = sum(r.memory_usage_mb for r in results) / len(results)
                avg_throughput = sum(r.throughput_samples_per_sec for r in results) / len(results)
                total_params = results[0].parameters if results else 0
                
                print(f"{model_name}:")
                print(f"  Avg Inference: {avg_time:.2f}ms")
                print(f"  Avg Memory: {avg_memory:.2f}MB")
                print(f"  Avg Throughput: {avg_throughput:.2f} samples/s")
                print(f"  Parameters: {total_params:,}")
                print()
        
        return {}

def generate_detailed_report(optimized_suite, original_suite=None, improvements=None):
    """Generate a detailed benchmark report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_lines = [
        "# Optimized Model Variants - Performance Benchmark Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report compares the performance of optimized model variants against their original implementations.",
        "The optimized variants include several performance enhancements:",
        "",
        "- **Flash Attention**: Memory-efficient attention computation",
        "- **Gradient Checkpointing**: Reduced memory usage during training",
        "- **Optimized MoE**: Improved mixture-of-experts routing",
        "- **Efficient Fusion**: Streamlined multi-modal processing",
        "- **Streaming Inference**: Real-time processing capabilities",
        "",
    ]
    
    if improvements:
        report_lines.extend([
            "## Performance Improvements",
            "",
            "| Model | Time Improvement | Memory Improvement | Optimized Time (ms) | Original Time (ms) |",
            "|-------|------------------|-------------------|-------------------|------------------|"
        ])
        
        for model_name, metrics in improvements.items():
            time_imp = metrics['time_improvement_percent']
            mem_imp = metrics['memory_improvement_percent']
            opt_time = metrics['optimized_time_ms']
            orig_time = metrics['original_time_ms']
            
            report_lines.append(
                f"| {model_name} | {time_imp:+.1f}% | {mem_imp:+.1f}% | {opt_time:.2f} | {orig_time:.2f} |"
            )
        
        report_lines.append("")
    
    optimized_report = optimized_suite.generate_report()
    report_lines.extend([
        "## Optimized Models Detailed Results",
        "",
        optimized_report,
        ""
    ])
    
    if original_suite:
        original_report = original_suite.generate_report()
        report_lines.extend([
            "## Original Models Detailed Results",
            "",
            original_report,
            ""
        ])
    
    report_lines.extend([
        "## Optimization Techniques Applied",
        "",
        "### DeepSeek-V3 Optimizations",
        "- Optimized Multi-Head Latent Attention with memory-efficient projections",
        "- Enhanced MoE routing with load balancing",
        "- Flash attention for reduced memory footprint",
        "- Gradient checkpointing for training efficiency",
        "",
        "### Viral Clipper Optimizations", 
        "- Efficient multi-modal feature fusion",
        "- Streaming inference buffer for long sequences",
        "- Optimized attention mechanisms for video processing",
        "- Batch processing optimizations",
        "",
        "### Brandkit Optimizations",
        "- Efficient cross-modal attention",
        "- Cached embeddings for repeated computations",
        "- Optimized content generation pipeline",
        "- Memory-efficient brand profile storage",
        "",
        "## Technical Specifications",
        "",
        f"- PyTorch Version: {torch.__version__}",
        f"- CUDA Available: {torch.cuda.is_available()}",
        f"- Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}",
        ""
    ])
    
    report_content = "\n".join(report_lines)
    
    report_filename = f"benchmark_report_{timestamp}.md"
    with open(report_filename, 'w') as f:
        f.write(report_content)
    
    print(f"Detailed report saved to: {report_filename}")
    return report_filename

def save_benchmark_data(optimized_suite, original_suite=None, improvements=None):
    """Save benchmark data to JSON for further analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    data = {
        'timestamp': timestamp,
        'optimized_results': [r.to_dict() for r in optimized_suite.results],
        'optimized_comparison': optimized_suite.get_performance_comparison()
    }
    
    if original_suite:
        data['original_results'] = [r.to_dict() for r in original_suite.results]
        data['original_comparison'] = original_suite.get_performance_comparison()
    
    if improvements:
        data['improvements'] = improvements
    
    filename = f"benchmark_data_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Benchmark data saved to: {filename}")
    return filename

def main():
    """Run comprehensive benchmarks and generate reports."""
    print("Comprehensive Model Benchmark Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        optimized_suite, optimized_results = benchmark_optimized_models()
        original_suite, original_results = benchmark_original_models()
        
        improvements = compare_performance(optimized_results, original_results)
        
        report_file = generate_detailed_report(optimized_suite, original_suite, improvements)
        data_file = save_benchmark_data(optimized_suite, original_suite, improvements)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n" + "=" * 50)
        print("✓ Benchmark suite completed successfully!")
        print(f"✓ Total execution time: {total_time:.2f} seconds")
        print(f"✓ Report generated: {report_file}")
        print(f"✓ Data saved: {data_file}")
        
        if improvements:
            print(f"\n📊 Performance Summary:")
            for model_name, metrics in improvements.items():
                time_imp = metrics['time_improvement_percent']
                mem_imp = metrics['memory_improvement_percent']
                print(f"  {model_name}: {time_imp:+.1f}% time, {mem_imp:+.1f}% memory")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
