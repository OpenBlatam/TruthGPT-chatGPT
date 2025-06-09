"""
Comprehensive Qwen benchmarking script with detailed analysis.
"""

import torch
import yaml
import json
import time
from qwen_variant import create_qwen_model, apply_qwen_optimizations, run_qwen_benchmarks

def load_config():
    """Load configuration from YAML file."""
    with open('qwen_variant/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_benchmark_report(results, model_name):
    """Create detailed benchmark report."""
    report = {
        'model_name': model_name,
        'timestamp': int(time.time()),
        'model_analysis': results.get('model_analysis', {}),
        'model_size': results.get('model_size', {}),
        'moe_analysis': results.get('moe_analysis', {}),
        'performance_summary': {},
        'detailed_results': results.get('performance', {})
    }
    
    if 'performance' in results:
        best_throughput = 0
        best_latency = float('inf')
        best_memory = float('inf')
        
        for test_key, test_results in results['performance'].items():
            if 'error' not in test_results:
                if 'throughput' in test_results:
                    throughput = test_results['throughput']['tokens_per_second']
                    if throughput > best_throughput:
                        best_throughput = throughput
                        report['performance_summary']['best_throughput_config'] = test_key
                
                if 'latency' in test_results:
                    latency = test_results['latency']['mean_time_ms']
                    if latency < best_latency:
                        best_latency = latency
                        report['performance_summary']['best_latency_config'] = test_key
                
                if 'memory' in test_results:
                    memory = test_results['memory']['peak_memory_mb']
                    if memory < best_memory:
                        best_memory = memory
                        report['performance_summary']['best_memory_config'] = test_key
        
        report['performance_summary']['best_throughput_tokens_per_sec'] = best_throughput
        report['performance_summary']['best_latency_ms'] = best_latency
        report['performance_summary']['best_memory_mb'] = best_memory
    
    return report

def benchmark_qwen_models():
    """Benchmark all Qwen model variants."""
    print("🚀 Starting Comprehensive Qwen Benchmarking")
    print("=" * 80)
    
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🖥️  Device: {device}")
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    models_to_test = ['qwen_7b', 'qwen_14b', 'qwen_72b']
    all_results = {}
    
    for model_name in models_to_test:
        if model_name not in config:
            print(f"⚠️  Skipping {model_name} - configuration not found")
            continue
        
        print(f"\n🔬 Benchmarking {model_name.upper()}")
        print("-" * 60)
        
        try:
            model_config = config[model_name]
            
            print(f"📊 Creating {model_name} model...")
            model = create_qwen_model(model_config)
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"🔢 Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
            
            if config.get('qwen_optimizations', {}).get('enable_compilation', False):
                print("⚡ Applying optimizations...")
                opt_config = config['qwen_optimizations']
                example_input = torch.randint(0, model_config['vocab_size'], (1, 512))
                model = apply_qwen_optimizations(model, opt_config, example_input)
            
            benchmark_config = config['qwen_benchmarks'].copy()
            
            if model_name == 'qwen_72b':
                benchmark_config['batch_sizes'] = [1, 2]
                benchmark_config['sequence_lengths'] = [512, 1024]
                benchmark_config['num_benchmark_runs'] = 10
            elif model_name == 'qwen_14b':
                benchmark_config['batch_sizes'] = [1, 2, 4]
                benchmark_config['sequence_lengths'] = [512, 1024, 2048]
                benchmark_config['num_benchmark_runs'] = 15
            
            print("🏃 Running benchmarks...")
            results = run_qwen_benchmarks(model, benchmark_config, device)
            
            report = create_benchmark_report(results, model_name)
            all_results[model_name] = report
            
            print(f"\n📈 {model_name.upper()} RESULTS:")
            print(f"   🔢 Parameters: {report['model_analysis']['total_parameters_billions']:.2f}B")
            print(f"   📦 Model Size: {report['model_size']['total_size_gb']:.2f} GB")
            
            if 'performance_summary' in report:
                perf = report['performance_summary']
                if 'best_throughput_tokens_per_sec' in perf:
                    print(f"   🚄 Best Throughput: {perf['best_throughput_tokens_per_sec']:.0f} tokens/s")
                if 'best_latency_ms' in perf:
                    print(f"   ⚡ Best Latency: {perf['best_latency_ms']:.1f} ms")
                if 'best_memory_mb' in perf:
                    print(f"   💾 Best Memory: {perf['best_memory_mb']:.1f} MB")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error benchmarking {model_name}: {str(e)}")
            all_results[model_name] = {'error': str(e)}
    
    timestamp = int(time.time())
    results_file = f"qwen_comprehensive_benchmarks_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Complete results saved to {results_file}")
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)
    
    for model_name, report in all_results.items():
        if 'error' not in report:
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   Parameters: {report['model_analysis']['total_parameters_billions']:.2f}B")
            print(f"   Model Size: {report['model_size']['total_size_gb']:.2f} GB")
            
            if 'performance_summary' in report:
                perf = report['performance_summary']
                if 'best_throughput_tokens_per_sec' in perf:
                    print(f"   Throughput: {perf['best_throughput_tokens_per_sec']:.0f} tokens/s")
                if 'best_latency_ms' in perf:
                    print(f"   Latency: {perf['best_latency_ms']:.1f} ms")
        else:
            print(f"\n❌ {model_name.upper()}: {report['error']}")
    
    print("=" * 80)
    
    return all_results

if __name__ == "__main__":
    results = benchmark_qwen_models()
