"""
Comprehensive benchmarking suite for generative AI models
"""

import torch
import time
import json
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class QualityMetrics:
    """Quality metrics for generative models."""
    perplexity: float = 0.0
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    diversity_score: float = 0.0
    coherence_score: float = 0.0

@dataclass
class PerformanceMetrics:
    """Performance metrics for generative models."""
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    tokens_generated: int = 0

class HumanEvaluationSimulator:
    """Simulator for human evaluation metrics."""
    
    def __init__(self):
        self.quality_threshold = 0.7
        self.coherence_threshold = 0.8
        
    def evaluate_quality(self, generated_text: str) -> float:
        """Simulate human quality evaluation."""
        return min(1.0, len(generated_text) / 100.0)
    
    def evaluate_coherence(self, generated_text: str) -> float:
        """Simulate human coherence evaluation."""
        return min(1.0, len(set(generated_text.split())) / 50.0)
    
    def evaluate_relevance(self, generated_text: str, prompt: str) -> float:
        """Simulate human relevance evaluation."""
        return 0.8  # Mock implementation

@dataclass
class GenerativeBenchmarkConfig:
    """Configuration for generative benchmarking."""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    generation_lengths: List[int] = None
    num_warmup_runs: int = 5
    num_benchmark_runs: int = 20
    measure_memory: bool = True
    measure_quality: bool = True
    measure_diversity: bool = True
    measure_latency: bool = True
    measure_throughput: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]
        if self.sequence_lengths is None:
            self.sequence_lengths = [512, 1024, 2048]
        if self.generation_lengths is None:
            self.generation_lengths = [128, 256, 512]

class GenerativeBenchmarkSuite:
    """Comprehensive benchmarking for generative models."""
    
    def __init__(self, model: torch.nn.Module, config: GenerativeBenchmarkConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 Starting Generative AI Comprehensive Benchmark")
        print("=" * 80)
        
        results = {
            'model_analysis': self._analyze_model(),
            'performance': {},
            'quality_metrics': {},
            'generation_samples': {}
        }
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                for gen_len in self.config.generation_lengths:
                    test_key = f"bs{batch_size}_seq{seq_len}_gen{gen_len}"
                    
                    print(f"🔄 Testing {test_key}")
                    
                    try:
                        perf_results = self._benchmark_generation_performance(
                            batch_size, seq_len, gen_len
                        )
                        results['performance'][test_key] = perf_results
                        
                        if batch_size <= 2 and seq_len <= 1024:
                            quality_results = self._benchmark_generation_quality(
                                batch_size, seq_len, gen_len
                            )
                            results['quality_metrics'][test_key] = quality_results
                            
                    except Exception as e:
                        print(f"❌ Error in {test_key}: {str(e)}")
                        results['performance'][test_key] = {'error': str(e)}
        
        results['generation_samples'] = self._generate_samples()
        
        if self.config.save_results:
            timestamp = int(time.time())
            filename = f"generative_benchmark_results_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {filename}")
        
        self._print_summary(results)
        return results
    
    def _analyze_model(self) -> Dict[str, Any]:
        """Analyze model architecture and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = param_size + buffer_size
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_parameters_billions': total_params / 1e9,
            'model_size_mb': total_size / (1024 * 1024),
            'model_size_gb': total_size / (1024 * 1024 * 1024),
            'device': str(self.device),
            'model_type': type(self.model).__name__
        }
    
    def _benchmark_generation_performance(self, 
                                        batch_size: int, 
                                        seq_len: int, 
                                        gen_len: int) -> Dict[str, Any]:
        """Benchmark generation performance."""
        vocab_size = getattr(self.model.config, 'vocab_size', 50000)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        if self.config.measure_memory:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            memory_before = self._get_memory_usage()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.config.num_warmup_runs):
                _ = self._generate_sequence(input_ids, gen_len)
        
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.config.num_benchmark_runs):
                start_time = time.time()
                generated = self._generate_sequence(input_ids, gen_len)
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        if self.config.measure_memory:
            memory_after = self._get_memory_usage()
            peak_memory = memory_after
        else:
            peak_memory = 0
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        total_tokens = batch_size * gen_len
        throughput = total_tokens / (avg_latency / 1000)  # tokens per second
        
        return {
            'latency': {
                'mean_time_ms': avg_latency,
                'std_time_ms': std_latency,
                'min_time_ms': min(latencies),
                'max_time_ms': max(latencies)
            },
            'throughput': {
                'tokens_per_second': throughput,
                'samples_per_second': batch_size / (avg_latency / 1000)
            },
            'memory': {
                'peak_memory_mb': peak_memory
            },
            'config': {
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'generation_length': gen_len
            }
        }
    
    def _benchmark_generation_quality(self, 
                                    batch_size: int, 
                                    seq_len: int, 
                                    gen_len: int) -> Dict[str, Any]:
        """Benchmark generation quality metrics."""
        vocab_size = getattr(self.model.config, 'vocab_size', 50000)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            generated = self._generate_sequence(input_ids, gen_len)
        
        diversity_score = self._calculate_diversity(generated)
        coherence_score = self._calculate_coherence(generated)
        novelty_score = self._calculate_novelty(generated, input_ids)
        
        return {
            'diversity_score': diversity_score,
            'coherence_score': coherence_score,
            'novelty_score': novelty_score,
            'generation_length': gen_len,
            'actual_generated_tokens': generated.shape[-1] - seq_len
        }
    
    def _generate_sequence(self, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """Generate sequence using the model."""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                
                if generated.shape[-1] >= seq_len + max_length:
                    break
        
        return generated
    
    def _calculate_diversity(self, generated: torch.Tensor) -> float:
        """Calculate diversity score (unique token ratio)."""
        unique_tokens = len(torch.unique(generated))
        total_tokens = generated.numel()
        return unique_tokens / total_tokens
    
    def _calculate_coherence(self, generated: torch.Tensor) -> float:
        """Calculate coherence score (mock implementation)."""
        batch_size, seq_len = generated.shape
        transitions = torch.abs(generated[:, 1:] - generated[:, :-1])
        avg_transition = transitions.float().mean().item()
        
        coherence = 1.0 / (1.0 + avg_transition / 1000.0)
        return coherence
    
    def _calculate_novelty(self, generated: torch.Tensor, input_ids: torch.Tensor) -> float:
        """Calculate novelty score (how different from input)."""
        gen_only = generated[:, input_ids.shape[-1]:]
        if gen_only.numel() == 0:
            return 0.0
        
        input_tokens = set(input_ids.flatten().tolist())
        gen_tokens = set(gen_only.flatten().tolist())
        
        overlap = len(input_tokens.intersection(gen_tokens))
        total_unique_gen = len(gen_tokens)
        
        if total_unique_gen == 0:
            return 0.0
        
        novelty = 1.0 - (overlap / total_unique_gen)
        return novelty
    
    def _generate_samples(self) -> Dict[str, Any]:
        """Generate sample outputs for qualitative analysis."""
        vocab_size = getattr(self.model.config, 'vocab_size', 50000)
        
        samples = {}
        
        sample_configs = [
            {'batch_size': 1, 'seq_len': 128, 'gen_len': 64, 'name': 'short_generation'},
            {'batch_size': 1, 'seq_len': 256, 'gen_len': 128, 'name': 'medium_generation'},
            {'batch_size': 2, 'seq_len': 128, 'gen_len': 64, 'name': 'batch_generation'}
        ]
        
        self.model.eval()
        with torch.no_grad():
            for config in sample_configs:
                try:
                    input_ids = torch.randint(
                        0, vocab_size, 
                        (config['batch_size'], config['seq_len']), 
                        device=self.device
                    )
                    
                    generated = self._generate_sequence(input_ids, config['gen_len'])
                    
                    samples[config['name']] = {
                        'input_shape': list(input_ids.shape),
                        'output_shape': list(generated.shape),
                        'input_tokens': input_ids.tolist(),
                        'generated_tokens': generated.tolist()
                    }
                    
                except Exception as e:
                    samples[config['name']] = {'error': str(e)}
        
        return samples
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("📊 GENERATIVE AI BENCHMARK SUMMARY")
        print("=" * 80)
        
        model_info = results['model_analysis']
        print(f"🤖 Model: {model_info['model_type']}")
        print(f"🔢 Parameters: {model_info['total_parameters_billions']:.2f}B")
        print(f"📦 Model Size: {model_info['model_size_gb']:.2f} GB")
        print(f"🖥️  Device: {model_info['device']}")
        
        if results['performance']:
            best_throughput = 0
            best_latency = float('inf')
            
            for test_key, perf in results['performance'].items():
                if 'error' not in perf:
                    throughput = perf['throughput']['tokens_per_second']
                    latency = perf['latency']['mean_time_ms']
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                    if latency < best_latency:
                        best_latency = latency
            
            print(f"🚄 Best Throughput: {best_throughput:.0f} tokens/s")
            print(f"⚡ Best Latency: {best_latency:.1f} ms")
        
        if results['quality_metrics']:
            avg_diversity = np.mean([
                q['diversity_score'] for q in results['quality_metrics'].values()
                if 'error' not in q
            ])
            avg_coherence = np.mean([
                q['coherence_score'] for q in results['quality_metrics'].values()
                if 'error' not in q
            ])
            
            print(f"🎨 Avg Diversity: {avg_diversity:.3f}")
            print(f"🧠 Avg Coherence: {avg_coherence:.3f}")
        
        print("=" * 80)

def run_generative_benchmarks(model: torch.nn.Module, 
                            config: Dict[str, Any],
                            device: str = 'cpu') -> Dict[str, Any]:
    """Run comprehensive generative benchmarks."""
    benchmark_config = GenerativeBenchmarkConfig(**config)
    
    model = model.to(device)
    
    benchmark_suite = GenerativeBenchmarkSuite(model, benchmark_config)
    
    return benchmark_suite.run_comprehensive_benchmark()
