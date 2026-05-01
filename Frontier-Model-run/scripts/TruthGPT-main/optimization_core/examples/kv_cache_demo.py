"""
K/V Cache Optimization Demo for TruthGPT
Demonstrates the improved K/V caching and efficient decoding design
"""

import torch
import torch.nn as nn
import time
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import the optimization modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from optimizers.kv_cache_optimizer import (
    KVCacheOptimizer, 
    KVCacheOptimizationConfig,
    create_kv_cache_optimizer,
    create_kv_cache_config
)
from modules.attention.efficient_kv_cache import KVCacheConfig
from modules.transformer.efficient_decoder import create_decoder_config
from config.transformer_config import OptimizationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthGPTKVCacheDemo:
    """
    Demonstration of K/V cache optimization for TruthGPT.
    
    This demo shows:
    - Efficient K/V cache reuse for sequential token generation
    - Separate prefill and decode phases
    - Memory-optimized attention computation
    - Performance comparison with and without K/V cache
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize configurations
        self.optimization_config = OptimizationConfig(
            learning_rate=1e-4,
            use_mixed_precision=True,
            use_gradient_checkpointing=True
        )
        
        self.kv_config = create_kv_cache_config(
            max_cache_size=2048,
            cache_dtype=torch.float16,
            use_compression=True,
            use_flash_attention=True,
            use_memory_efficient_attention=True
        )
        
        # Initialize optimizer
        self.optimizer = create_kv_cache_optimizer(
            self.optimization_config,
            self.kv_config
        )
        
        # Test prompts
        self.test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important aspect of machine learning is",
            "When considering the ethics of AI,",
            "The development of neural networks has led to"
        ]
    
    def create_dummy_model(self) -> nn.Module:
        """Create a dummy transformer model for demonstration."""
        class DummyTransformer(nn.Module):
            def __init__(self, vocab_size=50000, d_model=512, n_heads=8, n_layers=6):
                super().__init__()
                self.vocab_size = vocab_size
                self.d_model = d_model
                self.n_heads = n_heads
                self.n_layers = n_layers
                
                # Embedding layers
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Embedding(2048, d_model)
                
                # Transformer layers
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=2048,
                        dropout=0.1,
                        batch_first=True
                    )
                    for _ in range(n_layers)
                ])
                
                # Output projection
                self.output_projection = nn.Linear(d_model, vocab_size)
                
                # Dummy tokenizer
                self.tokenizer = DummyTokenizer(vocab_size)
            
            def forward(self, input_ids, attention_mask=None, use_cache=False, cache_position=None, **kwargs):
                batch_size, seq_len = input_ids.size()
                
                # Get embeddings
                x = self.embedding(input_ids)
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                pos_emb = self.pos_embedding(positions)
                x = x + pos_emb
                
                # Process through layers
                for layer in self.layers:
                    x = layer(x, x)  # Self-attention
                
                # Output projection
                logits = self.output_projection(x)
                
                return {
                    'logits': logits,
                    'hidden_states': x
                }
        
        return DummyTransformer()
    
    def run_performance_comparison(self):
        """Run performance comparison with and without K/V cache."""
        logger.info("Starting performance comparison...")
        
        # Create dummy model
        model = self.create_dummy_model().to(self.device)
        
        # Test without K/V cache
        logger.info("Testing without K/V cache...")
        start_time = time.time()
        
        for prompt in self.test_prompts:
            # Simulate generation without cache
            input_ids = torch.randint(0, 1000, (1, 10)).to(self.device)
            for _ in range(50):  # Generate 50 tokens
                with torch.no_grad():
                    outputs = model(input_ids)
                    # Simulate next token selection
                    next_token = torch.randint(0, 1000, (1, 1)).to(self.device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
        
        no_cache_time = time.time() - start_time
        logger.info(f"Without K/V cache: {no_cache_time:.4f}s")
        
        # Test with K/V cache
        logger.info("Testing with K/V cache...")
        self.optimizer.model = model
        optimized_model = self.optimizer.optimize_model(model)
        
        start_time = time.time()
        
        for prompt in self.test_prompts:
            # Simulate generation with cache
            input_ids = torch.randint(0, 1000, (1, 10)).to(self.device)
            
            # Prefill phase
            self.optimizer.prefill_phase(input_ids)
            
            # Decode phase
            for i in range(50):  # Generate 50 tokens
                next_token = torch.randint(0, 1000, (1, 1)).to(self.device)
                self.optimizer.decode_phase(next_token, cache_position=10 + i)
        
        with_cache_time = time.time() - start_time
        logger.info(f"With K/V cache: {with_cache_time:.4f}s")
        
        # Calculate improvement
        speedup = no_cache_time / with_cache_time
        logger.info(f"Speedup: {speedup:.2f}x")
        
        return {
            'no_cache_time': no_cache_time,
            'with_cache_time': with_cache_time,
            'speedup': speedup
        }
    
    def demonstrate_cache_efficiency(self):
        """Demonstrate cache efficiency metrics."""
        logger.info("Demonstrating cache efficiency...")
        
        # Create model and optimizer
        model = self.create_dummy_model().to(self.device)
        self.optimizer.model = model
        optimized_model = self.optimizer.optimize_model(model)
        
        # Generate text and collect metrics
        for i, prompt in enumerate(self.test_prompts):
            logger.info(f"Processing prompt {i+1}/{len(self.test_prompts)}: {prompt}")
            
            # Simulate text generation
            input_ids = torch.randint(0, 1000, (1, 10)).to(self.device)
            
            # Prefill phase
            start_time = time.time()
            self.optimizer.prefill_phase(input_ids)
            prefill_time = time.time() - start_time
            
            # Decode phase
            decode_times = []
            for j in range(20):  # Generate 20 tokens
                next_token = torch.randint(0, 1000, (1, 1)).to(self.device)
                
                start_time = time.time()
                self.optimizer.decode_phase(next_token, cache_position=10 + j)
                decode_time = time.time() - start_time
                decode_times.append(decode_time)
            
            # Get cache stats
            cache_stats = self.optimizer.get_cache_stats()
            performance_stats = self.optimizer.get_performance_stats()
            
            logger.info(f"  Prefill time: {prefill_time:.4f}s")
            logger.info(f"  Average decode time: {sum(decode_times)/len(decode_times):.4f}s")
            logger.info(f"  Cache hit rate: {performance_stats.get('cache_hit_rate', 0.0):.2%}")
            logger.info(f"  Throughput: {performance_stats.get('throughput', 0.0):.2f} tokens/s")
    
    def benchmark_different_sequence_lengths(self):
        """Benchmark performance with different sequence lengths."""
        logger.info("Benchmarking different sequence lengths...")
        
        model = self.create_dummy_model().to(self.device)
        self.optimizer.model = model
        optimized_model = self.optimizer.optimize_model(model)
        
        sequence_lengths = [50, 100, 200, 500, 1000]
        results = {}
        
        for seq_len in sequence_lengths:
            logger.info(f"Testing sequence length: {seq_len}")
            
            # Clear cache
            self.optimizer.clear_cache()
            
            # Generate sequence
            input_ids = torch.randint(0, 1000, (1, 10)).to(self.device)
            
            start_time = time.time()
            
            # Prefill phase
            self.optimizer.prefill_phase(input_ids)
            
            # Decode phase
            for i in range(seq_len - 10):
                next_token = torch.randint(0, 1000, (1, 1)).to(self.device)
                self.optimizer.decode_phase(next_token, cache_position=10 + i)
            
            total_time = time.time() - start_time
            
            # Get performance stats
            performance_stats = self.optimizer.get_performance_stats()
            
            results[seq_len] = {
                'total_time': total_time,
                'throughput': seq_len / total_time,
                'cache_hit_rate': performance_stats.get('cache_hit_rate', 0.0),
                'avg_decode_time': performance_stats.get('avg_decode_time', 0.0)
            }
            
            logger.info(f"  Total time: {total_time:.4f}s")
            logger.info(f"  Throughput: {results[seq_len]['throughput']:.2f} tokens/s")
            logger.info(f"  Cache hit rate: {results[seq_len]['cache_hit_rate']:.2%}")
        
        return results
    
    def visualize_performance(self, results: Dict[int, Dict[str, Any]]):
        """Visualize performance results."""
        try:
            import matplotlib.pyplot as plt
            
            seq_lengths = list(results.keys())
            throughputs = [results[seq_len]['throughput'] for seq_len in seq_lengths]
            cache_hit_rates = [results[seq_len]['cache_hit_rate'] for seq_len in seq_lengths]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Throughput plot
            ax1.plot(seq_lengths, throughputs, 'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel('Sequence Length')
            ax1.set_ylabel('Throughput (tokens/s)')
            ax1.set_title('K/V Cache Performance: Throughput vs Sequence Length')
            ax1.grid(True, alpha=0.3)
            
            # Cache hit rate plot
            ax2.plot(seq_lengths, cache_hit_rates, 'r-o', linewidth=2, markersize=8)
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Cache Hit Rate')
            ax2.set_title('K/V Cache Performance: Hit Rate vs Sequence Length')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('kv_cache_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Performance visualization saved as 'kv_cache_performance.png'")
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        logger.info("Starting TruthGPT K/V Cache Optimization Demo")
        logger.info("=" * 60)
        
        # Performance comparison
        logger.info("1. Performance Comparison")
        comparison_results = self.run_performance_comparison()
        
        # Cache efficiency demonstration
        logger.info("\n2. Cache Efficiency Demonstration")
        self.demonstrate_cache_efficiency()
        
        # Sequence length benchmarking
        logger.info("\n3. Sequence Length Benchmarking")
        benchmark_results = self.benchmark_different_sequence_lengths()
        
        # Visualization
        logger.info("\n4. Performance Visualization")
        self.visualize_performance(benchmark_results)
        
        # Summary
        logger.info("\n5. Summary")
        logger.info(f"Performance improvement: {comparison_results['speedup']:.2f}x speedup")
        logger.info(f"Tested sequence lengths: {list(benchmark_results.keys())}")
        logger.info(f"Average throughput: {np.mean([r['throughput'] for r in benchmark_results.values()]):.2f} tokens/s")
        
        logger.info("\nDemo completed successfully!")
        return {
            'comparison_results': comparison_results,
            'benchmark_results': benchmark_results
        }

class DummyTokenizer:
    """Dummy tokenizer for demonstration purposes."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
    
    def encode(self, text: str, return_tensors: str = None) -> torch.Tensor:
        """Encode text to token IDs."""
        # Simple encoding: convert characters to integers
        tokens = [ord(c) % self.vocab_size for c in text]
        tensor = torch.tensor(tokens).unsqueeze(0)
        return tensor
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        # Simple decoding: convert integers to characters
        text = ''.join([chr(token_id % 256) for token_id in token_ids.squeeze()])
        return text

def main():
    """Main function to run the demo."""
    demo = TruthGPTKVCacheDemo()
    results = demo.run_complete_demo()
    
    print("\n" + "=" * 60)
    print("TruthGPT K/V Cache Optimization Demo Results")
    print("=" * 60)
    print(f"Performance improvement: {results['comparison_results']['speedup']:.2f}x speedup")
    print(f"Tested sequence lengths: {list(results['benchmark_results'].keys())}")
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()





