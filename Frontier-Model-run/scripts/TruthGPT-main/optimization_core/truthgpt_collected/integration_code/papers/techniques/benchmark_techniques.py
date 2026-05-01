#!/usr/bin/env python3
"""
Benchmark script for measuring performance improvements of implemented techniques.
Compares baseline vs Paper 2503 (Flash Attention) vs Paper 2506 (Sparse Attention & CoD).
"""
import torch
import torch.nn as nn
import time
import logging
from typing import List, Dict, Any
import statistics
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports from local modules
from paper_2503_00735v3 import Paper2503_00735v3Config, EfficientFlashAttention
from paper_2506_10987v1 import Paper2506_10987v1Config, AdaptiveSparseAttention
from paper_2506_10987v1_chain_of_draft import ChainOfDraftConfig, ChainOfDraftModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkSuite:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        logger.info(f"Running benchmarks on device: {self.device}")
        self.results = {}

    def _measure_time(self, model: nn.Module, x: torch.Tensor, repetitions: int = 50) -> float:
        """Measure average forward pass time in milliseconds."""
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                
            model.eval()
            model.to(self.device)
            x = x.to(self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)
            
            # Measurement
            start_time = time.time()
            with torch.no_grad():
                for _ in range(repetitions):
                    _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / repetitions * 1000  # ms
            return avg_time
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM caught during benchmark: {e}")
                return float('inf')
            raise e

    def benchmark_attention(self, seq_lens: List[int] = [128, 512, 1024]):
        """Benchmark different attention mechanisms."""
        logger.info("--- Benchmarking Attention Mechanisms ---")
        hidden_dim = 512
        num_heads = 8
        batch_size = 2  # Reduced from 4 to avoid MPS OOM
        
        results = {seq_len: {} for seq_len in seq_lens}
        
        for seq_len in seq_lens:
            logger.info(f"Testing sequence length: {seq_len}")
            x = torch.randn(batch_size, seq_len, hidden_dim)
            
            # 1. Baseline (Standard PyTorch MultiheadAttention)
            class BaselineWrapper(nn.Module):
                def __init__(self): 
                    super().__init__()
                    self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                def forward(self, x): 
                    return self.attn(x, x, x)[0]
            
            baseline_wrapper = BaselineWrapper()
            time_baseline = self._measure_time(baseline_wrapper, x)
            results[seq_len]['Baseline'] = time_baseline
            
            # 2. Paper 2503 (Efficient Flash Attention)
            config_2503 = Paper2503_00735v3Config(hidden_dim=hidden_dim, num_heads=num_heads, chunk_size=64)
            model_2503 = EfficientFlashAttention(hidden_dim, num_heads, chunk_size=config_2503.chunk_size)
            time_2503 = self._measure_time(model_2503, x)
            results[seq_len]['Paper 2503 (Flash)'] = time_2503
            
            # 3. Paper 2506 (Adaptive Sparse Attention)
            model_2506 = AdaptiveSparseAttention(hidden_dim, num_heads, sparsity_ratio=0.5)
            time_2506 = self._measure_time(model_2506, x)
            results[seq_len]['Paper 2506 (Sparse)'] = time_2506
            
            logger.info(f"Seq {seq_len}: Baseline={time_baseline:.2f}ms, Flash={time_2503:.2f}ms, Sparse={time_2506:.2f}ms")

        self.results['attention'] = results
        return results

    def benchmark_chain_of_draft(self):
        """Benchmark Chain of Draft token efficiency simulation."""
        logger.info("--- Benchmarking Chain of Draft (Token Efficiency) ---")
        hidden_dim = 512
        seq_len = 1024
        batch_size = 1
        
        config = ChainOfDraftConfig(hidden_dim=hidden_dim, max_words_per_step=5)
        model = ChainOfDraftModule(config)
        model.to(self.device)
        
        x = torch.randn(batch_size, seq_len, hidden_dim).to(self.device)
        
        # Simulate different draft steps scenarios
        draft_steps_list = [3, 5, 10]
        
        print("\n" + "="*60)
        print(f"CHAIN OF DRAFT EFFICIENCY METRICS")
        print("="*60)
        print(f"{'Draft Steps':<15} | {'Total Tokens':<15} | {'Reduction vs Full':<20} | {'Conciseness':<12}")
        print("-" * 65)
        
        for steps in draft_steps_list:
            with torch.no_grad():
                output, metadata = model(x, num_draft_steps=steps)
            
            # Calculate theoretical reduction assuming standard CoT would be ~2x tokens (rough estimate from paper)
            # Efficiency ratio from metadata is tokens_used / seq_len
            reduction = (1.0 - metadata['efficiency_ratio']) * 100
            
            print(f"{steps:<15} | {metadata['total_tokens']:<15} | {reduction:>18.2f}% | {metadata['conciseness_score']:.2f}")
            
    def print_summary(self):
        print("\n" + "="*80)
        print("ATTENTION LATENCY BENCHMARK SUMMARY (Lower is Better)")
        print("="*80)
        print(f"{'Seq Len':<10} | {'Baseline (ms)':<15} | {'Flash (ms)':<15} | {'Sparse (ms)':<15} | {'Best Speedup':<15}")
        print("-" * 80)
        
        for seq_len, res in self.results.get('attention', {}).items():
            base = res.get('Baseline', float('inf'))
            flash = res.get('Paper 2503 (Flash)', float('inf'))
            sparse = res.get('Paper 2506 (Sparse)', float('inf'))
            
            # Handle OOM/inf
            base_str = f"{base:.2f}" if base != float('inf') else "OOM"
            flash_str = f"{flash:.2f}" if flash != float('inf') else "OOM"
            sparse_str = f"{sparse:.2f}" if sparse != float('inf') else "OOM"
            
            # Calculate speedup relative to baseline
            valid_times = [t for t in [flash, sparse] if t != float('inf')]
            best_time = min(valid_times) if valid_times else float('inf')
            
            if base != float('inf') and best_time != float('inf') and best_time > 0:
                speedup = base / best_time
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            print(f"{seq_len:<10} | {base_str:<15} | {flash_str:<15} | {sparse_str:<15} | {speedup_str:<15}")
        print("="*80 + "\n")

if __name__ == "__main__":
    print(f"Starting benchmark suite...")
    suite = BenchmarkSuite()
    
    # Run benchmarks
    # Using moderate sequence lengths to respect local machine resources while showing scaling
    suite.benchmark_attention(seq_lens=[128, 512, 1024]) 
    
    suite.benchmark_chain_of_draft()
    
    suite.print_summary()


