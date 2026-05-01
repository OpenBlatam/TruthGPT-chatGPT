"""
K/V Cache Optimizer for TruthGPT
Integrates efficient K/V caching with the existing TruthGPT optimization system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import time
import logging
from dataclasses import dataclass
import math
from contextlib import contextmanager

from ..modules.attention.efficient_kv_cache import (
    KVCacheConfig, 
    EfficientMultiHeadAttention,
    PrefillDecodeOptimizer,
    MemoryEfficientAttention
)
from ..modules.transformer.efficient_decoder import (
    EfficientTransformerDecoder,
    DecoderConfig
)
from optimization_core.modules.optimizers.core.pytorch_optimizer_base import PyTorchOptimizerBase, OptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class KVCacheOptimizationConfig:
    """Configuration for K/V cache optimization."""
    # Cache settings
    max_cache_size: int = 2048
    cache_dtype: torch.dtype = torch.float16
    use_compression: bool = True
    compression_ratio: float = 0.5
    cache_eviction_policy: str = "lru"
    
    # Performance settings
    use_flash_attention: bool = True
    use_memory_efficient_attention: bool = True
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    # Decode settings
    max_sequence_length: int = 2048
    batch_size: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None

class KVCacheOptimizer(PyTorchOptimizerBase):
    """
    K/V Cache Optimizer for TruthGPT.
    
    This optimizer implements the suggested improvements:
    - Efficient K/V cache reuse for sequential token generation
    - Separate prefill and decode phases
    - Memory-optimized attention computation
    - Automatic cache management
    """
    
    def __init__(self, config: OptimizationConfig, kv_config: Optional[KVCacheOptimizationConfig] = None):
        super().__init__(config)
        self.kv_config = kv_config or KVCacheOptimizationConfig()
        
        # Initialize components
        self._setup_kv_cache_optimization()
        self._setup_performance_tracking()
        
        # Performance metrics
        self.prefill_times = []
        self.decode_times = []
        self.cache_hit_rates = []
        self.memory_usage = []
    
    def _setup_kv_cache_optimization(self):
        """Setup K/V cache optimization components."""
        # Create cache configuration
        self.cache_config = KVCacheConfig(
            max_cache_size=self.kv_config.max_cache_size,
            cache_dtype=self.kv_config.cache_dtype,
            use_compression=self.kv_config.use_compression,
            compression_ratio=self.kv_config.compression_ratio,
            cache_eviction_policy=self.kv_config.cache_eviction_policy
        )
        
        # Create decoder configuration
        self.decoder_config = DecoderConfig(
            d_model=512,  # Will be updated based on model
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            dropout=0.1,
            activation="gelu",
            use_kv_cache=True,
            cache_config=self.cache_config,
            use_flash_attention=self.kv_config.use_flash_attention,
            use_memory_efficient_attention=self.kv_config.use_memory_efficient_attention,
            max_sequence_length=self.kv_config.max_sequence_length
        )
        
        # Initialize decoder
        self.decoder = None  # Will be initialized when model is loaded
        
        # Memory optimizer
        self.memory_optimizer = MemoryEfficientAttention(
            use_checkpointing=self.kv_config.use_gradient_checkpointing,
            use_mixed_precision=self.kv_config.use_mixed_precision
        )
    
    def _setup_performance_tracking(self):
        """Setup performance tracking."""
        self.performance_stats = {
            'total_prefill_time': 0.0,
            'total_decode_time': 0.0,
            'total_tokens_generated': 0,
            'cache_hit_rate': 0.0,
            'memory_usage': 0.0,
            'throughput': 0.0
        }
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Optimize model with K/V cache improvements.
        
        Args:
            model: The model to optimize
            
        Returns:
            Optimized model
        """
        self.logger.info("Applying K/V cache optimizations...")
        
        # Update decoder configuration based on model
        self._update_decoder_config(model)
        
        # Initialize decoder if needed
        if self.decoder is None:
            self.decoder = EfficientTransformerDecoder(self.decoder_config)
            self.decoder.to(self.device)
        
        # Apply memory optimizations
        if self.memory_optimizer:
            self.memory_optimizer.optimize_memory_usage(model)
        
        # Enable K/V caching for attention layers
        self._enable_kv_caching(model)
        
        self.logger.info("K/V cache optimizations applied successfully")
        return model
    
    def _update_decoder_config(self, model: nn.Module):
        """Update decoder configuration based on model architecture."""
        # Extract model dimensions from the actual model
        if hasattr(model, 'config'):
            config = model.config
            self.decoder_config.d_model = getattr(config, 'd_model', 512)
            self.decoder_config.n_heads = getattr(config, 'n_heads', 8)
            self.decoder_config.n_layers = getattr(config, 'n_layers', 6)
            self.decoder_config.d_ff = getattr(config, 'd_ff', 2048)
            self.decoder_config.vocab_size = getattr(config, 'vocab_size', 50000)
    
    def _enable_kv_caching(self, model: nn.Module):
        """Enable K/V caching for attention layers in the model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Replace with efficient attention
                efficient_attention = EfficientMultiHeadAttention(
                    d_model=module.embed_dim,
                    n_heads=module.num_heads,
                    dropout=module.dropout,
                    use_kv_cache=True,
                    cache_config=self.cache_config
                )
                
                # Replace the module
                parent = model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], efficient_attention)
    
    def prefill_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process the prefill phase (initial prompt processing).
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        start_time = time.time()
        
        # Use memory-efficient forward if enabled
        if self.memory_optimizer:
            with self.memory_optimizer.memory_efficient_forward(self.model):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **kwargs
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **kwargs
            )
        
        prefill_time = time.time() - start_time
        self.prefill_times.append(prefill_time)
        self.performance_stats['total_prefill_time'] += prefill_time
        
        self.logger.info(f"Prefill phase completed in {prefill_time:.4f}s")
        return outputs
    
    def decode_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process the decode phase (token-by-token generation).
        
        Args:
            input_ids: New token IDs (typically single token)
            attention_mask: Attention mask
            cache_position: Position in sequence for caching
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        start_time = time.time()
        
        # Use memory-efficient forward if enabled
        if self.memory_optimizer:
            with self.memory_optimizer.memory_efficient_forward(self.model):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    cache_position=cache_position,
                    **kwargs
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                cache_position=cache_position,
                **kwargs
            )
        
        decode_time = time.time() - start_time
        self.decode_times.append(decode_time)
        self.performance_stats['total_decode_time'] += decode_time
        self.performance_stats['total_tokens_generated'] += 1
        
        # Track cache performance
        cache_stats = self.get_cache_stats()
        if cache_stats:
            hit_rate = cache_stats.get('hit_rate', 0.0)
            self.cache_hit_rates.append(hit_rate)
            self.performance_stats['cache_hit_rate'] = sum(self.cache_hit_rates) / len(self.cache_hit_rates)
        
        return outputs
    
    def generate_text(
        self,
        input_text: str,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using efficient K/V caching.
        
        Args:
            input_text: Initial text prompt
            max_length: Maximum length to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Please load a model first.")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Process prefill phase
        outputs = self.prefill_phase(input_ids, attention_mask, **kwargs)
        generated_ids = input_ids.clone()
        
        # Generate tokens one by one
        for i in range(max_length - input_ids.size(1)):
            # Get logits for the last token
            logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Process decode phase for next token
            outputs = self.decode_phase(
                next_token, 
                cache_position=input_ids.size(1) + i,
                **kwargs
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from the model."""
        stats = {}
        
        # Get stats from all attention layers
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_cache_stats'):
                module_stats = module.get_cache_stats()
                if module_stats:
                    stats[name] = module_stats
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all K/V caches."""
        for module in self.model.modules():
            if hasattr(module, 'clear_cache'):
                module.clear_cache()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        # Calculate throughput
        total_time = self.performance_stats['total_prefill_time'] + self.performance_stats['total_decode_time']
        total_tokens = self.performance_stats['total_tokens_generated']
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        
        # Calculate average times
        avg_prefill_time = sum(self.prefill_times) / len(self.prefill_times) if self.prefill_times else 0.0
        avg_decode_time = sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0.0
        
        # Calculate cache hit rate
        cache_hit_rate = sum(self.cache_hit_rates) / len(self.cache_hit_rates) if self.cache_hit_rates else 0.0
        
        return {
            'performance_stats': self.performance_stats,
            'avg_prefill_time': avg_prefill_time,
            'avg_decode_time': avg_decode_time,
            'cache_hit_rate': cache_hit_rate,
            'throughput': throughput,
            'cache_stats': self.get_cache_stats()
        }
    
    def benchmark_performance(
        self,
        test_prompts: List[str],
        max_length: int = 100,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark the performance of K/V cache optimization.
        
        Args:
            test_prompts: List of test prompts
            max_length: Maximum length to generate
            num_runs: Number of runs for averaging
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting performance benchmark with {len(test_prompts)} prompts, {num_runs} runs each")
        
        results = {
            'prefill_times': [],
            'decode_times': [],
            'cache_hit_rates': [],
            'memory_usage': [],
            'throughput': []
        }
        
        for run in range(num_runs):
            self.logger.info(f"Benchmark run {run + 1}/{num_runs}")
            
            # Clear cache between runs
            self.clear_cache()
            
            for prompt in test_prompts:
                # Generate text and collect metrics
                start_time = time.time()
                generated_text = self.generate_text(prompt, max_length=max_length)
                total_time = time.time() - start_time
                
                # Collect performance metrics
                run_stats = self.get_performance_stats()
                results['prefill_times'].append(run_stats['avg_prefill_time'])
                results['decode_times'].append(run_stats['avg_decode_time'])
                results['cache_hit_rates'].append(run_stats['cache_hit_rate'])
                results['throughput'].append(run_stats['throughput'])
        
        # Calculate averages
        benchmark_results = {
            'avg_prefill_time': sum(results['prefill_times']) / len(results['prefill_times']),
            'avg_decode_time': sum(results['decode_times']) / len(results['decode_times']),
            'avg_cache_hit_rate': sum(results['cache_hit_rates']) / len(results['cache_hit_rates']),
            'avg_throughput': sum(results['throughput']) / len(results['throughput']),
            'total_prompts': len(test_prompts) * num_runs,
            'total_runs': num_runs
        }
        
        self.logger.info(f"Benchmark completed. Average throughput: {benchmark_results['avg_throughput']:.2f} tokens/s")
        return benchmark_results

# Factory functions
def create_kv_cache_optimizer(
    config: OptimizationConfig,
    kv_config: Optional[KVCacheOptimizationConfig] = None
) -> KVCacheOptimizer:
    """Create a K/V cache optimizer."""
    return KVCacheOptimizer(config, kv_config)

def create_kv_cache_config(**kwargs) -> KVCacheOptimizationConfig:
    """Create a K/V cache optimization configuration."""
    return KVCacheOptimizationConfig(**kwargs)





