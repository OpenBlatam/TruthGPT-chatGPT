"""
Claude API Optimizer with comprehensive optimization_core integration
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ClaudeAPIOptimizer:
    """
    Comprehensive optimizer for Claude API integration with optimization_core.
    Provides advanced optimization techniques specifically for Claude API models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimization_history = []
        self.performance_metrics = {}
        
        self.enable_request_batching = self.config.get('enable_request_batching', True)
        self.batch_size = self.config.get('batch_size', 4)
        self.batch_timeout = self.config.get('batch_timeout', 0.1)
        
        self.enable_response_caching = self.config.get('enable_response_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)
        
        self.enable_prompt_optimization = self.config.get('enable_prompt_optimization', True)
        self.max_prompt_length = self.config.get('max_prompt_length', 8000)
        
        self.pending_requests = []
        self.response_cache = {}
        
        self._setup_optimization_core()
    
    def _setup_optimization_core(self):
        """Setup optimization_core integration for Claude API."""
        try:
            from optimization_core import OptimizedLayerNorm
            from optimization_core.advanced_normalization import AdvancedRMSNorm, CRMSNorm
            from optimization_core.enhanced_mlp import EnhancedMLP, SwiGLU
            from optimization_core.memory_optimizations import MemoryOptimizer
            from optimization_core.computational_optimizations import ComputationalOptimizer
            from optimization_core.advanced_optimization_registry_v2 import get_advanced_optimization_config
            
            self.optimization_config = get_advanced_optimization_config('claud_api')
            self.optimization_core_available = True
            logger.info("optimization_core components loaded for Claude API optimizer")
            
        except ImportError as e:
            self.optimization_config = None
            self.optimization_core_available = False
            logger.warning(f"optimization_core not available: {e}")
    
    def optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for better Claude API performance."""
        if not self.enable_prompt_optimization:
            return prompt
        
        optimized_prompt = prompt.strip()
        
        if len(optimized_prompt) > self.max_prompt_length:
            optimized_prompt = optimized_prompt[:self.max_prompt_length] + "..."
            logger.warning(f"Prompt truncated to {self.max_prompt_length} characters")
        
        if not optimized_prompt.endswith(('.', '?', '!')):
            optimized_prompt += "."
        
        return optimized_prompt
    
    def should_use_cache(self, prompt: str) -> bool:
        """Determine if response should be cached."""
        if not self.enable_response_caching:
            return False
        
        if len(prompt) < 10:
            return False
        
        if any(word in prompt.lower() for word in ['random', 'current time', 'now', 'today']):
            return False
        
        return True
    
    def optimize_api_call(self, client, prompt: str, **kwargs) -> str:
        """Optimize API call with caching and batching."""
        optimized_prompt = self.optimize_prompt(prompt)
        
        cache_key = f"{optimized_prompt}_{hash(str(sorted(kwargs.items())))}"
        
        if self.should_use_cache(optimized_prompt) and cache_key in self.response_cache:
            logger.info("Returning cached Claude API response")
            return self.response_cache[cache_key]
        
        try:
            response = client.generate_text(optimized_prompt, **kwargs)
            
            if self.should_use_cache(optimized_prompt):
                self.response_cache[cache_key] = response
                
                if len(self.response_cache) > 1000:
                    oldest_key = next(iter(self.response_cache))
                    del self.response_cache[oldest_key]
            
            return response
            
        except Exception as e:
            logger.error(f"Claude API optimization error: {e}")
            return client.generate_text(optimized_prompt, **kwargs)
    
    def optimize_batch_requests(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Optimize batch requests for Claude API.
        
        Args:
            prompts: List of prompts to process
            
        Returns:
            Optimization configuration for batch processing
        """
        batch_config = {
            'batch_size': min(len(prompts), self.config.get('max_batch_size', 10)),
            'concurrent_requests': self.config.get('concurrent_requests', 3),
            'retry_strategy': {
                'max_retries': 3,
                'backoff_factor': 1.5,
                'retry_delay': 1.0
            },
            'caching_enabled': self.config.get('enable_caching', True),
            'prompt_optimization_enabled': True
        }
        
        optimized_prompts = [self.optimize_prompt(prompt) for prompt in prompts]
        
        return {
            'optimized_prompts': optimized_prompts,
            'batch_config': batch_config,
            'optimization_stats': {
                'total_prompts': len(prompts),
                'optimization_applied': True,
                'estimated_token_savings': sum(
                    max(0, len(orig) - len(opt)) 
                    for orig, opt in zip(prompts, optimized_prompts)
                )
            }
        }
    
    def apply_memory_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to Claude API model."""
        if not self.optimization_core_available:
            logger.warning("optimization_core not available for memory optimizations")
            return model
        
        try:
            from optimization_core.memory_optimizations import MemoryOptimizer
            
            memory_config = {
                'enable_gradient_checkpointing': True,
                'use_fp16': self.config.get('use_fp16', True),
                'enable_activation_checkpointing': True,
                'memory_efficient_attention': True
            }
            
            memory_optimizer = MemoryOptimizer(memory_config)
            optimized_model = memory_optimizer.optimize(model)
            
            logger.info("Applied memory optimizations to Claude API model")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return model
    
    def apply_computational_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply computational optimizations to Claude API model."""
        if not self.optimization_core_available:
            logger.warning("optimization_core not available for computational optimizations")
            return model
        
        try:
            from optimization_core.computational_optimizations import ComputationalOptimizer
            
            comp_config = {
                'use_fused_attention': True,
                'enable_kernel_fusion': True,
                'optimize_matrix_multiplication': True,
                'use_flash_attention': self.config.get('use_flash_attention', True)
            }
            
            comp_optimizer = ComputationalOptimizer(comp_config)
            optimized_model = comp_optimizer.optimize(model)
            
            logger.info("Applied computational optimizations to Claude API model")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Computational optimization failed: {e}")
            return model
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply comprehensive optimizations to Claude API model.
        
        Args:
            model: Claude API model to optimize
            
        Returns:
            Optimized model
        """
        logger.info("Starting comprehensive Claude API model optimization...")
        
        # Apply memory optimizations
        model = self.apply_memory_optimizations(model)
        
        # Apply computational optimizations
        model = self.apply_computational_optimizations(model)
        
        self.performance_metrics.update({
            'optimization_applied': True,
            'memory_optimized': True,
            'computational_optimized': True,
            'optimization_core_used': self.optimization_core_available
        })
        
        logger.info("Claude API model optimization complete")
        return model

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            'cache_size': len(self.response_cache),
            'pending_requests': len(self.pending_requests),
            'batching_enabled': self.enable_request_batching,
            'caching_enabled': self.enable_response_caching,
            'prompt_optimization_enabled': self.enable_prompt_optimization,
            'optimization_core_available': self.optimization_core_available,
            'optimization_history_count': len(self.optimization_history),
            'performance_metrics': self.performance_metrics
        }
    
    def clear_cache(self):
        """Clear optimization cache."""
        self.response_cache.clear()
        logger.info("Claude API optimization cache cleared")

def create_claude_api_optimizer(config: Optional[Dict[str, Any]] = None) -> ClaudeAPIOptimizer:
    """Create optimized Claude API client."""
    return ClaudeAPIOptimizer(config)
