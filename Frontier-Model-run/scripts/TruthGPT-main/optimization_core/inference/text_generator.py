"""
Professional text generator with caching and optimization.
"""
import logging
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from .inference_engine import InferenceEngine
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Professional text generator with caching and advanced features.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize text generator.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer
            device: Device for inference
            use_cache: Enable caching
            cache_dir: Cache directory
            **kwargs: Additional inference engine arguments
        """
        self.engine = InferenceEngine(model, tokenizer, device=device, **kwargs)
        self.use_cache = use_cache
        
        if use_cache:
            self.cache = CacheManager(cache_dir=cache_dir)
        else:
            self.cache = None
    
    def generate(
        self,
        prompt: str,
        use_cache: Optional[bool] = None,
        **generation_kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            use_cache: Override global cache setting
            **generation_kwargs: Generation arguments
        
        Returns:
            Generated text
        """
        # Check cache if enabled
        use_cache_flag = use_cache if use_cache is not None else self.use_cache
        
        if use_cache_flag and self.cache:
            cached_result = self.cache.get(prompt, **generation_kwargs)
            if cached_result is not None:
                logger.debug("Using cached result")
                return cached_result
        
        # Generate
        result = self.engine.generate(prompt, **generation_kwargs)
        
        # Store in cache
        if use_cache_flag and self.cache:
            self.cache.set(prompt, result, **generation_kwargs)
        
        return result
    
    def generate_batch(
        self,
        prompts: list[str],
        **generation_kwargs
    ) -> list[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of prompts
            **generation_kwargs: Generation arguments
        
        Returns:
            List of generated texts
        """
        # Check cache for each prompt
        results = []
        uncached_prompts = []
        uncached_indices = []
        
        if self.use_cache and self.cache:
            for i, prompt in enumerate(prompts):
                cached = self.cache.get(prompt, **generation_kwargs)
                if cached is not None:
                    results.append((i, cached))
                else:
                    uncached_prompts.append(prompt)
                    uncached_indices.append(i)
        
        # Generate for uncached prompts
        if uncached_prompts:
            generated = self.engine.generate_batched(uncached_prompts, **generation_kwargs)
            
            # Store in cache and add to results
            for idx, prompt, result in zip(uncached_indices, uncached_prompts, generated):
                if self.use_cache and self.cache:
                    self.cache.set(prompt, result, **generation_kwargs)
                results.append((idx, result))
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def clear_cache(self) -> None:
        """Clear generation cache."""
        if self.cache:
            self.cache.clear()
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return None



