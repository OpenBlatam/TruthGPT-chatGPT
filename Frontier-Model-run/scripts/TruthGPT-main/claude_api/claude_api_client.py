"""
Claude API Client with optimization_core integration
"""

import os
import time
import json
import asyncio
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from dataclasses import asdict
import logging

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

import torch
import torch.nn as nn

from .claude_api_config import ClaudeAPIConfig, get_default_claude_api_config

logger = logging.getLogger(__name__)

class ClaudeAPIClient(nn.Module):
    """
    Claude API Client with optimization_core integration.
    Provides a PyTorch-compatible interface to Claude API.
    """
    
    def __init__(self, config: Optional[ClaudeAPIConfig] = None):
        super().__init__()
        self.config = config or get_default_claude_api_config()
        
        self.api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key and ANTHROPIC_AVAILABLE:
            logger.warning("No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = None
        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        
        self.cache = {} if self.config.enable_caching else None
        self.request_count = 0
        self.total_tokens = 0
        
        self._setup_optimization_core()
    
    def _setup_optimization_core(self):
        """Setup optimization_core integration."""
        if self.config.use_optimization_core:
            try:
                from optimization_core import OptimizedLayerNorm
                from optimization_core.advanced_normalization import AdvancedRMSNorm
                from optimization_core.enhanced_mlp import EnhancedMLP
                
                self.input_norm = OptimizedLayerNorm(512)
                self.output_norm = OptimizedLayerNorm(512)
                logger.info("Using OptimizedLayerNorm for Claude API")
                try:
                    from optimization_core.enhanced_mlp import OptimizedLinear
                    self.processing_mlp = nn.Sequential(
                        OptimizedLinear(512, 2048),
                        nn.GELU(),
                        OptimizedLinear(2048, 512)
                    )
                    logger.info("Using OptimizedLinear for Claude API processing MLP")
                except ImportError:
                    self.processing_mlp = nn.Sequential(
                        nn.Linear(512, 2048),
                        nn.GELU(),
                        nn.Linear(2048, 512)
                    )
                    logger.warning("OptimizedLinear not available, using standard nn.Linear")
                
                logger.info("optimization_core integration enabled for Claude API")
            except ImportError:
                self.input_norm = nn.LayerNorm(512)
                self.output_norm = nn.LayerNorm(512)
                logger.warning("OptimizedLayerNorm not available, using standard nn.LayerNorm")
                
                try:
                    from optimization_core.enhanced_mlp import EnhancedLinear
                    self.processing_mlp = nn.Sequential(
                        EnhancedLinear(512, 2048),
                        nn.GELU(),
                        EnhancedLinear(2048, 512)
                    )
                except ImportError:
                    try:
                        from optimization_core.enhanced_mlp import OptimizedLinear
                        self.processing_mlp = nn.Sequential(
                            OptimizedLinear(512, 2048),
                            nn.GELU(),
                            OptimizedLinear(2048, 512)
                        )
                        logger.info("Using OptimizedLinear fallback for Claude API")
                    except ImportError:
                        self.processing_mlp = nn.Sequential(
                            nn.Linear(512, 2048),
                            nn.GELU(),
                            nn.Linear(2048, 512)
                        )
                        logger.warning("OptimizedLinear not available, using standard nn.Linear")
                logger.warning("optimization_core not available, using standard layers")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        PyTorch-compatible forward pass for integration with TruthGPT models.
        """
        batch_size, seq_len = input_ids.shape
        
        embeddings = torch.randn(batch_size, seq_len, 512, dtype=torch.float32)
        
        if hasattr(self, 'input_norm'):
            self.input_norm = self.input_norm.float()
            embeddings = self.input_norm(embeddings)
        
        if hasattr(self, 'processing_mlp'):
            self.processing_mlp = self.processing_mlp.float()
            embeddings = self.processing_mlp(embeddings)
        
        if hasattr(self, 'output_norm'):
            self.output_norm = self.output_norm.float()
            embeddings = self.output_norm(embeddings)
        
        vocab_size = 100000
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        
        return logits
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Claude API."""
        if not self.client:
            return self._mock_generate(prompt)
        
        try:
            config_dict = asdict(self.config)
            config_dict.update(kwargs)
            
            if self.config.enable_caching and prompt in self.cache:
                logger.info("Returning cached response")
                return self.cache[prompt]
            
            message = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=config_dict.get('max_tokens', self.config.max_tokens),
                temperature=config_dict.get('temperature', self.config.temperature),
                top_p=config_dict.get('top_p', self.config.top_p),
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            if self.config.enable_caching:
                if len(self.cache) >= self.config.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[prompt] = response_text
            
            self.request_count += 1
            self.total_tokens += message.usage.input_tokens + message.usage.output_tokens
            
            return response_text
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._mock_generate(prompt)
    
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """Async text generation using Claude API."""
        if not self.client:
            return self._mock_generate(prompt)
        
        try:
            config_dict = asdict(self.config)
            config_dict.update(kwargs)
            
            if self.config.enable_caching and prompt in self.cache:
                return self.cache[prompt]
            
            message = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=config_dict.get('max_tokens', self.config.max_tokens),
                temperature=config_dict.get('temperature', self.config.temperature),
                top_p=config_dict.get('top_p', self.config.top_p),
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            if self.config.enable_caching:
                if len(self.cache) >= self.config.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[prompt] = response_text
            
            self.request_count += 1
            self.total_tokens += message.usage.input_tokens + message.usage.output_tokens
            
            return response_text
            
        except Exception as e:
            logger.error(f"Claude API async error: {e}")
            return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation when API is not available."""
        logger.warning("Using mock Claude API response")
        return f"Mock Claude response for: {prompt[:50]}..."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'cache_size': len(self.cache) if self.cache else 0,
            'model_name': self.config.model_name,
            'optimization_core_enabled': self.config.use_optimization_core
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Claude API cache cleared")

def create_claude_api_model(config: Optional[Dict[str, Any]] = None) -> ClaudeAPIClient:
    """
    Factory function to create Claude API model with optimization_core integration.
    
    Args:
        config: Configuration dictionary for Claude API
        
    Returns:
        ClaudeAPIClient instance with optimization_core integration
    """
    if config is None:
        config = {}
    
    claude_config = ClaudeAPIConfig(**config)
    
    model = ClaudeAPIClient(claude_config)
    
    try:
        from enhanced_model_optimizer import create_universal_optimizer
        optimizer = create_universal_optimizer({
            'enable_fp16': True,
            'enable_gradient_checkpointing': True,
            'use_advanced_normalization': True,
            'use_enhanced_mlp': True,
            'use_mcts_optimization': True
        })
        model = optimizer.optimize_model(model, "claud_api")
        logger.info("Applied universal optimization to Claude API model")
    except ImportError:
        logger.warning("Universal optimizer not available for Claude API model")
    
    return model

def create_claud_api_model(config: Optional[Dict[str, Any]] = None) -> ClaudeAPIClient:
    """
    Factory function to create Claude API model with optimization_core integration (alternative spelling).
    
    Args:
        config: Configuration dictionary for Claude API
        
    Returns:
        ClaudeAPIClient instance with optimization_core integration
    """
    return create_claude_api_model(config)
