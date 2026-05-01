"""
Base Router Module
Abstract base class and interfaces for all routing strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
import time
import logging

class RoutingStrategy(Enum):
    """Routing strategy types."""
    ATTENTION_BASED = "attention_based"
    HIERARCHICAL = "hierarchical"
    NEURAL = "neural"
    ADAPTIVE = "adaptive"
    LOAD_BALANCING = "load_balancing"
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"

class ExpertType(Enum):
    """Expert type classifications."""
    REASONING = "reasoning"
    COMPUTATION = "computation"
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    LANGUAGE = "language"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    SPECIALIZED = "specialized"

@dataclass
class RoutingResult:
    """Result of routing decision."""
    expert_indices: List[int]
    expert_weights: List[float]
    routing_confidence: float
    routing_time: float
    strategy_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouterConfig:
    """Base router configuration."""
    strategy: RoutingStrategy = RoutingStrategy.ATTENTION_BASED
    num_experts: int = 8
    hidden_size: int = 512
    max_tokens_per_expert: int = 4
    min_tokens_per_expert: int = 1
    load_balancing_weight: float = 0.1
    confidence_threshold: float = 0.5
    enable_caching: bool = True
    cache_size: int = 1000
    enable_metrics: bool = True
    enable_logging: bool = True

class BaseRouter(ABC):
    """
    Abstract base class for all routing strategies.
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.metrics = {}
        self.cache = {} if config.enable_caching else None
        self._initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the router."""
        pass
    
    @abstractmethod
    def route_tokens(
        self, 
        input_tokens: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """
        Route tokens to experts.
        
        Args:
            input_tokens: Input token tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len]
            context: Optional routing context
            
        Returns:
            RoutingResult with expert assignments
        """
        pass
    
    @abstractmethod
    def get_router_info(self) -> Dict[str, Any]:
        """Get router information and statistics."""
        pass
    
    def validate_input(self, input_tokens: torch.Tensor) -> None:
        """Validate input tensor."""
        if not isinstance(input_tokens, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if input_tokens.dim() != 3:
            raise ValueError("Input must be 3D tensor [batch, seq_len, hidden_size]")
        
        if input_tokens.size(-1) != self.config.hidden_size:
            raise ValueError(f"Hidden size mismatch: expected {self.config.hidden_size}, got {input_tokens.size(-1)}")
    
    def get_cache_key(self, input_tokens: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for input."""
        if not self.cache:
            return None
        
        # Create hash from tensor and context
        tensor_hash = hash(input_tokens.data.tobytes())
        context_hash = hash(str(context)) if context else 0
        return f"{tensor_hash}_{context_hash}"
    
    def get_cached_result(self, cache_key: str) -> Optional[RoutingResult]:
        """Get cached routing result."""
        if not self.cache or cache_key not in self.cache:
            return None
        
        cached_result, timestamp = self.cache[cache_key]
        
        # Check cache expiry (5 minutes)
        if time.time() - timestamp > 300:
            del self.cache[cache_key]
            return None
        
        return cached_result
    
    def cache_result(self, cache_key: str, result: RoutingResult) -> None:
        """Cache routing result."""
        if not self.cache:
            return
        
        # Implement LRU cache
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = (result, time.time())
    
    def record_metrics(self, result: RoutingResult) -> None:
        """Record routing metrics."""
        if not self.config.enable_metrics:
            return
        
        self.metrics.setdefault('routing_time', []).append(result.routing_time)
        self.metrics.setdefault('confidence', []).append(result.routing_confidence)
        self.metrics.setdefault('expert_utilization', []).append(len(result.expert_indices))
        
        # Keep only recent metrics
        max_metrics = 1000
        for key in self.metrics:
            if len(self.metrics[key]) > max_metrics:
                self.metrics[key] = self.metrics[key][-max_metrics:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        if not self.metrics:
            return {}
        
        metrics = {}
        for key, values in self.metrics.items():
            if values:
                metrics[f'{key}_mean'] = np.mean(values)
                metrics[f'{key}_std'] = np.std(values)
                metrics[f'{key}_min'] = np.min(values)
                metrics[f'{key}_max'] = np.max(values)
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset routing metrics."""
        self.metrics.clear()
    
    def log_routing(self, result: RoutingResult, input_shape: Tuple[int, ...]) -> None:
        """Log routing information."""
        if not self.config.enable_logging:
            return
        
        self.logger.info(
            f"Routing completed: "
            f"experts={len(result.expert_indices)}, "
            f"confidence={result.routing_confidence:.3f}, "
            f"time={result.routing_time:.4f}s, "
            f"strategy={result.strategy_used}, "
            f"input_shape={input_shape}"
        )
    
    @property
    def is_initialized(self) -> bool:
        """Check if router is initialized."""
        return self._initialized
    
    def shutdown(self) -> None:
        """Shutdown the router."""
        self.cache.clear() if self.cache else None
        self.reset_metrics()
        self._initialized = False





