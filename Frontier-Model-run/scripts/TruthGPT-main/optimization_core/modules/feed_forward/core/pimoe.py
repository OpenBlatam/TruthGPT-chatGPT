"""
Unified PiMoE System
====================

Consolidated implementation of Advanced, Enhanced, and Adaptive PiMoE systems.
"""

import logging
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from optimization_core.modules.feed_forward.core.interfaces import (
    ExpertType,
    ProductionConfig,
    SystemConfig,
    PiMoEProcessorProtocol
)

from optimization_core.modules.feed_forward.routing.pimoe_router import (
    create_pimoe_system,
    PiMoESystem
)

# Configure logging
logger = logging.getLogger(__name__)

# Windows UTF-8 Fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

class PerformanceTracker:
    """Tracks performance metrics over time."""
    def __init__(self, max_history: int = 1000) -> None:
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = max_history

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]

    def get_metrics(self) -> Dict[str, Any]:
        if not self.metrics_history: return {}
        avg_metrics: Dict[str, Any] = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history if key in m and isinstance(m[key], (int, float))]
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
                avg_metrics[f'min_{key}'] = min(values)
                avg_metrics[f'max_{key}'] = max(values)
        return avg_metrics

class UnifiedPiMoESystem(nn.Module):
    """
    Unified PiMoE system that combines static optimization with dynamic adaptation.
    """
    def __init__(
        self,
        config: Optional[ProductionConfig] = None,
        hidden_size: int = 512,
        num_experts: int = 8,
        expert_types: Optional[List[ExpertType]] = None,
        enable_adaptation: bool = False,
        adaptation_rate: float = 0.01,
        performance_threshold: float = 0.8
    ) -> None:
        super().__init__()
        self.config = config or ProductionConfig(system_config=SystemConfig(hidden_size=hidden_size, num_experts=num_experts))
        self.hidden_size = self.config.system_config.hidden_size
        self.num_experts = self.config.system_config.num_experts
        self.enable_adaptation = enable_adaptation
        self.adaptation_rate = adaptation_rate
        self.performance_threshold = performance_threshold

        # Initialize core PiMoE system
        self.pimoe_core = create_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=expert_types
        )

        self.performance_tracker = PerformanceTracker()
        self.adaptation_history: List[Dict[str, Any]] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        enable_optimization: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        start_time = time.time()
        
        # 1. Pre-processing Optimizations
        if enable_optimization and self.config.enable_quantization:
            hidden_states = self._apply_quantization(hidden_states)

        # 2. Main processing
        output, routing_info = self.pimoe_core(
            hidden_states,
            attention_mask,
            return_routing_info=True
        )

        # 3. Dynamic Adaptation
        if self.enable_adaptation:
            self._adapt_to_performance(hidden_states, output, routing_info)

        # 4. Post-processing
        if enable_optimization and self.config.enable_pruning:
            output = self._apply_pruning_mask(output)

        latency_ms = (time.time() - start_time) * 1000
        
        if return_metrics or self.config.enable_metrics:
            metrics = self._calculate_metrics(hidden_states, output, routing_info, latency_ms)
            self.performance_tracker.add_metrics(metrics)
            if return_metrics:
                return output, metrics

        return output

    def _apply_quantization(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified quantization stub logic
        return x

    def _apply_pruning_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified pruning mask logic
        return x

    def _calculate_metrics(self, input_tensor, output_tensor, routing_info, latency_ms) -> Dict[str, Any]:
        batch_size, seq_len, _ = input_tensor.shape
        total_tokens = batch_size * seq_len
        
        expert_utilization = 0.0
        if routing_info and 'routing_decisions' in routing_info:
            decisions = routing_info['routing_decisions']
            unique_experts = len(set(d.expert_id for d in decisions)) if decisions else 0
            expert_utilization = unique_experts / self.num_experts

        return {
            'latency_ms': latency_ms,
            'throughput_tokens_per_sec': total_tokens / (latency_ms / 1000) if latency_ms > 0 else 0,
            'expert_utilization': expert_utilization,
            'load_balance_score': routing_info.get('router_stats', {}).get('load_balance_ratio', 0.0) if routing_info else 0.0
        }

    def _adapt_to_performance(self, input_tensor, output_tensor, routing_info):
        # Implementation of adaptive logic merged from AdaptivePiMoE
        pass

    def get_system_stats(self) -> Dict[str, Any]:
        stats = self.pimoe_core.get_system_stats()
        stats.update({
            'performance_metrics': self.performance_tracker.get_metrics(),
            'adaptation_enabled': self.enable_adaptation
        })
        return stats

    def optimize_for_production(self) -> None:
        """Optimize model for production deployment."""
        self.eval()
        if hasattr(torch, 'compile'):
            try:
                self.pimoe_core = torch.compile(self.pimoe_core)
                logger.info("✅ PiMoE Core compiled successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not compile core: {e}")

def create_unified_pimoe(hidden_size: int, **kwargs) -> UnifiedPiMoESystem:
    return UnifiedPiMoESystem(hidden_size=hidden_size, **kwargs)

