"""
Optimization Metrics
====================
Improved metrics calculation for optimization results.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate performance metrics for optimized models."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate(self, 
                  original_model: nn.Module,
                  optimized_model: nn.Module,
                  techniques_applied: List[str],
                  optimization_time: float) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            original_model: Original model before optimization
            optimized_model: Model after optimization
            techniques_applied: List of techniques that were applied
            optimization_time: Time taken for optimization (ms)
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'speed_improvement': self._estimate_speed_improvement(techniques_applied),
            'memory_reduction': self._estimate_memory_reduction(original_model, optimized_model, techniques_applied),
            'accuracy_preservation': self._estimate_accuracy_preservation(techniques_applied),
            'energy_efficiency': self._estimate_energy_efficiency(techniques_applied),
            'optimization_time': optimization_time,
        }
        
        return metrics
    
    def _estimate_speed_improvement(self, techniques: List[str]) -> float:
        """Estimate speed improvement based on applied techniques."""
        improvements = {
            'gradient_checkpointing': 1.2,
            'mixed_precision': 1.5,
            'torch_compile': 1.3,
            'tf32': 1.2,
            'fused_adamw': 1.1,
            'quantization': 2.0,
            'pruning': 1.3,
        }
        
        total_improvement = 1.0
        for technique in techniques:
            improvement = improvements.get(technique, 1.0)
            total_improvement *= improvement
        
        return total_improvement
    
    def _estimate_memory_reduction(self, 
                                   original: nn.Module,
                                   optimized: nn.Module,
                                   techniques: List[str]) -> float:
        """Estimate memory reduction."""
        reductions = {
            'gradient_checkpointing': 0.3,
            'mixed_precision': 0.5,
            'quantization': 0.75,
            'pruning': 0.5,
        }
        
        total_reduction = 0.0
        for technique in techniques:
            reduction = reductions.get(technique, 0.0)
            total_reduction = 1 - (1 - total_reduction) * (1 - reduction)
        
        return total_reduction
    
    def _estimate_accuracy_preservation(self, techniques: List[str]) -> float:
        """Estimate accuracy preservation."""
        preservation = {
            'gradient_checkpointing': 1.0,
            'mixed_precision': 0.99,
            'torch_compile': 1.0,
            'tf32': 1.0,
            'fused_adamw': 1.0,
            'quantization': 0.95,
            'pruning': 0.90,
        }
        
        min_preservation = 1.0
        for technique in techniques:
            pres = preservation.get(technique, 1.0)
            min_preservation = min(min_preservation, pres)
        
        return min_preservation
    
    def _estimate_energy_efficiency(self, techniques: List[str]) -> float:
        """Estimate energy efficiency improvement."""
        efficiencies = {
            'gradient_checkpointing': 1.1,
            'mixed_precision': 1.3,
            'torch_compile': 1.2,
            'quantization': 1.5,
            'pruning': 1.2,
        }
        
        total_efficiency = 1.0
        for technique in techniques:
            efficiency = efficiencies.get(technique, 1.0)
            total_efficiency *= efficiency
        
        return total_efficiency


_global_metrics_calculator = MetricsCalculator()


def calculate_optimization_metrics(original_model: nn.Module,
                                   optimized_model: nn.Module,
                                   techniques_applied: List[str],
                                   optimization_time: float) -> Dict[str, float]:
    """Calculate optimization metrics using the global calculator."""
    return _global_metrics_calculator.calculate(
        original_model, optimized_model, techniques_applied, optimization_time
    )








