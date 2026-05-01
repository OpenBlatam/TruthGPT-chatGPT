"""
Metrics utilities for optimization_core.

This module re-exports common metrics utilities from modules.base.core_system.core.metrics_base
for backward compatibility.
"""
from optimization_core.core.metrics_base import (
    calculate_perplexity,
    calculate_tokens_per_second,
    calculate_throughput,
    calculate_percentage,
    calculate_rate,
    format_metric_value,
    BaseMetrics,
    MetricsCollectorBase,
)

# Backward compatibility aliases
perplexity_from_loss = calculate_perplexity
tokens_per_second = calculate_tokens_per_second

__all__ = [
    "calculate_perplexity",
    "perplexity_from_loss",  # Alias
    "calculate_tokens_per_second",
    "tokens_per_second",  # Alias
    "calculate_throughput",
    "calculate_percentage",
    "calculate_rate",
    "format_metric_value",
    "BaseMetrics",
    "MetricsCollectorBase",
]






