"""
Examples for optimization_core.

Provides example code and usage patterns for all modules.
"""
from .inference_examples import (
    example_vllm_engine,
    example_tensorrt_engine,
    example_engine_factory,
    example_base_engine,
    example_with_decorators,
)
from .data_examples import (
    example_polars_processor,
    example_processor_factory,
    example_large_dataset,
)
from .benchmark_examples import (
    example_simple_benchmark,
    example_compare_engines,
    example_collect_metrics,
    example_benchmark_runner,
)
from .advanced_examples import (
    example_tracing,
    example_metrics,
    example_hyperparameter_optimization,
    example_plugin,
)

__all__ = [
    # Inference examples
    "example_vllm_engine",
    "example_tensorrt_engine",
    "example_engine_factory",
    "example_base_engine",
    "example_with_decorators",
    # Data examples
    "example_polars_processor",
    "example_processor_factory",
    "example_large_dataset",
    # Benchmark examples
    "example_simple_benchmark",
    "example_compare_engines",
    "example_collect_metrics",
    "example_benchmark_runner",
    # Advanced examples
    "example_tracing",
    "example_metrics",
    "example_hyperparameter_optimization",
    "example_plugin",
]

