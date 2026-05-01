"""
Inference utilities module.

Provides shared utilities for inference engines including validation,
error handling, logging, decorators, and common operations.
"""
from .validators import (
    validate_model_path,
    validate_generation_params,
    validate_sampling_params,
    validate_batch_size,
)
from .prompt_utils import (
    normalize_prompts,
    extract_generated_text,
    handle_single_prompt,
)
from .decorators import (
    validate_inputs,
    handle_errors,
    log_execution_time,
    retry_on_failure,
    cache_result,
    performance_monitor,
)
from .logging_utils import (
    InferenceMetrics,
    MetricsCollector,
    setup_inference_logging,
    log_operation,
    log_metrics,
)

__all__ = [
    # Validators
    "validate_model_path",
    "validate_generation_params",
    "validate_sampling_params",
    "validate_batch_size",
    # Prompt utilities
    "normalize_prompts",
    "extract_generated_text",
    "handle_single_prompt",
    # Decorators
    "validate_inputs",
    "handle_errors",
    "log_execution_time",
    "retry_on_failure",
    "cache_result",
    "performance_monitor",
    # Logging utilities
    "InferenceMetrics",
    "MetricsCollector",
    "setup_inference_logging",
    "log_operation",
    "log_metrics",
]

