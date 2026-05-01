"""
Utility modules for polyglot_core.

Logging, validation, errors, context managers, decorators, events, common utilities, factory, builder, and registry.
"""

from ..logging import (
    PolyglotLogger,
    PolyglotFormatter,
    StructuredFormatter,
    get_logger,
    setup_logging,
)

from ..validation import (
    ValidationError,
    validate_tensor,
    validate_attention_inputs,
    validate_cache_key,
    validate_config,
    validate_backend,
    validate_range,
    validate_positive,
    validate_non_negative,
)

from ..errors import (
    PolyglotError,
    BackendError,
    BackendNotAvailableError,
    BackendSelectionError,
    CacheError,
    CacheFullError,
    CacheKeyError,
    AttentionError,
    AttentionShapeError,
    CompressionError,
    CompressionFailedError,
    InferenceError,
    GenerationError,
    TokenizationError,
    QuantizationError,
    ValidationError as PolyglotValidationError,
    ConfigurationError,
    HealthCheckError,
    handle_polyglot_error,
)

from ..context import (
    operation_context,
    backend_context,
    performance_context,
    resource_context,
)

from ..decorators import (
    profile_operation,
    validate_inputs,
    handle_errors,
    retry_on_failure,
    cache_result,
    log_operation,
    measure_performance,
    ensure_backend,
)

from ..events import (
    EventType,
    Event,
    EventEmitter,
    get_event_emitter,
    emit_event,
    on_event,
    off_event,
)

from ..utils import (
    format_bytes,
    format_time,
    validate_shape,
    ensure_contiguous,
    pad_sequence,
    truncate_sequence,
    batch_tensors,
    get_device_info,
    print_device_info,
    estimate_memory_usage,
    create_random_tensor,
    check_backend_compatibility,
)

from ..factory import (
    ComponentType,
    FactoryConfig,
    ComponentFactory,
    get_factory,
    create_component,
)

from ..builder import (
    CacheBuilder,
    AttentionBuilder,
    InferenceBuilder,
    cache_builder,
    attention_builder,
    inference_builder,
)

from ..registry import (
    RegistryType,
    RegistryEntry,
    ComponentRegistry,
    get_registry,
    register_component,
    get_component,
)

__all__ = [
    # Logging
    "PolyglotLogger",
    "PolyglotFormatter",
    "StructuredFormatter",
    "get_logger",
    "setup_logging",
    # Validation
    "ValidationError",
    "validate_tensor",
    "validate_attention_inputs",
    "validate_cache_key",
    "validate_config",
    "validate_backend",
    "validate_range",
    "validate_positive",
    "validate_non_negative",
    # Errors
    "PolyglotError",
    "BackendError",
    "BackendNotAvailableError",
    "BackendSelectionError",
    "CacheError",
    "CacheFullError",
    "CacheKeyError",
    "AttentionError",
    "AttentionShapeError",
    "CompressionError",
    "CompressionFailedError",
    "InferenceError",
    "GenerationError",
    "TokenizationError",
    "QuantizationError",
    "PolyglotValidationError",
    "ConfigurationError",
    "HealthCheckError",
    "handle_polyglot_error",
    # Context
    "operation_context",
    "backend_context",
    "performance_context",
    "resource_context",
    # Decorators
    "profile_operation",
    "validate_inputs",
    "handle_errors",
    "retry_on_failure",
    "cache_result",
    "log_operation",
    "measure_performance",
    "ensure_backend",
    # Events
    "EventType",
    "Event",
    "EventEmitter",
    "get_event_emitter",
    "emit_event",
    "on_event",
    "off_event",
    # Common Utils
    "format_bytes",
    "format_time",
    "validate_shape",
    "ensure_contiguous",
    "pad_sequence",
    "truncate_sequence",
    "batch_tensors",
    "get_device_info",
    "print_device_info",
    "estimate_memory_usage",
    "create_random_tensor",
    "check_backend_compatibility",
    # Factory
    "ComponentType",
    "FactoryConfig",
    "ComponentFactory",
    "get_factory",
    "create_component",
    # Builder
    "CacheBuilder",
    "AttentionBuilder",
    "InferenceBuilder",
    "cache_builder",
    "attention_builder",
    "inference_builder",
    # Registry
    "RegistryType",
    "RegistryEntry",
    "ComponentRegistry",
    "get_registry",
    "register_component",
    "get_component",
]

