"""
Core Module - Foundation for Optimization Core.

This module provides organized access to core components:
- Interfaces: Base classes and ABCs for modular architecture
- Validators: Input validation utilities
- File Utils: File operations and format detection
- Factory Base: Factory pattern implementations
- Config: Configuration management
- Services: Base services (training, inference, model)
- Adapters: Adapter pattern implementations
- Composition: Workflow and component assembly
"""

from __future__ import annotations

import importlib

# ════════════════════════════════════════════════════════════════════════════════
# CORE VALIDATORS (Eager Import - Frequently Used)
# ════════════════════════════════════════════════════════════════════════════════

from .validators import (
    ValidationError,
    validate_non_empty_string,
    validate_path,
    validate_model_path,
    validate_file_path,
    validate_positive_number,
    validate_positive_int,
    validate_float_range,
    validate_generation_params,
    validate_sampling_params,
    validate_batch_size,
    validate_precision,
    validate_quantization,
    validate_dataframe_schema,
    validate_column_exists,
)

# ════════════════════════════════════════════════════════════════════════════════
# FILE UTILITIES (Eager Import - Frequently Used)
# ════════════════════════════════════════════════════════════════════════════════

from .file_utils import (
    detect_file_format,
    validate_file_format,
    ensure_output_directory,
    get_file_size,
    get_file_size_mb,
    list_files,
    get_file_info,
    safe_remove,
    safe_rename,
    get_temp_path,
    SUPPORTED_FORMATS,
)

# ════════════════════════════════════════════════════════════════════════════════
# FACTORY BASE (Eager Import - Foundation)
# ════════════════════════════════════════════════════════════════════════════════

from .factory_base import (
    FactoryError,
    ComponentNotFoundError,
    ComponentNotAvailableError,
    BaseFactory,
    SimpleFactory,
    CallableFactory,
    FactoryRegistry,
)

# ════════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS (Load on Demand)
# ════════════════════════════════════════════════════════════════════════════════

_LAZY_IMPORTS = {
    # Interfaces
    'interfaces': '.interfaces',
    'BaseModelManager': '.interfaces',
    'BaseDataLoader': '.interfaces',
    'BaseEvaluator': '.interfaces',
    'BaseCheckpointManager': '.interfaces',
    'BaseTrainer': '.interfaces',
    # Configuration
    'config': '.config',
    'ModelConfig': '.config',
    'TrainingConfig': '.config',
    'DataConfig': '.config',
    'HardwareConfig': '.config',
    'CheckpointConfig': '.config',
    'EMAConfig': '.config',
    'ResumeConfig': '.config',
    'load_config': '.config',
    'save_config': '.config',
    # Services
    'services': '.services',
    'BaseService': '.services.base_service',
    'TrainingService': '.services.training_service',
    'InferenceService': '.services.inference_service',
    'ModelService': '.services.model_service',
    # Adapters
    'adapters': '.adapters',
    'ModelAdapter': '.adapters.model_adapter',
    'DataAdapter': '.adapters.data_adapter',
    'OptimizerAdapter': '.adapters.optimizer_adapter',
    # Composition
    'composition': '.composition',
    'WorkflowBuilder': '.composition.workflow_builder',
    'ComponentAssembler': '.composition.component_assembler',
    # Papers
    'papers': '.papers',
    'PaperRegistry': '.papers.paper_registry',
    'get_paper_registry': '.papers.paper_registry',
    'PaperAdapter': '.papers.paper_adapter',
    'ModelEnhancer': '.papers.paper_adapter',
    'PaperFactory': '.papers.paper_factory',
    'create_paper_component': '.papers.paper_factory',
    'PaperMetadata': '.papers.paper_metadata',
    'PaperModule': '.papers.paper_metadata',
    # Framework
    'framework': '.framework',
    'AIExtremeOptimizer': '.framework.ai_extreme_optimizer',
    'OptimizationPipeline': '.framework.optimization_pipeline',
    # Configuration
    'config_base': '.config_base',
    'ConfigError': '.config_base',
    'ConfigValidationError': '.config_base',
    'BaseConfig': '.config_base',
    'ValidatedConfig': '.config_base',
    # Logging
    'logging_utils': '.logging_utils',
    'get_logger': '.logging_utils',
    'setup_logging': '.logging_utils',
    'LoggerMixin': '.logging_utils',
    'log_function_call': '.logging_utils',
    'log_execution_time': '.logging_utils',
    # Exceptions
    'exceptions': '.exceptions',
    'OptimizationCoreError': '.exceptions',
    'ValidationError': '.exceptions',
    'ConfigurationError': '.exceptions',
    'ResourceError': '.exceptions',
    'PerformanceError': '.exceptions',
    'ModelError': '.exceptions',
    'InferenceError': '.exceptions',
    'DataError': '.exceptions',
    'ErrorSeverity': '.exceptions',
    # Helpers
    'helpers': '.helpers',
    'ensure_initialized': '.helpers',
    'timing': '.helpers',
    'handle_errors': '.helpers',
    'retry': '.helpers',
    'timing_context': '.helpers',
    'error_context': '.helpers',
    'batch_items': '.helpers',
    'format_duration': '.helpers',
    'format_size': '.helpers',
    'safe_get': '.helpers',
    'chunk_list': '.helpers',
    # Metrics
    'metrics_base': '.metrics_base',
    'BaseMetrics': '.metrics_base',
    'MetricsCollectorBase': '.metrics_base',
    'calculate_perplexity': '.metrics_base',
    'calculate_tokens_per_second': '.metrics_base',
    'calculate_throughput': '.metrics_base',
    'calculate_percentage': '.metrics_base',
    'calculate_rate': '.metrics_base',
    'format_metric_value': '.metrics_base',
    # Serialization
    'serialization': '.serialization',
    'to_dict': '.serialization',
    'from_dict': '.serialization',
    'to_json': '.serialization',
    'from_json': '.serialization',
    'to_pickle': '.serialization',
    'from_pickle': '.serialization',
    'safe_serialize': '.serialization',
    # Decorators
    'decorators': '.decorators',
    'retry': '.decorators',
    'timeout': '.decorators',
    'cache_result': '.decorators',
    'rate_limit': '.decorators',
    'circuit_breaker': '.decorators',
    'log_execution_time': '.decorators',
    'handle_errors': '.decorators',
    # Types
    'types': '.types',
    'PathLike': '.types',
    'Number': '.types',
    'DictStrAny': '.types',
    'Serializable': '.types',
    'Validatable': '.types',
    'Resettable': '.types',
    'Precision': '.types',
    'QuantizationType': '.types',
    'BackendType': '.types',
    'DeviceType': '.types',
    # Test utilities
    'test_utils': '.test_utils',
    'create_test_config': '.test_utils',
    'assert_tensor_close': '.test_utils',
    'assert_shape_equal': '.test_utils',
    'assert_dtype_equal': '.test_utils',
    'measure_execution_time': '.test_utils',
    'compare_performance': '.test_utils',
    'MemoryTracker': '.test_utils',
    'create_mock_service': '.test_utils',
    'create_async_mock_service': '.test_utils',
    'retry_test': '.test_utils',
    'skip_if_unavailable': '.test_utils',
    'performance_context': '.test_utils',
    'cleanup_context': '.test_utils',
    # String utilities
    'string_utils': '.string_utils',
    'clean_text': '.string_utils',
    'slugify': '.string_utils',
    'truncate': '.string_utils',
    'camel_to_snake': '.string_utils',
    'extract_words': '.string_utils',
    'generate_random_string': '.string_utils',
    'is_email': '.string_utils',
    # Math utilities
    'math_utils': '.math_utils',
    'clamp': '.math_utils',
    'round_to': '.math_utils',
    'percentage': '.math_utils',
    'mean': '.math_utils',
    'median': '.math_utils',
    'lerp': '.math_utils',
    # Datetime utilities
    'datetime_utils': '.datetime_utils',
    'now_utc': '.datetime_utils',
    'now_iso': '.datetime_utils',
    'parse_datetime': '.datetime_utils',
    'format_datetime': '.datetime_utils',
    'format_relative_time': '.datetime_utils',
    'add_days': '.datetime_utils',
    # Collection utilities
    'collection_utils': '.collection_utils',
    'chunk_list': '.collection_utils',
    'flatten_list': '.collection_utils',
    'group_by': '.collection_utils',
    'partition': '.collection_utils',
    'unique_list': '.collection_utils',
    'filter_dict': '.collection_utils',
    'merge_dicts': '.collection_utils',
    # Async utilities
    'async_utils': '.async_utils',
    'gather_with_limit': '.async_utils',
    'async_map': '.async_utils',
    'async_filter': '.async_utils',
    'async_timeout': '.async_utils',
    'async_batch_process': '.async_utils',
    'run_async': '.async_utils',
    # Encoding utilities
    'encoding_utils': '.encoding_utils',
    'encode_base64': '.encoding_utils',
    'decode_base64': '.encoding_utils',
    'hash_data': '.encoding_utils',
    'hash_file': '.encoding_utils',
    'create_hmac': '.encoding_utils',
    'generate_random_bytes': '.encoding_utils',
    # Environment utilities
    'env_utils': '.env_utils',
    'get_env': '.env_utils',
    'get_env_str': '.env_utils',
    'get_env_int': '.env_utils',
    'get_env_bool': '.env_utils',
    'load_env_file': '.env_utils',
    'get_environment': '.env_utils',
    'is_production': '.env_utils',
    # URL utilities
    'url_utils': '.url_utils',
    'parse_url': '.url_utils',
    'get_domain': '.url_utils',
    'get_path': '.url_utils',
    'build_url': '.url_utils',
    'is_valid_url': '.url_utils',
    # Retry utilities
    'retry_utils': '.retry_utils',
    'BackoffStrategy': '.retry_utils',
    'retry_with_backoff': '.retry_utils',
    'async_retry_with_backoff': '.retry_utils',
    # Rate limiting utilities
    'rate_limit_utils': '.rate_limit_utils',
    'RateLimiter': '.rate_limit_utils',
    'AsyncRateLimiter': '.rate_limit_utils',
    'Throttler': '.rate_limit_utils',
    # Circuit breaker utilities
    'circuit_breaker_utils': '.circuit_breaker_utils',
    'CircuitState': '.circuit_breaker_utils',
    'CircuitBreaker': '.circuit_breaker_utils',
    'AsyncCircuitBreaker': '.circuit_breaker_utils',
    # Health check utilities
    'health_check_utils': '.health_check_utils',
    'HealthStatus': '.health_check_utils',
    'HealthChecker': '.health_check_utils',
    'AsyncHealthChecker': '.health_check_utils',
    # Performance utilities
    'performance_utils': '.performance_utils',
    'PerformanceMetrics': '.performance_utils',
    'BenchmarkResult': '.performance_utils',
    'FunctionStats': '.performance_utils',
    'FunctionProfiler': '.performance_utils',
    'PerformanceProfiler': '.performance_utils',
    'TorchModelProfiler': '.performance_utils',
    'profile_context': '.performance_utils',
    'profile_operation': '.performance_utils',
    'profile_function': '.performance_utils',
    'profile_decorator': '.performance_utils',
    'measure_memory_usage': '.performance_utils',
    'benchmark_model': '.performance_utils',
    'profile_model': '.performance_utils',
    'optimize_model': '.performance_utils',
    'create_optimization_report': '.performance_utils',
    # Utilities
    'module_loader': '.module_loader',
    'plugin_system': '.plugin_system',
    'event_system': '.event_system',
    'service_registry': '.service_registry',
    'dynamic_factory': '.dynamic_factory',
    # Optimizers
    'optimizers': '.optimizers',
    # Validation
    'validation': '.validation',
    'ConfigValidator': '.validation.config_validator',
    'DataValidator': '.validation.data_validator',
    'ModelValidator': '.validation.model_validator',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for core submodules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        if name in ['interfaces', 'config', 'services', 'adapters', 'composition', 'framework', 'validation', 'optimizers']:
            # Import entire module
            module = importlib.import_module(module_path, package=__package__)
        else:
            # Import specific class/function
            module = importlib.import_module(module_path, package=__package__)
            if hasattr(module, name):
                _import_cache[name] = getattr(module, name)
                return _import_cache[name]
            else:
                # Return module if class not found
                _import_cache[name] = module
                return module
        
        _import_cache[name] = module
        return module
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


# ════════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_core_component(component_type: str, config: dict = None):
    """
    Unified factory function to create core components.
    
    Args:
        component_type: Type of component. Options:
            - "service": Create a service (training/inference/model)
            - "adapter": Create an adapter (model/data/optimizer)
            - "validator": Create a validator (config/data/model)
            - "workflow": Create a workflow builder
        config: Optional configuration dictionary
    
    Returns:
        The requested component instance
    
    Examples:
        >>> # Create training service
        >>> service = create_core_component(
        ...     "service",
        ...     {"service_type": "training", "config": {...}}
        ... )
        
        >>> # Create model adapter
        >>> adapter = create_core_component(
        ...     "adapter",
        ...     {"adapter_type": "model", "model": model}
        ... )
    """
    if config is None:
        config = {}
    
    component_type = component_type.lower()
    
    if component_type == "service":
        from .services import BaseService
        service_type = config.get("service_type", "training")
        if service_type == "training":
            from .services.training_service import TrainingService
            return TrainingService(**config)
        elif service_type == "inference":
            from .services.inference_service import InferenceService
            return InferenceService(**config)
        elif service_type == "model":
            from .services.model_service import ModelService
            return ModelService(**config)
        else:
            raise ValueError(f"Unknown service type: {service_type}")
    
    elif component_type == "adapter":
        adapter_type = config.get("adapter_type", "model")
        if adapter_type == "model":
            from .adapters.model_adapter import ModelAdapter
            return ModelAdapter(**config)
        elif adapter_type == "data":
            from .adapters.data_adapter import DataAdapter
            return DataAdapter(**config)
        elif adapter_type == "optimizer":
            from .adapters.optimizer_adapter import OptimizerAdapter
            return OptimizerAdapter(**config)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    elif component_type == "validator":
        validator_type = config.get("validator_type", "config")
        if validator_type == "config":
            from .validation.config_validator import ConfigValidator
            return ConfigValidator(**config)
        elif validator_type == "data":
            from .validation.data_validator import DataValidator
            return DataValidator(**config)
        elif validator_type == "model":
            from .validation.model_validator import ModelValidator
            return ModelValidator(**config)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")
    
    elif component_type == "workflow":
        from .composition.workflow_builder import WorkflowBuilder
        return WorkflowBuilder(**config)
    
    else:
        available = ["service", "adapter", "validator", "workflow"]
        raise ValueError(
            f"Unknown core component type: '{component_type}'. "
            f"Available types: {available}"
        )


# ════════════════════════════════════════════════════════════════════════════════
# REGISTRY SYSTEM
# ════════════════════════════════════════════════════════════════════════════════

CORE_COMPONENT_REGISTRY = {
    "validators": {
        "module": "core.validators",
        "description": "Input validation utilities",
        "exports": [
            "ValidationError",
            "validate_non_empty_string",
            "validate_path",
            "validate_model_path",
            "validate_file_path",
            "validate_positive_number",
            "validate_positive_int",
            "validate_float_range",
            "validate_generation_params",
            "validate_sampling_params",
            "validate_batch_size",
            "validate_precision",
            "validate_quantization",
            "validate_dataframe_schema",
            "validate_column_exists",
        ],
    },
    "file_utils": {
        "module": "core.file_utils",
        "description": "File operations and format detection",
        "exports": [
            "detect_file_format",
            "validate_file_format",
            "ensure_output_directory",
            "get_file_size",
            "get_file_size_mb",
            "list_files",
            "get_file_info",
            "safe_remove",
            "safe_rename",
            "get_temp_path",
            "SUPPORTED_FORMATS",
        ],
    },
    "factory_base": {
        "module": "core.factory_base",
        "description": "Factory pattern implementations",
        "exports": [
            "FactoryError",
            "ComponentNotFoundError",
            "ComponentNotAvailableError",
            "BaseFactory",
            "SimpleFactory",
            "CallableFactory",
            "FactoryRegistry",
        ],
    },
    "config_base": {
        "module": "core.config_base",
        "description": "Base configuration classes with validation",
        "exports": [
            "ConfigError",
            "ConfigValidationError",
            "BaseConfig",
            "ValidatedConfig",
        ],
    },
    "logging_utils": {
        "module": "core.logging_utils",
        "description": "Logging utilities and decorators",
        "exports": [
            "get_logger",
            "setup_logging",
            "LoggerMixin",
            "log_function_call",
            "log_execution_time",
        ],
    },
    "exceptions": {
        "module": "core.exceptions",
        "description": "Unified exception hierarchy",
        "exports": [
            "ErrorSeverity",
            "OptimizationCoreError",
            "ValidationError",
            "ConfigurationError",
            "ResourceError",
            "PerformanceError",
            "ModelError",
            "InferenceError",
            "DataError",
        ],
    },
    "helpers": {
        "module": "core.helpers",
        "description": "Common helper utilities (decorators, context managers)",
        "exports": [
            "ensure_initialized",
            "timing",
            "handle_errors",
            "retry",
            "timing_context",
            "error_context",
            "batch_items",
            "format_duration",
            "format_size",
            "safe_get",
            "chunk_list",
        ],
    },
    "metrics_base": {
        "module": "core.metrics_base",
        "description": "Base metrics classes and utilities",
        "exports": [
            "BaseMetrics",
            "MetricsCollectorBase",
            "calculate_perplexity",
            "calculate_tokens_per_second",
            "calculate_throughput",
            "calculate_percentage",
            "calculate_rate",
            "format_metric_value",
        ],
    },
    "serialization": {
        "module": "core.serialization",
        "description": "Serialization utilities (JSON, pickle, dict)",
        "exports": [
            "to_dict",
            "from_dict",
            "to_json",
            "from_json",
            "to_pickle",
            "from_pickle",
            "safe_serialize",
        ],
    },
    "types": {
        "module": "core.types",
        "description": "Common type definitions and type aliases",
        "exports": [
            "PathLike",
            "Number",
            "DictStrAny",
            "Serializable",
            "Validatable",
            "Resettable",
            "Precision",
            "QuantizationType",
            "BackendType",
            "DeviceType",
        ],
    },
    "decorators": {
        "module": "core.decorators",
        "description": "Common decorators (retry, timeout, cache, rate limit, circuit breaker)",
        "exports": [
            "retry",
            "timeout",
            "cache_result",
            "rate_limit",
            "circuit_breaker",
            "log_execution_time",
            "handle_errors",
            "RateLimiter",
            "CircuitBreaker",
            "CircuitState",
        ],
    },
    "test_utils": {
        "module": "core.test_utils",
        "description": "Common testing utilities (assertions, performance, mocks)",
        "exports": [
            "create_test_config",
            "assert_tensor_close",
            "assert_shape_equal",
            "assert_dtype_equal",
            "measure_execution_time",
            "compare_performance",
            "MemoryTracker",
            "create_mock_service",
            "create_async_mock_service",
            "retry_test",
            "skip_if_unavailable",
            "performance_context",
            "cleanup_context",
        ],
    },
    "string_utils": {
        "module": "core.string_utils",
        "description": "Common string manipulation utilities",
        "exports": [
            "clean_text",
            "normalize_whitespace",
            "remove_html_tags",
            "slugify",
            "truncate",
            "camel_to_snake",
            "snake_to_camel",
            "extract_words",
            "extract_hashtags",
            "extract_emails",
            "generate_random_string",
            "is_email",
            "is_url",
            "sanitize_filename",
        ],
    },
    "math_utils": {
        "module": "core.math_utils",
        "description": "Common math and numerical utilities",
        "exports": [
            "clamp",
            "in_range",
            "round_to",
            "ceil_to",
            "floor_to",
            "percentage",
            "percentage_of",
            "ratio",
            "mean",
            "median",
            "std_dev",
            "is_positive",
            "is_non_negative",
            "is_finite",
            "lerp",
            "map_range",
        ],
    },
    "datetime_utils": {
        "module": "core.datetime_utils",
        "description": "Common datetime manipulation utilities",
        "exports": [
            "now_utc",
            "now_local",
            "now_iso",
            "parse_datetime",
            "parse_datetime_format",
            "format_datetime",
            "format_datetime_iso",
            "format_relative_time",
            "to_utc",
            "add_days",
            "add_hours",
            "add_minutes",
            "add_seconds",
            "start_of_day",
            "end_of_day",
        ],
    },
    "collection_utils": {
        "module": "core.collection_utils",
        "description": "Common collection manipulation utilities",
        "exports": [
            "chunk_list",
            "chunk_list_iter",
            "flatten_list",
            "flatten_list_deep",
            "group_by",
            "group_by_key",
            "partition",
            "unique_list",
            "filter_dict",
            "merge_dicts",
            "get_nested_value",
            "set_nested_value",
        ],
    },
    "async_utils": {
        "module": "core.async_utils",
        "description": "Common async/concurrency utilities",
        "exports": [
            "gather_with_limit",
            "async_map",
            "async_filter",
            "async_timeout",
            "async_batch_process",
            "async_retry_context",
            "run_async",
        ],
    },
    "encoding_utils": {
        "module": "core.encoding_utils",
        "description": "Common encoding, decoding, and hashing utilities",
        "exports": [
            "encode_base64",
            "decode_base64",
            "encode_base64_urlsafe",
            "decode_base64_urlsafe",
            "encode_hex",
            "decode_hex",
            "hash_data",
            "hash_file",
            "create_hmac",
            "verify_hmac",
            "generate_random_bytes",
            "generate_random_hex",
            "generate_random_urlsafe",
        ],
    },
    "env_utils": {
        "module": "core.env_utils",
        "description": "Common environment variable utilities",
        "exports": [
            "get_env",
            "get_env_str",
            "get_env_int",
            "get_env_float",
            "get_env_bool",
            "get_env_list",
            "load_env_file",
            "get_environment",
            "is_production",
            "is_development",
            "is_testing",
        ],
    },
    "url_utils": {
        "module": "core.url_utils",
        "description": "Common URL manipulation utilities",
        "exports": [
            "parse_url",
            "get_domain",
            "get_path",
            "get_query_params",
            "get_query_param",
            "build_url",
            "join_url",
            "encode_url",
            "decode_url",
            "is_valid_url",
            "normalize_url",
        ],
    },
    "retry_utils": {
        "module": "core.retry_utils",
        "description": "Advanced retry utilities with various backoff strategies",
        "exports": [
            "BackoffStrategy",
            "calculate_backoff",
            "add_jitter",
            "retry_with_backoff",
            "async_retry_with_backoff",
        ],
    },
    "rate_limit_utils": {
        "module": "core.rate_limit_utils",
        "description": "Rate limiting and throttling utilities",
        "exports": [
            "RateLimiter",
            "AsyncRateLimiter",
            "Throttler",
        ],
    },
    "circuit_breaker_utils": {
        "module": "core.circuit_breaker_utils",
        "description": "Circuit breaker utilities for fault tolerance",
        "exports": [
            "CircuitState",
            "CircuitBreakerConfig",
            "CircuitBreaker",
            "AsyncCircuitBreaker",
            "CircuitBreakerOpenError",
        ],
    },
    "health_check_utils": {
        "module": "core.health_check_utils",
        "description": "Health check utilities for services/components",
        "exports": [
            "HealthStatus",
            "HealthCheckResult",
            "HealthChecker",
            "AsyncHealthChecker",
            "create_simple_check",
            "create_async_simple_check",
        ],
    },
    "performance_utils": {
        "module": "core.performance_utils",
        "description": "Profiling, benchmarking, and optimization utilities",
        "exports": [
            "PerformanceMetrics",
            "BenchmarkResult",
            "FunctionStats",
            "FunctionProfiler",
            "PerformanceProfiler",
            "TorchModelProfiler",
            "profile_context",
            "profile_operation",
            "profile_function",
            "profile_decorator",
            "measure_memory_usage",
            "benchmark_model",
            "profile_model",
            "optimize_model",
            "create_optimization_report",
        ],
    },
    "interfaces": {
        "module": "core.interfaces",
        "description": "Base classes and ABCs for modular architecture",
        "exports": [
            "BaseModelManager",
            "BaseDataLoader",
            "BaseEvaluator",
            "BaseCheckpointManager",
            "BaseTrainer",
        ],
    },
    "config": {
        "module": "core.config",
        "description": "Configuration management with validation",
        "exports": [
            "ModelConfig",
            "TrainingConfig",
            "DataConfig",
            "HardwareConfig",
            "CheckpointConfig",
            "EMAConfig",
            "ResumeConfig",
            "load_config",
            "save_config",
        ],
    },
    "services": {
        "module": "core.services",
        "description": "Base services (training, inference, model)",
        "exports": [
            "BaseService",
            "TrainingService",
            "InferenceService",
            "ModelService",
        ],
    },
    "adapters": {
        "module": "core.adapters",
        "description": "Adapter pattern implementations",
        "exports": [
            "ModelAdapter",
            "DataAdapter",
            "OptimizerAdapter",
        ],
    },
    "composition": {
        "module": "core.composition",
        "description": "Workflow and component assembly",
        "exports": [
            "WorkflowBuilder",
            "ComponentAssembler",
        ],
    },
    "papers": {
        "module": "core.papers",
        "description": "Research papers integration for model enhancement",
        "exports": [
            "PaperRegistry",
            "get_paper_registry",
            "PaperAdapter",
            "ModelEnhancer",
            "PaperFactory",
            "create_paper_component",
            "PaperMetadata",
            "PaperModule",
        ],
    },
}


def list_available_core_components() -> list[str]:
    """
    List all available core component types.
    
    Returns:
        List of component type names
    
    Examples:
        >>> components = list_available_core_components()
        >>> print(components)
        ['validators', 'file_utils', 'factory_base', 'interfaces', ...]
    """
    return list(CORE_COMPONENT_REGISTRY.keys())


def get_core_component_info(component_type: str) -> dict[str, any]:
    """
    Get information about a core component.
    
    Args:
        component_type: Component type name
    
    Returns:
        Dictionary with component information
    
    Examples:
        >>> info = get_core_component_info("validators")
        >>> print(info['description'])
        'Input validation utilities'
    """
    if component_type not in CORE_COMPONENT_REGISTRY:
        raise ValueError(
            f"Unknown core component: {component_type}. "
            f"Available: {list_available_core_components()}"
        )
    
    registry_entry = CORE_COMPONENT_REGISTRY[component_type]
    return {
        'name': component_type,
        'module': registry_entry['module'],
        'description': registry_entry['description'],
        'exports': registry_entry.get('exports', []),
    }


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Validators (eager)
    "ValidationError",
    "validate_non_empty_string",
    "validate_path",
    "validate_model_path",
    "validate_file_path",
    "validate_positive_number",
    "validate_positive_int",
    "validate_float_range",
    "validate_generation_params",
    "validate_sampling_params",
    "validate_batch_size",
    "validate_precision",
    "validate_quantization",
    "validate_dataframe_schema",
    "validate_column_exists",
    # File utilities (eager)
    "detect_file_format",
    "validate_file_format",
    "ensure_output_directory",
    "get_file_size",
    "get_file_size_mb",
    "list_files",
    "get_file_info",
    "safe_remove",
    "safe_rename",
    "get_temp_path",
    "SUPPORTED_FORMATS",
    # Factory base (eager)
    "FactoryError",
    "ComponentNotFoundError",
    "ComponentNotAvailableError",
    "BaseFactory",
    "SimpleFactory",
    "CallableFactory",
    "FactoryRegistry",
    # Factory functions
    "create_core_component",
    # Registry
    "CORE_COMPONENT_REGISTRY",
    "list_available_core_components",
    "get_core_component_info",
    # Lazy imports (available via __getattr__)
    "interfaces",
    "config",
    "config_base",
    "services",
    "adapters",
    "composition",
    "framework",
    "validation",
    "optimizers",
    "logging_utils",
    "exceptions",
    "helpers",
    "types",
    "decorators",
    "test_utils",
    "string_utils",
    "math_utils",
    "datetime_utils",
    "collection_utils",
    "async_utils",
    "encoding_utils",
    "env_utils",
    "url_utils",
    "retry_utils",
    "rate_limit_utils",
    "circuit_breaker_utils",
    "health_check_utils",
    "module_loader",
    "plugin_system",
    "event_system",
    "service_registry",
    "dynamic_factory",
    "papers",
]

