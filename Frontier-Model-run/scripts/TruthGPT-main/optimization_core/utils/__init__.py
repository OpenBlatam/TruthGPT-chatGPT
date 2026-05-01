"""
Shared utilities for the entire optimization_core module.

Provides common utilities for validation, error handling, configuration,
and other cross-cutting concerns.
"""
from .base import (
    BaseOptimizationModel,
    CudaResourceManager,
    system_metrics_collector,
)
from .shared_validators import (
    validate_not_none,
    validate_not_empty,
    validate_type,
    validate_in_range,
    validate_one_of,
    validate_path_exists,
    validate_callable,
    validate_dict_keys,
    validate_list_items,
)
from .error_handling import (
    OptimizationCoreError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    PerformanceError,
    ErrorSeverity,
    handle_error,
    safe_execute,
    retry_with_backoff,
    error_context,
)
from .config_utils import (
    load_config,
    save_config,
    merge_configs,
    validate_config,
    get_config_value,
)
from .integration_utils import (
    ComponentRegistry,
    Pipeline,
    register_component,
    get_component,
    list_components,
    create_pipeline,
)
from .serialization_utils import (
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_pickle,
    load_pickle,
    serialize_to_dict,
)
from .event_system import (
    EventType,
    Event,
    EventEmitter,
    EventBus,
    get_event_bus,
    get_emitter,
)
from .version_utils import (
    get_version,
    parse_version,
    check_version_compatibility,
    get_version_info,
    format_version,
)
from .health_check import (
    HealthStatus,
    HealthCheckResult,
    HealthChecker,
    check_vllm_available,
    check_tensorrt_llm_available,
    check_polars_available,
    check_gpu_available,
    create_default_health_checker,
)
from .profiling_utils import (
    profile_context,
    profile_function,
    PerformanceProfiler,
    profile_decorator,
)
from .cache_utils import (
    MemoryCache,
    DiskCache,
    cached,
)
from .migration_utils import (
    Migration,
    MigrationManager,
    migrate_config,
)
from .plugin_system import (
    PluginInfo,
    PluginRegistry,
    BasePlugin,
    register_plugin,
    get_plugin,
    list_plugins,
)
from .observability_utils import (
    TraceLevel,
    TraceSpan,
    Tracer,
    MetricsExporter,
    get_tracer,
    get_metrics_exporter,
)
from .optimization_utils import (
    OptimizationResult,
    HyperparameterOptimizer,
    optimize_batch_size,
)
from .ci_cd_utils import (
    TestResult,
    CIRunner,
    run_ci_checks,
)
from .monitoring_utils import (
    AlertLevel,
    Alert,
    AlertManager,
    SystemMonitor,
    get_alert_manager,
    get_system_monitor,
)
from .code_analysis_utils import (
    CodeAnalyzer,
    analyze_codebase,
)
from .doc_utils import (
    DocGenerator,
    generate_documentation,
)
from .deployment_utils import (
    EnvironmentConfig,
    DeploymentManager,
    get_deployment_config,
)
from .security_utils import (
    hash_string,
    generate_token,
    validate_file_hash,
    sanitize_path,
    SecureConfig,
)
from .networking_utils import (
    HTTPMethod,
    APIResponse,
    APIClient,
    RateLimiter,
)
from .task_scheduler import (
    TaskStatus,
    Task,
    TaskScheduler,
)
from .backup_utils import (
    BackupInfo,
    BackupManager,
)
from .performance_tuning import (
    TuningResult,
    PerformanceTuner,
    auto_tune_performance,
)
from .schema_validation import (
    ValidationError,
    FieldSchema,
    SchemaValidator,
    validate_dataclass,
)
from .advanced_logging import (
    JSONFormatter,
    StructuredLogger,
    setup_logging,
)
from .integration_testing import (
    IntegrationTestResult,
    IntegrationTestRunner,
    create_integration_test_runner,
)
from .dependency_manager import (
    Dependency,
    DependencyManager,
    get_dependency_manager,
    register_dependency,
)
from .data_transformation import (
    Transformation,
    DataTransformer,
    create_transformer,
)
from .middleware import (
    Middleware,
    MiddlewareStack,
    middleware_decorator,
    create_middleware_stack,
)
from .metrics_advanced import (
    MetricValue,
    MetricStats,
    AdvancedMetricsCollector,
    create_metrics_collector,
)
from .batch_processing import (
    BatchResult,
    BatchProcessor,
    create_batch_processor,
)
from .retry_advanced import (
    RetryConfig,
    RetryHandler,
    retry,
    with_retry,
)
from .circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerOpenError,
    circuit_breaker,
)
from .worker_pool import (
    WorkerTask,
    WorkerPool,
    create_worker_pool,
)
from .rate_limiter_advanced import (
    RateLimitStrategy,
    RateLimitConfig,
    AdvancedRateLimiter,
    create_rate_limiter,
)
from .reporting_utils import (
    ReportSection,
    ReportGenerator,
    create_report,
)
from .notification_utils import (
    NotificationLevel,
    Notification,
    NotificationManager,
    get_notification_manager,
    notify,
)

__all__ = [
    # Base utilities
    "BaseOptimizationModel",
    "CudaResourceManager",
    "system_metrics_collector",
    # Validators
    "validate_not_none",
    "validate_not_empty",
    "validate_type",
    "validate_in_range",
    "validate_one_of",
    "validate_path_exists",
    "validate_callable",
    "validate_dict_keys",
    "validate_list_items",
    # Error handling
    "OptimizationCoreError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    "PerformanceError",
    "ErrorSeverity",
    "handle_error",
    "safe_execute",
    "retry_with_backoff",
    "error_context",
    # Configuration
    "load_config",
    "save_config",
    "merge_configs",
    "validate_config",
    "get_config_value",
    # Integration
    "ComponentRegistry",
    "Pipeline",
    "register_component",
    "get_component",
    "list_components",
    "create_pipeline",
    # Serialization
    "save_json",
    "load_json",
    "save_yaml",
    "load_yaml",
    "save_pickle",
    "load_pickle",
    "serialize_to_dict",
    # Events
    "EventType",
    "Event",
    "EventEmitter",
    "EventBus",
    "get_event_bus",
    "get_emitter",
    # Version
    "get_version",
    "parse_version",
    "check_version_compatibility",
    "get_version_info",
    "format_version",
    # Health Check
    "HealthStatus",
    "HealthCheckResult",
    "HealthChecker",
    "check_vllm_available",
    "check_tensorrt_llm_available",
    "check_polars_available",
    "check_gpu_available",
    "create_default_health_checker",
    # Profiling
    "profile_context",
    "profile_function",
    "PerformanceProfiler",
    "profile_decorator",
    # Cache
    "MemoryCache",
    "DiskCache",
    "cached",
    # Migration
    "Migration",
    "MigrationManager",
    "migrate_config",
    # Plugins
    "PluginInfo",
    "PluginRegistry",
    "BasePlugin",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    # Observability
    "TraceLevel",
    "TraceSpan",
    "Tracer",
    "MetricsExporter",
    "get_tracer",
    "get_metrics_exporter",
    # Optimization
    "OptimizationResult",
    "HyperparameterOptimizer",
    "optimize_batch_size",
    # CI/CD
    "TestResult",
    "CIRunner",
    "run_ci_checks",
    # Monitoring
    "AlertLevel",
    "Alert",
    "AlertManager",
    "SystemMonitor",
    "get_alert_manager",
    "get_system_monitor",
    # Code Analysis
    "CodeAnalyzer",
    "analyze_codebase",
    # Documentation
    "DocGenerator",
    "generate_documentation",
    # Deployment
    "EnvironmentConfig",
    "DeploymentManager",
    "get_deployment_config",
    # Security
    "hash_string",
    "generate_token",
    "validate_file_hash",
    "sanitize_path",
    "SecureConfig",
    # Networking
    "HTTPMethod",
    "APIResponse",
    "APIClient",
    "RateLimiter",
    # Task Scheduling
    "TaskStatus",
    "Task",
    "TaskScheduler",
    # Backup
    "BackupInfo",
    "BackupManager",
    # Performance Tuning
    "TuningResult",
    "PerformanceTuner",
    "auto_tune_performance",
    # Schema Validation
    "ValidationError",
    "FieldSchema",
    "SchemaValidator",
    "validate_dataclass",
    # Advanced Logging
    "JSONFormatter",
    "StructuredLogger",
    "setup_logging",
    # Integration Testing
    "IntegrationTestResult",
    "IntegrationTestRunner",
    "create_integration_test_runner",
    # Dependency Management
    "Dependency",
    "DependencyManager",
    "get_dependency_manager",
    "register_dependency",
    # Data Transformation
    "Transformation",
    "DataTransformer",
    "create_transformer",
    # Middleware
    "Middleware",
    "MiddlewareStack",
    "middleware_decorator",
    "create_middleware_stack",
    # Advanced Metrics
    "MetricValue",
    "MetricStats",
    "AdvancedMetricsCollector",
    "create_metrics_collector",
    # Batch Processing
    "BatchResult",
    "BatchProcessor",
    "create_batch_processor",
    # Advanced Retry
    "RetryConfig",
    "RetryHandler",
    "retry",
    "with_retry",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "circuit_breaker",
    # Worker Pool
    "WorkerTask",
    "WorkerPool",
    "create_worker_pool",
    # Advanced Rate Limiting
    "RateLimitStrategy",
    "RateLimitConfig",
    "AdvancedRateLimiter",
    "create_rate_limiter",
    # Reporting
    "ReportSection",
    "ReportGenerator",
    "create_report",
    # Notifications
    "NotificationLevel",
    "Notification",
    "NotificationManager",
    "get_notification_manager",
    "notify",
]

