"""
TruthGPT Polyglot Core
======================

Unified Python API for high-performance multi-language backends.

This module automatically selects the best backend for each operation:
- **Rust**: KV Cache, Compression, Tokenization
- **C++**: Flash Attention, CUDA Kernels, Inference
- **Go**: HTTP/gRPC API, Distributed Coordination, Messaging

Example:
    >>> from optimization_core.polyglot_core import KVCache, Attention, Compressor
    >>> 
    >>> # Auto-select best backend
    >>> cache = KVCache(max_size=8192)
    >>> attention = Attention(d_model=768, n_heads=12)
    >>> compressor = Compressor(algorithm="lz4")
    >>> 
    >>> # Force specific backend
    >>> cache = KVCache(max_size=8192, backend=Backend.RUST)
    
Modular Imports:
    >>> from optimization_core.polyglot_core.core import KVCache, Attention
    >>> from optimization_core.polyglot_core.monitoring import get_profiler
    >>> from optimization_core.polyglot_core.enterprise import get_security_manager
"""

# Core modules
from .backend import (
    Backend,
    BackendInfo,
    get_available_backends,
    get_best_backend,
    is_backend_available,
    get_backend_info,
    print_backend_status,
)

from .cache import (
    KVCache,
    KVCacheConfig,
    EvictionStrategy,
    CacheStats,
)

from .attention import (
    Attention,
    AttentionConfig,
    AttentionPattern,
    PositionEncoding,
    FlashAttention,
    SparseAttention,
)

from .compression import (
    Compressor,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionStats,
    compress,
    decompress,
)

from .inference import (
    InferenceEngine,
    InferenceConfig,
    GenerationConfig,
    GenerationResult,
    TokenSampler,
)

from .distributed import (
    DistributedClient,
    GoClient,
    ServiceEndpoint,
)

from .tokenization import (
    Tokenizer,
    TokenizerConfig,
)

from .quantization import (
    Quantizer,
    QuantizationConfig,
    QuantizationType,
    QuantizationStats,
    quantize_weights,
    dequantize_weights,
)

from .profiling import (
    Profiler,
    PerformanceMetrics,
    ResourceUsage,
    get_profiler,
    profile,
)

from .benchmarking import (
    Benchmark,
    BenchmarkResult,
    benchmark,
    compare_backends_quick,
)

from .metrics import (
    MetricsCollector,
    Metric,
    MetricSummary,
    get_metrics_collector,
    record_metric,
)

from .reporting import (
    ReportGenerator,
    PerformanceReport,
    ReportSection,
    generate_benchmark_report,
)

from .utils import (
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

from .integration import (
    UnifiedKVCache,
    UnifiedCompressor,
    check_polyglot_availability,
    print_polyglot_status,
    get_test_compatibility_info,
    PolyglotTestHelper,
    get_test_helper,
)

from .config import (
    PolyglotConfig,
    ConfigManager,
    Environment,
    get_config_manager,
    get_config,
    load_config,
    save_config,
)

from .logging import (
    PolyglotLogger,
    PolyglotFormatter,
    StructuredFormatter,
    get_logger,
    setup_logging,
)

from .validation import (
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

from .health import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthChecker,
    get_health_checker,
    check_health,
    print_health_status,
)

from .optimization import (
    OptimizationResult,
    AutoOptimizer,
    get_optimizer,
    optimize_backend,
)

from .decorators import (
    profile_operation,
    validate_inputs,
    handle_errors,
    retry_on_failure,
    cache_result,
    log_operation,
    measure_performance,
    ensure_backend,
)

from .events import (
    EventType,
    Event,
    EventEmitter,
    get_event_emitter,
    emit_event,
    on_event,
    off_event,
)

from .errors import (
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

from .context import (
    operation_context,
    backend_context,
    performance_context,
    resource_context,
)

from .serialization import (
    Serializer,
    serialize_cache_entry,
    deserialize_cache_entry,
)

from .testing import (
    PolyglotTestFixtures,
    TestHelpers,
    assert_tensor_equal,
    assert_tensor_shape,
)

from .batch import (
    BatchProcessor,
    batch,
    process_batches,
    pad_batch,
)

from .streaming import (
    StreamProcessor,
    TokenStreamer,
    stream_process,
    stream_tokens_async,
)

from .observability import (
    Tracer,
    TraceSpan,
    Observability,
    get_observability,
    trace,
)

from .rate_limiting import (
    RateLimit,
    RateLimiter,
    RateLimitExceeded,
    rate_limit,
)

from .cli import (
    create_cli_parser,
    main_cli,
)

from .plugins import (
    Plugin,
    PluginManager,
    get_plugin_manager,
    register_plugin,
    get_plugin,
)

from .version import (
    Version,
    get_version,
    get_version_info,
    check_compatibility,
)

from .migration import (
    Migration,
    MigrationManager,
    get_migration_manager,
    register_migration,
)

from .docs import (
    DocumentationGenerator,
    get_documentation_generator,
    generate_docs,
)

from .scheduler import (
    TaskStatus,
    Task,
    TaskScheduler,
    get_scheduler,
    schedule_task,
)

from .workflow import (
    StepStatus,
    WorkflowStep,
    Workflow,
    create_workflow,
)

from .feature_flags import (
    FlagType,
    FeatureFlag,
    FeatureFlagManager,
    get_feature_flag_manager,
    is_feature_enabled,
    register_feature_flag,
)

from .security import (
    SecurityManager,
    SecretsManager,
    get_security_manager,
    get_secrets_manager,
)

from .telemetry import (
    TelemetryEvent,
    TelemetryCollector,
    get_telemetry,
    track_event,
)

from .alerts import (
    AlertSeverity,
    Alert,
    AlertRule,
    AlertManager,
    get_alert_manager,
    create_alert,
)

from .analytics import (
    AnalyticsInsight,
    AnalyticsEngine,
    get_analytics,
    record_data_point,
)

from .backup import (
    BackupManager,
    get_backup_manager,
    create_backup,
)

from .performance_tuning import (
    TuningRecommendation,
    PerformanceTuner,
    get_performance_tuner,
    analyze_performance,
)

from .compliance import (
    ComplianceLevel,
    ComplianceCheck,
    AuditLog,
    ComplianceChecker,
    AuditLogger,
    get_compliance_checker,
    get_audit_logger,
    log_audit_event,
)

from .cost_optimization import (
    CostEntry,
    CostOptimization,
    CostTracker,
    CostOptimizer,
    get_cost_tracker,
    get_cost_optimizer,
    record_cost,
)

from .resource_management import (
    ResourceType,
    ResourceUsage,
    ResourceQuota,
    ResourceManager,
    get_resource_manager,
    allocate_resources,
)

from .api import (
    APIEndpoint,
    APIRouter,
    get_api_router,
    register_endpoint,
)

from .service_discovery import (
    ServiceStatus,
    ServiceInfo,
    ServiceRegistry,
    get_service_registry,
    register_service,
)

from .load_balancer import (
    LoadBalanceStrategy,
    BackendInstance,
    LoadBalancer,
    get_load_balancer,
    create_load_balancer,
)

from .factory import (
    ComponentType,
    FactoryConfig,
    ComponentFactory,
    get_factory,
    create_component,
)

from .builder import (
    CacheBuilder,
    AttentionBuilder,
    InferenceBuilder,
    cache_builder,
    attention_builder,
    inference_builder,
)

from .registry import (
    RegistryType,
    RegistryEntry,
    ComponentRegistry,
    get_registry,
    register_component,
    get_component,
)

# Also export modular subpackages for direct access
from . import core
from . import processing
from . import monitoring
from . import infrastructure
from . import utils
from . import management
from . import enterprise
from . import orchestration
from . import testing
from . import integration
from . import benchmarking
from . import optimization

__all__ = [
    # Backend
    "Backend",
    "BackendInfo",
    "get_available_backends",
    "get_best_backend",
    "is_backend_available",
    "get_backend_info",
    "print_backend_status",
    
    # Cache
    "KVCache",
    "KVCacheConfig",
    "EvictionStrategy",
    "CacheStats",
    
    # Attention
    "Attention",
    "AttentionConfig",
    "AttentionPattern",
    "PositionEncoding",
    "FlashAttention",
    "SparseAttention",
    
    # Compression
    "Compressor",
    "CompressionConfig",
    "CompressionAlgorithm",
    "CompressionStats",
    "compress",
    "decompress",
    
    # Inference
    "InferenceEngine",
    "InferenceConfig",
    "GenerationConfig",
    "GenerationResult",
    "TokenSampler",
    
    # Distributed
    "DistributedClient",
    "GoClient",
    "ServiceEndpoint",
    
    # Tokenization
    "Tokenizer",
    "TokenizerConfig",
    
    # Quantization
    "Quantizer",
    "QuantizationConfig",
    "QuantizationType",
    "QuantizationStats",
    "quantize_weights",
    "dequantize_weights",
    
    # Profiling
    "Profiler",
    "PerformanceMetrics",
    "ResourceUsage",
    "get_profiler",
    "profile",
    
    # Benchmarking
    "Benchmark",
    "BenchmarkResult",
    "benchmark",
    "compare_backends_quick",
    
    # Metrics
    "MetricsCollector",
    "Metric",
    "MetricSummary",
    "get_metrics_collector",
    "record_metric",
    
    # Reporting
    "ReportGenerator",
    "PerformanceReport",
    "ReportSection",
    "generate_benchmark_report",
    
    # Utils
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
    
    # Integration
    "UnifiedKVCache",
    "UnifiedCompressor",
    "check_polyglot_availability",
    "print_polyglot_status",
    "get_test_compatibility_info",
    "PolyglotTestHelper",
    "get_test_helper",
    
    # Config
    "PolyglotConfig",
    "ConfigManager",
    "Environment",
    "get_config_manager",
    "get_config",
    "load_config",
    "save_config",
    
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
    
    # Health
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "HealthChecker",
    "get_health_checker",
    "check_health",
    "print_health_status",
    
    # Optimization
    "OptimizationResult",
    "AutoOptimizer",
    "get_optimizer",
    "optimize_backend",
    
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
    
    # Serialization
    "Serializer",
    "serialize_cache_entry",
    "deserialize_cache_entry",
    
    # Testing
    "PolyglotTestFixtures",
    "TestHelpers",
    "assert_tensor_equal",
    "assert_tensor_shape",
    
    # Batch
    "BatchProcessor",
    "batch",
    "process_batches",
    "pad_batch",
    
    # Streaming
    "StreamProcessor",
    "TokenStreamer",
    "stream_process",
    "stream_tokens_async",
    
    # Observability
    "Tracer",
    "TraceSpan",
    "Observability",
    "get_observability",
    "trace",
    
    # Rate Limiting
    "RateLimit",
    "RateLimiter",
    "RateLimitExceeded",
    "rate_limit",
    
    # CLI
    "create_cli_parser",
    "main_cli",
    
    # Plugins
    "Plugin",
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "get_plugin",
    
    # Version
    "Version",
    "get_version",
    "get_version_info",
    "check_compatibility",
    
    # Migration
    "Migration",
    "MigrationManager",
    "get_migration_manager",
    "register_migration",
    
    # Documentation
    "DocumentationGenerator",
    "get_documentation_generator",
    "generate_docs",
    
    # Scheduler
    "TaskStatus",
    "Task",
    "TaskScheduler",
    "get_scheduler",
    "schedule_task",
    
    # Workflow
    "StepStatus",
    "WorkflowStep",
    "Workflow",
    "create_workflow",
    
    # Feature Flags
    "FlagType",
    "FeatureFlag",
    "FeatureFlagManager",
    "get_feature_flag_manager",
    "is_feature_enabled",
    "register_feature_flag",
    
    # Security
    "SecurityManager",
    "SecretsManager",
    "get_security_manager",
    "get_secrets_manager",
    
    # Telemetry
    "TelemetryEvent",
    "TelemetryCollector",
    "get_telemetry",
    "track_event",
    
    # Alerts
    "AlertSeverity",
    "Alert",
    "AlertRule",
    "AlertManager",
    "get_alert_manager",
    "create_alert",
    
    # Analytics
    "AnalyticsInsight",
    "AnalyticsEngine",
    "get_analytics",
    "record_data_point",
    
    # Backup
    "BackupManager",
    "get_backup_manager",
    "create_backup",
    
    # Performance Tuning
    "TuningRecommendation",
    "PerformanceTuner",
    "get_performance_tuner",
    "analyze_performance",
    
    # Compliance
    "ComplianceLevel",
    "ComplianceCheck",
    "AuditLog",
    "ComplianceChecker",
    "AuditLogger",
    "get_compliance_checker",
    "get_audit_logger",
    "log_audit_event",
    
    # Cost Optimization
    "CostEntry",
    "CostOptimization",
    "CostTracker",
    "CostOptimizer",
    "get_cost_tracker",
    "get_cost_optimizer",
    "record_cost",
    
    # Resource Management
    "ResourceType",
    "ResourceUsage",
    "ResourceQuota",
    "ResourceManager",
    "get_resource_manager",
    "allocate_resources",
    
    # API
    "APIEndpoint",
    "APIRouter",
    "get_api_router",
    "register_endpoint",
    
    # Service Discovery
    "ServiceStatus",
    "ServiceInfo",
    "ServiceRegistry",
    "get_service_registry",
    "register_service",
    
    # Load Balancer
    "LoadBalanceStrategy",
    "BackendInstance",
    "LoadBalancer",
    "get_load_balancer",
    "create_load_balancer",
    
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
    
    # Modular subpackages
    "core",
    "processing",
    "monitoring",
    "infrastructure",
    "utils",
    "management",
    "enterprise",
    "orchestration",
    "testing",
    "integration",
    "benchmarking",
    "optimization",
]

__version__ = "2.0.0"

