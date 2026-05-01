# 📚 Índice Completo de Utilidades - optimization_core

## 📋 Resumen

Este documento proporciona un índice completo de todas las utilidades disponibles en `optimization_core`.

---

## 🔧 Utilidades de Inferencia (`inference/utils/`)

### `validators.py`
- `validate_generation_params()` - Valida parámetros de generación
- `validate_positive_int()` - Valida enteros positivos
- `validate_float_range()` - Valida rangos de flotantes
- `validate_precision()` - Valida precisiones
- `validate_quantization()` - Valida tipos de cuantización

### `prompt_utils.py`
- `normalize_prompts()` - Normaliza prompts
- `handle_single_prompt()` - Maneja prompts únicos
- `extract_generated_text()` - Extrae texto generado

### `decorators.py`
- `@validate_input()` - Valida entrada
- `@handle_errors()` - Maneja errores
- `@log_execution()` - Registra ejecución
- `@retry_on_failure()` - Reintenta en fallo
- `@measure_performance()` - Mide rendimiento
- `@cache_result()` - Cachea resultados

### `logging_utils.py`
- `MetricsCollector` - Colector de métricas
- `setup_logging()` - Configura logging

---

## 📊 Utilidades de Datos (`data/utils/`)

### `validators.py`
- `validate_file_path()` - Valida paths de archivos
- `validate_dataframe_schema()` - Valida esquemas de DataFrames
- `validate_positive_number()` - Valida números positivos
- `validate_non_empty_string()` - Valida strings no vacíos

### `file_utils.py`
- `detect_file_format()` - Detecta formato de archivo
- `ensure_output_directory()` - Asegura directorio de salida

---

## 🌐 Utilidades Globales (`utils/`)

### `shared_validators.py`
- `validate_non_empty_string()` - Valida strings no vacíos
- `validate_positive_int()` - Valida enteros positivos
- `validate_float_range()` - Valida rangos de flotantes
- `validate_file_path()` - Valida paths
- `validate_directory_path()` - Valida paths de directorios
- `validate_url()` - Valida URLs
- `validate_email()` - Valida emails
- `validate_positive_number()` - Valida números positivos
- `validate_list_not_empty()` - Valida listas no vacías

### `error_handling.py`
- `OptimizationCoreError` - Error base
- `ValidationError` - Error de validación
- `ConfigurationError` - Error de configuración
- `ExecutionError` - Error de ejecución
- `@handle_errors()` - Decorador de manejo de errores
- `error_context()` - Context manager de errores

### `config_utils.py`
- `load_config()` - Carga configuración
- `save_config()` - Guarda configuración
- `merge_configs()` - Fusiona configuraciones
- `validate_config()` - Valida configuración
- `get_config_value()` - Obtiene valor de configuración

### `integration_utils.py`
- `ComponentRegistry` - Registro de componentes
- `Pipeline` - Pipeline de procesamiento

### `serialization_utils.py`
- `serialize_json()` - Serializa a JSON
- `deserialize_json()` - Deserializa desde JSON
- `serialize_yaml()` - Serializa a YAML
- `deserialize_yaml()` - Deserializa desde YAML
- `serialize_pickle()` - Serializa a Pickle
- `deserialize_pickle()` - Deserializa desde Pickle
- `compress_data()` - Comprime datos

### `event_system.py`
- `EventEmitter` - Emisor de eventos
- `EventBus` - Bus de eventos

### `version_utils.py`
- `get_version()` - Obtiene versión
- `compare_versions()` - Compara versiones
- `is_compatible()` - Verifica compatibilidad
- `parse_version()` - Parsea versión
- `format_version()` - Formatea versión

### `health_check.py`
- `HealthCheck` - Health check base
- `ComponentHealthCheck` - Health check de componente
- `SystemHealthCheck` - Health check de sistema
- `register_health_check()` - Registra health check
- `run_health_checks()` - Ejecuta health checks
- `get_health_status()` - Obtiene estado de salud
- `HealthStatus` - Estado de salud
- `check_dependencies()` - Verifica dependencias

### `profiling_utils.py`
- `profile_context()` - Context manager de profiling
- `Profiler` - Profiler
- `measure_time()` - Mide tiempo
- `get_memory_usage()` - Obtiene uso de memoria

### `cache_utils.py`
- `MemoryCache` - Caché en memoria
- `DiskCache` - Caché en disco
- `@cache_result()` - Decorador de caché

### `migration_utils.py`
- `Migration` - Migración base
- `MigrationManager` - Gestor de migraciones
- `run_migrations()` - Ejecuta migraciones

### `plugin_system.py`
- `BasePlugin` - Plugin base
- `PluginRegistry` - Registro de plugins
- `register_plugin()` - Registra plugin
- `load_plugin()` - Carga plugin

### `observability_utils.py`
- `Tracer` - Tracer distribuido
- `TraceSpan` - Span de traza
- `MetricsExporter` - Exportador de métricas

### `optimization_utils.py`
- `OptimizationResult` - Resultado de optimización
- `HyperparameterOptimizer` - Optimizador de hiperparámetros
- `optimize_batch_size()` - Optimiza tamaño de batch

### `ci_cd_utils.py`
- `TestResult` - Resultado de test
- `CIRunner` - Runner de CI
- `run_ci_checks()` - Ejecuta checks de CI

### `monitoring_utils.py`
- `AlertLevel` - Nivel de alerta
- `Alert` - Alerta
- `AlertManager` - Gestor de alertas
- `SystemMonitor` - Monitor de sistema
- `get_alert_manager()` - Obtiene gestor de alertas
- `get_system_monitor()` - Obtiene monitor de sistema

### `code_analysis_utils.py`
- `CodeAnalyzer` - Analizador de código
- `analyze_codebase()` - Analiza codebase

### `doc_utils.py`
- `DocGenerator` - Generador de documentación
- `generate_documentation()` - Genera documentación

### `deployment_utils.py`
- `EnvironmentConfig` - Configuración de entorno
- `DeploymentManager` - Gestor de deployment
- `get_deployment_config()` - Obtiene configuración de deployment

### `security_utils.py`
- `hash_string()` - Hashea string
- `generate_token()` - Genera token
- `validate_file_hash()` - Valida hash de archivo
- `sanitize_path()` - Sanitiza path
- `SecureConfig` - Configuración segura

### `networking_utils.py`
- `HTTPMethod` - Método HTTP
- `APIResponse` - Respuesta de API
- `APIClient` - Cliente de API
- `RateLimiter` - Limitador de tasa

### `task_scheduler.py`
- `TaskStatus` - Estado de tarea
- `Task` - Tarea
- `TaskScheduler` - Scheduler de tareas

### `backup_utils.py`
- `BackupInfo` - Información de backup
- `BackupManager` - Gestor de backups

### `performance_tuning.py`
- `TuningResult` - Resultado de tuning
- `PerformanceTuner` - Tuner de rendimiento
- `auto_tune_performance()` - Tunea rendimiento automáticamente

### `schema_validation.py`
- `ValidationError` - Error de validación
- `FieldSchema` - Esquema de campo
- `SchemaValidator` - Validador de esquemas
- `validate_dataclass()` - Valida dataclass

### `advanced_logging.py`
- `JSONFormatter` - Formateador JSON
- `StructuredLogger` - Logger estructurado
- `setup_logging()` - Configura logging

### `integration_testing.py`
- `IntegrationTestResult` - Resultado de test de integración
- `IntegrationTestRunner` - Runner de tests de integración
- `create_integration_test_runner()` - Crea runner de tests

### `dependency_manager.py`
- `Dependency` - Dependencia
- `DependencyManager` - Gestor de dependencias
- `get_dependency_manager()` - Obtiene gestor de dependencias
- `register_dependency()` - Registra dependencia

### `data_transformation.py`
- `Transformation` - Transformación
- `DataTransformer` - Transformer de datos
- `create_transformer()` - Crea transformer

### `middleware.py`
- `Middleware` - Middleware
- `MiddlewareStack` - Stack de middleware
- `middleware_decorator()` - Decorador de middleware
- `create_middleware_stack()` - Crea stack de middleware

### `metrics_advanced.py`
- `MetricValue` - Valor de métrica
- `MetricStats` - Estadísticas de métrica
- `AdvancedMetricsCollector` - Colector avanzado de métricas
- `create_metrics_collector()` - Crea colector de métricas

### `batch_processing.py`
- `BatchResult` - Resultado de batch
- `BatchProcessor` - Procesador de batches
- `create_batch_processor()` - Crea procesador de batches

### `retry_advanced.py`
- `RetryConfig` - Configuración de retry
- `RetryHandler` - Handler de retry
- `retry()` - Decorador de retry
- `with_retry()` - Wrapper de retry

### `circuit_breaker.py`
- `CircuitState` - Estado de circuit breaker
- `CircuitBreakerConfig` - Configuración de circuit breaker
- `CircuitBreaker` - Circuit breaker
- `CircuitBreakerOpenError` - Error de circuit breaker abierto
- `circuit_breaker()` - Decorador de circuit breaker

---

## 🧪 Utilidades de Testing (`tests/utils/`)

### `test_helpers.py`
- `create_mock_engine()` - Crea mock de engine
- `create_mock_processor()` - Crea mock de processor
- `create_test_data()` - Crea datos de test
- `assert_engine_works()` - Assert engine funciona
- `assert_processor_works()` - Assert processor funciona

### `test_fixtures.py`
- `TestConfig` - Configuración de test
- `MockInferenceEngine` - Mock de inference engine
- `MockDataProcessor` - Mock de data processor
- `TestDataGenerator` - Generador de datos de test

### `test_assertions.py`
- `assert_valid_generation()` - Assert generación válida
- `assert_valid_dataframe()` - Assert DataFrame válido
- `assert_performance_improved()` - Assert rendimiento mejorado

### `base_test_case.py`
- `BaseOptimizationCoreTestCase` - Clase base de tests

---

## 📈 Benchmarks (`benchmarks/`)

### `benchmark_runner.py`
- `BenchmarkResult` - Resultado de benchmark
- `BenchmarkRunner` - Runner de benchmarks

### `performance_metrics.py`
- `PerformanceMetrics` - Métricas de rendimiento
- `MetricsCollector` - Colector de métricas

---

## 📖 Ejemplos (`examples/`)

### `inference_examples.py`
- Ejemplos de uso de inference engines

### `data_examples.py`
- Ejemplos de procesamiento de datos

### `benchmark_examples.py`
- Ejemplos de benchmarking

### `advanced_examples.py`
- Ejemplos avanzados

---

## 🎯 Uso Rápido

```python
# Importar utilidades
from utils import (
    # Validación
    validate_non_empty_string,
    SchemaValidator,
    
    # Error handling
    handle_errors,
    
    # Configuración
    load_config,
    
    # Logging
    setup_logging,
    
    # Métricas
    create_metrics_collector,
    
    # Retry
    retry,
    
    # Circuit breaker
    circuit_breaker,
    
    # Batch processing
    create_batch_processor,
    
    # Y muchas más...
)
```

---

*Última actualización: Noviembre 2025*












