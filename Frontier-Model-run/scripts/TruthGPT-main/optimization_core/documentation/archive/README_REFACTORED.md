# 🚀 Optimization Core - Guía Completa

## 📋 Resumen

`optimization_core` es un framework de alto rendimiento para inferencia de LLMs y procesamiento de datos, diseñado con una arquitectura polyglot que aprovecha las mejores herramientas de cada lenguaje.

---

## ✨ Características Principales

### 🎯 Inferencia de Alto Rendimiento
- **vLLM** - 5-10x más rápido que PyTorch
- **TensorRT-LLM** - 2-10x más rápido en GPUs NVIDIA
- **PagedAttention** - Reducción de memoria 3-5x
- **Continuous Batching** - Utilización óptima de GPU

### 📊 Procesamiento de Datos
- **Polars** - 10-100x más rápido que pandas
- **Lazy Evaluation** - Optimización automática de queries
- **Multi-threaded** - Procesamiento paralelo nativo
- **Streaming** - Soporte para datasets grandes

### 🔧 Utilidades Compartidas
- Validación robusta
- Manejo de errores centralizado
- Sistema de eventos
- Benchmarks estandarizados
- Testing completo

---

## 🏗️ Arquitectura

```
optimization_core/
├── inference/          # Motores de inferencia
│   ├── vllm_engine.py
│   ├── tensorrt_llm_engine.py
│   ├── base_engine.py
│   ├── engine_factory.py
│   └── utils/          # Utilidades de inferencia
├── data/               # Procesamiento de datos
│   ├── polars_processor.py
│   ├── processor_factory.py
│   └── utils/          # Utilidades de datos
├── benchmarks/         # Benchmarks y métricas
│   ├── benchmark_runner.py
│   └── performance_metrics.py
├── utils/              # Utilidades globales
│   ├── shared_validators.py
│   ├── error_handling.py
│   ├── config_utils.py
│   ├── integration_utils.py
│   ├── serialization_utils.py
│   └── event_system.py
├── tests/              # Testing
│   ├── utils/          # Utilidades de testing
│   └── base_test_case.py
└── examples/           # Ejemplos de uso
    ├── inference_examples.py
    ├── data_examples.py
    └── benchmark_examples.py
```

---

## 🚀 Inicio Rápido

### Instalación

```bash
# Instalar dependencias
pip install -r requirements_lock.txt

# Para vLLM
pip install vllm>=0.2.0

# Para Polars
pip install polars

# Para TensorRT-LLM (NVIDIA GPUs)
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

### Uso Básico

#### Inferencia con vLLM

```python
from inference.vllm_engine import VLLMEngine

# Crear engine
engine = VLLMEngine(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    tensor_parallel_size=1,
    dtype="float16"
)

# Generar texto
result = engine.generate(
    "What is machine learning?",
    max_tokens=100,
    temperature=0.7
)

print(result)
```

#### Procesamiento de Datos con Polars

```python
from data.polars_processor import PolarsProcessor

# Crear processor
processor = PolarsProcessor(lazy=True)

# Leer datos
df = processor.read_parquet("data.parquet")

# Procesar
result = processor.process_training_data(
    input_path="input.parquet",
    output_path="output.parquet",
    min_tokens=1000
)
```

#### Usando Factory (Auto-selección)

```python
from inference.engine_factory import create_inference_engine, EngineType

# Auto-seleccionar mejor engine disponible
engine = create_inference_engine(
    model="mistral-7b",
    engine_type=EngineType.AUTO
)

result = engine.generate("Hello, world!")
```

---

## 📚 Módulos Principales

### Inference Engines

#### VLLMEngine
```python
from inference.vllm_engine import VLLMEngine

engine = VLLMEngine(
    model="mistral-7b",
    tensor_parallel_size=2,  # Multi-GPU
    gpu_memory_utilization=0.9,
    quantization="awq"  # Opcional
)
```

#### TensorRTLLMEngine
```python
from inference.tensorrt_llm_engine import TensorRTLLMEngine

engine = TensorRTLLMEngine(
    model_path="/path/to/model",
    precision="fp16",
    max_batch_size=8
)
```

### Data Processors

#### PolarsProcessor
```python
from data.polars_processor import PolarsProcessor

processor = PolarsProcessor(lazy=True)

# Lazy evaluation - optimización automática
df = processor.read_parquet("data.parquet")
result = (
    df
    .filter(pl.col("tokens") > 1000)
    .group_by("category")
    .agg([pl.mean("loss"), pl.count()])
    .collect()  # Ejecutar query
)
```

### Utilidades

#### Validación
```python
from utils import validate_not_none, validate_in_range

validate_not_none(value, "value")
validate_in_range(temperature, "temperature", 0.0, 2.0)
```

#### Manejo de Errores
```python
from utils import handle_error, error_context

with error_context("model_loading", model_name="mistral-7b"):
    model = load_model("mistral-7b")
```

#### Eventos
```python
from utils import get_emitter, EventType

emitter = get_emitter("my_engine")
emitter.on(EventType.GENERATION_COMPLETED, lambda e: print("Done!"))
emitter.emit(EventType.GENERATION_COMPLETED, data={"result": "..."})
```

#### Benchmarks
```python
from benchmarks import run_benchmark

result = run_benchmark(
    "my_benchmark",
    engine.generate,
    "test prompt",
    num_runs=10
)

print(f"Duration: {result.duration:.3f}s")
print(f"Throughput: {result.throughput:.2f} ops/s")
```

---

## 🧪 Testing

### Usando Base Test Case

```python
from tests.base_test_case import BaseOptimizationCoreTestCase

class TestMyEngine(BaseOptimizationCoreTestCase):
    def test_generation(self):
        engine = self.create_mock_engine()
        self.assert_engine_works(engine)
```

### Usando Fixtures

```python
from tests.utils import MockInferenceEngine, TestDataGenerator

def test_engine():
    engine = MockInferenceEngine()
    prompts = TestDataGenerator.generate_prompts(10)
    results = engine.generate(prompts)
    assert len(results) == 10
```

---

## 📊 Benchmarks

### Ejecutar Benchmark

```python
from benchmarks import BenchmarkRunner

runner = BenchmarkRunner(warmup_runs=3, num_runs=10)

result = runner.run(
    "vllm_generation",
    engine.generate,
    "test prompt"
)
```

### Comparar Múltiples Engines

```python
from benchmarks import run_benchmark, compare_benchmarks

results = [
    run_benchmark("vllm", vllm_engine.generate, "prompt"),
    run_benchmark("tensorrt", tensorrt_engine.generate, "prompt"),
]

comparison = compare_benchmarks(results)
print(f"Best: {comparison['best']}")
```

---

## 🔧 Configuración

### Cargar Configuración

```python
from utils import load_config, validate_config

config = load_config("config.yaml")
validate_config(config, required_keys=["model", "batch_size"])
```

### Guardar Configuración

```python
from utils import save_config

config = {
    "model": {"name": "mistral-7b"},
    "inference": {"max_tokens": 100}
}

save_config(config, "config.yaml")
```

---

## 📖 Ejemplos

Ver módulos en `examples/`:
- `inference_examples.py` - Ejemplos de inferencia
- `data_examples.py` - Ejemplos de procesamiento
- `benchmark_examples.py` - Ejemplos de benchmarks

---

## 🎯 Mejores Prácticas

### 1. Usar Factory para Auto-selección
```python
# ✅ Bueno
engine = create_inference_engine(model="mistral-7b", engine_type=EngineType.AUTO)

# ❌ Evitar
if vllm_available:
    engine = VLLMEngine(...)
elif tensorrt_available:
    engine = TensorRTLLMEngine(...)
```

### 2. Usar Validadores Compartidos
```python
# ✅ Bueno
from utils import validate_generation_params
validate_generation_params(max_tokens=100, temperature=0.7, top_p=0.95)

# ❌ Evitar
if max_tokens < 1:
    raise ValueError("...")
```

### 3. Usar Utilidades de Prompts
```python
# ✅ Bueno
from inference.utils import normalize_prompts, handle_single_prompt
prompts_list, was_single = normalize_prompts(prompts)
# ... procesamiento
return handle_single_prompt(results, was_single)

# ❌ Evitar
if isinstance(prompts, str):
    prompts = [prompts]
```

### 4. Usar Manejo de Errores Centralizado
```python
# ✅ Bueno
from utils import handle_error, error_context

with error_context("operation"):
    result = risky_operation()

# ❌ Evitar
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Error: {e}")
    raise
```

---

## 📈 Rendimiento

### Mejoras Esperadas

| Componente | Mejora |
|------------|--------|
| vLLM vs PyTorch | 5-10x |
| Polars vs pandas | 10-100x |
| TensorRT-LLM vs PyTorch | 2-10x |
| Reducción de memoria (PagedAttention) | 3-5x |

---

## 🔍 Troubleshooting

### vLLM no disponible
```bash
pip install vllm>=0.2.0
```

### TensorRT-LLM no disponible
```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

### Polars no disponible
```bash
pip install polars
```

### Errores de GPU
- Verificar que CUDA esté instalado
- Verificar que el driver de NVIDIA esté actualizado
- Reducir `gpu_memory_utilization` si hay OOM

---

## 📚 Documentación Adicional

- `REFACTORING_COMPLETE.md` - Resumen completo de refactorización
- `REFACTORING_PHASE2.md` - Utilidades compartidas
- `REFACTORING_PHASE3.md` - Decoradores y métricas
- `REFACTORING_PHASE4.md` - Utilidades globales
- `REFACTORING_PHASE5.md` - Utilidades de testing
- `REFACTORING_PHASE6.md` - Benchmarks e integración
- `REFACTORING_PHASE7.md` - Serialización y eventos

---

## 🤝 Contribuir

1. Usar validadores compartidos
2. Seguir patrones establecidos
3. Agregar tests para nuevas features
4. Documentar cambios

---

## 📝 Licencia

[Tu licencia aquí]

---

*Última actualización: Noviembre 2025*












