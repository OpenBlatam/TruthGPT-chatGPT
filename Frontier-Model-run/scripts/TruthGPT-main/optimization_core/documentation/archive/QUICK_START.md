# ⚡ Quick Start Guide

## 🚀 Inicio Rápido en 5 Minutos

### 1. Instalación

```bash
# Dependencias básicas
pip install vllm polars

# Opcional: TensorRT-LLM (solo NVIDIA GPUs)
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

### 2. Inferencia Simple

```python
from inference.engine_factory import create_inference_engine, EngineType

# Crear engine (auto-selecciona el mejor disponible)
engine = create_inference_engine(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    engine_type=EngineType.AUTO
)

# Generar texto
result = engine.generate(
    "What is machine learning?",
    max_tokens=100,
    temperature=0.7
)

print(result)
```

### 3. Procesamiento de Datos

```python
from data.processor_factory import create_data_processor, ProcessorType

# Crear processor
processor = create_data_processor(processor_type=ProcessorType.AUTO)

# Leer y procesar
df = processor.read_parquet("data.parquet")
result = processor.process_training_data(
    input_path="input.parquet",
    output_path="output.parquet"
)
```

### 4. Health Check

```python
from utils import create_default_health_checker

# Verificar estado del sistema
checker = create_default_health_checker()
results = checker.check()

for component, result in results.items():
    print(f"{component}: {result.status.value}")
```

### 5. Benchmark

```python
from benchmarks import run_benchmark

# Ejecutar benchmark
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

## 📚 Siguiente Paso

Ver `README_REFACTORED.md` para documentación completa.

---

*Última actualización: Noviembre 2025*












