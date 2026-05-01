# 🚀 Guía de Implementación Paso a Paso

## 📋 Resumen

Esta guía proporciona instrucciones detalladas para implementar los 7 gaps críticos identificados en `GAPS_CRITICOS_RECOMENDACIONES.md`.

---

## ✅ Estado de Implementación

### 1. ✅ vLLM (Python) - COMPLETADO
**Archivo:** `inference/vllm_engine.py`

**Instalación:**
```bash
pip install vllm>=0.2.0
```

**Uso:**
```python
from inference.vllm_engine import VLLMEngine

engine = VLLMEngine(
    model="mistralai/Mistral-7B-v0.1",
    tensor_parallel_size=2
)

outputs = engine.generate(
    prompts=["Hello, my name is"],
    max_tokens=64,
    temperature=0.7
)
```

**Integración:**
- Reemplazar `inference/inference_engine.py` con `vllm_engine.py`
- Actualizar `inference/api.py` para usar VLLMEngine

---

### 2. ✅ Polars (Python) - COMPLETADO
**Archivo:** `data/polars_processor.py`

**Instalación:**
```bash
pip install polars
```

**Uso:**
```python
from data.polars_processor import PolarsProcessor

processor = PolarsProcessor(lazy=True)

# Leer datos
df = processor.read_parquet("training_data.parquet")

# Procesar con lazy evaluation
result = processor.process_training_data(
    input_path="data.parquet",
    output_path="processed.parquet",
    min_tokens=1000
)
```

**Integración:**
- Reemplazar pandas en `data/dataset_manager.py`
- Usar PolarsProcessor para operaciones de DataFrame

---

### 3. ✅ TensorRT-LLM (Python) - COMPLETADO
**Archivo:** `inference/tensorrt_llm_engine.py`

**Instalación:**
```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

**Uso:**
```python
from inference.tensorrt_llm_engine import TensorRTLLMEngine

engine = TensorRTLLMEngine(
    model_path="model/",
    engine_path="engine.trt",
    precision="fp16"
)

outputs = engine.generate(
    prompts=["Hello"],
    max_new_tokens=64
)
```

**Nota:** Requiere CUDA 11.8+ y TensorRT instalado.

---

### 4. ✅ Apache Spark (Scala) - COMPLETADO
**Archivo:** `scala_core/src/main/scala/truthgpt/SparkDataPipeline.scala`

**Dependencias:** Ya incluidas en `scala_core/build.sbt`

**Uso:**
```scala
import truthgpt.SparkDataPipeline

val pipeline = SparkDataPipeline.create(
  SparkPipelineConfig(
    appName = "TruthGPT Pipeline",
    master = "local[*]"  // O "spark://cluster:7077"
  )
)

val processed = pipeline.processTrainingData(
  inputPath = "hdfs://training-data/",
  outputPath = "hdfs://processed-data/",
  minTokens = 1000
)
```

**Compilación:**
```bash
cd scala_core
sbt compile
sbt package
```

---

### 5. ✅ JuMP.jl + Flux.jl (Julia) - COMPLETADO
**Archivos:**
- `julia_core/src/jump_optimization.jl`
- `julia_core/src/flux_ml.jl`

**Instalación:**
```julia
# En julia_core/
using Pkg
Pkg.add("JuMP")
Pkg.add("HiGHS")  # Solver open source
Pkg.add("Flux")
Pkg.add("CUDA")  # Para GPU
```

**Uso JuMP.jl:**
```julia
using TruthGPT.JumpOptimization

x, obj = optimize_linear(c, A, b, lb, ub)
```

**Uso Flux.jl:**
```julia
using TruthGPT.FluxML

model = create_model(768, [256, 128], 10)
trained = train_model(model, (x, y), epochs=10)
```

**Actualizar Project.toml:**
```toml
[deps]
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
```

---

### 6. ✅ ggml (C++) - COMPLETADO (Estructura)
**Archivos:**
- `cpp_core/include/inference/ggml_engine.hpp`
- `cpp_core/src/inference/ggml_engine.cpp`

**Instalación:**
```bash
cd cpp_core
git submodule add https://github.com/ggerganov/ggml.git third_party/ggml
```

**Próximos pasos:**
1. Integrar ggml como submodule
2. Actualizar `CMakeLists.txt` para incluir ggml
3. Implementar funciones de carga y generación
4. Crear bindings Python con PyBind11

**Uso (después de completar):**
```cpp
#include "inference/ggml_engine.hpp"

truthgpt::inference::GGMLEngine engine;
engine.load_model("model.ggml");
auto output = engine.generate(input_ids, max_tokens, temperature);
```

---

### 7. ✅ libcuckoo (C++) - COMPLETADO (Estructura)
**Archivos:**
- `cpp_core/include/memory/libcuckoo_cache.hpp`
- `cpp_core/src/memory/libcuckoo_cache.cpp`

**Instalación:**
```bash
cd cpp_core
git submodule add https://github.com/efficient/libcuckoo.git third_party/libcuckoo
```

**Próximos pasos:**
1. Integrar libcuckoo como submodule
2. Actualizar `CMakeLists.txt` para incluir libcuckoo
3. Reemplazar placeholder `std::unordered_map` con `cuckoohash_map`
4. Crear bindings Python con PyBind11

**Uso (después de completar):**
```cpp
#include "memory/libcuckoo_cache.hpp"

truthgpt::memory::LibCuckooCache cache(1024);
cache.put(key, keys, values);
bool found = cache.get(key, keys, values);
```

---

## 📦 Actualización de Dependencias

### requirements_lock.txt
Agregar:
```
vllm>=0.2.0
polars>=0.36.0
tensorrt-llm>=0.5.0  # Si disponible
```

### scala_core/build.sbt
Ya incluye Spark (marcado como "provided" - cambiar a "compile" si necesario):
```scala
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.0"
```

### julia_core/Project.toml
Agregar:
```toml
[deps]
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
```

---

## 🔧 Pasos de Integración

### Paso 1: Instalar Dependencias Python
```bash
pip install vllm>=0.2.0 polars>=0.36.0
```

### Paso 2: Actualizar inference_engine.py
```python
# En inference/inference_engine.py
from inference.vllm_engine import VLLMEngine

# Reemplazar InferenceEngine con VLLMEngine cuando esté disponible
```

### Paso 3: Actualizar dataset_manager.py
```python
# En data/dataset_manager.py
from data.polars_processor import PolarsProcessor

# Usar PolarsProcessor en lugar de pandas
```

### Paso 4: Compilar Scala
```bash
cd scala_core
sbt compile
```

### Paso 5: Configurar Julia
```bash
cd julia_core
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Paso 6: Integrar C++ (ggml y libcuckoo)
```bash
cd cpp_core
git submodule add https://github.com/ggerganov/ggml.git third_party/ggml
git submodule add https://github.com/efficient/libcuckoo.git third_party/libcuckoo
# Actualizar CMakeLists.txt
```

---

## 📊 Testing

### Test vLLM
```python
python -m pytest tests/test_vllm.py
```

### Test Polars
```python
python -m pytest tests/test_polars.py
```

### Test Spark
```bash
cd scala_core
sbt test
```

### Test Julia
```julia
cd julia_core
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## 🎯 Próximos Pasos

1. **Completar integración de ggml:**
   - Integrar submodule
   - Implementar funciones de carga/generación
   - Crear bindings Python

2. **Completar integración de libcuckoo:**
   - Integrar submodule
   - Reemplazar placeholder
   - Crear bindings Python

3. **Crear tests:**
   - Tests unitarios para cada componente
   - Tests de integración
   - Benchmarks de rendimiento

4. **Documentación:**
   - Ejemplos de uso
   - Guías de optimización
   - Troubleshooting

---

*Última actualización: Noviembre 2025*












