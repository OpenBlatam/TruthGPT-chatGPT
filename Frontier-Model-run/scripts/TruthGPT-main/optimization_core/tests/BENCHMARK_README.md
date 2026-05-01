# 🚀 Benchmark Polyglot vs Closed Source Models

Este directorio contiene tests completos para comparar el rendimiento del modelo Polyglot con modelos closed source (GPT-4, Claude, etc.).

## 📋 Descripción

Los tests validan:
1. ✅ **Todos los módulos del polyglot** (Rust, C++, Go, Python)
2. ✅ **Velocidad de inferencia** comparada con modelos closed source
3. ✅ **Funcionalidad de cada módulo** (KV Cache, Compression, Attention, Inference)
4. ✅ **Reportes detallados** de rendimiento

## 🛠️ Instalación

### Dependencias Básicas

```bash
# Instalar dependencias del proyecto
pip install -r requirements.txt

# Para benchmarks de closed source (opcional)
pip install openai anthropic
```

### Configurar API Keys (Opcional)

Para comparar con modelos closed source, configura las API keys:

```bash
# OpenAI (GPT-4, GPT-3.5)
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."
```

O crea un archivo `.env`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## 🏃 Uso

### Ejecutar Todos los Benchmarks

```bash
# Con pytest
pytest tests/test_polyglot_benchmark_vs_closed_source.py -v

# Con script directo
python scripts/run_polyglot_benchmarks.py --full

# Solo benchmarks polyglot (sin closed source)
python scripts/run_polyglot_benchmarks.py --polyglot-only
```

### Ejecutar Módulos Específicos

```bash
# Solo KV Cache y Compression
python scripts/run_polyglot_benchmarks.py --modules kv_cache compression

# Solo Inference
python scripts/run_polyglot_benchmarks.py --modules inference --model gpt2
```

### Opciones Avanzadas

```bash
# Especificar modelo para inference
python scripts/run_polyglot_benchmarks.py --full --model microsoft/DialoGPT-small

# Especificar directorio de salida
python scripts/run_polyglot_benchmarks.py --full --output ./my_reports

# Modo verbose
python scripts/run_polyglot_benchmarks.py --full --verbose
```

## 📊 Módulos Probados

### 1. KV Cache
- **Backends**: Rust, C++, Go, Python
- **Métricas**: Latencia PUT/GET, Throughput
- **Test**: Operaciones de cache concurrentes

### 2. Compression
- **Backends**: Rust, C++, Go, Python
- **Algoritmos**: LZ4, Zstd
- **Métricas**: Velocidad de compresión/descompresión, Ratio

### 3. Attention
- **Backends**: C++, Rust, Python
- **Métricas**: Latencia forward pass, Throughput (tokens/s)
- **Configuración**: Batch size, Sequence length, d_model

### 4. Inference
- **Engines**: vLLM, TensorRT-LLM
- **Métricas**: Latencia de generación, Tokens por segundo
- **Comparación**: Con GPT-4, Claude, etc.

## 📈 Reportes

Los reportes se guardan en `benchmark_reports/` con formato JSON:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "polyglot_results": [
    {
      "name": "kv_cache_put",
      "module": "kv_cache",
      "backend": "rust",
      "latency_ms": 0.123,
      "throughput_tokens_per_sec": 8123.45,
      "success": true
    }
  ],
  "closed_source_results": [
    {
      "model_name": "gpt-4",
      "latency_ms": 1250.5,
      "tokens_generated": 50,
      "throughput_tokens_per_sec": 40.0
    }
  ],
  "summary": {
    "comparison": {
      "latency_speedup": 10.17,
      "throughput_speedup": 203.0
    }
  }
}
```

## 🔍 Interpretación de Resultados

### Latency Speedup
- **> 1.0**: Polyglot es más rápido
- **< 1.0**: Closed source es más rápido

### Throughput Speedup
- **> 1.0**: Polyglot genera más tokens por segundo
- **< 1.0**: Closed source genera más tokens por segundo

### Ejemplo de Salida

```
REPORTE DE BENCHMARKS - POLYGLOT vs CLOSED SOURCE
================================================================================
Timestamp: 2025-01-15T10:30:00

Estado de Módulos:
  polyglot_core: ✓ Disponible
  rust: ✓ Disponible
  cpp: ✓ Disponible
  go: ✗ No disponible
  python: ✓ Disponible

Resultados Polyglot (12 tests):
  Exitosos: 10/12

  KV_CACHE:
    kv_cache_put (rust): 0.12ms, 8123 tokens/s
    kv_cache_get (rust): 0.08ms, 12500 tokens/s

  COMPRESSION:
    compression_compress (rust): 2.5ms, 45.2 MB/s
    compression_decompress (rust): 1.2ms, 94.1 MB/s

  INFERENCE:
    inference_vllm (vllm): 45.2ms, 1106 tokens/s

Resultados Closed Source (2 tests):
  Exitosos: 2/2
    gpt-4: 1250.5ms, 40.0 tokens/s
    gpt-3.5-turbo: 450.2ms, 111.1 tokens/s

Comparación:
  Latency: Polyglot es 27.67x más rápido
  Throughput: Polyglot es 27.65x más rápido
```

## 🐛 Troubleshooting

### Error: "Module not available"

Algunos módulos pueden no estar disponibles si:
- No están compilados (Rust, C++)
- No están instaladas las dependencias
- No hay GPU disponible (para engines GPU)

**Solución**: Los tests continuarán con los módulos disponibles.

### Error: "API key not found"

Para benchmarks de closed source, necesitas API keys.

**Solución**: 
- Configura las variables de entorno
- O ejecuta solo benchmarks polyglot: `--polyglot-only`

### Error: "Model not found"

Para inference, necesitas un modelo disponible.

**Solución**:
- Usa un modelo pequeño como `gpt2`
- O especifica la ruta: `--model /path/to/model`

## 📝 Tests Individuales

### Test de Disponibilidad de Módulos

```python
def test_all_modules_available():
    """Verifica que los módulos principales estén disponibles."""
    # ...
```

### Test de Benchmarks Polyglot

```python
def test_polyglot_benchmarks():
    """Ejecuta todos los benchmarks polyglot."""
    # ...
```

### Test de Benchmarks Closed Source

```python
def test_closed_source_benchmarks():
    """Ejecuta benchmarks de modelos closed source."""
    # ...
```

### Test Completo

```python
def test_full_benchmark_suite():
    """Ejecuta suite completa y genera reporte."""
    # ...
```

## 🔧 Configuración Avanzada

### Personalizar Prompts de Test

Edita `test_prompts` en `PolyglotBenchmarker` y `ClosedSourceBenchmarker`:

```python
self.test_prompts = [
    "Tu prompt personalizado aquí",
    # ...
]
```

### Ajustar Iteraciones

Modifica las iteraciones en cada método de benchmark:

```python
iterations = 100  # Aumentar para más precisión
```

### Cambiar Configuración de Modelos

Para inference, ajusta parámetros en `benchmark_inference()`:

```python
result = engine.generate(
    prompts=[prompt],
    max_new_tokens=100,  # Aumentar para tests más largos
    temperature=0.7
)
```

## 📚 Referencias

- [POLYGLOT_ARCHITECTURE.md](../POLYGLOT_ARCHITECTURE.md) - Arquitectura del sistema
- [IMPLEMENTATION_STATUS_AND_IMPROVEMENTS.md](../IMPLEMENTATION_STATUS_AND_IMPROVEMENTS.md) - Estado de implementación
- [benchmarks/polyglot_benchmarks.py](../benchmarks/polyglot_benchmarks.py) - Benchmarks internos

## 🤝 Contribuir

Para agregar nuevos benchmarks:

1. Agrega método en `PolyglotBenchmarker` o `ClosedSourceBenchmarker`
2. Llama al método en `run_all_benchmarks()`
3. Actualiza este README

## 📄 Licencia

Véase el archivo LICENSE del proyecto principal.












