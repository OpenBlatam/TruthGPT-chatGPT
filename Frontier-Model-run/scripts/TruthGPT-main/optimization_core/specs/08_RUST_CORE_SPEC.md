# 🦀 Especificación de Rust Core - Optimization Core

## 📋 Resumen

Este documento especifica la implementación del backend Rust (`truthgpt_rust`), que proporciona componentes de alto rendimiento con seguridad de memoria estricta y delegación de concurrencia *lock-free*. Diseñado integralmente para trabajar en conjunto con Python bajo puentes de latencia cero (Zero-Copy) usando PyO3.

## 🎯 Objetivos

1. **Alto Rendimiento**: Operaciones 10-50x más rápidas que Python.
2. **Seguridad de Memoria FFI**: Zero-copy (vía `PyBuffer` y vistas de memoria pre-asignadas) y memory safety garantizado en las barreras C/Rust.
3. **Concurrencia Async y GIL**: Desbloqueo explícito del Global Interpreter Lock (GIL) para cargas de trabajo CPU-bound (>1ms).
4. **SIMD**: Optimizaciones SIMD para operaciones numéricas de tokenización y descompresión.
5. **Telemetría Transparente**: Exposición de latencias internas al sistema de Observabilidad Base de Python sin costo adicional.

## 🏗️ Arquitectura

### Estructura del Proyecto

```
rust_core/
├── Cargo.toml              # Dependencias Rust (PyO3, DashMap, Rayon)
├── pyproject.toml          # Configuración Maturin para generación Wheels Python
├── src/
│   ├── lib.rs              # Punto de entrada de extensión y registro PyModule
│   ├── kv_cache.rs         # KV Cache lock-free con liberador GIL
│   ├── compression.rs      # Compresión LZ4/Zstd con zero-copy PyBytes
│   ├── tokenization.rs     # Tokenización rápida (HuggingFace tokenizers)
│   ├── data_loader.rs      # Carga paralela asíncrona de datos
│   ├── attention.rs        # Kernels de atención CPU
│   └── errors.rs           # Mapeo unificado PyErr <-> Rust Error
├── benches/                # Benchmarks (Criterion.rs)
└── tests/                  # Tests unitarios nativos (cargo test)
```

## 📦 Componentes

### Mapeo de Errores Base

**Propósito**: Garantizar que los errores de Rust se traduzcan limpiamente a las excepciones del núcleo estructurado (ej. `PolyglotError`).

```rust
// errors.rs
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};

pyo3::create_exception!(truthgpt_rust, RustCoreError, pyo3::exceptions::PyException);
pyo3::create_exception!(truthgpt_rust, MemoryAllocationError, RustCoreError);
```

### KV Cache (MemoryView y Release GIL)

**Propósito**: Cache lock-free concurrente para atención, permitiendo `aput` asíncrono desde el orquestador sin congelar la red.

**Especificación**:

```rust
use pyo3::prelude::*;
use pyo3::buffer::PyBuffer;
use std::sync::{Arc, RwLock};
use dashmap::DashMap;

#[pyclass]
pub struct PyKVCache {
    cache: Arc<DashMap<(usize, usize), Vec<u8>>>,
    max_size: usize,
    stats: Arc<RwLock<CacheStats>>,
}

#[pymethods]
impl PyKVCache {
    #[new]
    fn new(max_size: usize) -> Self {
        PyKVCache {
            cache: Arc::new(DashMap::new()),
            max_size,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
    
    // Usamos py.allow_threads para liberar el GIL si la inserción es masiva.
    // Aceptamos cualquier objeto que exponga el buffer protocol (memoryview)
    fn put(&self, py: Python, layer_idx: usize, position: usize, data_buf: PyBuffer<u8>) -> PyResult<()> {
        let key = (layer_idx, position);
        let data = data_buf.to_vec(py)?;
        
        py.allow_threads(|| {
            // Check size limit (Concurrent)
            if self.cache.len() >= self.max_size {
                if let Some(oldest) = self.cache.iter().next() {
                    self.cache.remove(oldest.key());
                }
            }
            
            self.cache.insert(key, data);
            
            let mut stats = self.stats.write().unwrap();
            stats.puts += 1;
        });
        
        Ok(())
    }
    
    fn get(&self, py: Python, layer_idx: usize, position: usize) -> PyResult<Option<PyObject>> {
        let key = (layer_idx, position);
        
        let result = py.allow_threads(|| self.cache.get(&key).map(|entry| entry.value().clone()));
        
        let mut stats = self.stats.write().unwrap();
        if let Some(data) = result {
            stats.hits += 1;
            // Retorna PyBytes para Zero-Copy en lectura si es posible.
            use pyo3::types::PyBytes;
            let py_bytes = PyBytes::new(py, &data);
            Ok(Some(py_bytes.into()))
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }
    
    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let stats = self.stats.read().unwrap();
        
        let dict = PyDict::new(py);
        dict.set_item("size", self.cache.len())?;
        dict.set_item("max_size", self.max_size)?;
        dict.set_item("hits", stats.hits)?;
        dict.set_item("misses", stats.misses)?;
        
        let hit_rate = if stats.hits + stats.misses > 0 {
             stats.hits as f64 / (stats.hits + stats.misses) as f64
        } else { 0.0 };
        dict.set_item("hit_rate", hit_rate)?;
        
        Ok(dict.into())
    }
}

#[derive(Default)]
struct CacheStats {
    hits: usize, misses: usize, puts: usize,
}
```

### Compression (Zero-Copy Transfer)

**Propósito**: Compresión LZ4/Zstd liberando el GIL explícitamente y usando `PyBytes` contiguos.

**Especificación**:

```rust
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use lz4_flex::{compress, decompress};

#[pyclass]
pub struct PyCompressor {
    algorithm: String,
}

#[pymethods]
impl PyCompressor {
    #[new]
    fn new(algorithm: String) -> Self { PyCompressor { algorithm } }
    
    fn compress(&self, py: Python, data: &[u8]) -> PyResult<PyObject> {
        let compressed = py.allow_threads(|| compress(data));
        // Casts contiguous memory payload back to unmanaged Python runtime space
        Ok(PyBytes::new(py, &compressed).into())
    }
    
    fn decompress(&self, py: Python, compressed_data: &[u8], original_size_hint: usize) -> PyResult<PyObject> {
        let decompressed = py.allow_threads(|| decompress(compressed_data, original_size_hint))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("LZ4 failure: {:?}", e)))?;
            
        Ok(PyBytes::new(py, &decompressed).into())
    }
}
```

## 🔧 Build y Configuración

### Cargo.toml

Requiere dependencias preparadas para extensiones de `Python`, incluyendo características estáticas fuertes.

```toml
[package]
name = "truthgpt-rust"
version = "1.1.0"
edition = "2021"

[lib]
name = "truthgpt_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py310"] }
dashmap = "5.5"
lz4-flex = "0.11"
zstd = "0.13"
tokenizers = "0.15"
rayon = "1.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "kv_cache_bench"
harness = false
```

### pyproject.toml (Maturin FFI Toolchain)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "truthgpt-rust"
version = "1.1.0"
requires-python = ">=3.10"
description = "Optimization Core Rust Backend for TruthGPT"
```

### Instrucciones de Build (CI/CD Ready)

```bash
# Desarrollo nativo (Virtual Env en Python)
maturin develop --release --features abi3-py310

# Build universal Wheel
maturin build --release
```

## 📊 Performance Targets

- **Lock-free KV Cache**: > 50M ops/s usando `DashMap` (Sharded Hashing).
- **Compresión Lz4 (CPU Fast Paths)**: > 5 GB/s liberando el GIL, no interrumpe orquestación web (SSE).
- **Memory Overhead FFI**: Constante `O(1)` (punteros `PyBytes`/`PyBuffer` transferidos sin serialización intermedia).

## 🧪 Testing

### Test Binding a Async Python

Integrados en `Polyglot` a alto nivel, pero probables directamente como módulos crudos:

```python
import pytest
import asyncio
import truthgpt_rust

def test_rust_gil_release():
    compressor = truthgpt_rust.PyCompressor("lz4")
    data = b"X" * 1024 * 1024 # 1MB test string
    
    # Compress debería liberar el GIL y permitir al framework iterar el loop en paralelo 
    resultado = compressor.compress(data)
    assert len(resultado) < len(data)

def test_kv_cache_memoryview():
    cache = truthgpt_rust.PyKVCache(1024)
    buffer = memoryview(b"\x00\x01\x02") # Zero-copy pointer handle
    
    # buffer es procesado como PyBuffer u8 en backend
    cache.put(0, 0, buffer)
    res = cache.get(0, 0)
    assert res == b"\x00\x01\x02"
```

## 📝 Reglas Estrictas de Backend Compilado (v1.1.0)

1. **GIL Release Imperativo**: Cualquier función en Rust esperada a durar más de 1 ms *debe* envolver su lógica intensiva (LZ4, Tokenización) con la macro/clausura `py.allow_threads(|| {...})`.
2. **Uso Exclusivo de Tipos Compatibles con Buffer Protocol**: No usar `String` de Rust ni conversiones `b"str"` si se puede evitar al pasar tensores de red a memoria. Preferir recibir `PyBuffer<u8>` y responder `PyBytes`.
3. **Pánicos Mapeables (`std::panic::catch_unwind`)**:  Un pánico en Rust no manejado causa un C-Segfault, tirando el contenedor entero (Docker u orquestador UVicorn). Todos los `Result::Err` deben transformarse en `PyResult` antes del FFI boundary.

---

**Versión**: 1.1.0  
**Última actualización**: Marzo 2026
