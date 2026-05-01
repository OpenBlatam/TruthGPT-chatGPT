# 🔌 Especificación de Interfaces Core - Optimization Core

## 📋 Resumen

Este documento especifica todas las interfaces y contratos base del sistema `optimization_core`. Estas interfaces definen los contratos asíncronos y arquitectónicos que deben cumplir de manera estricta todos los componentes de la suite para garantizar concurrencia no bloqueante y alta cohesión.

## 🎯 Principios de Diseño

1. **Interfaces Primero y Asíncronas**: Todas las funcionalidades se definen primero como interfaces `async-first`.
2. **Contratos Claros (Zero-Copy)**: Cada interfaz especifica explícitamente el uso de visualizaciones de memoria transitorias (MemoryViews) en fronteras intensivas.
3. **Extensibilidad Vía Factory**: El registro de implementaciones ocurre siempre a través de un patrón *Registry*. No instanciar clases concretas directamente.
4. **Resiliencia Pormenorizada**: Cada interfaz promueve delegación y recolección de métricas.

## 📦 Interfaces Base

### IComponent

**Propósito**: Interfaz general para todos los bloques de construcción (Motores, Caches, Procesadores).

**Especificación**:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional, List, Union, AsyncGenerator
from pathlib import Path

class IComponent(ABC):
    """Base asynchronous interface for all lifecycle components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the registry name (e.g., 'PolarsProcessor')."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Returns semantic version string."""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> 'IComponent':
        """
        Synchronous foundational setup (e.g., config parsing).
        Returns self for chaining: engine.initialize().load_model()
        """
        pass
        
    @abstractmethod
    async def ainitialize(self, **kwargs) -> 'IComponent':
         """
         Asynchronous network or heavy setup initialization.
         """
         pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Idempotent synchronous cleanup of resources (file handles, lockfiles).
        """
        pass
        
    @abstractmethod
    async def acleanup(self) -> None:
         """
         Asynchronous cleanup (closing aiohttp sessions, asyncio queues).
         """
         pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get standard observability payload. Must never block.
        Returns:
            {
                "name": str, "version": str, "health": "healthy|degraded",
                "metrics": Dict, "last_error": Optional[str]
            }
        """
        pass
```

### IInferenceEngine

**Propósito**: Interfaz estandarizada asíncrona para motores de texto (vLLM, TensorRT).

**Especificación**:

```python
class IInferenceEngine(IComponent):
    """Async interface for LLM operations."""
    
    @abstractmethod
    async def agenerate(
        self,
        prompts: Union[str, List[str]],
        config: Optional['GenerationConfig'] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Single-shot async text generation.
        """
        pass
        
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        config: Optional['GenerationConfig'] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Yields tokens asynchronously (SSE/WebSocket friendly).
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Hardware footprint and dimension data."""
        pass
    
    @abstractmethod
    def load_model(self, model: Union[str, Path], **kwargs) -> bool:
        """Loads weights into memory."""
        pass
    
    @property
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Boolean guard for initialization."""
        pass
```

### IDataProcessor

**Propósito**: Transformador matricial o de eventos (Polars/Pandas).

**Especificación**:

```python
class IDataProcessor(IComponent):
    """Interface for Lazy data evaluation and Streaming pipelines."""
    
    @abstractmethod
    def process(self, data: Any, operations: List[Dict[str, Any]], **kwargs) -> Any:
        """Applies a sequence of logical operations (AST-like build)."""
        pass
    
    @abstractmethod
    async def aread(self, path: Union[str, Path], format: Optional[str] = None, **kwargs) -> Any:
        """Asynchronous disk/network file ingest."""
        pass
    
    @abstractmethod
    async def awrite(self, data: Any, path: Union[str, Path], format: Optional[str] = None, **kwargs) -> bool:
        """Asynchronous disk/network flush."""
        pass
```

## 📊 Tipos de Datos (Value Objects)

### GenerationConfig

```python
from pydantic import BaseModel, Field

class GenerationConfig(BaseModel):
    """Pydantic-based configuration for strict validation at system ingress."""
    
    max_tokens: int = Field(default=100, ge=1, description="Maximum tokens to output.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
```

## 🏭 ComponentFactory (Registry)

**Propósito**: Mecanismo global para resolver componentes en tiempo de ejecución.

**Especificación**:

```python
class IComponentFactory(ABC):
    """Interface for the strict Component Registry."""
    
    @classmethod
    @abstractmethod
    def register(cls, name: str):
        """Decorator for appending a class to the registry."""
        pass
    
    @classmethod
    @abstractmethod
    def create(cls, name: str, **kwargs) -> IComponent:
        """
        Instantiates a registered component. 
        Raises ComponentNotFoundError if the string key is absent.
        """
        pass
```

## ✅ Validación y Errores

### Jerarquía de Excepciones Core

```python
class OptimizationCoreError(Exception):
    """Base top-level error."""
    pass

class ComponentLifecycleError(OptimizationCoreError):
    """Caught during initialize(), cleanup() or instantiation constraints."""
    pass

class MemoryConstraintError(OptimizationCoreError):
    """Raised when an operation would exceed the configured zero-copy buffer limits."""
    pass

class IOTimeoutError(OptimizationCoreError):
    """Raised on asynchronous aread/awrite blocking thresholds breached."""
    pass
```

## 🧪 Testing

### Principios de Mocking en Async

En la era asíncrona (`v1.1.0`), las interfaces se mochean usando `AsyncMock`.

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_engine_delegation():
    mock_engine = AsyncMock(spec=IInferenceEngine)
    mock_engine.agenerate.return_value = "Result"
    
    res = await mock_engine.agenerate("Hello")
    assert res == "Result"
```

## 📝 Convenciones Restrictivas v1.1.0

- **Type Annotations obligatorias**: Requerimos que todo código de Optimization Core pase validadores estáticos (`mypy --strict`). 
- **Pydantic**: En lugar de simples diccionarios o `@dataclass` crudas nativas para la entrada de parámetros sucios (`GenerationConfig/Inputs`), preferimos `Pydantic` o equivalente validación cruzada.
- **Asincronía Integral**: Los componentes FFI (Rust/CPP) que consuman la interfaz `IComponent` usarán `run_in_executor` silenciosamente para adaptar su código bloqueante y comportarse como corrutinas frente a Python.

---

**Versión**: 1.1.0  
**Última actualización**: Marzo 2026
