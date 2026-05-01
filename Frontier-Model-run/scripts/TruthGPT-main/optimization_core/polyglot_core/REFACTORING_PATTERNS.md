# 🏗️ Polyglot Core - Design Patterns Refactoring

## ✅ Patrones de Diseño Implementados

### 1. Factory Pattern (factory.py)
**Propósito**: Crear componentes con selección automática de backend.

**Características**:
- Selección automática de backend
- Configuración centralizada
- Fallback inteligente
- Modo estricto opcional

**Uso**:
```python
from optimization_core.polyglot_core import get_factory, FactoryConfig

factory = get_factory(FactoryConfig(preferred_backend=Backend.RUST))
cache = factory.create_cache(max_size=8192)
attention = factory.create_attention(d_model=768, n_heads=12)
```

### 2. Builder Pattern (builder.py)
**Propósito**: Construir configuraciones complejas de forma fluida.

**Características**:
- API fluida (method chaining)
- Configuraciones complejas
- Validación en build
- Metadata personalizada

**Uso**:
```python
from optimization_core.polyglot_core import cache_builder, attention_builder

cache = (cache_builder()
    .with_max_size(32768)
    .with_eviction_strategy(EvictionStrategy.ADAPTIVE)
    .with_compression(True)
    .with_quantization(True)
    .build())

attention = (attention_builder()
    .with_dimensions(d_model=1024, n_heads=16)
    .with_flash_attention(True)
    .with_dropout(0.1)
    .build())
```

### 3. Registry Pattern (registry.py)
**Propósito**: Registrar y descubrir componentes dinámicamente.

**Características**:
- Registro dinámico
- Múltiples tipos de componentes
- Priorización
- Metadata asociada

**Uso**:
```python
from optimization_core.polyglot_core import get_registry, RegistryType

registry = get_registry()
registry.register("my_cache", cache_instance, RegistryType.COMPONENT, priority=10)
component = registry.get("my_cache")
best = registry.get_best(RegistryType.COMPONENT)
```

## 📊 Comparación de Patrones

| Patrón | Cuándo Usar | Ventajas |
|--------|-------------|----------|
| **Factory** | Creación simple con auto-selección | Automático, fácil de usar |
| **Builder** | Configuraciones complejas | Flexible, legible, validación |
| **Registry** | Componentes dinámicos | Extensible, descubrimiento |

## 🎯 Casos de Uso

### Factory
- ✅ Creación rápida de componentes
- ✅ Selección automática de backend
- ✅ Configuración por defecto

### Builder
- ✅ Configuraciones complejas
- ✅ Múltiples opciones
- ✅ Validación avanzada

### Registry
- ✅ Plugins dinámicos
- ✅ Estrategias intercambiables
- ✅ Componentes personalizados

## 📚 Ejemplos

Ver `examples/example_factory_builder.py` para ejemplos completos.

---

**Versión**: 2.0.0  
**Estado**: ✅ Patrones de Diseño Implementados  
**Fecha**: 2025-01-XX












