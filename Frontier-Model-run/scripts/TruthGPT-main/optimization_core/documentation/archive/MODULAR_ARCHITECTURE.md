# 🏗️ Arquitectura Modular - optimization_core

## 📋 Resumen

Este documento describe la arquitectura modular de `optimization_core`, diseñada para máxima flexibilidad, extensibilidad y mantenibilidad.

---

## 🎯 Principios de Diseño Modular

### 1. Separación de Concerns
- **Core**: Interfaces y abstracciones base
- **Inference**: Motores de inferencia
- **Data**: Procesadores de datos
- **Utils**: Utilidades compartidas
- **Tests**: Testing
- **Examples**: Ejemplos de uso

### 2. Interfaces y Abstracciones
- Todas las interfaces en `core/interfaces.py`
- Implementaciones base en `core/base_classes.py`
- Factories para creación de instancias

### 3. Inversión de Dependencias
- Componentes dependen de interfaces, no de implementaciones
- Fácil intercambio de implementaciones
- Testing simplificado con mocks

---

## 📁 Estructura Modular

```
optimization_core/
├── core/                      # Núcleo del framework
│   ├── __init__.py           # Exports principales
│   ├── interfaces.py         # Interfaces abstractas
│   ├── base_classes.py       # Clases base
│   └── factories.py          # Factories
│
├── inference/                 # Motores de inferencia
│   ├── __init__.py
│   ├── base_engine.py        # Clase base (usa core)
│   ├── vllm_engine.py        # Implementación vLLM
│   ├── tensorrt_llm_engine.py # Implementación TensorRT-LLM
│   ├── inference_engine.py   # Implementación genérica
│   ├── engine_factory.py     # Factory específico
│   └── utils/                # Utilidades de inferencia
│
├── data/                      # Procesadores de datos
│   ├── __init__.py
│   ├── polars_processor.py   # Implementación Polars
│   ├── processor_factory.py  # Factory específico
│   └── utils/                # Utilidades de datos
│
├── utils/                     # Utilidades globales (39 módulos)
│   ├── __init__.py
│   ├── validation/           # Validación
│   ├── error_handling/       # Manejo de errores
│   ├── logging/              # Logging
│   ├── metrics/              # Métricas
│   ├── networking/           # Networking
│   ├── scheduling/            # Scheduling
│   ├── security/             # Seguridad
│   └── ...                   # Más utilidades
│
├── tests/                     # Testing
│   ├── utils/                # Utilidades de testing
│   └── base_test_case.py     # Clase base de tests
│
├── benchmarks/                # Benchmarks
│   ├── benchmark_runner.py
│   └── performance_metrics.py
│
└── examples/                  # Ejemplos
    ├── inference_examples.py
    ├── data_examples.py
    └── ...
```

---

## 🔌 Interfaces Principales

### `IComponent`
Interfaz base para todos los componentes:
- `name`: Nombre del componente
- `version`: Versión del componente
- `initialize()`: Inicialización
- `cleanup()`: Limpieza
- `get_status()`: Estado

### `IInferenceEngine`
Interfaz para motores de inferencia:
- `generate()`: Generación de texto
- `get_model_info()`: Información del modelo

### `IDataProcessor`
Interfaz para procesadores de datos:
- `process()`: Procesamiento de datos
- `validate()`: Validación de datos

---

## 🏭 Factories

### `ComponentFactory`
Factory genérico para componentes:
- `register()`: Registrar componente
- `create()`: Crear instancia
- `list_components()`: Listar componentes

### `InferenceEngineFactory`
Factory especializado para engines:
- `create_engine()`: Crear engine

### `DataProcessorFactory`
Factory especializado para procesadores:
- `create_processor()`: Crear procesador

---

## 📦 Módulos por Categoría

### Core (3 módulos)
- `interfaces.py` - Interfaces abstractas
- `base_classes.py` - Clases base
- `factories.py` - Factories

### Inference (6 módulos)
- `base_engine.py` - Clase base
- `vllm_engine.py` - vLLM
- `tensorrt_llm_engine.py` - TensorRT-LLM
- `inference_engine.py` - Genérico
- `engine_factory.py` - Factory
- `utils/` - 4 módulos de utilidades

### Data (3 módulos)
- `polars_processor.py` - Polars
- `processor_factory.py` - Factory
- `utils/` - 2 módulos de utilidades

### Utils (39 módulos)
Organizados por funcionalidad:
- Validación
- Error handling
- Logging
- Métricas
- Networking
- Scheduling
- Security
- Y más...

---

## 🔄 Flujo de Uso Modular

### 1. Crear Componente
```python
from core import InferenceEngineFactory

factory = InferenceEngineFactory()
engine = factory.create_engine("vllm", "mistral-7b")
```

### 2. Usar Interface
```python
from core import IInferenceEngine

def process_with_engine(engine: IInferenceEngine, prompt: str):
    return engine.generate(prompt)
```

### 3. Extender Framework
```python
from core import BaseInferenceEngine, GenerationConfig

class MyEngine(BaseInferenceEngine):
    def _generate_impl(self, prompts, config, **kwargs):
        # Tu implementación
        pass
```

---

## ✅ Beneficios de la Arquitectura Modular

1. **Flexibilidad**
   - Fácil intercambio de implementaciones
   - Múltiples implementaciones de la misma interfaz

2. **Extensibilidad**
   - Agregar nuevos componentes sin modificar existentes
   - Herencia clara de clases base

3. **Testabilidad**
   - Mocks fáciles usando interfaces
   - Testing aislado de componentes

4. **Mantenibilidad**
   - Separación clara de concerns
   - Código organizado y fácil de navegar

5. **Reusabilidad**
   - Componentes reutilizables
   - Utilidades compartidas

---

## 🚀 Próximos Pasos

1. **Organizar Utils por Categorías**
   - Crear subdirectorios por funcionalidad
   - Mejorar organización

2. **Documentación de Interfaces**
   - Documentar todas las interfaces
   - Ejemplos de implementación

3. **Plugin System Mejorado**
   - Integración con core interfaces
   - Carga dinámica de plugins

---

*Última actualización: Noviembre 2025*
