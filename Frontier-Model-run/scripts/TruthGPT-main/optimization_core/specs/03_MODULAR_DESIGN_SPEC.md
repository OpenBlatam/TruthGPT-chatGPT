# 🏗️ Especificación de Diseño Modular - Optimization Core

## 📋 Resumen

Este documento especifica los principios y patrones de diseño modular utilizados en `optimization_core`, asegurando máxima flexibilidad, extensibilidad y mantenibilidad.

## 🎯 Objetivos

1. **Separación de Concerns**: Cada módulo tiene una responsabilidad única y bien definida
2. **Bajo Acoplamiento**: Módulos independientes con interfaces claras
3. **Alta Cohesión**: Funcionalidad relacionada agrupada lógicamente
4. **Extensibilidad**: Fácil agregar nuevos módulos sin modificar existentes
5. **Testabilidad**: Módulos fácilmente testeables de forma aislada

## 🏗️ Principios de Diseño Modular

### 1. Single Responsibility Principle (SRP)

Cada módulo debe tener una única razón para cambiar.

**Ejemplo**:
```python
# ✅ Bueno: Separación clara
class VLLMEngine(BaseInferenceEngine):
    """Solo responsable de inferencia con vLLM."""
    pass

class PolarsProcessor(BaseDataProcessor):
    """Solo responsable de procesamiento con Polars."""
    pass

# ❌ Malo: Múltiples responsabilidades
class EngineAndProcessor:
    """Hace inferencia Y procesamiento."""
    pass
```

### 2. Open/Closed Principle (OCP)

Módulos abiertos para extensión, cerrados para modificación.

**Ejemplo**:
```python
# ✅ Bueno: Extensible sin modificar
class BaseInferenceEngine(ABC):
    @abstractmethod
    def _generate_impl(self, prompts, **kwargs):
        pass

class VLLMEngine(BaseInferenceEngine):
    def _generate_impl(self, prompts, **kwargs):
        # Nueva implementación sin modificar base
        pass

# ❌ Malo: Requiere modificar base
class InferenceEngine:
    def generate(self, prompts, engine_type):
        if engine_type == "vllm":
            # Modificar para agregar nuevo engine
            pass
```

### 3. Dependency Inversion Principle (DIP)

Módulos dependen de abstracciones, no de implementaciones concretas.

**Ejemplo**:
```python
# ✅ Bueno: Depende de interfaz
def process_with_engine(engine: IInferenceEngine, prompt: str):
    return engine.generate(prompt)

# ❌ Malo: Depende de implementación concreta
def process_with_engine(engine: VLLMEngine, prompt: str):
    return engine.generate(prompt)
```

### 4. Interface Segregation Principle (ISP)

Interfaces específicas en lugar de interfaces generales.

**Ejemplo**:
```python
# ✅ Bueno: Interfaces específicas
class IInferenceEngine(IComponent):
    def generate(self, prompts, **kwargs):
        pass

class IDataProcessor(IComponent):
    def process(self, data, **kwargs):
        pass

# ❌ Malo: Interface monolítica
class IEverything(IComponent):
    def generate(self, prompts, **kwargs):
        pass
    def process(self, data, **kwargs):
        pass
    def train(self, model, data, **kwargs):
        pass
    # ... muchas más responsabilidades
```

## 📦 Estructura Modular

### Organización por Capas

```
optimization_core/
├── core/                    # Capa de abstracción
│   ├── interfaces.py       # Interfaces abstractas
│   ├── base_classes.py     # Clases base
│   └── factories.py        # Factories
│
├── inference/              # Capa de implementación - Inferencia
│   ├── base_engine.py      # Clase base
│   ├── vllm_engine.py     # Implementación vLLM
│   └── tensorrt_llm_engine.py  # Implementación TensorRT
│
├── data/                   # Capa de implementación - Datos
│   ├── polars_processor.py # Implementación Polars
│   └── processor_factory.py
│
├── polyglot_core/          # Capa de unificación
│   ├── backend.py          # Detección de backends
│   ├── cache.py            # Cache unificado
│   └── compression.py     # Compresión unificada
│
└── utils/                  # Capa de utilidades
    ├── validation/         # Validación
    ├── error_handling/     # Manejo de errores
    └── metrics/            # Métricas
```

### Organización por Funcionalidad

Cada módulo agrupa funcionalidad relacionada:

```
inference/
├── engines/                # Motores de inferencia
│   ├── vllm_engine.py
│   └── tensorrt_llm_engine.py
├── batching/               # Batching
│   └── continuous_batcher.py
├── caching/                # Caching
│   └── kv_cache_manager.py
└── utils/                  # Utilidades de inferencia
    └── prompt_utils.py
```

## 🔌 Interfaces y Contratos

### Definición de Interfaces

Todas las interfaces se definen en `core/interfaces.py`:

```python
# core/interfaces.py
class IComponent(ABC):
    """Interfaz base para todos los componentes."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        pass

class IInferenceEngine(IComponent):
    """Interfaz para motores de inferencia."""
    @abstractmethod
    def generate(self, prompts, **kwargs):
        pass
```

### Implementación de Interfaces

Las implementaciones concretas están en sus respectivos módulos:

```python
# inference/vllm_engine.py
class VLLMEngine(BaseInferenceEngine):
    """Implementación vLLM de IInferenceEngine."""
    
    def generate(self, prompts, **kwargs):
        # Implementación específica
        pass
```

## 🏭 Factories

### Factory Pattern

Las factories crean instancias sin exponer la lógica de creación:

```python
# inference/engine_factory.py
class EngineFactory:
    @staticmethod
    def create_engine(
        model: str,
        engine_type: EngineType = EngineType.AUTO,
        **kwargs
    ) -> IInferenceEngine:
        """Crea engine según tipo especificado."""
        if engine_type == EngineType.VLLM:
            return VLLMEngine(model, **kwargs)
        elif engine_type == EngineType.TENSORRT:
            return TensorRTLLMEngine(model, **kwargs)
        # ...
```

### Registry Pattern

Registro dinámico de componentes:

```python
# core/factories.py
class ComponentRegistry:
    _engines: Dict[str, Type[IInferenceEngine]] = {}
    
    @classmethod
    def register_engine(
        cls,
        name: str,
        engine_class: Type[IInferenceEngine]
    ):
        """Registra un nuevo engine."""
        cls._engines[name] = engine_class
    
    @classmethod
    def create_engine(cls, name: str, **kwargs) -> IInferenceEngine:
        """Crea engine por nombre."""
        engine_class = cls._engines.get(name)
        if not engine_class:
            raise ValueError(f"Engine {name} not registered")
        return engine_class(**kwargs)
```

## 🔄 Flujos Modulares

### Flujo de Creación

```
1. Usuario → Factory.create_engine()
2. Factory → Selecciona implementación
3. Factory → Crea instancia
4. Factory → Inicializa componente
5. Factory → Retorna componente
```

### Flujo de Uso

```
1. Usuario → Obtiene componente (via factory)
2. Usuario → Usa interfaz (IInferenceEngine)
3. Componente → Implementación específica
4. Componente → Retorna resultado
```

## 📊 Módulos por Categoría

### Core (3 módulos)
- `interfaces.py` - Interfaces abstractas
- `base_classes.py` - Clases base
- `factories.py` - Factories

### Inference (6 módulos)
- `base_engine.py` - Clase base
- `vllm_engine.py` - vLLM
- `tensorrt_llm_engine.py` - TensorRT-LLM
- `engine_factory.py` - Factory
- `utils/` - 4 módulos de utilidades

### Data (3 módulos)
- `polars_processor.py` - Polars
- `processor_factory.py` - Factory
- `utils/` - 2 módulos de utilidades

### Utils (39 módulos)
Organizados por funcionalidad:
- `validation/` - Validación
- `error_handling/` - Manejo de errores
- `logging/` - Logging
- `metrics/` - Métricas
- `networking/` - Networking
- `scheduling/` - Scheduling
- `security/` - Seguridad

## ✅ Beneficios del Diseño Modular

### 1. Mantenibilidad
- Código organizado y fácil de encontrar
- Cambios localizados en módulos específicos
- Fácil entender el propósito de cada módulo

### 2. Testabilidad
- Tests aislados por módulo
- Mocks fáciles usando interfaces
- Tests de integración claros

### 3. Extensibilidad
- Agregar nuevos engines sin modificar existentes
- Nuevos procesadores independientes
- Plugins y extensiones fáciles

### 4. Reusabilidad
- Componentes reutilizables en diferentes contextos
- Utilidades compartidas
- Factories genéricas

### 5. Escalabilidad
- Módulos independientes escalan por separado
- Carga bajo demanda (lazy imports)
- Distribución de módulos

## 🔧 Convenciones

### Naming
- Interfaces: Prefijo `I` (IComponent, IInferenceEngine)
- Clases base: Prefijo `Base` (BaseInferenceEngine)
- Implementaciones: Sin prefijo (VLLMEngine)
- Factories: Sufijo `Factory` (EngineFactory)

### Organización
- Un archivo por clase principal
- Utilidades en subdirectorios `utils/`
- Tests en directorio `tests/` paralelo

### Dependencias
- Módulos superiores no dependen de módulos inferiores
- Dependencias solo hacia abajo en la jerarquía
- Interfaces en `core/` sin dependencias externas

## 🧪 Testing Modular

### Tests Unitarios

```python
# tests/test_vllm_engine.py
def test_vllm_engine_generate():
    """Test VLLMEngine de forma aislada."""
    engine = VLLMEngine("mistral-7b")
    engine.initialize()
    result = engine.generate("Hello")
    assert isinstance(result, str)
```

### Tests de Integración

```python
# tests/test_inference_integration.py
def test_engine_factory():
    """Test integración factory con engines."""
    engine = EngineFactory.create_engine(
        "mistral-7b",
        EngineType.VLLM
    )
    assert isinstance(engine, IInferenceEngine)
```

### Mocks

```python
# tests/test_with_mocks.py
def test_with_mock_engine():
    """Test usando mock de interfaz."""
    mock_engine = Mock(spec=IInferenceEngine)
    mock_engine.generate.return_value = "test"
    
    result = process_with_engine(mock_engine, "prompt")
    assert result == "test"
```

## 📝 Ejemplos de Uso

### Uso Básico

```python
from optimization_core.inference import EngineFactory, EngineType

# Crear engine usando factory
engine = EngineFactory.create_engine(
    "mistral-7b",
    EngineType.AUTO
)

# Usar interfaz unificada
result = engine.generate("Hello, world!")
```

### Extensión

```python
from optimization_core.core import BaseInferenceEngine

class MyCustomEngine(BaseInferenceEngine):
    """Engine personalizado."""
    
    def _generate_impl(self, prompts, **kwargs):
        # Implementación personalizada
        pass

# Registrar
from optimization_core.core import ComponentRegistry
ComponentRegistry.register_engine("my_custom", MyCustomEngine)
```

## ⚠️ Anti-Patrones a Evitar

### 1. God Object
❌ Una clase que hace todo
✅ Separar en múltiples clases especializadas

### 2. Circular Dependencies
❌ Módulo A depende de B, B depende de A
✅ Usar interfaces o eventos para desacoplar

### 3. Tight Coupling
❌ Dependencias directas entre implementaciones
✅ Depender de interfaces

### 4. Feature Envy
❌ Clase que usa más métodos de otra clase que de sí misma
✅ Mover funcionalidad a la clase correcta

## 🔄 Refactoring Modular

### Cuándo Refactorizar

1. **Módulo muy grande**: > 1000 líneas
2. **Múltiples responsabilidades**: Violación de SRP
3. **Dependencias circulares**: Problemas de importación
4. **Difícil de testear**: Muchas dependencias

### Cómo Refactorizar

1. Identificar responsabilidades
2. Extraer a módulos separados
3. Definir interfaces claras
4. Actualizar dependencias
5. Actualizar tests

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2025




