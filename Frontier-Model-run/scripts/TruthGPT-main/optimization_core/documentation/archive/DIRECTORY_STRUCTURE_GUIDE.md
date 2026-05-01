# 📁 Guía de Estructura de Directorios - Optimization Core

## 🎯 Principios de Organización

Esta guía establece las convenciones para organizar código en `optimization_core`. Sigue estos principios para mantener la estructura clara y mantenible.

## 📂 Estructura de Directorios

### Directorio Raíz

```
optimization_core/
├── configs/              # ✅ Configuración (YAMLs y clases)
├── factories/            # ✅ Registries y factories
├── trainers/             # ✅ Entrenamiento
├── optimizers/           # ✅ Optimizadores
├── core/                 # ✅ Framework base
├── utils/                # ✅ Utilidades
├── modules/              # ✅ Módulos de modelo
├── data/                 # ✅ Procesamiento de datos
├── inference/            # ✅ Inferencia
└── tests/                # ✅ Tests
```

## 📋 Reglas de Ubicación

### 1. Configuración → `configs/`

**Ubicación**: Todo código relacionado con configuración va en `configs/`

**Estructura**:
```
configs/
├── __init__.py           # Exports principales
├── loader.py             # Cargadores de configuración
├── schema.py             # Schemas de validación
├── *.yaml                # Archivos de configuración YAML
├── presets/              # Presets predefinidos
│   ├── debug.yaml
│   ├── lora_fast.yaml
│   └── performance_max.yaml
└── core/                 # Clases Python de configuración
    ├── __init__.py
    ├── config_manager.py
    ├── transformer_config.py
    └── ...
```

**Qué va aquí**:
- ✅ Archivos YAML de configuración
- ✅ Clases Python para manejo de configuración
- ✅ Loaders y parsers de configuración
- ✅ Schemas de validación
- ✅ Presets y templates

**Qué NO va aquí**:
- ❌ Configuración de entrenamiento (va en `trainers/config.py`)
- ❌ Configuración de producción (va en `production/config/`)

### 2. Utilidades → `utils/`

**Ubicación**: Todo código de utilidades va en `utils/`

**Estructura**:
```
utils/
├── __init__.py
├── adapters/             # Adapters
├── monitoring/            # Monitoreo
├── training/              # Utilidades de entrenamiento
├── memory/                # Utilidades de memoria
├── gpu/                   # Utilidades de GPU
└── ...
```

**Qué va aquí**:
- ✅ Funciones de utilidad general
- ✅ Helpers y utilities
- ✅ Adapters y wrappers
- ✅ Utilidades de monitoreo
- ✅ Utilidades de logging

**Qué NO va aquí**:
- ❌ Código específico de optimización (va en `optimizers/`)
- ❌ Código de framework base (va en `core/`)

### 3. Framework Base → `core/`

**Ubicación**: Código de framework base, interfaces, servicios genéricos

**Estructura**:
```
core/
├── __init__.py
├── interfaces.py         # Interfaces y protocols
├── services/              # Servicios base
│   ├── base_service.py
│   ├── training_service.py
│   └── inference_service.py
├── validation/            # Validación
│   ├── config_validator.py
│   └── data_validator.py
└── ...
```

**Qué va aquí**:
- ✅ Interfaces y protocols base
- ✅ Servicios genéricos
- ✅ Validación genérica
- ✅ Framework base reutilizable

**Qué NO va aquí**:
- ❌ Implementaciones específicas de optimización (va en `optimizers/`)
- ❌ Código específico de modelos (va en `modules/`)

### 4. Optimizadores → `optimizers/`

**Ubicación**: Implementaciones específicas de optimización

**Estructura**:
```
optimizers/
├── __init__.py
├── core/                  # Base y estrategias
│   ├── base_truthgpt_optimizer.py
│   ├── unified_optimizer.py
│   └── strategies/
├── production/            # Optimizadores de producción
├── specialized/          # Optimizadores especializados
└── ...
```

**Qué va aquí**:
- ✅ Implementaciones de optimizadores
- ✅ Estrategias de optimización
- ✅ Técnicas de optimización
- ✅ Optimizadores especializados

**Qué NO va aquí**:
- ❌ Framework base (va en `core/`)
- ❌ Utilidades generales (va en `utils/`)

### 5. Factories → `factories/`

**Ubicación**: Registries y factories de componentes

**Estructura**:
```
factories/
├── __init__.py
├── registry.py           # Registry base
├── attention.py          # Factory de attention
├── optimizer.py           # Factory de optimizers
├── datasets.py            # Factory de datasets
└── ...
```

**Qué va aquí**:
- ✅ Registries de componentes
- ✅ Factories para crear componentes
- ✅ Sistema de registro

### 6. Entrenamiento → `trainers/`

**Ubicación**: Código relacionado con entrenamiento

**Estructura**:
```
trainers/
├── __init__.py
├── trainer.py            # GenericTrainer
├── config.py             # TrainerConfig
└── callbacks.py          # Sistema de callbacks
```

**Qué va aquí**:
- ✅ Trainers
- ✅ Configuración de entrenamiento
- ✅ Callbacks de entrenamiento

### 7. Módulos → `modules/`

**Ubicación**: Módulos de modelo (attention, feed_forward, etc.)

**Estructura**:
```
modules/
├── __init__.py
├── attention/            # Módulos de attention
├── feed_forward/        # Módulos feed-forward
└── ...
```

**Qué va aquí**:
- ✅ Módulos de modelo
- ✅ Componentes de arquitectura
- ✅ Bloques de transformer

### 8. Datos → `data/`

**Ubicación**: Procesamiento de datos

**Estructura**:
```
data/
├── __init__.py
├── datasets.py           # Datasets
├── collate.py            # Funciones de collate
└── ...
```

**Qué va aquí**:
- ✅ Procesamiento de datos
- ✅ Datasets
- ✅ Data loaders

### 9. Inferencia → `inference/`

**Ubicación**: Código de inferencia

**Estructura**:
```
inference/
├── __init__.py
├── core/                 # Core de inferencia
├── middleware/           # Middleware
└── monitoring/            # Monitoreo de inferencia
```

**Qué va aquí**:
- ✅ Servicios de inferencia
- ✅ Middleware de inferencia
- ✅ Monitoreo de inferencia

### 10. Tests → `tests/`

**Ubicación**: Todos los tests

**Estructura**:
```
tests/
├── __init__.py
├── unit/                 # Tests unitarios
├── integration/          # Tests de integración
└── fixtures/             # Fixtures
```

**Qué va aquí**:
- ✅ Todos los tests
- ✅ Fixtures
- ✅ Helpers de testing

## 🔄 Migración de Archivos

### Reglas de Migración

1. **Configuración**:
   - `config/` → `configs/core/`
   - `configs/` → Mantener (ya está bien)
   - `configurations/` → Usar como wrapper o eliminar

2. **Utilidades**:
   - `utils_mod/` → `utils/`
   - `utils/` → Mantener

3. **Core vs Optimizers**:
   - Si es framework base → `core/`
   - Si es optimización específica → `optimizers/`

## ⚠️ Anti-Patrones a Evitar

### ❌ NO Hacer:

1. **Crear directorios duplicados**:
   - ❌ `config/`, `configs/`, `configurations/` (usar solo `configs/`)
   - ❌ `utils/`, `utils_mod/` (usar solo `utils/`)

2. **Mezclar responsabilidades**:
   - ❌ Poner utilidades en `core/`
   - ❌ Poner optimizadores en `core/`
   - ❌ Poner configuración en múltiples lugares

3. **Estructura inconsistente**:
   - ❌ Algunos módulos con `__init__.py`, otros sin
   - ❌ Nombres inconsistentes
   - ❌ Estructura diferente en subdirectorios similares

## ✅ Mejores Prácticas

### ✅ Hacer:

1. **Seguir la estructura establecida**
2. **Usar `__init__.py` para exports**
3. **Mantener imports relativos consistentes**
4. **Documentar decisiones de estructura**
5. **Actualizar esta guía cuando cambie la estructura**

## 📝 Ejemplos

### ✅ Correcto

```python
# configs/core/config_manager.py
from configs.core.transformer_config import TransformerConfig

# utils/monitoring/logger.py
from utils.adapters import Adapter

# optimizers/core/unified_optimizer.py
from optimizers.core.strategies import EnhancedStrategy
```

### ❌ Incorrecto

```python
# config/config_manager.py (debería estar en configs/core/)
from config.transformer_config import TransformerConfig

# utils_mod/logging.py (debería estar en utils/)
from utils_mod.adapters import Adapter

# core/unified_optimizer.py (debería estar en optimizers/core/)
from core.strategies import EnhancedStrategy
```

## 🔗 Referencias

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Arquitectura general
- [PHASE2_DIRECTORY_REORGANIZATION.md](./PHASE2_DIRECTORY_REORGANIZATION.md) - Plan de reorganización
- [ARCHITECTURE_IMPROVEMENTS.md](./ARCHITECTURE_IMPROVEMENTS.md) - Mejoras arquitectónicas

---

**Última Actualización**: 2024
**Mantenedor**: Equipo de desarrollo




