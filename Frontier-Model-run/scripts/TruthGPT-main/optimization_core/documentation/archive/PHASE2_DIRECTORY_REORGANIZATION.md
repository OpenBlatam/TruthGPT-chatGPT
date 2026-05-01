# 📁 FASE 2: Reorganización de Directorios - Plan de Consolidación

## 📊 Análisis de Directorios Superpuestos

### Directorios de Configuración

#### Situación Actual:
- **`config/`** (8 archivos): Clases de configuración Python (ConfigManager, TransformerConfig, etc.)
- **`configs/`** (7 archivos): Archivos YAML y loaders (llm_default.yaml, presets/, loader.py, schema.py)
- **`configurations/`** (1 archivo): Wrapper/unificador que importa de config/ y otros lugares

#### Propósito:
- `config/` → Código Python para manejo de configuración
- `configs/` → Archivos YAML de configuración y utilidades de carga
- `configurations/` → Punto de acceso unificado (wrapper)

#### Decisión de Consolidación:
**Mantener `configs/` como directorio principal** porque:
- Ya contiene los YAMLs principales (llm_default.yaml, presets/)
- Es el directorio más usado según README
- Tiene la estructura más clara

**Plan:**
1. Mover `config/` → `configs/core/` (clases Python)
2. `configurations/` → Convertir en shim de compatibilidad o eliminar
3. Actualizar todos los imports

### Directorios de Utilidades

#### Situación Actual:
- **`utils/`** (186 archivos): Utilidades principales
- **`utils_mod/`** (1 archivo): Solo contiene `logging.py`

#### Decisión de Consolidación:
**Mover `utils_mod/` → `utils/`** porque:
- `utils_mod/` solo tiene 1 archivo
- No hay razón para mantener un directorio separado
- Simplifica la estructura

## 🎯 Plan de Consolidación

### Paso 1: Consolidar Directorios de Configuración

#### Estructura Propuesta:
```
configs/
├── __init__.py              # Exports principales
├── loader.py                # ✅ Ya existe
├── schema.py                # ✅ Ya existe
├── llm_default.yaml         # ✅ Ya existe
├── presets/                 # ✅ Ya existe
│   ├── debug.yaml
│   ├── lora_fast.yaml
│   └── performance_max.yaml
└── core/                    # 🆕 Nuevo (de config/)
    ├── __init__.py
    ├── config_manager.py
    ├── transformer_config.py
    ├── environment_config.py
    ├── validation_rules.py
    ├── architecture.py
    └── optimization_config.yaml
```

#### Acciones:
1. Crear `configs/core/` directory
2. Mover archivos de `config/` a `configs/core/`
3. Actualizar imports en `configs/core/__init__.py`
4. Actualizar `configs/__init__.py` para incluir exports de core
5. Crear shim de compatibilidad en `config/__init__.py`
6. Actualizar `configurations/__init__.py` para usar nuevos paths
7. Buscar y actualizar todos los imports en el código

### Paso 2: Consolidar Directorios de Utilidades

#### Acciones:
1. Mover `utils_mod/logging.py` → `utils/logging_mod.py` (o integrar si no hay conflicto)
2. Eliminar `utils_mod/` directory
3. Buscar y actualizar imports de `utils_mod`

### Paso 3: Clarificar Separación core/ vs optimizers/

#### Situación Actual:
- **`core/`**: Framework base, interfaces, servicios, validación
- **`optimizers/`**: Implementaciones específicas de optimización

#### Acciones:
1. Documentar claramente la separación
2. Verificar que no haya solapamiento
3. Mover archivos si es necesario

## 📝 Guía de Estructura de Directorios

### Estructura Final Propuesta

```
optimization_core/
├── configs/                 # ✅ Configuración unificada
│   ├── core/               # Clases Python de configuración
│   ├── presets/            # Presets YAML
│   └── *.yaml              # Archivos de configuración
│
├── factories/               # ✅ Registries y factories
│   ├── registry.py
│   ├── attention.py
│   ├── optimizer.py
│   └── ...
│
├── trainers/               # ✅ Entrenamiento
│   ├── trainer.py
│   └── config.py
│
├── optimizers/             # ✅ Optimizadores
│   ├── core/              # Base y estrategias
│   ├── production/        # Optimizadores de producción
│   └── specialized/       # Optimizadores especializados
│
├── core/                   # ✅ Framework base
│   ├── interfaces.py      # Interfaces y protocols
│   ├── services/          # Servicios base
│   └── validation/        # Validación
│
├── utils/                  # ✅ Utilidades consolidadas
│   ├── adapters/
│   ├── monitoring/
│   └── ...
│
├── modules/                # ✅ Módulos de modelo
├── data/                   # ✅ Procesamiento de datos
├── inference/              # ✅ Inferencia
└── tests/                  # ✅ Tests
```

## 🔄 Convenciones de Estructura

### Reglas de Ubicación

1. **Configuración**: Todo en `configs/`
   - YAMLs en raíz de `configs/`
   - Clases Python en `configs/core/`
   - Presets en `configs/presets/`

2. **Utilidades**: Todo en `utils/`
   - No crear `utils_*` adicionales
   - Subdirectorios por categoría (adapters/, monitoring/, etc.)

3. **Core vs Optimizers**:
   - `core/` → Framework base, interfaces, servicios genéricos
   - `optimizers/` → Implementaciones específicas de optimización

4. **Factories**: Todo en `factories/`
   - Registries y factories de componentes

5. **Tests**: Todo en `tests/`
   - Estructura refleja estructura del código

## ⚠️ Consideraciones de Migración

### Backward Compatibility

1. **Shims de Compatibilidad**:
   - `config/__init__.py` → Redirigir a `configs.core`
   - `configurations/__init__.py` → Mantener como wrapper

2. **Deprecation Warnings**:
   - Agregar warnings cuando se use `config/` directamente
   - Documentar migración

3. **Actualización Gradual**:
   - Mantener shims durante período de transición
   - Actualizar código interno primero
   - Documentar cambios

## 📋 Checklist de Implementación

### Consolidación de Configuración
- [ ] Crear `configs/core/` directory
- [ ] Mover archivos de `config/` a `configs/core/`
- [ ] Actualizar `configs/core/__init__.py`
- [ ] Actualizar `configs/__init__.py`
- [ ] Crear shim en `config/__init__.py`
- [ ] Actualizar `configurations/__init__.py`
- [ ] Buscar y actualizar imports en código
- [ ] Tests de compatibilidad

### Consolidación de Utilidades
- [ ] Verificar contenido de `utils_mod/logging.py`
- [ ] Mover a `utils/` (renombrar si necesario)
- [ ] Eliminar `utils_mod/`
- [ ] Buscar y actualizar imports
- [ ] Tests

### Documentación
- [ ] Actualizar README con nueva estructura
- [ ] Crear guía de migración
- [ ] Documentar convenciones

## 🎯 Métricas de Éxito

- ✅ Un solo directorio de configuración (`configs/`)
- ✅ Un solo directorio de utilidades (`utils/`)
- ✅ Separación clara entre `core/` y `optimizers/`
- ✅ 100% backward compatibility (shims)
- ✅ Todos los imports actualizados
- ✅ Tests pasando

---

**Estado**: Plan listo para implementación
**Próximos Pasos**: Comenzar con Paso 1 (consolidación de configuración)




