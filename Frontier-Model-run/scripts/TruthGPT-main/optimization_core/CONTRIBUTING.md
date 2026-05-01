# 🤝 Guía de Contribución

## 📋 Cómo Contribuir

Gracias por tu interés en contribuir a `optimization_core`! Esta guía te ayudará a entender cómo contribuir efectivamente.

---

## 🎯 Estándares de Código

### Validación
- ✅ Usa validadores compartidos de `utils.shared_validators`
- ✅ Valida todos los parámetros de entrada
- ✅ Proporciona mensajes de error informativos

### Manejo de Errores
- ✅ Usa el sistema de manejo de errores centralizado
- ✅ Usa excepciones personalizadas cuando sea apropiado
- ✅ Incluye contexto en los errores

### Testing
- ✅ Hereda de `BaseOptimizationCoreTestCase`
- ✅ Usa fixtures compartidas
- ✅ Usa assertions personalizadas
- ✅ Mantén cobertura de tests alta

### Documentación
- ✅ Incluye docstrings completos
- ✅ Documenta parámetros y valores de retorno
- ✅ Proporciona ejemplos de uso

---

## 🔧 Patrones de Código

### Crear un Nuevo Engine

```python
from inference.base_engine import BaseInferenceEngine
from inference.utils.validators import validate_generation_params
from inference.utils.prompt_utils import normalize_prompts, handle_single_prompt

class MyEngine(BaseInferenceEngine):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        # Validación usando utilidades compartidas
        # Inicialización
        self._set_initialized(True)
    
    def generate(self, prompts, **kwargs):
        # Validar parámetros
        validate_generation_params(...)
        
        # Normalizar prompts
        prompts_list, was_single = normalize_prompts(prompts)
        
        # Procesar
        results = self._generate_impl(prompts_list, **kwargs)
        
        # Retornar en formato correcto
        return handle_single_prompt(results, was_single)
```

### Crear un Nuevo Processor

```python
from data.utils.validators import validate_file_path
from data.utils.file_utils import detect_file_format

class MyProcessor:
    def read_data(self, path):
        # Validar path
        path_obj = validate_file_path(path, must_exist=True)
        
        # Detectar formato
        format = detect_file_format(path_obj)
        
        # Procesar
        ...
```

### Crear un Plugin

```python
from utils import BasePlugin, PluginInfo, register_plugin

class MyPlugin(BasePlugin):
    name = "my_plugin"
    version = "1.0.0"
    description = "My custom plugin"
    
    def execute(self, data, **kwargs):
        # Tu lógica aquí
        return processed_data

# Registrar
plugin_info = PluginInfo(
    name="my_plugin",
    version="1.0.0",
    description="My custom plugin"
)
register_plugin(plugin_info, MyPlugin())
```

---

## ✅ Checklist Antes de PR

- [ ] Código sigue los estándares
- [ ] Usa validadores compartidos
- [ ] Manejo de errores apropiado
- [ ] Tests agregados/actualizados
- [ ] Documentación actualizada
- [ ] Ejemplos agregados si aplica
- [ ] No hay código duplicado
- [ ] Type hints completos
- [ ] Logging apropiado

---

## 📝 Proceso de PR

1. Fork el repositorio
2. Crea una rama para tu feature
3. Implementa cambios siguiendo estándares
4. Agrega tests
5. Actualiza documentación
6. Crea PR con descripción clara

---

## 🎯 Áreas de Contribución

### Nuevas Features
- Nuevos engines de inferencia
- Nuevos procesadores de datos
- Nuevos plugins
- Nuevas utilidades

### Mejoras
- Optimizaciones de rendimiento
- Mejoras de UX
- Mejor documentación
- Más ejemplos

### Bugs
- Reportar bugs
- Fix bugs existentes
- Mejorar manejo de errores

---

*Última actualización: Noviembre 2025*
