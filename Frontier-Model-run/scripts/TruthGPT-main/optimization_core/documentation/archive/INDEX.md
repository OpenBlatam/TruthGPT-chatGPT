# 📚 TruthGPT Optimization Core - Índice de Documentación

Guía rápida para encontrar la documentación que necesitas.

## 🚀 Inicio Rápido

- **¿Primera vez?** → [`README.md`](README.md) - Quick Start
- **¿Comandos rápidos?** → [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)
- **¿Setup automático?** → `./setup_dev.sh` (Linux/Mac) o `.\setup_dev.ps1` (Windows)

## 📖 Documentación Principal

| Documento | Descripción | Cuándo Usar |
|-----------|-------------|-------------|
| [`README.md`](README.md) | Documentación completa principal | Guía principal, ejemplos, troubleshooting |
| [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) | Comandos y configuraciones rápidas | Referencia rápida durante trabajo |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Arquitectura y diseño del sistema | Entender cómo funciona internamente |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Guía para contribuidores | Quieres agregar features o hacer PR |
| [`CHANGELOG.md`](CHANGELOG.md) | Historial de cambios | Ver qué se ha agregado/cambiado |

## 🔧 Herramientas

| Herramienta | Descripción | Uso |
|-------------|-------------|-----|
| `validate_config.py` | Valida configuración YAML | Antes de entrenar |
| `init_project.py` | Crea proyecto nuevo | Nuevo experimento |
| `Makefile` | Comandos comunes | `make help` para ver todos |
| `setup_dev.sh/.ps1` | Setup automático | Primera instalación |

## 📁 Configuraciones

| Archivo | Descripción |
|---------|-------------|
| `configs/llm_default.yaml` | Configuración por defecto completa |
| `configs/presets/lora_fast.yaml` | LoRA rápido y eficiente |
| `configs/presets/performance_max.yaml` | Máxima performance en GPU |
| `configs/presets/debug.yaml` | Modo debug para desarrollo |

## 💻 Scripts y Ejemplos

| Script | Descripción |
|--------|-------------|
| `train_llm.py` | CLI principal de entrenamiento |
| `demo_gradio_llm.py` | Demo interactiva Gradio |
| `examples/benchmark_tokens_per_sec.py` | Benchmark de performance |
| `examples/train_with_datasets.py` | Uso de datasets modulares |
| `examples/switch_attention_backend.py` | Cambio de backends |
| `examples/complete_workflow.py` | Demo completo con 6 configuraciones |

## 🏗️ Código Core

| Módulo | Descripción |
|--------|-------------|
| `trainers/trainer.py` | GenericTrainer principal |
| `build_trainer.py` | Builder que ensambla componentes |
| `build.py` | Construcción modular de componentes |
| `factories/*.py` | 8 registries modulares |

## 🧪 Tests

| Test | Descripción |
|------|-------------|
| `tests/test_basic.py` | Tests unitarios básicos |

## 🎯 Casos de Uso Comunes

### "Quiero entrenar rápido con LoRA"
1. `make train-lora` o
2. `python train_llm.py --config configs/presets/lora_fast.yaml`

### "Quiero máxima performance"
1. `make train-perf` o
2. `python train_llm.py --config configs/presets/performance_max.yaml`

### "Quiero debuggear un problema"
1. `make train-debug` o
2. `python train_llm.py --config configs/presets/debug.yaml`

### "Quiero crear un nuevo proyecto"
```bash
python init_project.py mi_proyecto --preset lora_fast --model gpt2
python train_llm.py --config configs/mi_proyecto.yaml
```

### "Quiero validar mi config antes de entrenar"
```bash
python validate_config.py configs/mi_config.yaml
# o
make validate
```

### "Quiero hacer benchmark"
```bash
make benchmark
# o
python examples/benchmark_tokens_per_sec.py --model gpt2 --dtype bf16
```

## 🔍 Búsqueda Rápida

- **Configuración YAML**: Ver `README.md` sección "Configuración YAML Completa"
- **Registries disponibles**: Ver `ARCHITECTURE.md` sección "Componentes Modulares"
- **Troubleshooting**: Ver `README.md` sección "Troubleshooting"
- **Cómo contribuir**: Ver `CONTRIBUTING.md`
- **Comandos Make**: `make help` o ver `QUICK_REFERENCE.md`

## 📞 Ayuda

- Revisa `README.md` primero
- Consulta `QUICK_REFERENCE.md` para comandos
- Ver `ARCHITECTURE.md` para entender el diseño
- Abre un issue en el repositorio para bugs/features

---

**Última actualización**: v1.0.0 - Sistema Modular Completo


