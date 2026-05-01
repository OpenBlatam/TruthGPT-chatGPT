# 🚀 TruthGPT Optimization Core - Quick Reference

## 🎯 Comandos Esenciales

```bash
# Setup
./setup_dev.sh              # Linux/Mac
.\setup_dev.ps1             # Windows

# Health Check
make health
python utils/health_check.py

# Validar config
make validate
# o
python validate_config.py configs/llm_default.yaml

# Entrenar
make train                  # GPU (default)
make train-lora             # LoRA fast preset
make train-perf             # Performance max preset
make train-debug            # Debug preset
make train-cpu              # CPU

# Visualización y Análisis
make visualize              # Ver resumen de runs
make compare                # Comparar múltiples runs
make monitor                # Monitorear en tiempo real
python utils/compare_runs.py --runs-dir runs
python utils/monitor_training.py runs/mi_run

# Limpieza
python utils/cleanup_runs.py --days 30 --old-runs
python utils/cleanup_runs.py --keep-checkpoints 3 --checkpoints

# Exportar Config
python utils/export_config.py runs/mi_run --output configs/reproduce.yaml

# Benchmark
make benchmark

# Demo
make demo

# Tests
make test
pytest tests/test_basic.py -v
```

## 📋 Cambios Rápidos en YAML

### Activar LoRA
```yaml
model:
  lora:
    enabled: true
```

### Cambiar Optimizer
```yaml
optimizer:
  type: lion  # adamw|lion|adafactor
```

### Activar W&B
```yaml
training:
  callbacks: [print, wandb]
logging:
  project: my-project
```

### Optimizar Performance
```yaml
training:
  allow_tf32: true
  torch_compile: true
data:
  bucket_by_length: true
```

### Auto-resume
```yaml
resume:
  enabled: true
```

## 🔧 Registries Disponibles

| Registry | Valores | Archivo |
|----------|---------|---------|
| Attention | `sdpa\|flash\|triton` | `factories/attention.py` |
| KV Cache | `none\|paged` | `factories/kv_cache.py` |
| Memory | `adaptive\|static` | `factories/memory.py` |
| Optimizer | `adamw\|lion\|adafactor` | `factories/optimizer.py` |
| Callbacks | `print\|wandb\|tensorboard` | `factories/callbacks.py` |
| Datasets | `hf\|jsonl\|webdataset` | `factories/datasets.py` |
| Collate | `lm\|cv` | `factories/collate.py` |
| Metrics | `loss\|ppl` | `factories/metrics.py` |

## 📁 Archivos Clave

- `configs/llm_default.yaml` - Configuración principal
- `train_llm.py` - CLI de entrenamiento
- `trainers/trainer.py` - GenericTrainer principal
- `build_trainer.py` - Builder modular
- `validate_config.py` - Validador de YAML
- `README.md` - Documentación completa

## 🐛 Soluciones Rápidas

**OOM:** Reduce `train_batch_size`, activa `gradient_checkpointing`, usa `mixed_precision: bf16`

**Lento:** Activa `allow_tf32: true`, `torch_compile: true`, `bucket_by_length: true`

**No logea:** Instala `wandb` o `tensorboard`, verifica `training.callbacks` en YAML

## 📞 Ayuda

- `make help` - Ver todos los comandos
- `python examples/complete_workflow.py` - Ver ejemplos
- Ver `README.md` para documentación completa

