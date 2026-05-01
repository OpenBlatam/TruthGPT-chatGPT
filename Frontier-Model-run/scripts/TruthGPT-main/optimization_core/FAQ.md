# ❓ FAQ - Preguntas Frecuentes

## 🔧 Instalación y Configuración

### ¿Qué versión de Python necesito?
Python 3.8 o superior. Verifica con:
```bash
python --version
# o
make health
```

### ¿Cómo instalo dependencias opcionales?
```bash
# Para W&B logging
pip install wandb

# Para TensorBoard
pip install tensorboard

# Para monitoring avanzado
pip install psutil

# Para aceleración
pip install accelerate peft
```

### ¿Cómo verifico que todo está instalado correctamente?
```bash
make health
# o
python utils/health_check.py
```

## 🚀 Entrenamiento

### ¿Cómo empiezo rápido?
```bash
# Opción 1: Preset LoRA (más rápido)
make train-lora

# Opción 2: Configuración por defecto
make train
```

### ¿Cómo cambio el modelo sin editar código?
Edita `configs/llm_default.yaml`:
```yaml
model:
  name_or_path: gpt2  # Cambia aquí
```

O usa el inicializador:
```bash
python init_project.py mi_proyecto --model microsoft/DialoGPT-small
```

### ¿Cómo activo LoRA para ahorrar memoria?
```yaml
model:
  lora:
    enabled: true
    r: 16
    alpha: 32
```

### ¿Mi GPU se queda sin memoria, qué hago?
1. Reduce `train_batch_size` (ej: de 32 a 16)
2. Reduce `max_seq_len` (ej: de 512 a 256)
3. Activa `gradient_checkpointing: true`
4. Usa `mixed_precision: bf16`
5. Aumenta `grad_accum_steps`

### ¿Cómo entreno más rápido?
Activa optimizaciones en YAML:
```yaml
training:
  allow_tf32: true
  torch_compile: true
  fused_adamw: true
data:
  bucket_by_length: true
  num_workers: 8
  prefetch_factor: 4
```

O usa el preset de performance:
```bash
make train-perf
```

## 📊 Logging y Monitoreo

### ¿Cómo activo W&B?
1. Instala: `pip install wandb`
2. Login: `wandb login`
3. En YAML:
```yaml
training:
  callbacks: [print, wandb]
logging:
  project: mi-proyecto
  run_name: experimento-1
```

### ¿Cómo veo métricas en tiempo real?
```bash
# Monitorear directorio de run
make monitor

# O monitorear archivo de log
python utils/monitor_training.py logs/training.log --file
```

### ¿Cómo comparo diferentes runs?
```bash
make compare
# o
python utils/compare_runs.py --runs-dir runs
```

## 💾 Checkpoints y Resume

### ¿Cómo activo auto-resume?
```yaml
resume:
  enabled: true
  checkpoint_dir: null  # Usa output_dir si null
```

### ¿Dónde se guardan los checkpoints?
En `output_dir` configurado en YAML (default: `runs/<run_name>/`).

### ¿Cómo exporto configuración desde un checkpoint?
```bash
python utils/export_config.py runs/mi_run/best.pt --output configs/reproduce.yaml
```

## 🧹 Limpieza y Mantenimiento

### ¿Cómo limpio runs antiguos?
```bash
# Ver qué se eliminaría (dry run)
make cleanup-dry

# Ejecutar limpieza
make cleanup

# Personalizado
python utils/cleanup_runs.py --days 30 --old-runs --execute
```

### ¿Cómo limpio checkpoints antiguos pero mantengo los últimos N?
```bash
python utils/cleanup_runs.py --keep-checkpoints 3 --checkpoints --execute
```

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
- Reduce `train_batch_size`
- Reduce `max_seq_len`
- Activa `gradient_checkpointing: true`
- Usa `mixed_precision: bf16`
- Aumenta `grad_accum_steps`

### Error: "ModuleNotFoundError"
```bash
# Instala dependencias básicas
pip install -r requirements_advanced.txt

# Verifica con health check
make health
```

### El entrenamiento es muy lento
- Activa `allow_tf32: true` (GPUs Ampere+)
- Activa `torch_compile: true`
- Aumenta `num_workers` y `prefetch_factor`
- Activa `bucket_by_length: true`

### W&B/TensorBoard no funciona
1. Verifica instalación: `pip install wandb` o `pip install tensorboard`
2. Verifica que está en `training.callbacks` en YAML
3. Verifica `logging.project` y `logging.run_name`

### NaN en loss
El sistema detecta NaN automáticamente y omite el step. Si persiste:
- Reduce `learning_rate`
- Verifica datos (pueden tener valores inválidos)
- Activa `detect_anomaly: true` en YAML para debug

## 🔄 Configuración Avanzada

### ¿Cómo cambio el optimizer?
```yaml
optimizer:
  type: adamw  # adamw|lion|adafactor
  fused: true
```

### ¿Cómo cambio el backend de atención?
```yaml
model:
  attention:
    backend: sdpa  # sdpa|flash|triton
```

### ¿Cómo uso datasets en streaming?
```yaml
data:
  source: hf
  streaming: true
```

### ¿Cómo activo EMA para evaluación?
```yaml
ema:
  enabled: true
  decay: 0.999
```

## 📚 Recursos

### ¿Dónde encuentro ejemplos?
- `examples/benchmark_tokens_per_sec.py` - Benchmark
- `examples/train_with_datasets.py` - Datasets
- `examples/complete_workflow.py` - Workflow completo

### ¿Dónde está la documentación completa?
- `README.md` - Documentación principal
- `QUICK_REFERENCE.md` - Referencia rápida
- `ARCHITECTURE.md` - Arquitectura del sistema
- `INDEX.md` - Índice de documentación

### ¿Cómo contribuyo?
Ver `CONTRIBUTING.md` para guía completa.

## 💡 Tips y Mejores Prácticas

1. **Siempre valida tu config antes de entrenar:**
   ```bash
   make validate
   ```

2. **Usa presets para empezar rápido:**
   ```bash
   make train-lora  # Para pruebas rápidas
   make train-perf  # Para máximo performance
   ```

3. **Monitorea tu entrenamiento:**
   ```bash
   # En otra terminal
   make monitor
   ```

4. **Usa health check antes de entrenar:**
   ```bash
   make health
   ```

5. **Limpia runs antiguos regularmente:**
   ```bash
   make cleanup-dry  # Ver qué se eliminaría
   make cleanup      # Ejecutar limpieza
   ```

6. **Exporta configs para reproducibilidad:**
   ```bash
   python utils/export_config.py runs/mi_run --output configs/reproduce.yaml
   ```

---

**¿No encuentras tu respuesta?** Abre un issue en el repositorio con:
- Tu pregunta
- Configuración relevante
- Logs de error (si aplica)


