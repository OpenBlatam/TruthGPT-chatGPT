# 🏗️ TruthGPT Optimization Core - Architecture

## 📐 Arquitectura Modular

```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration YAML                        │
│              (configs/llm_default.yaml)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Build System                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ build.py     │  │build_trainer │  │validate_config│      │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Registries (Factories)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │Attention │ │Optimizer │ │Datasets  │ │Callbacks │        │
│  │KV Cache  │ │Memory    │ │Collate   │ │Metrics   │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  GenericTrainer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Mixed Precision (bf16/fp16)                          │   │
│  │ • TF32 Acceleration                                    │   │
│  │ • torch.compile Support                                │   │
│  │ • Fused AdamW Optimizer                               │   │
│  │ • EMA Weights                                          │   │
│  │ • Gradient Clipping                                    │   │
│  │ • NaN Detection                                        │   │
│  │ • Periodic Checkpointing                               │   │
│  │ • Auto-resume                                          │   │
│  │ • Dynamic Padding + Bucketing                         │   │
│  │ • Tokens/sec Tracking                                  │   │
│  │ • Early Stopping                                       │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output                                    │
│  • Checkpoints (best.pt, last.pt, step_*.pt)               │
│  • W&B/TensorBoard Logs                                     │
│  • Model Artifacts                                          │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Flujo de Datos

```
YAML Config
    ↓
build_components() → [attention, kv_cache, memory, optimizer, ...]
    ↓
build_trainer() → GenericTrainer
    ↓
train() → [Load Data → Forward → Backward → Optimize → Log → Eval → Checkpoint]
    ↓
Output: Checkpoints + Logs
```

## 📦 Componentes Modulares

### 1. Attention Backends
- **sdpa**: PyTorch SDPA (default, siempre disponible)
- **flash**: Flash Attention (fallback a sdpa si no disponible)
- **triton**: Triton kernels (fallback a sdpa si no disponible)

### 2. KV Cache
- **none**: Sin cache (para entrenamiento)
- **paged**: PagedKVCache (para inferencia eficiente)

### 3. Memory Management
- **adaptive**: AdvancedMemoryManager con detección GPU
- **static**: Configuración estática básica

### 4. Optimizers
- **adamw**: AdamW fused (default)
- **lion**: Lion optimizer (stub, fallback a AdamW)
- **adafactor**: Adafactor (stub, fallback a AdamW)

### 5. Callbacks
- **print**: PrintLogger (siempre disponible)
- **wandb**: Weights & Biases (requiere `pip install wandb`)
- **tensorboard**: TensorBoard (requiere `pip install tensorboard`)

### 6. Datasets
- **hf**: HuggingFace datasets (streaming opcional)
- **jsonl**: JSONL files (iterable)
- **webdataset**: WebDataset (stub para futuro)

### 7. Collate Functions
- **lm**: Language modeling (dynamic padding)
- **cv**: Computer vision (stub)

### 8. Metrics
- **loss**: Validation loss
- **ppl**: Perplexity (exp(loss))

## 🎛️ Configuración por Capas

```
Layer 1: Model Configuration
├── name_or_path
├── attention.backend
├── kv_cache.type
├── memory.policy
└── lora.* (opcional)

Layer 2: Training Configuration
├── epochs, batch_size, lr
├── optimizer.type
├── callbacks
├── performance flags (tf32, compile)
└── early stopping

Layer 3: Data Configuration
├── source (hf/jsonl/webdataset)
├── streaming
├── collate
└── bucketing

Layer 4: Infrastructure
├── checkpoint.*
├── ema.*
├── resume.*
├── eval.*
└── logging.*
```

## 🔌 Extension Points

Para agregar nuevos componentes sin tocar código core:

1. **Nuevo Backend de Attention:**
   ```python
   # En factories/attention.py
   @ATTENTION_BACKENDS.register("mi_backend")
   def build_mi_backend():
       return mi_attention_function
   ```

2. **Nuevo Optimizer:**
   ```python
   # En factories/optimizer.py
   @OPTIMIZERS.register("mi_optimizer")
   def build_mi_optimizer(params, lr, **kwargs):
       return MiOptimizer(params, lr=lr)
   ```

3. **Nuevo Callback:**
   ```python
   # En factories/callbacks.py
   @CALLBACKS.register("mi_callback")
   def build_mi_callback(**kwargs):
       return MiCallback(**kwargs)
   ```

## 🚀 Performance Tuning Guide

### Nivel 1: Básico
- `allow_tf32: true`
- `mixed_precision: bf16`
- `fused_adamw: true`

### Nivel 2: Intermedio
- `torch_compile: true`
- `compile_mode: reduce-overhead`
- `bucket_by_length: true`

### Nivel 3: Avanzado
- `torch_compile: true`
- `compile_mode: max-autotune`
- `data.prefetch_factor: 4`
- `data.num_workers: 8`

## 📊 Métricas y Observabilidad

### Durante Entrenamiento
- `loss` - Loss promedio por log_interval
- `tokens_per_sec` - Throughput en tiempo real
- `lr` - Learning rate actual (vía callbacks)

### Durante Evaluación
- `val_loss` - Loss de validación
- `ppl` - Perplexity (exp(val_loss))
- `best_metric` - Mejor métrica según `select_best_by`

### En W&B/TensorBoard
- Todos los logs automáticamente
- Gráficas de loss, ppl, tokens/s
- Comparación de runs

## 🔒 Principios de Diseño

1. **Modularidad**: Componentes intercambiables vía registries
2. **Configurabilidad**: Todo desde YAML, sin código
3. **Fallbacks**: Degradación elegante si componentes opcionales no disponibles
4. **Performance**: Optimizaciones aplicadas automáticamente cuando es posible
5. **Observabilidad**: Logging y métricas integradas
6. **Robustez**: Manejo de errores, NaN detection, auto-resume

---

**Para más detalles, ver `README.md` y `QUICK_REFERENCE.md`**


