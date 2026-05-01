# Guía de Uso de TruthGPT

Esta guía te enseñará cómo usar TruthGPT de manera efectiva, desde la instalación hasta el uso avanzado.

## 📋 Tabla de Contenidos

1. [Instalación y Configuración](#instalación-y-configuración)
2. [Uso Básico](#uso-básico)
3. [Configuración Avanzada](#configuración-avanzada)
4. [Optimizaciones](#optimizaciones)
5. [Troubleshooting](#troubleshooting)

## 🚀 Instalación y Configuración

### Requisitos Previos

```bash
# Verificar Python
python --version  # Debe ser 3.8+

# Verificar PyTorch
python -c "import torch; print(torch.__version__)"
```

### Instalación Paso a Paso

```bash
# 1. Clonar el repositorio
git clone <repository-url>
cd optimization_core

# 2. Crear entorno virtual
python -m venv truthgpt_env
source truthgpt_env/bin/activate  # Linux/Mac
# truthgpt_env\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements_modern.txt

# 4. Verificar instalación
python -c "from optimization_core import *; print('TruthGPT instalado correctamente')"
```

### Configuración Inicial

```python
# config/initial_config.yaml
model:
  name: "microsoft/DialoGPT-medium"
  max_length: 512
  temperature: 0.7

optimization:
  use_mixed_precision: true
  use_gradient_checkpointing: true
  use_flash_attention: true

training:
  batch_size: 4
  learning_rate: 5e-5
  num_epochs: 3
```

## 🎯 Uso Básico

### 1. Importación y Configuración

```python
from optimization_core import (
    ModernTruthGPTOptimizer, 
    TruthGPTConfig,
    create_ultra_optimization_core
)

# Configuración básica
config = TruthGPTConfig(
    model_name="microsoft/DialoGPT-medium",
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    use_flash_attention=True
)

# Inicializar optimizador
optimizer = ModernTruthGPTOptimizer(config)
```

### 2. Generación de Texto

```python
# Generación simple
text = optimizer.generate(
    input_text="Hola, ¿cómo estás?",
    max_length=100,
    temperature=0.7
)
print(text)

# Generación con parámetros personalizados
text = optimizer.generate(
    input_text="Explica la inteligencia artificial",
    max_length=200,
    temperature=0.5,
    top_p=0.9,
    repetition_penalty=1.1
)
```

### 3. Entrenamiento Básico

```python
from optimization_core import create_training_pipeline

# Crear pipeline de entrenamiento
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    experiment_name="mi_experimento",
    use_wandb=True
)

# Preparar datos
train_data = [...]  # Tu dataset
val_data = [...]    # Datos de validación

# Entrenar
results = pipeline.train(train_data, val_data)
```

## ⚙️ Configuración Avanzada

### Configuración de Optimizaciones

```python
from optimization_core import (
    UltraOptimizationCore,
    create_ultra_optimization_core
)

# Configuración ultra optimizada
ultra_config = {
    "use_quantization": True,
    "use_kernel_fusion": True,
    "use_memory_pooling": True,
    "use_adaptive_precision": True
}

# Crear optimizador ultra
ultra_optimizer = create_ultra_optimization_core(ultra_config)
```

### Configuración de GPU

```python
from optimization_core import GPUAccelerator, create_gpu_accelerator

# Configurar aceleración GPU
gpu_config = {
    "cuda_device": 0,
    "memory_fraction": 0.8,
    "use_tensor_cores": True,
    "use_mixed_precision": True
}

# Crear acelerador GPU
gpu_accelerator = create_gpu_accelerator(gpu_config)
```

### Configuración de Memoria

```python
from optimization_core import MemoryOptimizer, create_memory_optimizer

# Configurar optimización de memoria
memory_config = {
    "use_gradient_checkpointing": True,
    "use_activation_checkpointing": True,
    "memory_efficient_attention": True,
    "use_offload": True
}

# Crear optimizador de memoria
memory_optimizer = create_memory_optimizer(memory_config)
```

## 🚀 Optimizaciones

### 1. Optimización de Velocidad

```python
from optimization_core import (
    UltraFastOptimizer,
    create_ultra_fast_optimizer
)

# Configuración de velocidad ultra
speed_config = {
    "use_parallel_processing": True,
    "use_batch_optimization": True,
    "use_kernel_fusion": True,
    "use_quantization": True
}

# Crear optimizador ultra rápido
speed_optimizer = create_ultra_fast_optimizer(speed_config)
```

### 2. Optimización de Memoria

```python
from optimization_core import (
    MemoryPoolingOptimizer,
    create_memory_pooling_optimizer
)

# Configuración de pooling de memoria
pooling_config = {
    "pool_size": 1024,
    "use_activation_cache": True,
    "use_gradient_cache": True,
    "compression_ratio": 0.5
}

# Crear optimizador de pooling
pooling_optimizer = create_memory_pooling_optimizer(pooling_config)
```

### 3. Optimización de Compilación

```python
from optimization_core import (
    CompilerCore,
    create_compiler_core
)

# Configuración de compilación
compiler_config = {
    "target": "cuda",
    "optimization_level": "O3",
    "use_jit": True,
    "use_aot": True
}

# Crear compilador
compiler = create_compiler_core(compiler_config)
```

## 🔧 Casos de Uso Avanzados

### 1. Entrenamiento Distribuido

```python
from optimization_core import (
    DistributedOptimizer,
    create_distributed_optimizer
)

# Configuración distribuida
distributed_config = {
    "num_nodes": 4,
    "gpus_per_node": 8,
    "strategy": "ddp",
    "use_gradient_accumulation": True
}

# Crear optimizador distribuido
distributed_optimizer = create_distributed_optimizer(distributed_config)
```

### 2. Fine-tuning con LoRA

```python
from optimization_core import (
    LoRAConfig,
    create_lora_optimizer
)

# Configuración LoRA
lora_config = {
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "use_peft": True
}

# Crear optimizador LoRA
lora_optimizer = create_lora_optimizer(lora_config)
```

### 3. Quantización Avanzada

```python
from optimization_core import (
    AdvancedQuantizationOptimizer,
    create_quantization_optimizer
)

# Configuración de quantización
quantization_config = {
    "quantization_type": "int8",
    "use_dynamic_quantization": True,
    "use_static_quantization": True,
    "calibration_dataset": calibration_data
}

# Crear optimizador de quantización
quantization_optimizer = create_quantization_optimizer(quantization_config)
```

## 📊 Monitoreo y Logging

### 1. Configuración de WandB

```python
import wandb
from optimization_core import create_training_pipeline

# Inicializar WandB
wandb.init(
    project="truthgpt-experiments",
    config={
        "model_name": "microsoft/DialoGPT-medium",
        "learning_rate": 5e-5,
        "batch_size": 4
    }
)

# Crear pipeline con logging
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    use_wandb=True,
    experiment_name="mi_experimento"
)
```

### 2. Configuración de TensorBoard

```python
from optimization_core import create_training_pipeline

# Crear pipeline con TensorBoard
pipeline = create_training_pipeline(
    model_name="microsoft/DialoGPT-medium",
    use_tensorboard=True,
    log_dir="./logs"
)
```

## 🐛 Troubleshooting

### Problemas Comunes

#### 1. Error de Memoria GPU

```python
# Solución: Reducir batch size y usar gradient checkpointing
config = TruthGPTConfig(
    batch_size=1,  # Reducir batch size
    use_gradient_checkpointing=True,
    use_mixed_precision=True
)
```

#### 2. Error de CUDA

```python
# Verificar disponibilidad de CUDA
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión CUDA: {torch.version.cuda}")

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### 3. Error de Dependencias

```bash
# Reinstalar dependencias
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Logs y Debugging

```python
# Habilitar logging detallado
import logging
logging.basicConfig(level=logging.DEBUG)

# Configurar optimizador con debugging
config = TruthGPTConfig(
    debug_mode=True,
    verbose_logging=True,
    save_intermediate_results=True
)
```

## 📈 Mejores Prácticas

### 1. Configuración de Hardware

- **GPU**: Usar GPU con al menos 8GB VRAM
- **RAM**: Mínimo 16GB RAM
- **CPU**: Procesador multi-core recomendado

### 2. Configuración de Software

- **Python**: Usar Python 3.8+
- **PyTorch**: Usar PyTorch 2.0+
- **CUDA**: Usar CUDA 11.8+

### 3. Optimización de Rendimiento

- Usar mixed precision training
- Habilitar gradient checkpointing
- Usar flash attention
- Configurar batch size apropiado

## 🎯 Próximos Pasos

1. **Experimenta** con diferentes configuraciones
2. **Optimiza** según tu hardware
3. **Monitorea** el rendimiento
4. **Escala** según tus necesidades

---

*¡Ahora estás listo para usar TruthGPT de manera efectiva! Consulta los tutoriales y ejemplos para casos de uso más específicos.*


