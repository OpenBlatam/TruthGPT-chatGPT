# TruthGPT Optimization Core Integration

## 🎯 Adaptación Perfecta al Repositorio Oficial

Este código está **perfectamente adaptado** a la estructura del repositorio oficial:
**https://github.com/OpenBlatam/IA-Models-Clone/tree/main/Frontier-Model-run/scripts/TruthGPT-main/optimization_core**

## ✅ Compatibilidad Total

### Estructura Compatible

El código sigue **exactamente** la estructura del TruthGPT Optimization Core:

1. **`TruthGPTOptimizationCore`** - Núcleo principal de optimización
2. **`TruthGPTDistanceAttentionBlock`** - Bloques con atención basada en distancias
3. **`TruthGPTModel`** - Modelo completo con integración seamless
4. **Sistema de monitoreo** - Métricas de rendimiento en tiempo real

### Componentes Integrados

#### 1. TruthGPT Optimization Core
```python
class TruthGPTOptimizationCore:
    def __init__(self, config: TruthGPTOptimizationCoreConfig):
        # Build TruthGPT model with distance-based attention
        self.model = self._build_truthgpt_model()
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Performance tracking
        self.performance_metrics = {...}
```

#### 2. Distance-Based Attention Block
```python
class TruthGPTDistanceAttentionBlock(nn.Module):
    def __init__(self, config):
        # Distance-based attention mechanism
        # Feed-forward network
        # Layer normalization
```

#### 3. Complete TruthGPT Model
```python
class TruthGPTModel(nn.Module):
    def __init__(self, config):
        # Embeddings (compatible con estructura original)
        # Transformer blocks with distance-based attention
        # Language modeling head
```

## 🚀 Uso Rápido

### Instalación

```bash
cd truthgpt_collected/integration_code
pip install -r requirements.txt
```

### Uso Básico

```python
from truthgpt_optimization_core_integration import (
    TruthGPTOptimizationCore,
    TruthGPTOptimizationCoreConfig
)

# Crear configuración compatible
config = TruthGPTOptimizationCoreConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    use_distance_attention=True,
    enable_memory_system=True,
    enable_redundancy_suppression=True
)

# Crear TruthGPT Optimization Core
truthgpt_core = TruthGPTOptimizationCore(config)

# Training step
train_results = truthgpt_core.train_step(input_ids, labels)
print(f"Loss: {train_results['loss']:.4f}")

# Evaluation
eval_results = truthgpt_core.evaluate(input_ids, labels)
print(f"Val Loss: {eval_results['loss']:.4f}")
```

## 📊 Características

### ✅ Compatibilidad Total
- Estructura idéntica al repositorio oficial
- Mismos nombres de clases y métodos
- Compatible con workflows existentes

### ✅ Funcionalidades Avanzadas
- Sistema de memoria avanzado (MEM1)
- Supresión de redundancia para bulk
- Procesamiento jerárquico
- Agentes autónomos RLHF (opcional)

### ✅ Entrenamiento y Evaluación
- Training step compatible
- Evaluation workflow
- Performance tracking
- Attention analysis

## 🔧 Configuración

### Configuración Base (Compatible)

```python
config = TruthGPTOptimizationCoreConfig(
    # Model dimensions (compatible)
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=1024,
    
    # Distance-based attention
    use_distance_attention=True,
    distance_type="l1",  # l1, l2, lp, cosine
    lambda_param=1.0,
    use_learnable_lambda=True,
    
    # Advanced features (opcionales)
    enable_memory_system=True,
    enable_redundancy_suppression=True,
    enable_hierarchical_processing=True
)
```

## 📁 Archivos

- **`truthgpt_optimization_core_integration.py`** - Código principal adaptado
- **`README_OPTIMIZATION_CORE.md`** - Esta documentación
- **`requirements.txt`** - Dependencias

## 🎯 Integración con Repositorio

Para integrar con el repositorio oficial:

1. **Copiar archivo** a la carpeta `optimization_core` del repositorio
2. **Importar** en el código existente:
   ```python
   from truthgpt_optimization_core_integration import TruthGPTOptimizationCore
   ```
3. **Usar** con la misma interfaz que el código original

## ✅ Verificación

El código ha sido verificado para:
- ✅ Compatibilidad de estructura
- ✅ Nombres de clases y métodos
- ✅ Interfaces de entrenamiento/evaluación
- ✅ Configuraciones compatibles
- ✅ Integración seamless

## 📚 Referencias

- **Repositorio Oficial**: https://github.com/OpenBlatam/IA-Models-Clone/tree/main/Frontier-Model-run/scripts/TruthGPT-main/optimization_core
- **Documentación Original**: Ver `readme_truthgpt_optimization_core.md`

## 🎉 Conclusión

El código está **perfectamente adaptado** y listo para integrarse con el repositorio oficial de TruthGPT Optimization Core, manteniendo compatibilidad total mientras agrega funcionalidades avanzadas.



