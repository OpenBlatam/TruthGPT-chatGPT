# Quick Start - TruthGPT Advanced Integration

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
cd truthgpt_collected/integration_code
pip install -r requirements.txt
```

### 2. Uso Básico

```python
from truthgpt_advanced_integration import TruthGPTAdvanced, TruthGPTAdvancedConfig

# Crear configuración
config = TruthGPTAdvancedConfig(
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    use_bulk_processing=True,
    enable_autonomous_agents=True,
    enable_memory_system=True
)

# Crear modelo
model = TruthGPTAdvanced(config)

# Procesar inputs
import torch
inputs = torch.randn(4, 32, 512)  # [batch, seq_len, hidden_dim]
outputs = model(inputs, use_memory=True, suppress_redundancy=True)

print(f"Output shape: {outputs['output'].shape}")
```

### 3. Ejecutar Ejemplos

```bash
cd truthgpt_collected/integration_code
python example_usage.py
```

## 📚 Documentación

- `README_INTEGRATION.md` - Documentación completa
- `INTEGRATION_SUMMARY.md` - Resumen de componentes
- `PAPERS_AND_REPOS.md` - Lista de papers y repositorios

## 🎯 Características Principales

1. **Memoria Avanzada**: Almacenamiento y recuperación contextual
2. **Supresión de Redundancia**: Para procesamiento masivo
3. **Agentes Autónomos**: Con RLHF para tareas largas
4. **Procesamiento Jerárquico**: Multi-nivel para diferentes documentos

## 💡 Ejemplos Incluidos

- Uso básico
- Sistema de memoria
- Supresión de redundancia
- Agentes autónomos RLHF
- Entrenamiento completo
- Pipeline completo

Ver `example_usage.py` para más detalles.
