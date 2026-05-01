# TruthGPT Adapters - Universal Integration Layer

## 🎯 Overview

The TruthGPT Adapters module provides a universal integration layer for converting models between different frameworks and optimizing them with TruthGPT's advanced capabilities.

## 🚀 Key Features

### 1. **Universal Framework Conversion**
- PyTorch ↔️ TensorRT
- PyTorch ↔️ ONNX
- Support for multiple optimization levels
- Preserve accuracy during conversion

### 2. **TruthGPT Integration**
- Seamless optimization
- Performance monitoring
- Automatic optimization application
- Model information extraction

### 3. **Production Ready**
- Error handling
- Comprehensive logging
- Validation support
- Performance metrics

## 📦 Quick Start

### Basic Usage

```python
from optimization_core.utils.truthgpt_adapters import (
    create_truthgpt_adapter,
    adapt_model,
    FrameworkType
)
import torch.nn as nn

# Create a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 10)
    
    def forward(self, x):
        return self.linear(x)

model = MyModel()

# Adapt with TruthGPT
truthgpt_model = create_truthgpt_adapter(model)

# Get model info
info = truthgpt_model.get_model_info()
print(info)

# Use adapted model
output = truthgpt_model(input_tensor)
```

### Framework Conversion

```python
from optimization_core.utils.truthgpt_adapters import (
    adapt_model,
    FrameworkType,
    AdapterConfig,
    OptimizationLevel
)

# Configure adapter
config = AdapterConfig(
    source_framework=FrameworkType.PYTORCH,
    target_framework=FrameworkType.ONNX,
    optimization_level=OptimizationLevel.ADVANCED,
    enable_quantization=True,
    preserve_accuracy=True
)

# Adapt model
adapted_model = adapt_model(
    model=my_pytorch_model,
    source="pytorch",
    target="onnx",
    config=config
)

# Export to ONNX
truthgpt_adapter = create_truthgpt_adapter(adapted_model)
truthgpt_adapter.export_to_onnx("model.onnx", input_shape=(1, 100))
```

### Advanced Usage

```python
from optimization_core.utils.truthgpt_adapters import (
    create_universal_adapter,
    FrameworkType,
    AdapterConfig
)

# Create universal adapter
adapter = create_universal_adapter()

# Get adapter information
info = adapter.get_adapter_info()
print(info)

# Adapt between frameworks
onnx_model = adapter.adapt(
    model=my_pytorch_model,
    source=FrameworkType.PYTORCH,
    target=FrameworkType.ONNX
)
```

## 🎨 Component Overview

### 1. **BaseAdapter**
Abstract base class for all adapters.
- Handles configuration
- Provides common interface
- Error handling
- Logging support

### 2. **PyTorchToTensorRTAdapter**
Converts PyTorch models to TensorRT.
- Optimized inference
- Quantization support
- Validation support

### 3. **PyTorchToONNXAdapter**
Converts PyTorch models to ONNX.
- Cross-platform compatibility
- ONNX Runtime optimization
- Dynamic shapes support

### 4. **UniversalAdapter**
Universal adapter for any framework.
- Multiple framework support
- Automatic adapter selection
- Flexible configuration

### 5. **TruthGPTModelAdapter**
TruthGPT-specific model adapter.
- Automatic optimizations
- Performance monitoring
- ONNX export
- Model information

## 🔧 Configuration

### AdapterConfig

```python
from optimization_core.utils.truthgpt_adapters import (
    AdapterConfig,
    FrameworkType,
    OptimizationLevel
)

config = AdapterConfig(
    source_framework=FrameworkType.PYTORCH,
    target_framework=FrameworkType.TENSORRT,
    optimization_level=OptimizationLevel.ADVANCED,
    preserve_accuracy=True,
    enable_quantization=False,
    enable_pruning=True,
    batch_size=32,
    device=torch.device('cuda'),
    metadata={'input_shape': [224, 224]}
)
```

### Optimization Levels

1. **BASIC**: Minimal optimization
2. **ADVANCED**: Balanced optimization
3. **AGGRESSIVE**: Maximum optimization
4. **PRODUCTION**: Production-ready optimization

## 📊 Performance Monitoring

```python
# Create adapter with monitoring
adapter = create_truthgpt_adapter(
    model=my_model,
    enable_optimizations=True,
    enable_monitoring=True
)

# Forward pass with automatic monitoring
output = adapter(input_tensor)

# Get performance info
info = adapter.get_model_info()
print(f"Total parameters: {info['total_parameters']}")
print(f"Optimizations: {info['optimizations_enabled']}")
```

## 🎯 Best Practices

### 1. **Model Optimization**
```python
# Optimize before deploying
adapter = create_truthgpt_adapter(
    model=my_model,
    enable_optimizations=True
)

# Export optimized model
adapter.export_to_onnx("optimized_model.onnx", input_shape=(1, 100))
```

### 2. **Framework Conversion**
```python
# Use appropriate adapter for your needs
if need_speed:
    # Use TensorRT
    adapter = adapt_model(
        model, source="pytorch", target="tensorrt"
    )
elif need_cross_platform:
    # Use ONNX
    adapter = adapt_model(
        model, source="pytorch", target="onnx"
    )
```

### 3. **Error Handling**
```python
try:
    adapter = adapt_model(
        model, source="pytorch", target="onnx"
    )
except ValueError as e:
    print(f"Conversion failed: {e}")
    # Fallback to original model
    adapter = create_truthgpt_adapter(model)
```

## 🔬 Advanced Features

### Custom Transformations

```python
from optimization_core.utils.truthgpt_adapters import AdapterConfig

def custom_transform(model):
    # Apply custom transformations
    model.eval()
    return model

config = AdapterConfig(
    source_framework=FrameworkType.PYTORCH,
    target_framework=FrameworkType.ONNX,
    custom_transforms=[custom_transform]
)
```

### Metadata Passing

```python
config = AdapterConfig(
    source_framework=FrameworkType.PYTORCH,
    target_framework=FrameworkType.TENSORRT,
    metadata={
        'input_shape': [1, 3, 224, 224],
        'input_names': ['input'],
        'output_names': ['output']
    }
)
```

## 🧪 Testing

### Unit Tests

```python
import unittest
from optimization_core.utils.truthgpt_adapters import *

class TestAdapters(unittest.TestCase):
    def test_pytorch_to_onnx(self):
        # Create model
        model = nn.Linear(100, 10)
        
        # Adapt
        adapter = adapt_model(model, "pytorch", "onnx")
        
        # Validate
        self.assertIsNotNone(adapter)
    
    def test_truthgpt_adapter(self):
        # Create model
        model = nn.Linear(100, 10)
        
        # Adapt
        adapter = create_truthgpt_adapter(model)
        
        # Get info
        info = adapter.get_model_info()
        
        # Assert
        self.assertIn('total_parameters', info)
```

## 📚 API Reference

### Main Functions

- `create_universal_adapter()`: Create universal adapter
- `create_truthgpt_adapter()`: Create TruthGPT model adapter
- `adapt_model()`: Adapt model between frameworks

### Classes

- `BaseAdapter`: Base adapter class
- `PyTorchToTensorRTAdapter`: PyTorch to TensorRT
- `PyTorchToONNXAdapter`: PyTorch to ONNX
- `UniversalAdapter`: Universal adapter
- `TruthGPTModelAdapter`: TruthGPT adapter

### Configuration

- `AdapterConfig`: Adapter configuration dataclass
- `FrameworkType`: Supported frameworks
- `OptimizationLevel`: Optimization levels

## 🚀 Deployment

### Production Deployment

```python
# Create production-ready adapter
adapter = create_truthgpt_adapter(
    model=production_model,
    enable_optimizations=True,
    enable_monitoring=True
)

# Export for deployment
adapter.export_to_onnx(
    "production_model.onnx",
    input_shape=(1, 100)
)

# Use in production
output = adapter(input_tensor)
```

### Edge Deployment

```python
# Optimize for edge devices
edge_adapter = adapt_model(
    model=my_model,
    source="pytorch",
    target="tflite",  # TensorFlow Lite for edge
    config=AdapterConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_quantization=True
    )
)
```

## 🙏 Contributing

Contributions are welcome! Please follow the project guidelines.

## 📝 License

Part of the TruthGPT optimization core project.

---

*For more information, see the [main documentation](../docs/README.md)*


