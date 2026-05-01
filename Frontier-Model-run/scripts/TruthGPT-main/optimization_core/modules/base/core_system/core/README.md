# 🏗️ Core Module - Foundation for Optimization Core

[![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)]()

Foundation module providing core utilities, interfaces, and base classes for the TruthGPT optimization core.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Components](#components)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Best Practices](#best-practices)

## 🚀 Quick Start

### Installation

Core module is part of optimization_core:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from core import (
    validate_non_empty_string,
    validate_file_path,
    detect_file_format,
    create_core_component
)

# Validation
validate_non_empty_string("model_name", "model_name")
validate_file_path("./data.json", must_exist=True)

# File operations
format = detect_file_format("./data.json")
# Returns: "json"

# Create components
service = create_core_component(
    "service",
    {"service_type": "training", "config": {...}}
)
```

## 🧩 Components

### 1. Validators

Input validation utilities for type checking and validation:

```python
from core import (
    validate_non_empty_string,
    validate_file_path,
    validate_positive_number,
    validate_generation_params
)

# String validation
validate_non_empty_string("model_name", "model_name")

# File path validation
validate_file_path("./data.json", must_exist=True, allowed_extensions=['.json'])

# Number validation
validate_positive_number(100, "batch_size", min_value=1, max_value=1024)

# Generation params validation
validate_generation_params({
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95
})
```

### 2. File Utils

File operations and format detection:

```python
from core import (
    detect_file_format,
    ensure_output_directory,
    get_file_size_mb,
    list_files
)

# Detect file format
format = detect_file_format("./data.json")
# Returns: "json"

# Ensure output directory exists
ensure_output_directory("./output/checkpoints")

# Get file size
size_mb = get_file_size_mb("./data.json")
# Returns: 1.5

# List files
files = list_files("./data", pattern="*.json")
```

### 3. Factory Base

Factory pattern implementations:

```python
from core import (
    BaseFactory,
    SimpleFactory,
    FactoryRegistry
)

# Create factory
factory = SimpleFactory({
    "optimizer": lambda: AdamW(...),
    "scheduler": lambda: CosineScheduler(...)
})

# Get component
optimizer = factory.create("optimizer")
```

### 4. Interfaces

Base classes and ABCs for modular architecture:

```python
from core.interfaces import (
    BaseModelManager,
    BaseEvaluator,
    BaseCheckpointManager,
    BaseTrainer
)

# Implement interface
class MyModelManager(BaseModelManager):
    def load_model(self, model_name: str, **kwargs):
        # Implementation
        pass
```

### 5. Configuration

Configuration management with validation:

```python
from core.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    load_config,
    save_config
)

# Create config
config = TrainingConfig(
    epochs=3,
    train_batch_size=8,
    learning_rate=5e-5
)

# Load from file
config = load_config("./config.yaml")

# Save to file
save_config(config, "./config.yaml")
```

### 6. Services

Base services for training, inference, and model management:

```python
from core.services import (
    TrainingService,
    InferenceService,
    ModelService
)

# Create service
service = TrainingService(config=training_config)

# Use service
service.train(model, train_loader, val_loader)
```

### 7. Adapters

Adapter pattern implementations:

```python
from core.adapters import (
    ModelAdapter,
    DataAdapter,
    OptimizerAdapter
)

# Create adapter
adapter = ModelAdapter(model=model)

# Adapt model
adapted_model = adapter.adapt(target_framework="onnx")
```

### 8. Composition

Workflow and component assembly:

```python
from core.composition import WorkflowBuilder

# Build workflow
workflow = WorkflowBuilder() \
    .add_step("load_data", load_data_fn) \
    .add_step("preprocess", preprocess_fn) \
    .add_step("train", train_fn) \
    .build()

# Execute workflow
result = workflow.execute()
```

## 📖 API Reference

### Factory Functions

#### `create_core_component(component_type, config)`

Unified factory for creating core components.

```python
# Create service
service = create_core_component(
    "service",
    {"service_type": "training", "config": {...}}
)

# Create adapter
adapter = create_core_component(
    "adapter",
    {"adapter_type": "model", "model": model}
)

# Create validator
validator = create_core_component(
    "validator",
    {"validator_type": "config"}
)

# Create workflow
workflow = create_core_component(
    "workflow",
    {"steps": [...]}
)
```

#### `list_available_core_components()`

List all available core component types.

```python
components = list_available_core_components()
# Returns: ['validators', 'file_utils', 'factory_base', ...]
```

#### `get_core_component_info(component_type)`

Get information about a core component.

```python
info = get_core_component_info("validators")
# Returns: {'name': 'validators', 'module': 'core.validators', ...}
```

## 🏗️ Architecture

### Module Structure

```
core/
├── __init__.py              # Main exports with lazy imports
├── validators.py            # Input validation utilities
├── file_utils.py            # File operations
├── factory_base.py          # Factory pattern
├── interfaces.py            # Base classes and ABCs
├── config.py                # Configuration management
├── services/                # Base services
│   ├── base_service.py
│   ├── training_service.py
│   ├── inference_service.py
│   └── model_service.py
├── adapters/                # Adapter pattern
│   ├── model_adapter.py
│   ├── data_adapter.py
│   └── optimizer_adapter.py
├── composition/             # Workflow assembly
│   ├── workflow_builder.py
│   └── component_assembler.py
├── validation/              # Validators
│   ├── config_validator.py
│   ├── data_validator.py
│   └── model_validator.py
└── framework/              # Framework components
    └── ...
```

### Lazy Loading

Core module uses lazy loading for heavy imports:

```python
# Eager imports (always loaded)
from core import validate_non_empty_string  # ✅ Fast

# Lazy imports (loaded on demand)
from core import interfaces  # ✅ Loaded when accessed
from core import config      # ✅ Loaded when accessed
```

## 💡 Examples

### Complete Workflow

```python
from core import (
    validate_file_path,
    detect_file_format,
    create_core_component,
    load_config
)

# Validate input
validate_file_path("./config.yaml", must_exist=True)

# Load configuration
config = load_config("./config.yaml")

# Create services
training_service = create_core_component(
    "service",
    {"service_type": "training", "config": config.training}
)

inference_service = create_core_component(
    "service",
    {"service_type": "inference", "config": config.inference}
)

# Use services
training_service.train(model, train_loader)
results = inference_service.infer(model, prompts)
```

### Custom Validator

```python
from core.validators import ValidationError

def validate_model_size(model, max_params: int):
    """Custom validator for model size."""
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > max_params:
        raise ValidationError(
            f"Model has {num_params} parameters, "
            f"exceeds maximum of {max_params}"
        )
```

### Factory Pattern

```python
from core import FactoryRegistry

# Register components
registry = FactoryRegistry()
registry.register("optimizer", lambda: AdamW(lr=1e-4))
registry.register("scheduler", lambda: CosineScheduler())

# Create components
optimizer = registry.create("optimizer")
scheduler = registry.create("scheduler")
```

## 🎯 Best Practices

### 1. Use Validators

Always validate inputs:

```python
from core import validate_non_empty_string, validate_positive_number

def train_model(model_name: str, batch_size: int):
    validate_non_empty_string(model_name, "model_name")
    validate_positive_number(batch_size, "batch_size", min_value=1)
    # ... rest of function
```

### 2. Use Factory Pattern

Create components using factory:

```python
service = create_core_component("service", config)
```

### 3. Lazy Loading

Import heavy modules only when needed:

```python
# ✅ Good - lazy import
from core import interfaces
manager = interfaces.BaseModelManager()

# ❌ Avoid - direct import (unless needed immediately)
from core.interfaces import BaseModelManager
```

### 4. Configuration Management

Use config classes for type safety:

```python
from core.config import TrainingConfig

config = TrainingConfig(
    epochs=3,
    train_batch_size=8,
    learning_rate=5e-5
)
```

## 📊 Component Registry

All components are registered in `CORE_COMPONENT_REGISTRY`:

```python
from core import CORE_COMPONENT_REGISTRY

for name, info in CORE_COMPONENT_REGISTRY.items():
    print(f"{name}: {info['description']}")
```

## 🔧 Configuration

### Validator Config

```python
{
    "strict": True,        # Strict validation mode
    "raise_on_error": True # Raise exception on validation error
}
```

### File Utils Config

```python
{
    "supported_formats": [".json", ".yaml", ".jsonl"],
    "max_file_size_mb": 1000
}
```

## 📚 Additional Resources

- [Interfaces Documentation](./interfaces.py)
- [Configuration Guide](./config.py)
- [Factory Pattern Guide](./factory_base.py)
- [Validation Guide](./validators.py)

---

**Version:** 2.0.0  
**Status:** ✅ Production Ready
