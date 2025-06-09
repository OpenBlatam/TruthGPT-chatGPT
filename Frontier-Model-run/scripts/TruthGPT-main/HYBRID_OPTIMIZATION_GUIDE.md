# Hybrid Optimization Core Guide

## Overview

The Hybrid Optimization Core provides advanced optimization techniques that combine multiple optimization strategies and use candidate selection to choose the best performing variants for neural network models.

## Key Features

### 1. Candidate Selection
- **Tournament Selection**: Selects best candidates through tournament-style competition
- **Roulette Wheel Selection**: Probabilistic selection based on fitness scores
- **Rank-based Selection**: Selection based on relative ranking of candidates

### 2. Optimization Strategies
- **Kernel Fusion**: Combines operations for better performance
- **Quantization**: Reduces precision for memory and speed improvements
- **Memory Pooling**: Efficient memory management and reuse
- **Attention Fusion**: Optimizes attention mechanisms

### 3. Ensemble Optimization
- Combines multiple optimization strategies
- Evaluates strategy combinations for optimal results
- Adaptive learning from optimization history

## Usage Examples

### Basic Usage

```python
from optimization_core import create_hybrid_optimization_core
import torch.nn as nn

# Create hybrid optimization core
hybrid_core = create_hybrid_optimization_core()

# Define your model
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.LayerNorm(256),
    nn.Linear(256, 64)
)

# Apply hybrid optimization
optimized_model, result = hybrid_core.hybrid_optimize_module(model)

print(f"Selected strategy: {result['selected_strategy']}")
print(f"Performance improvement: {result['performance_metrics']}")
```

### Custom Configuration

```python
config = {
    'enable_candidate_selection': True,
    'enable_ensemble_optimization': True,
    'num_candidates': 8,
    'tournament_size': 4,
    'selection_strategy': 'tournament',
    'optimization_strategies': [
        'kernel_fusion', 
        'quantization', 
        'memory_pooling', 
        'attention_fusion'
    ],
    'objective_weights': {
        'speed': 0.5,
        'memory': 0.3,
        'accuracy': 0.2
    }
}

hybrid_core = create_hybrid_optimization_core(config)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_candidate_selection` | bool | True | Enable candidate selection algorithms |
| `enable_tournament_selection` | bool | True | Use tournament selection |
| `enable_ensemble_optimization` | bool | True | Enable ensemble optimization strategies |
| `num_candidates` | int | 5 | Number of optimization candidates to generate |
| `tournament_size` | int | 3 | Size of tournament for selection |
| `selection_strategy` | str | "tournament" | Selection algorithm ("tournament", "roulette", "rank") |
| `optimization_strategies` | List[str] | ["kernel_fusion", "quantization", "memory_pooling", "attention_fusion"] | Available optimization strategies |

## Performance Metrics

The hybrid optimization system tracks three key metrics:

1. **Speed Improvement**: Ratio of optimized vs original execution time
2. **Memory Efficiency**: Ratio of memory usage reduction
3. **Accuracy Preservation**: Ratio of maintained model accuracy

## Integration with Existing Optimizations

The hybrid optimization core integrates seamlessly with existing optimization modules:

- Uses `advanced_kernel_fusion` for kernel fusion strategies
- Uses `advanced_quantization` for quantization strategies  
- Uses `memory_pooling` for memory optimization strategies
- Uses `advanced_attention_fusion` for attention optimization strategies

## Best Practices

1. **Start with default configuration** for most use cases
2. **Adjust objective weights** based on your priorities (speed vs memory vs accuracy)
3. **Use ensemble optimization** for complex models that benefit from multiple strategies
4. **Monitor optimization reports** to understand which strategies work best for your models
5. **Experiment with different selection strategies** for different model types

## Integration with Enhanced Model Optimizer

The hybrid optimization can be enabled in the enhanced model optimizer:

```python
from enhanced_model_optimizer import EnhancedModelOptimizer

optimizer = EnhancedModelOptimizer({
    'enable_hybrid_optimization': True,
    'enable_candidate_selection': True,
    'enable_ensemble_optimization': True,
    'num_candidates': 5,
    'hybrid_strategies': ['kernel_fusion', 'quantization', 'memory_pooling', 'attention_fusion']
})

optimized_model = optimizer.optimize_model(model)
```
