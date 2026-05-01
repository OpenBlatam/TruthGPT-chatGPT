---
title: "Distance Attention Documentation"
category: "08_ai_artificial_intelligence"
tags: ["ai", "artificial-intelligence"]
created: "2025-10-29"
path: "08_ai_artificial_intelligence/Other/distance_attention_documentation.md"
---

# Distance-Based Attention Mechanism Documentation

## Overview

This module implements a novel distance-based attention mechanism that replaces the traditional scaled dot-product attention with distance-based computations. The implementation is based on the mathematical formulation where attention weights are computed using distance metrics between queries and keys.

## Mathematical Foundation

### Traditional Attention
The standard scaled dot-product attention is defined as:
```
α = softmax(QK^T / √d_k)
O = αV
```

### Distance-Based Attention
Our novel approach uses distance-based computations:
```
L_ij = -distance(Q_i, K_j)
α_new = softmax(λL / √d_k)
O = α_new V
```

Where:
- `L` is the distance matrix
- `λ` is a tuning parameter
- `distance()` can be L1, L2, Lp, cosine, or Chebyshev distance

## Key Features

### 1. Multiple Distance Metrics
- **L1 (Manhattan) Distance**: `||Q_i - K_j||_1`
- **L2 (Euclidean) Distance**: `||Q_i - K_j||_2`
- **Lp Distance**: `||Q_i - K_j||_p` for any p ≥ 1
- **Cosine Distance**: `1 - cosine_similarity(Q_i, K_j)`
- **Chebyshev Distance**: `max|Q_i - K_j|`

### 2. Configurable Parameters
- **Lambda Parameter (λ)**: Controls attention scaling
- **Learnable Lambda**: Optional learnable λ parameter
- **P-Norm**: Configurable p value for Lp distance
- **Multi-head Support**: Full multi-head attention support

### 3. Integration Features
- **TruthGPT Integration**: Drop-in replacement for standard attention
- **Performance Monitoring**: Built-in performance tracking
- **Gradient Flow**: Proper gradient computation and flow
- **Numerical Stability**: Robust handling of edge cases

## Installation and Setup

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib seaborn
```

### Basic Usage

#### 1. Simple Distance-Based Attention
```python
from distance_based_attention import DistanceBasedAttention, DistanceType

# Create attention mechanism
attention = DistanceBasedAttention(
    hidden_size=768,
    num_heads=12,
    distance_type=DistanceType.L1,
    lambda_param=1.0
)

# Forward pass
x = torch.randn(2, 128, 768)  # (batch_size, seq_len, hidden_size)
output, attn_weights = attention(x, output_attentions=True)
```

#### 2. Multi-Head Distance Attention
```python
from distance_based_attention import MultiHeadDistanceAttention

# Create multi-head attention with residual connections
attention = MultiHeadDistanceAttention(
    hidden_size=768,
    num_heads=12,
    distance_type=DistanceType.L2,
    lambda_param=1.0,
    use_residual=True,
    use_layer_norm=True
)

# Forward pass
output, attn_weights = attention(x, output_attentions=True)
```

#### 3. Configuration-Based Creation
```python
from distance_based_attention import DistanceAttentionConfig, create_distance_attention

# Create configuration
config = DistanceAttentionConfig(
    hidden_size=768,
    num_heads=12,
    distance_type="l1",
    lambda_param=1.0,
    use_learnable_lambda=True
)

# Create attention mechanism
attention = create_distance_attention(config)
```

## TruthGPT Integration

### 1. Basic Integration
```python
from truthgpt_distance_attention_integration import TruthGPTDistanceAttentionModel, TruthGPTAttentionConfig

# Create TruthGPT configuration
config = TruthGPTAttentionConfig(
    hidden_size=768,
    num_heads=12,
    num_layers=6,
    vocab_size=1000,
    distance_type="l1",
    lambda_param=1.0
)

# Create model
model = TruthGPTDistanceAttentionModel(config)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 128))
attention_mask = torch.ones(2, 128)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True
)
```

### 2. Performance Monitoring
```python
# Enable performance monitoring
config.enable_performance_monitoring = True

# After training/inference, get performance stats
perf_stats = model.get_performance_stats()
print(f"Layer 0 average forward time: {perf_stats['layer_0']['avg_forward_time']:.4f}s")
```

### 3. Lambda Parameter Optimization
```python
from truthgpt_distance_attention_integration import DistanceAttentionOptimizer

# Create optimizer
optimizer = DistanceAttentionOptimizer(model)

# Optimize lambda parameters
optimized_lambdas = optimizer.optimize_lambda_parameters(dataloader, num_epochs=5)
print(f"Optimized lambda values: {optimized_lambdas}")
```

## Advanced Usage

### 1. Custom Distance Metrics
```python
class CustomDistanceAttention(DistanceBasedAttention):
    def compute_distance_matrix(self, Q, K):
        # Implement custom distance computation
        # Example: Weighted L2 distance
        weights = torch.ones_like(Q[..., 0:1])  # Example weights
        Q_weighted = Q * weights
        K_weighted = K * weights
        
        Q_expanded = Q_weighted.unsqueeze(-2)
        K_expanded = K_weighted.unsqueeze(-3)
        distances = torch.sqrt(torch.sum((Q_expanded - K_expanded) ** 2, dim=-1) + 1e-8)
        
        return -distances
```

### 2. Learnable Lambda Parameters
```python
# Enable learnable lambda
attention = DistanceBasedAttention(
    hidden_size=768,
    num_heads=12,
    distance_type=DistanceType.L1,
    lambda_param=1.0
)

attention.set_learnable_lambda(True)

# Lambda parameter is now learnable
print(f"Learnable lambda: {attention.learnable_lambda}")
```

### 3. Attention Pattern Analysis
```python
# Analyze attention patterns
optimizer = DistanceAttentionOptimizer(model)
attention_analysis = optimizer.analyze_attention_patterns(dataloader, num_samples=100)

for layer_idx, analysis in attention_analysis.items():
    print(f"Layer {layer_idx}:")
    print(f"  Mean attention: {analysis['mean_attention']:.4f}")
    print(f"  Attention entropy: {analysis['attention_entropy']:.4f}")
```

## Testing and Validation

### 1. Run Comprehensive Tests
```python
from distance_attention_tests import DistanceAttentionTester, TestConfig

# Create test configuration
config = TestConfig(
    batch_size=4,
    seq_len=128,
    hidden_size=768,
    num_heads=12,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Run all tests
tester = DistanceAttentionTester(config)
results = tester.run_all_tests()

# Print summary
summary = results['summary']
print(f"Success Rate: {summary['success_rate']:.2%}")
```

### 2. Unit Tests
```python
import unittest
from distance_attention_tests import TestDistanceAttention

# Run unit tests
unittest.main(argv=[''], exit=False, verbosity=2)
```

### 3. Performance Benchmarking
```python
# Benchmark against standard attention
results = tester.benchmark_performance()

for distance_type, metrics in results.items():
    print(f"{distance_type}:")
    print(f"  Speedup: {metrics['speedup']:.2f}x")
    print(f"  Memory ratio: {metrics['memory_ratio']:.2f}")
```

## Configuration Options

### DistanceAttentionConfig Parameters
- `hidden_size`: Model hidden size (default: 768)
- `num_heads`: Number of attention heads (default: 12)
- `distance_type`: Distance metric ("l1", "l2", "lp", "cosine", "chebyshev")
- `lambda_param`: Tuning parameter λ (default: 1.0)
- `p_norm`: P value for Lp distance (default: 2.0)
- `dropout`: Dropout probability (default: 0.1)
- `use_learnable_lambda`: Enable learnable λ (default: False)
- `use_residual`: Use residual connections (default: True)
- `use_layer_norm`: Use layer normalization (default: True)

### TruthGPTAttentionConfig Parameters
- `hidden_size`: Model hidden size (default: 768)
- `num_heads`: Number of attention heads (default: 12)
- `num_layers`: Number of transformer layers (default: 12)
- `vocab_size`: Vocabulary size (default: 1000)
- `max_position_embeddings`: Maximum sequence length (default: 2048)
- `distance_type`: Distance metric (default: "l1")
- `lambda_param`: Tuning parameter λ (default: 1.0)
- `enable_performance_monitoring`: Enable performance tracking (default: True)
- `log_attention_weights`: Log attention weights (default: False)

## Performance Considerations

### 1. Computational Complexity
- **L1 Distance**: O(d) per attention pair
- **L2 Distance**: O(d) per attention pair
- **Lp Distance**: O(d) per attention pair
- **Cosine Distance**: O(d) per attention pair
- **Chebyshev Distance**: O(d) per attention pair

Where d is the head dimension.

### 2. Memory Usage
- Distance matrix storage: O(N²) where N is sequence length
- Similar to standard attention memory requirements
- Consider gradient checkpointing for long sequences

### 3. Optimization Tips
- Use appropriate batch sizes for your hardware
- Enable gradient checkpointing for memory efficiency
- Consider mixed precision training for speed
- Monitor performance statistics during training

## Troubleshooting

### Common Issues

#### 1. NaN or Inf Values
```python
# Check for numerical stability
tester = DistanceAttentionTester(config)
stability_results = tester.test_numerical_stability()

# If issues found, try:
# - Reducing lambda parameter
# - Using different distance metric
# - Adding numerical stability checks
```

#### 2. Gradient Flow Issues
```python
# Test gradient flow
gradient_results = tester.test_gradient_flow()

# If issues found, try:
# - Checking learning rate
# - Using gradient clipping
# - Verifying loss computation
```

#### 3. Performance Issues
```python
# Benchmark performance
perf_results = tester.benchmark_performance()

# If slow, try:
# - Using GPU acceleration
# - Reducing sequence length
# - Using fewer attention heads
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
attention = DistanceBasedAttention(
    hidden_size=768,
    num_heads=12,
    distance_type=DistanceType.L1,
    lambda_param=1.0
)
```

## Examples and Use Cases

### 1. Natural Language Processing
```python
# For NLP tasks, L1 distance often works well
config = DistanceAttentionConfig(
    hidden_size=768,
    num_heads=12,
    distance_type="l1",
    lambda_param=1.0
)
```

### 2. Computer Vision
```python
# For vision tasks, L2 distance might be more appropriate
config = DistanceAttentionConfig(
    hidden_size=512,
    num_heads=8,
    distance_type="l2",
    lambda_param=1.2
)
```

### 3. Time Series Analysis
```python
# For time series, cosine distance can capture patterns
config = DistanceAttentionConfig(
    hidden_size=256,
    num_heads=4,
    distance_type="cosine",
    lambda_param=0.8
)
```

## Research and Development

### 1. Experimenting with New Distance Metrics
```python
class CustomDistanceType(DistanceType):
    CUSTOM = "custom"

# Implement custom distance computation
def compute_custom_distance(Q, K):
    # Your custom distance implementation
    pass
```

### 2. Hyperparameter Tuning
```python
# Grid search for optimal parameters
lambda_values = [0.5, 1.0, 1.5, 2.0]
distance_types = ["l1", "l2", "cosine"]

best_config = None
best_performance = 0

for lambda_val in lambda_values:
    for dist_type in distance_types:
        config = DistanceAttentionConfig(
            distance_type=dist_type,
            lambda_param=lambda_val
        )
        
        # Train and evaluate
        performance = train_and_evaluate(config)
        
        if performance > best_performance:
            best_performance = performance
            best_config = config
```

### 3. Ablation Studies
```python
# Compare different distance metrics
distance_metrics = ["l1", "l2", "lp", "cosine", "chebyshev"]
results = {}

for metric in distance_metrics:
    config = DistanceAttentionConfig(distance_type=metric)
    model = create_distance_attention(config)
    
    # Train and evaluate
    results[metric] = train_and_evaluate(model)

# Analyze results
for metric, performance in results.items():
    print(f"{metric}: {performance:.4f}")
```

## Contributing

### 1. Adding New Distance Metrics
1. Add new distance type to `DistanceType` enum
2. Implement distance computation in `compute_distance_matrix`
3. Add tests for the new metric
4. Update documentation

### 2. Performance Optimizations
1. Profile the code to identify bottlenecks
2. Implement optimizations (e.g., vectorization, caching)
3. Add performance tests
4. Document performance improvements

### 3. Integration Improvements
1. Add support for new model architectures
2. Improve compatibility with existing frameworks
3. Add more configuration options
4. Enhance error handling and logging

## License and Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{distance_attention_2024,
  title={Distance-Based Attention Mechanism for Transformer Models},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository},
  url={https://github.com/your-repo/distance-attention}
}
```

## Support and Contact

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: your-email@example.com
- Documentation: [Link to documentation]

---

*This documentation is continuously updated. Please check for the latest version.*





