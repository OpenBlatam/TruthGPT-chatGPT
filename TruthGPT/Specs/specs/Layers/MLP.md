## 🔧 MLP Module

This module defines a Multi-Layer Perceptron (MLP) commonly used as the Position-wise Feed-Forward Network (FFN) in Transformer architectures.

Modular Design:
Component Reusability: Each part (layer, optimizer, loss function, training/evaluation pipelines) can be easily replaced, making the system flexible.

Configurable: Using a configuration-driven approach, you can modify and extend the model without changing the core logic.

Scalable: It’s easier to scale by adding new components (e.g., new layers, optimizers, and loss functions).

Testable: Each part of the system can be unit tested independently, ensuring high test coverage and reliability.

Extensible: You can easily add new components or replace existing ones, e.g., adding support for new optimizers or loss functions.

Layer Independence:

The Transformer architecture can be modularized as a distinct component in the system.

If you change the Transformer architecture, for example, from a standard Vanilla Transformer to a GPT-style Transformer or a T5-style model, you can update the implementation of the TransformerLayer class without needing to change the rest of the system.

```python
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### 📥 Parameters

| Parameter   | Type   | Description                                      |
|-------------|--------|--------------------------------------------------|
| `input_dim` | `int`  | Dimensionality of input and output features.     |
| `hidden_dim`| `int`  | Size of the hidden layer (usually 4× input_dim). |
| `dropout`   | `float`| Dropout probability (default: `0.1`).            |


Input / Output Shape
Input: (batch_size, seq_len, input_dim)

Output: (batch_size, seq_len, input_dim)