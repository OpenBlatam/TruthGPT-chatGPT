## 🔧 Layers in Modern Neural Architectures

State-of-the-art neural networks use a range of custom layers to improve performance, generalization, and efficiency. Below is a summary of the most common architectural components:

### 🧠 Attention Mechanisms
- **Multi-Head Attention** – Enables simultaneous attention to different parts of the input sequence.
- **Cross-Attention** – Allows a model to attend to a separate sequence (e.g., encoder-decoder architectures).
- **Relative Position Encodings** – Improves contextual understanding by encoding relative positions of tokens.

### 🚪 Gated Feedforward Networks (FFN)
- **GLU / GELU / SwiGLU** – Variants of gated layers that improve non-linear capacity and training stability.
  - [SwiGLU Paper](https://arxiv.org/abs/2002.05202)

### 📏 Normalization Layers

## Suvey

https://dl.acm.org/doi/abs/10.1145/3569928

- **RMSNorm**, **ScaleNorm**, **CRMSNorm** – Alternatives to LayerNorm that improve training dynamics and model stability.
  - [RMSNorm](https://arxiv.org/abs/1910.07467)
  - [ScaleNorm](https://arxiv.org/abs/2002.04745)
  - [CRMSNorm](https://arxiv.org/abs/2310.01564)

### 🧮 Mixture-of-Experts (MoE)
- **Dynamic MoE Layers** – Selects a sparse set of expert networks per input using learned routing for compute-efficient scaling.
  - [MoE Routing](https://arxiv.org/abs/2405.14297)

### ➕ Residual Connections
- **Skip Connections** – Help gradients flow through deep networks and enable better convergence.
  - [Residual Networks (ResNet)](https://arxiv.org/abs/1512.03385)

These layers, often backed by recent research, greatly expand the expressive power of modern neural models.
