### Table 1. Keywords and definitions in the domain of optimization-based transformers and LLMs models

| Keyword                 | Definition                                                                                          |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| Transformer             | A DL architecture that uses self-attention mechanisms to model sequential data.                    |
| LLMs                    | A transformer-based model trained on massive text datasets to perform diverse NLP tasks.           |
| Sparse Attention        | A modification of the attention mechanism that reduces computational complexity by focusing on specific parts of the input. |
| Mixed-Precision Training| A technique that uses lower-precision data types (e.g., FP16) to speed up computations and reduce memory usage. |
| Gradient Checkpointing  | A memory-saving technique that recomputes intermediate activations during backpropagation instead of storing them. |
| Quantization            | The process of reducing the precision of model weights and activations to decrease model size and inference time. |
| Pruning                 | Removing unnecessary parameters from a model to reduce its size and computational demands.         |
| Knowledge Distillation | A training method where a smaller model learns from the outputs of a larger, pre-trained model.    |
| LoRA (Low-Rank Adaptation)| A fine-tuning technique that injects task-specific parameters into pre-trained models for low-resource adaptation. |
| ZeRO Optimization       | A memory optimization technique that partitions model states across devices during distributed training. |
| Sparsity                | A paradigm that reduces model parameters by representing weights or activations with fewer non-zero elements. |
| Attention Mechanism     | A component of transformer models that assign weights to input elements to focus on the most relevant parts. |
| Self-Attention          | A mechanism where a model relates different parts of the input sequence to capture dependencies.     |
| Zero-shot Learning      | A learning paradigm where models perform tasks without being explicitly trained on them by leveraging general knowledge. |
| Transfer Learning       | Leveraging a pre-trained model on a new task, often with fine-tuning.                               |
| Feedforward Network     | A fully connected neural network within each transformer layer processes outputs from the attention mechanism. |
| Layer Normalization     | A normalization technique applied within each transformer layer to stabilize training and improve performance. |

## Papers 

https://dl.acm.org/doi/abs/10.1145/3647782.3647803