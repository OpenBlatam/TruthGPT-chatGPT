# TensorFlow - Customizable Generative AI

## 🚀 Overview

This project focuses on the development of customizable Generative AI models using **TensorFlow**, with a particular emphasis on **GPT** models and transformer architectures. TensorFlow's flexibility allows for the construction of generative models on the fly, but there are still several limitations and challenges when working with older TensorFlow versions (e.g., 0.5). The goal is to adapt TensorFlow for modern AI practices while addressing these challenges.

## ⚠️ Key Challenges

### 1. **Training Large GPT Models**
   - Training large GPT models using TensorFlow 0.5 is slow and inefficient due to the absence of modern optimizations.
   - Key optimizations like mixed precision, eager execution, and graph optimizations are essential for improving training efficiency, but are not fully supported in TensorFlow 0.5.

### 2. **TensorFlow 0.5 Limitations**
   - **tf.keras**: Missing support for TensorFlow’s high-level Keras API, which simplifies model building and training.
   - **Multi-head attention**: Essential for transformers, but not implemented natively in TensorFlow 0.5 (e.g., `tf.nn.multi_head_attention`).
   - **Eager execution and dynamic control flow**: These features are required for dynamic computation graphs and are only available in TensorFlow 2.x or newer.
   - **Layer normalization**: Crucial for transformer architectures to stabilize training and improve convergence, but not efficiently supported in older versions.

## 🔧 TODO

To overcome these limitations, the following tasks are critical for adapting TensorFlow to modern requirements for large generative models:

### 1. **tf.keras**
   - Migrate to `tf.keras` for easier model construction and training, replacing older TensorFlow models.
   
### 2. **Eager Execution**
   - Enable **eager execution** to support dynamic computation graphs, which are essential for transformer-based models.

### 3. **Dynamic Control Flow**
   - Implement dynamic control flow mechanisms to adapt the architecture during training and inference.

### 4. **Layer Normalization**
   - Develop custom layer normalization that is efficient and compatible with large transformer models.

### 5. **Multi-head Attention**
   - Implement **multi-head attention** layers compatible with TensorFlow 0.5 and optimize them for better performance.

### 6. **Custom Op Fusions**
   - Explore creating **custom op fusions** for better memory and computation optimization during model training.

## 🔧 Framework Recommendations

### 1. **Use Modern Frameworks for Training Large GPT Models**
   - **PyTorch + HuggingFace**: Recommended for flexibility and ease of use in training and finetuning large GPT models.
   - **TensorFlow 2.x**: If TensorFlow is a must, upgrading to TensorFlow 2.x will provide better support for modern optimizations like eager execution, `tf.keras`, and `tf.nn.multi_head_attention`.
   - **ONNX Runtime, Triton, or vLLM**: For efficient inference of trained models, consider using optimized runtimes like **ONNX**, **Triton** or **vLLM**.

### 2. **Consider Alternative Backends for Performance**
   - **Older Torch-style Backend**: While TensorFlow 0.5 may be useful for smaller, toy models, consider using a more modern backend like **PyTorch** or a TensorFlow 2.x structure to better utilize advanced features.

## ⚙️ How to Get Started

1. **Set Up TensorFlow**
   - If you're using TensorFlow 2.x, install the latest version:
     ```bash
     pip install tensorflow
     ```
   - For older versions like TensorFlow 0.5, you may need to manually configure dependencies and compatibility layers.

2. **Create Your Model**
   - Use `tf.keras` or custom layers for model architecture.
   - Implement or import **multi-head attention** layers and **layer normalization**.

3. **Training on Custom Data**
   - Customize data pipelines using **tf.data**.
   - Train the model with your specific dataset and experiment with different optimizations.

4. **Explore Optimized Inference Frameworks**
   - After training, export your model to **ONNX** or other optimized formats for faster inference.

## 📚 References

- **TensorFlow 2.x Overview**: [TensorFlow Docs](https://www.tensorflow.org)
- **GPT Models and Transformer Architecture**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **ONNX Runtime for Efficient Inference**: [ONNX Docs](https://onnxruntime.ai/)
- **Triton Inference Server**: [Triton Docs](https://github.com/triton-inference-server/server)

---

🔧 **For Advanced Users**:
- Explore building custom ops and fusions to improve the efficiency of large-scale model training.
- Use **TPU** support for massive scale if your workload requires it.


### 🔗 GPT Implementation Stack (JAX + Flax)

| Component               | Tool / Module                               | GitHub / Docs Link                                                                 |
|------------------------|---------------------------------------------|------------------------------------------------------------------------------------|
| **Core Framework**      | JAX                                         | [google/jax](https://github.com/google/jax)                                       |
| **NN Layers**           | Flax                                        | [google/flax](https://github.com/google/flax)                                     |
| **Optimizer**           | Optax                                       | [deepmind/optax](https://github.com/deepmind/optax)                               |
| **Dataset**             | HuggingFace Datasets                        | [huggingface/datasets](https://github.com/huggingface/datasets)                   |
| **Pretrained GPT**      | Transformers (Flax support)                 | [huggingface/transformers](https://github.com/huggingface/transformers)           |
| **Visualization**       | TensorBoard                                 | [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard)               |
|                        | Weights & Biases (WandB)                    | [wandb/client](https://github.com/wandb/client)                                   |
| **Advanced GPT Training**| T5X                                        | [google-research/t5x](https://github.com/google-research/t5x)                     |
|                        | Scenic                                      | [google-research/scenic](https://github.com/google-research/scenic)               |
|                        | Orbax (Checkpointing)                       | [google/orbax](https://github.com/google/orbax)                                   |



| **Feature**                 | **Legacy Torch (TH/THC)**             | **GPT Backends (Modern)**                         |
|-----------------------------|----------------------------------------|---------------------------------------------------|
| **Language**                | C, CUDA                                | C++, CUDA, Triton, Python (JIT), MLIR             |
| **Invocation Style**        | Manually bound via `updateOutput`     | Declarative `forward()` or graph compilers        |
| **Softmax**                 | Manual C/CUDA                          | ATen, cuDNN, FlashAttention                       |
| **Optimization**            | None / Manual                          | Kernel fusion, tiling, caching                    |
| **Compilation / Fusion**    | Manual only                            | AOT / JIT compiled (TensorRT, Triton, XLA, etc.)  |

| **Backend**      | **GitHub Link / Docs**                                    | **Key Use**                        |
|------------------|-----------------------------------------------------------|------------------------------------|
| ATen (PyTorch)   | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) | Native ops (e.g. softmax)         |
| FlashAttention   | [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Fused softmax + attention         |
| Triton           | [github.com/openai/triton](https://github.com/openai/triton) | Custom GPU kernels                |
| xFormers         | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | Meta's fast transformer ops       |
| TensorRT         | [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) | Runtime inference optimization    |
| DeepSpeed        | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | Efficient training + fused ops    |

Alternatives:

| **Backend**         | **GitHub Link / Docs**                                                   | **Key Use**                                     |
|---------------------|--------------------------------------------------------------------------|-------------------------------------------------|
| ATen (PyTorch)      | [pytorch/pytorch](https://github.com/pytorch/pytorch)                   | Native PyTorch ops (e.g., softmax, matmul)     |
| FlashAttention      | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | Fused softmax + attention                       |
| Triton              | [openai/triton](https://github.com/openai/triton)                       | Custom GPU kernels in Python DSL                |
| xFormers            | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | Meta's fused transformer ops (Flash, block)     |
| TensorRT            | [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)  | Optimized inference (AOT compiled)              |
| DeepSpeed           | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)           | Training optimizer, fused ops, ZeRO             |
| TorchInductor       | [pytorch/torchdynamo](https://github.com/pytorch/torchdynamo)           | Compiler backend for PyTorch (JIT graph)        |
| XLA                 | [tensorflow/xla](https://github.com/tensorflow/xla)                     | MLIR-based compiler used in JAX/TensorFlow      |
| TVM                 | [apache/tvm](https://github.com/apache/tvm)                             | Deep learning compiler, supports GPT inference  |
| OpenXLA / StableHLO | [openxla/stablehlo](https://github.com/openxla/stablehlo)               | Portable compute IR used in modern compilers    |
| MLC AI              | [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)                     | LLM deployment on mobile/WebGPU using TVM       |
| OneDNN              | [oneapi-src/oneDNN](https://github.com/oneapi-src/oneDNN)               | Intel's backend for CPU fused ops               |
| ROCm / MIOpen       | [ROCmSoftwarePlatform/MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen) | AMD GPU backend, cuDNN analog                   |
| cuDNN               | [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)        | NVIDIA GPU primitives for deep learning         |
| ONNX Runtime        | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)       | Universal runtime (supports TensorRT, CPU, etc.)|



TensorFlow is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.


**Note: Currently we do not accept pull requests on github -- see
[CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute code
changes to TensorFlow through
[tensorflow.googlesource.com](https://tensorflow.googlesource.com/tensorflow)**

**We use [github issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, but please see
[Community](tensorflow/g3doc/resources/index.md#community) for general questions
and discussion.**

# Download and Setup

To install TensorFlow using a binary package, see the instructions below.  For
more detailed installation instructions, including installing from source, see
[here](tensorflow/g3doc/get_started/os_setup.md).

## Binary Installation

### Ubuntu/Linux

Make sure you have [pip](https://pypi.python.org/pypi/pip) installed:

```sh
$ sudo apt-get install python-pip
```

Install TensorFlow:

```sh
# For CPU-only version
$ sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled version.  See detailed install instructions
# for GPU configuration information.
$ sudo pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

### Mac OS X

Make sure you have [pip](https://pypi.python.org/pypi/pip) installed:

If using `easy_install`:

```sh
$ sudo easy_install pip
```

Install TensorFlow (only CPU binary version is currently available).

```sh
$ sudo pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

### Try your first TensorFlow program

```sh
$ python

>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print sess.run(a+b)
42
>>>

```


##For more information

* [TensorFlow website](http://tensorflow.org)
