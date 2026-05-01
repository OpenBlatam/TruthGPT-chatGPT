# Optimization Techniques

TruthGPT leverages state-of-the-art optimization techniques to maximize training throughput and minimize memory usage. This guide explains not just *what* features we have, but *why* they matter and *how* to tune them.

## ⚡ Performance Features

### 1. Torch.compile (PyTorch 2.0+)

We utilize `torch.compile` (Graph Mode) to fuse operations and reduce Python overhead.

-   **Why it matters**: Standard PyTorch "eager mode" executes operations one by one, launching a CUDA kernel for each (e.g., `add`, `mul`, `relu`). This causes "kernel launch overhead". `torch.compile` analyzes the entire graph and fuses these into a single kernel, reducing memory access and CPU overhead.
-   **Impact**: Can provide 1.3x - 2.0x speedup on modern GPUs.

**Modes**:
-   `default`: Good balance of compile time and performance.
-   `reduce-overhead`: Uses CUDA Graphs to minimize CPU-launch latency (great for small batches).
-   `max-autotune`: Aggressive optimization, profiling different Triton configs. Long compile times (minutes), but best runtime performance.

**Configuration**:
```yaml
training:
  torch_compile: true
  compile_mode: max-autotune
```

### 2. TF32 (TensorFloat-32)

-   **What is it?**: A math mode on NVIDIA Ampere (A100, RTX 30/40) GPUs that uses 19-bit precision for matrix multipliers (FP32 range, FP16 precision).
-   **Why it matters**: It is **8x faster** than standard FP32 on A100s, with no perceptible loss in model accuracy. It is "free speed".
-   **Note**: Enabled by default in TruthGPT for supported hardware.

**Configuration**:
```yaml
training:
  allow_tf32: true
```

### 3. Flash Attention & SDPA

Attention is the bottleneck of Transformers ($O(N^2)$ complexity). We support:

1.  **Flash Attention 2**: Hardware-aware IO optimization. It splits Q, K, V into blocks that fit in GPU SRAM, avoiding slow Global Memory reads.
2.  **SDPA (Scalable Dot Product Attention)**: PyTorch's native `F.scaled_dot_product_attention`. It automatically chooses between FlashAttention, MemEfficient, or Math backends.

**Why it matters**:
-   **Speed**: 2-4x faster than vanilla attention.
-   **Memory**: Linear memory usage with sequence length (vs Quadratic), allowing much longer sequences (e.g., 8k, 32k).

### 4. Mixed Precision (AMP)

Training in pure FP32 (32-bit float) is redundant for Deep Learning.

-   **BF16 (BFloat16)**:
    -   *Pros*: Same dynamic range as FP32, so no "loss scaling" issues. extremely stable.
    -   *Reqs*: NVIDIA Ampere (A100, 3090, 4090) or newer.
-   **FP16**:
    -   *Pros*: Supported on older hardware (V100, T4).
    -   *Cons*: Smaller dynamic range. Gradients can underflow to zero. Requires `GradScaler`.

**Configuration**:
```yaml
training:
  mixed_precision: bf16  # Recommended
  # mixed_precision: fp16 # Fallback
```

### 5. Gradient Checkpointing

-   **The Problem**: To calculate gradients, you need to store the input of every layer during the forward pass. This eats VRAM.
-   **The Solution**: Throw away the intermediate activations. During backwards pass, *recompute* them on the fly from the checkpoints.
-   **Trade-off**: Increases compute by ~20%, but **reduces memory usage by 75%**.

**When to use**: When you see `CUDA Out of Memory`. It allows you to increase batch size significantly, which often offsets the compute cost.

**Configuration**:
```yaml
model:
  gradient_checkpointing: true
```

## 🧠 Smart Data Pipeline

### Dynamic Padding & Bucketing

Standard data loaders pad every sequence in a batch to the maximum model length (e.g., 2048). If your real inputs are short (e.g., 100 tokens), you are computing 95% padding (zeroes).

**Our Approach**:
1.  **Bucketing**: We group sequences of similar lengths together in the dataset buffer.
2.  **Dynamic Padding**: When creating a batch, we pad to the length of the *longest sequence in that specific batch*, not the global max.

*Result*: Massive speedups (2x-3x) on datasets with variable lengths (like instruction tuning data).

**Configuration**:
```yaml
data:
  bucket_by_length: true
  bucket_bins: [64, 128, 256, 512, 1024, 2048]
```

## 🕵️ Troubleshooting Performance

| Symptom | Probable Cause | Fix |
| :--- | :--- | :--- |
| **Low GPU Utilization (<50%)** | Data Starvation | Increase `num_workers`, `prefetch_factor`, or `train_batch_size`. Use `torch_compile=True` with `mode="reduce-overhead"`. |
| **Training is Slow** | Computing Padding | Enable `bucket_by_length: True`. |
| **CUDA OOM (Memory)** | Batch size too large | 1. Enable `gradient_checkpointing: True`.<br>2. Use `mixed_precision: bf16`.<br>3. Reduce `train_batch_size` and increase `grad_accum_steps`. |
| **Loss is NaN** | Exploding Gradients | 1. Use `mixed_precision: bf16` instead of `fp16`.<br>2. Reduce `learning_rate`.<br>3. Check for bad data samples. |
