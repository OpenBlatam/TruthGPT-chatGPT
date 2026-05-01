# Configuration Reference

The TruthGPT system uses a unified configuration object, `TrainerConfig`. This centralized YAML approach ensures that every experiment is reproducible.

## `TrainerConfig`

**Location**: `optimization_core/trainers/trainer.py`

### 🏗️ Model Settings

| Parameter | Type | Default | Description | Best Practice |
| :--- | :--- | :--- | :--- | :--- |
| `model_name` | `str` | `"gpt2"` | HuggingFace model ID or path. | Use smaller models (gpt2) for debugging. |
| `gradient_checkpointing` | `bool` | `True` | Enable checkpointing to save RAM. | **Always Enable** for models > 1B params. |
| `save_safetensors` | `bool` | `True` | Save in `safetensors` format. | **Yes**. Much faster loading and safer than pickle. |

### ⚡ LoRA (Low-Rank Adaptation)

| Parameter | Type | Default | Description | Best Practice |
| :--- | :--- | :--- | :--- | :--- |
| `lora_enabled` | `bool` | `False` | Enable LoRA fine-tuning. | Set to `true` for fine-tuning on consumer GPUs. |
| `lora_r` | `int` | `16` | LoRA rank dimension. | `8` or `16` is usually sufficient. `64` for complex tasks. |
| `lora_alpha` | `int` | `32` | Scaling factor. | Rule of thumb: `alpha = 2 * r`. |
| `lora_dropout` | `float` | `0.05` | Dropout probability. | Keep low (`0.05` or `0.1`) to prevent underfitting. |

### 🚅 Training Hyperparameters

| Parameter | Type | Default | Description | Best Practice |
| :--- | :--- | :--- | :--- | :--- |
| `epochs` | `int` | `3` | Number of training passes. | `1-3` for fine-tuning. Pre-training needs more. |
| `train_batch_size` | `int` | `8` | Batch size per device. | Maximize this until OOM, then step back slightly. |
| `grad_accum_steps` | `int` | `2` | Steps to accumulate gradients. | effective_bs = `batch_size * accum_steps * num_gpus`. Aim for 32-128. |
| `learning_rate` | `float` | `5e-5` | Max learning rate. | `1e-4` to `2e-5` for LoRA. `1e-5` for full finetune. |
| `scheduler` | `str` | `"cosine"` | LR scheduler type. | `cosine` is generally superior to `linear` for LLMs. |

### 🚀 Optimization & Precision

| Parameter | Type | Default | Description | Best Practice |
| :--- | :--- | :--- | :--- | :--- |
| `mixed_precision` | `str` | `"bf16"` | `"bf16"`, `"fp16"`, or `"none"`. | Use `bf16` on Ampere (3090/A100). Use `fp16` on older (T4/V100). |
| `allow_tf32` | `bool` | `True` | Enable TensorFloat-32. | **Always True** on Ampere+. Free speedup. |
| `torch_compile` | `bool` | `False` | Enable JIT compilation. | Set `True` for long runs. Startup is slow but step time is faster. |
| `fused_adamw` | `bool` | `True` | Use Fused AdamW. | **Always True** if CUDA is available. |

### 💾 Data Loading

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_workers` | `int` | `4` | Number of dataloader subprocesses. |
| `prefetch_factor` | `int` | `2` | Number of batches to prefetch per worker. |
| `persistent_workers` | `bool` | `True` | Keep workers alive between epochs to reduce startup cost. |

### 📝 Logging & Checkpointing

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `output_dir` | `str` | `"runs/run"` | Directory to save checkpoints and logs. |
| `log_interval` | `int` | `50` | Steps between logging metrics. |
| `eval_interval` | `int` | `500` | Steps between validation runs. |
| `ckpt_interval_steps` | `int` | `1000` | Steps between saving checkpoints. |
| `ckpt_keep_last` | `int` | `3` | Number of recent checkpoints to restrict disk usage. |
| `ema_enabled` | `bool` | `True` | Maintain EMA (Exponential Moving Average) of weights. |
| `ema_decay` | `float` | `0.999` | Decay rate for EMA. |

### 🔄 Recovery

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `resume_enabled` | `bool` | `False` | Auto-detect and resume from latest checkpoint. |
| `resume_from_checkpoint` | `str` | `None` | Path to specific checkpoint file (`.pt`) to resume from. |
