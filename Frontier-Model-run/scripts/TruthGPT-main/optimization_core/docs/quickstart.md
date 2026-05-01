# Quick Start Guide

This guide will get you up and running with training your first model using TruthGPT Optimization Core in under 5 minutes.

## 🏁 Train Your First Model

The fastest way to start is using one of our pre-configured presets.

### 1. From the Command Line

```bash
# Verify installation first
python utils/health_check.py

# Run a quick training job using the 'lora_fast' preset
python train_llm.py --config configs/presets/lora_fast.yaml
```

This will:
- Load a small model (e.g., GPT-2).
- Apply LoRA (Low-Rank Adaptation) for efficiency.
- Train on a sample dataset.
- Save checkpoints to `runs/lora_fast_run`.

### 2. Using Python API

You can also train directly from your Python code:

```python
from trainers.trainer import GenericTrainer, TrainerConfig

# Define configuration
config = TrainerConfig(
    model_name="gpt2",
    output_dir="runs/my_first_run",
    epochs=1,
    train_batch_size=4,
    use_mps_device=True  # or use_cuda_device=True
)

# Dummy dataset
train_texts = ["Hello world", "This is a test"] * 100
val_texts = ["Validation sample"] * 10

# Initialize trainer
trainer = GenericTrainer(
    cfg=config,
    train_texts=train_texts,
    val_texts=val_texts
)

# Start training
trainer.train()
```

## 🛠️ Customize Your Training

TruthGPT uses a unified YAML configuration system. To customize your training, create a new YAML file.

### Step 1: Initialize Project

Use the `init_project.py` script to generate a boilerplate configuration.

```bash
python init_project.py my_custom_project --preset performance_max --model meta-llama/Llama-2-7b
```

This creates `configs/my_custom_project.yaml`.

### Step 2: Edit Configuration

Open `configs/my_custom_project.yaml` and adjust settings:

```yaml
model:
  name_or_path: meta-llama/Llama-2-7b
  lora:
    enabled: true
    r: 64
    alpha: 16

training:
  batch_size: 2
  grad_accum_steps: 16  # Simulate larger batch size
  learning_rate: 2e-4
  mixed_precision: bf16 # Recommended for newer GPUs (Ampere+)

data:
  dataset: wikitext
  subset: wikitext-103-raw-v1
```

### Step 3: Train

```bash
python train_llm.py --config configs/my_custom_project.yaml
```

## 📊 Monitor Progress

TruthGPT integrates with popular logging tools.

### TensorBoard

```bash
tensorboard --logdir runs
```

### Weights & Biases

Enable W&B in your YAML config:

```yaml
training:
  callbacks:
    - wandb

logging:
  project: truthgpt-experiment
  run_name: llama2-finetune
```

## ⏭️ Next Steps

- Explore the **[Architecture](architecture.md)** to understand how it works under the hood.
- Dive into **[Optimization Techniques](optimization.md)** to maximize GPU utilization.
- Check the **[API Reference](api/trainer.md)** for advanced usage.
