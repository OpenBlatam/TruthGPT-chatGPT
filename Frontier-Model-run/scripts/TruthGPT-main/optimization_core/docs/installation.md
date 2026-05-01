# Installation Guide

This guide provides detailed instructions for installing the TruthGPT Optimization Core in various environments, ensuring you have the correct CUDA and PyTorch versions for maximum performance.

## 📋 Prerequisites

Before you begin, ensure your system meets these requirements:

-   **Operating System**: Linux (Ubuntu 20.04+ Recommended), Windows (WSL2), or macOS (M1/M2/M3).
-   **Python**: Version 3.10 is recommended (3.8+ supported).
-   **GPU**: NVIDIA GPU with CUDA 11.8+ (for acceleration). CPU-only mode is supported but slow.

## 🚀 Recommended Method: Automatic Setup

We provide setup scripts that automatically detect your OS and configure a virtual environment.

### Linux / macOS

```bash
cd optimization_core
chmod +x setup_dev.sh
./setup_dev.sh
```

### Windows (PowerShell)

```powershell
cd optimization_core
.\setup_dev.ps1
```

*These scripts will create a `.venv` directory, upgrade pip, and install all core requirements.*

## 📦 Manual Installation

If you need granular control over the installation, follow these steps.

### 1. Create a Virtual Environment

Isolating your project dependencies is critical for stability.

```bash
# Standard Python venv
python -m venv venv
source venv/bin/activate      # Linux/macOS
# .\venv\Scripts\activate     # Windows

# OR using Conda (Alternative)
conda create -n truthgpt python=3.10
conda activate truthgpt
```

### 2. Install PyTorch with CUDA Support

This is the most important step. **Do not just run `pip install torch`**, as it often installs the CPU version by default.

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the exact command for your hardware.

**Example for CUDA 11.8 (Standard):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Example for CUDA 12.1 (Newer GPUs):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install TruthGPT Core Dependencies

Once PyTorch is installed, install the rest of the stack.

```bash
pip install -r requirements_advanced.txt
```

### 4. Install Optional Extensions

To keep the core lightweight, we separate heavy dependencies. Use our valid helper script:

```bash
# See all available extension groups
python install_extras.py --list

# Install specific features
python install_extras.py wandb        # For experiment tracking
python install_extras.py bitsandbytes   # For 8-bit optimizers (saves memory)

# Install everything (for power users)
python install_extras.py all
```

## 🐳 Docker Installation

For reproducible production environments, use the provided Docker support.

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app
COPY requirements_advanced.txt .
RUN pip install -r requirements_advanced.txt

COPY . .
CMD ["python", "train_llm.py", "--config", "configs/llm_default.yaml"]
```

Build and run:

```bash
docker build -t truthgpt-core .
docker run --gpus all -it truthgpt-core
```

## ✅ Verification

Verify your installation by running the health check script.

```bash
# Run the included health check
python utils/health_check.py

# Or use the Makefile shortcut
make health
```

You should see output confirming that PyTorch, CUDA, and key modules are importable.
