<div align="center">

# 🚀 TruthGPT Optimization Core

**The Enterprise-Grade, Modular Training System for Frontier Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/index.md)

[**Quick Start**](docs/quickstart.md) | [**Installation**](docs/installation.md) | [**Architecture**](docs/architecture.md) | [**API Reference**](docs/api/trainer.md)

</div>

---

## 🌟 Introduction

**TruthGPT Optimization Core** is a high-performance, modular framework designed to train Large Language Models (LLMs) with maximum efficiency. Built on PyTorch, it integrates state-of-the-art optimization techniques like **Flash Attention**, **TF32**, and **Torch.compile** into a unified, easy-to-use system.

Whether you are fine-tuning a 7B model on a single GPU or training a massive foundation model on a cluster, TruthGPT scales with you.

## ✨ Key Features (SOTA 2025)

-   **🐝 Swarm Intelligence**: Orchestrate dozens of specialized agents via the `SwarmOrchestrator`.
-   **📄 Research Integration**: Instant access to **48+ SOTA papers** (LongRoPE, etc.) via the `PaperRegistry`.
-   **⚡ Ultimate Performance**: Auto-configured for speed with TF32, mixed precision, and compile-time optimizations.
-   **🧠 Hierarchical Memory**: Persistent `CoreMemory` so your agents never forget your preferences.
-   **📡 Omnichannel**: Native support for Telegram, Signal, Email, and more.
-   **🖥️ Interactive Command Center**: A rich terminal UI to control the entire system effortlessly.

## 🚀 Interactive Command Center (Recommended)

The most powerful and intuitive way to use TruthGPT. Experience a full-featured terminal menu with categorical control.

```bash
# Enter the Command Center
python main.py
```

Inside the menu, you can access:
- **🚀 Inference**: Real-time text generation.
- **🐝 Swarm Intelligence**: Direct expert agent routing.
- **📄 Research Hub**: Browse 48+ SOTA papers.
- **🛠️ Diagnostics**: Run optimization and health tools.

---

## 💻 CLI Usage (Advanced)
# Ask the swarm anything
python main.py swarm ask "Explain the benefits of LongRoPE for 2M context"

# List available research papers
python main.py papers list
```

### 2. Researcher SDK (Python)
Use the entire infrastructure in your own scripts:
```python
from truthgpt import api

# Ask the swarm
response = await api.ask("Build a research plan for sparse attention")
print(response.content)
```

👉 **[Read the Full SOTA 2025 Quick Start Guide](docs/quickstart_sota.md)**

## 📖 Documentation

Dive deep into the system:

-   **[Installation Guide](docs/installation.md)**: Setup for Dev, Prod, Docker, and Conda.
-   **[System Architecture](docs/architecture.md)**: Understand the registry patterns and component flow.
-   **[Optimization Techniques](docs/optimization.md)**: Learn how we achieve high GPU utilization.
-   **[Configuration Reference](docs/api/configuration.md)**: Every YAML option explained.
-   **[Trainer API](docs/api/trainer.md)**: Developer documentation for the core engine.

## 🤖 OpenClaw Agents SDK & API

TruthGPT Optimization Core now includes full **OpenClaw-compatible** autonomous agent capabilities. You can use it via Python SDK or REST API:

### Python SDK
```python
import asyncio
from optimization_core.agents import AgentClient

async def main():
    # Inicializa el cliente (funciona igual que OpenClaw)
    client = AgentClient(use_swarm=False)
    
    # Añade herramientas (ej. file_read, python_execute, web_search)
    client.add_tool("web_search")
    client.add_tool("python_execute")
    
    # Ejecuta el agente autónomo
    response = await client.run(user_id="user123", prompt="Busca en internet el precio de Bitcoin y calcula el 10% en Python.")
    print(response)

asyncio.run(main())
```

### REST API
Start the inference server (`openclaw serve`) and call the agent endpoint:

```bash
curl -X POST http://localhost:8080/v1/agent/run \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer changeme" \\
  -d '{
    "prompt": "Escribe un script en un archivo de python que imprima hola mundo",
    "user_id": "user123",
    "tools": ["file_write"]
  }'
```

## 🛠️ Project Structure

```text
optimization_core/
├── configs/               # YAML Configurations
├── docs/                  # 📚 Comprehensive Documentation
├── factories/             # Component Registries (Optimizers, etc.)
├── trainers/              # Core Training Logic
├── scripts/               # Utility Scripts
└── train_llm.py           # Main Entry Point
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

<div align="center">
    Built with ❤️ by the TruthGPT Team
</div>
