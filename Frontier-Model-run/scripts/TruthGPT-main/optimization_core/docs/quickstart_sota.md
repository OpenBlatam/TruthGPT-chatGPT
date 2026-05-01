# 🚀 SOTA 2025 Quick Start Guide

Welcome to the **TruthGPT SOTA 2025** ecosystem. This guide will help you run the most advanced features of the platform in minutes.

## 📋 Prerequisites
- Python 3.10+
- CUDA-compatible GPU (Optional but recommended for papers)
- API Keys for LLMs (set in `.env`)

## 🐝 1. Using the Agent Swarm (CLI)
The `openclaw` command rutes your request to the best specialized agent using semantic routing.

```bash
# Direct question
openclaw swarm ask "What are the latest breakthroughs in multi-modal LLMs?"

# Context-aware chat (persists memory)
openclaw swarm ask "Help me write a research paper on focus-based attention" --user researcher_1
```

## 📚 2. Discovering Research Papers
TruthGPT has a built-in library of 48+ SOTA papers that can be applied to your models.

```bash
# List top papers
openclaw papers list

# Get details on a specific paper (e.g., LongRoPE)
openclaw papers info longrope_2024
```

## 🐍 3. Python SDK (OpenClaw)
Integrate OpenClaw into your own research scripts or Jupyter notebooks.

```python
import openclaw as oc
import asyncio

async def main():
    # 1. Ask the swarm
    res = await oc.ask("Can you summarize the FocusLLM paper?")
    print(f"Agent: {res.content}")
    
    # 2. List papers programmatically
    papers = oc.list_papers(category="attention")
    print(f"Found {len(papers)} matching papers.")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🌐 4. Production API
Run TruthGPT as a service for your applications.

```bash
# Start the server
python cli.py serve --port 8080 --workers 4
```

### Key Endpoints:
- `POST /v1/swarm/ask`: JSON body `{"prompt": "...", "user_id": "..."}`
- `GET /v1/research/papers`: Query `?category=...`
- `GET /v1/metrics`: Prometheus-compatible metrics.

## 🛠️ 5. Developer Guide: Building Your Own Agent
TruthGPT is a framework first. You can use its base classes to build specialized agents that integrate perfectly with the ecosystem.

### 🚀 Ejemplo SOTA 2025 (Recomendado)

Esta es la forma más profesional y escalable de usar el framework ahora:

```python
from openclaw import AgentClient, AgentConfig

config = AgentConfig(
    use_swarm=True,
    max_handoff_depth=8,
    default_agent_name="ResearchAgent"
)

client = AgentClient(config=config)

async def ask():
    response = await client.run(
        user_id="researcher_1", 
        prompt="¿Cuál es el estado del arte en LLMs?",
        return_response=True
    )
    print(f"Agente: {response.agent_name}")
    print(f"Respuesta: {response.content}")
```

### 🛠️ Guía del Desarrollador (Custom Agents)
To build a new agent, inherit from `BaseAgent` and implement the `process` method.

```python
from agents.arquitecturas_fundamentales.base_agent import BaseAgent
from agents.models import AgentResponse

class MyExpertAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ExpertAgent", role="Especialista en Datos")

    async def process(self, query: str, context: dict = None) -> AgentResponse:
        # Tu lógica personalizada aquí
        return AgentResponse(
            content=f"Análisis experto de: {query}",
            agent_name=self.name,
            action_type="final_answer"
        )
```

### Running Your Agent
You can run your agent using the `AgentClient`, which automatically provides:
- **Observability**: Tracing integration.
- **Memory**: Persistent memory management.
- **Safety**: HITL (Human-in-the-Loop) support.

See the full example in `examples/custom_agent_example.py`.

---
## 📊 6. API Integration
If you want to create agents in other languages, use the Unified API:
- `POST /v1/swarm/ask`: Entry point for the semantic orchestrator.
- `GET /v1/research/papers`: Direct access to the SOTA knowledge base.

---
**Build the future with TruthGPT.** 🚀
