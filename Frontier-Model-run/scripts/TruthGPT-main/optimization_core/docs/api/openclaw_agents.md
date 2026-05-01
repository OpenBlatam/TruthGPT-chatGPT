# OpenClaw Agents SDK

> [!NOTE]
> The OpenClaw Agents SDK provides a high-level, production-ready interface for creating, managing, and invoking autonomous agents capable of tool utilization (ReAct architecture) and environmental interaction (Embodied RL).

## Core Concepts

### 1. Basic Initialization (Single Agent)
To instantiate an agent, use the primary `AgentClient` class. This provides direct access to a ReAct agent where you can register basic tools.

```python
import asyncio
from optimization_core.agents.client import AgentClient

async def main():
    # Initialize the client in single-agent mode
    client = AgentClient(use_swarm=False)
    
    # Enable tools dynamically
    for tool in ["web_search", "python_execute", "file_read", "file_write"]:
        client.add_tool(tool)
    
    # Execute an instruction
    response = await client.run(
        user_id="demo_user", 
        prompt="Search for what OpenClaw is and save it to openclaw.txt"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Multi-Agent Swarm Mode
The SDK features a **Swarm Orchestrator** that uses an LLM to automatically route incoming user queries to the most qualified specialized agent (e.g., Marketing, RL).

```python
async def swarm_demo():
    client = AgentClient(use_swarm=True)
    
    # The swarm automatically detects this is a marketing query
    # and routes it to the ContentMarketingAgent
    res1 = await client.run("user_1", "I need an SEO content strategy for my SaaS.")
    
    # This query gets routed to the embodied RLAgent
    res2 = await client.run("user_1", "Optimize this advertising funnel.")
```

### 3. Memory Management
Every `AgentClient` maintains a localized SQLite database for episodic interactions. Agents recall past context based on the provided `user_id`.

```python
# Clear memory for a specific user to reset context
await client.clear_memory("user_1")
```

---

## Advanced Architecture (2024 Research Implementation)

OpenClaw incorporates cutting-edge LLM agent research natively:

### Long-Term Episodic & Semantic Vector Memory (RAG)
Agents can store and retrieve past interactions and learned rules using ChromaDB. This acts as a long-term context injection (RAG) without bloating the context window.
```python
client = AgentClient(use_vector_memory=True)
# The agent will automatically query ChromaDB to append relevant past
# episodes and semantics to the current reasoning prompt.
```

### Auto-Reflection (Reflexion Pattern)
Force the ReAct reasoning loop to critique its own final output before returning it to the user. If the response is incomplete or has code errors, it dynamically triggers a retry phase.
```python
client = AgentClient(use_reflexion=True)
```

### Graph-Based Multi-Agent Orchestrator (State-Machine)
For predefined, sequential, or complex conditional workflows (unlike the Swarm's dynamic routing), use the DAG Orchestrator:
```python
from agents.multi_agentes.graph_orchestrator import GraphOrchestrator
graph = GraphOrchestrator()
graph.add_node("DataScraper", scraper_agent)
graph.add_node("ReportWriter", writer_agent)
graph.add_edge("DataScraper", "ReportWriter")
graph.set_entry_point("DataScraper")
await graph.run("user_1", "Scrape Bitcoin news and write a report.")
```

### Hierarchical Delegation (`DelegateTaskTool`)
Agents can spawn sub-agents to complete complex tasks using the `delegate_task` tool.
```python
# A MarketingAgent can decide to call:
# <tool>delegate_task</tool>
# <cmd>CodeInterpreterAgent:::Write a python script to fetch the latest SEO trends</cmd>
client.add_tool("delegate_task")
```

---

## Internal Architecture

Agents are structurally isolated inside `optimization_core/agents`:

| Directory | Purpose |
|-----------|---------|
| `razonamiento_planificacion/` | Core ReAct orchestrator and base Tool implementations |
| `marketing_intelligence/` | Specialized agents with SEO/Marketing system prompts |
| `embodied_rl/` | Agents acting as Reinforcement Learning policies in simulations |
| `multi_agentes/` | Swarm orchestration, hierarchy, and intelligent routing |
| `messaging/` | Adapters for third-party messaging platforms (Telegram, WhatsApp, etc.) |

---

## Messaging Integrations (Webhooks)

OpenClaw allows your agents to interface directly with popular messaging platforms by utilizing built-in webhook adapters.

### Common Configuration

To run the webhook proxy, launch the core server:
```bash
openclaw serve
```
Most integrations require setting up specific environment variables and configuring a webhook URL pointing to `https://your-domain.com/webhooks/{platform}`.

### Platform Setup Requirements

| Platform | Required Environment Variables | Setup Instructions |
|----------|--------------------------------|--------------------|
| **Telegram** | `TELEGRAM_BOT_TOKEN` | Create via **@BotFather**. POST your URL to `/webhooks/telegram/setup` with `{"webhook_url": "..."}`. |
| **WhatsApp (Twilio)** | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` | Configure your Twilio Sandbox webhook to point to your WhatsApp endpoint via POST. |
| **Discord** | `DISCORD_BOT_TOKEN`, `DISCORD_APP_ID` | Set the Interactions Endpoint URL in the Discord Developer Portal. |
| **Signal** | `SIGNAL_CLI_API_URL`, `SIGNAL_SENDER_NUMBER` | Requires `signal-cli-rest-api` docker container forwarding to your endpoint. |
| **Slack** | `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` | Subscribe to `message.channels` events in the Slack API Dashboard. |
| **MS Teams** | `TEAMS_APP_ID`, `TEAMS_APP_PASSWORD` | Set the Messaging Endpoint in the Azure Bot Service Connectivity Gallery. |

> [!TIP]
> **Headless Execution:** You can bypass the REST API server and instantiate the messaging adapters directly in your Python code by importing them from `agents.messaging` and calling their `process_update()` methods manually.

---

## Specialized Agents

### Code Interpreter
An autonomous agent that writes Python code, executes it in a sandboxed environment, and iteratively debugs subsequent RuntimeErrors.
```python
from agents.code_interpreter import CodeInterpreterAgent
agent = CodeInterpreterAgent(llm_engine=my_llm)
result = await agent.process("Calculate fibonacci(30) and print the result")
```

### Data Analysis
Equipped with `pandas` and `matplotlib` contexts for executing dataframe manipulations.
```python
from agents.data_analysis import DataAnalysisAgent
agent = DataAnalysisAgent(llm_engine=my_llm)
result = await agent.process("Read sales.csv and plot the top 10 categories")
```

---

## Scheduled Tasks (Cron)

You can schedule agents to execute tasks on a recurring or delayed basis using the `AgentScheduler`.

```python
from agents.scheduler import AgentScheduler
from agents.client import AgentClient

client = AgentClient()
scheduler = AgentScheduler(client)

# Execute the daily summary prompt every 3600 seconds
scheduler.add_recurring(
    task_id="daily_report", 
    user_id="user1", 
    prompt="Generate a daily summary", 
    interval_seconds=3600
)
await scheduler.start()
```

---

## Observability & Tracing

OpenClaw includes a built-in tracer for tracking deeply nested tool calls and LLM invocations.

```python
from agents.observability import global_tracer

# Start a root trace
trace_id = global_tracer.start_trace("user_request", agent_name="CodeInterpreter")

# Start a child span for a tool execution
span = global_tracer.start_span(trace_id, "python_execute", kind="tool_call")
# ... tool execution ...
span.finish(output="Tool execution result")
```

---

## Core API Reference

The following REST endpoints are exposed when running the OpenClaw Agent API server.

| Endpoint | Method | Component Category |
|----------|--------|--------------------|
| `/v1/agent/run` | POST | Agent Execution |
| `/v1/agent/memory/{user_id}` | DELETE | Agent SDK Memory |
| `/v1/traces/stats` | GET | Observability |
| `/v1/traces/recent` | GET | Observability |
| `/v1/traces/{trace_id}` | GET | Observability |
| `/v1/scheduler/tasks` | GET / POST | Scheduled Tasks |
| `/v1/scheduler/tasks/{id}` | DELETE | Scheduled Tasks |
| `/v1/scheduler/start` | POST | Scheduled Tasks |
| `/v1/scheduler/stop` | POST | Scheduled Tasks |
