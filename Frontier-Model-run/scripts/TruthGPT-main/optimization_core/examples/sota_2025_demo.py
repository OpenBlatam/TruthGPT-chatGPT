import asyncio
from openclaw import AgentClient, AgentConfig
from rich.console import Console
from rich.panel import Panel

async def main():
    console = Console()
    console.print(Panel.fit("[bold blue]TruthGPT / OpenClaw SOTA 2025 Demo[/bold blue]"))

    # 1. Configuración estructurada (Nueva en Phase 19/21)
    config = AgentConfig(
        use_swarm=True,
        max_handoff_depth=10,
        memory_db_path="demo_memory.db",
        default_agent_name="ResearchAgent"
    )

    # 2. Inicialización limpia del Cliente
    client = AgentClient(config=config)

    console.print("[yellow]Investigando sobre Computación Cuántica...[/yellow]")
    
    # 3. Ejecución con el nuevo modelo unificado
    prompt = "Explica brevemente qué es la computación cuántica y busca si hay algún paper reciente sobre corrección de errores."
    
    # Usamos return_response=True para ver los metadatos del nuevo AgentResponse
    response = await client.run(user_id="user_123", prompt=prompt, return_response=True)

    # 4. Mostrar resultados profesionales con Rich
    console.print(Panel(
        response.content,
        title=f"Respuesta de: {response.agent_name}",
        subtitle=f"Acción: {response.action_type}",
        border_style="green"
    ))

    if response.metadata:
        console.print(f"[dim]Metadatos: {response.metadata}[/dim]")

if __name__ == "__main__":
    asyncio.run(main())

