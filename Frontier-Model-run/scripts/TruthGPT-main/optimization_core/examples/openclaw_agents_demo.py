"""
Demo de OpenClaw Agents SDK
Este script demuestra cómo usar el AgentClient interactuando con herramientas.
"""

import asyncio
import os
import sys

def _setup_path() -> None:
    """Ajustar el path para permitir imports directos si se ejecuta desde examples/."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

_setup_path()

try:
    from agents import AgentClient
except ImportError as e:
    print(f"Error de importación. Asegúrese de ejecutar el script en el entorno correcto: {e}")
    sys.exit(1)


async def run_single_agent() -> None:
    """Ejecuta una demostración de un agente único con capacidades web_search y python_execute."""
    print("=" * 50)
    print("Iniciando Demo de Agente Único (OpenClaw Base)")
    print("=" * 50)
    
    client = AgentClient(use_swarm=False)
    
    # Agregando capacidades al agente
    client.add_tool("web_search")
    client.add_tool("python_execute")
    
    user = "demo_developer"
    
    prompt_1 = "Busca información sobre el precio de Bitcoin hoy."
    print(f"\n[Usuario]: {prompt_1}")
    response = await client.run(user_id=user, prompt=prompt_1)
    print(f"\n[Agente]: {response}")

    prompt_2 = "Escribe un script en python que calcule 5 años de interés compuesto al 10% anual para 1000 dólares."
    print(f"\n[Usuario]: {prompt_2}")
    response = await client.run(user_id=user, prompt=prompt_2)
    print(f"\n[Agente]: {response}")


async def run_swarm() -> None:
    """Ejecuta una demostración de Multi-Agentes (Swarm)."""
    print("\n" + "=" * 50)
    print("Iniciando Demo de Swarm (Multi-Agentes)")
    print("=" * 50)
    
    # El modo swarm carga RLAgent, MarketingAgent, etc.
    client = AgentClient(use_swarm=True)
    user = "demo_manager"
    
    prompt_1 = "Quiero una estrategia SEO y de marketing de contenidos para vender software."
    print(f"\n[Usuario]: {prompt_1}")
    response = await client.run(user_id=user, prompt=prompt_1)
    print(f"\n[Swarm -> MarketingAgent]: {response}")

    prompt_2 = "Optimiza la conversión interactuando con mi entorno."
    print(f"\n[Usuario]: {prompt_2}")
    response = await client.run(user_id=user, prompt=prompt_2)
    print(f"\n[Swarm -> RLAgent]: {response}")


async def main() -> None:
    """Coroutine principal que coordina las demostraciones."""
    try:
        await run_single_agent()
        await run_swarm()
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la ejecución: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEjecución cancelada por el usuario.")


