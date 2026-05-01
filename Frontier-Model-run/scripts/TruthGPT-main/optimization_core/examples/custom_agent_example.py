import asyncio
import os
import sys
from pathlib import Path

# Configurar el path para que el ejemplo pueda importar el núcleo
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.arquitecturas_fundamentales.base_agent import BaseAgent
from agents.models import AgentResponse
from agents.client import AgentClient

class MyCustomResearcher(BaseAgent):
    """
    Un agente personalizado que simula una investigación profunda.
    Hereda de BaseAgent para ser compatible con el Swarm y el Client de OpenClaw/TruthGPT.
    """
    def __init__(self):
        super().__init__(name="CustomResearcher", role="Investigador de Vanguardia")

    async def process(self, query: str, context: dict = None) -> AgentResponse:
        print(f"[{self.name}] Analizando: {query}...")
        
        # Simulación de lógica personalizada (ej. llamar a una API externa, procesar datos locales)
        analysis_result = f"Resultado personalizado para '{query}': El mercado de LLMs está evolucionando hacia modelos de razonamiento elástico."
        
        # Guardar en memoria episódica propia (Historial de la sesión actual)
        self.add_to_memory("user", query)
        self.add_to_memory("assistant", analysis_result)
        
        # Retornar AgentResponse estructurado (Estándar SOTA 2025)
        return AgentResponse(
            content=analysis_result,
            agent_name=self.name,
            action_type="final_answer",
            metadata={"confidence": 0.95, "source": "Custom Logic"}
        )

async def main():
    # 1. Instanciar el agente personalizado
    my_agent = MyCustomResearcher()
    
    # 2. Usar el AgentClient para gestionar la ejecución
    # Esto habilita observabilidad, trazas y memoria de largo plazo automáticamente.
    client = AgentClient()
    
    print("--- Iniciando Ejecución ---")
    # Ejecutamos el agente
    response = await my_agent.process("¿Cuál es el futuro de los LLMs?")
    
    print(f"\nRespuesta Final: {response.content}")
    print(f"Metadatos de la respuesta: {response.metadata}")

if __name__ == "__main__":
    asyncio.run(main())

