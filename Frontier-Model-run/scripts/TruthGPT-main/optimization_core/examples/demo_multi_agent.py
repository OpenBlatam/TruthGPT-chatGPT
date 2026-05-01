import sys
import os
import asyncio

# Añadir optimization_core al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.razonamiento_planificacion.orchestrator import MultiUserReActAgent
from agents.razonamiento_planificacion.tools import WebSearchTool, SystemBashTool

# ---------------------------------------------------------
# 1. Mock de Inferencia Asíncrona
# ---------------------------------------------------------

async def mock_truthgpt_inference(prompt: str) -> str:
    """Simulación asíncrona del LLM."""
    await asyncio.sleep(0.1) # Simular latencia
    
    if "bitcoin" in prompt.lower() and "TOOL_RESULT" not in prompt:
        return "<tool>web_search</tool>\n<cmd>bitcoin price</cmd>"
    
    if "TOOL_RESULT: Resumen Web" in prompt:
        return "The web search indicates Bitcoin is around $60k-$70k USD."

    if "meetings" in prompt.lower() and "hate meetings" in prompt.lower():
        return "I remember you mentioned you hate meetings. I'll keep that in mind."
        
    return "Hello! I am the refactored TruthGPT. How can I help you in this async environment?"

# ---------------------------------------------------------
# 2. Ejecución de Demo
# ---------------------------------------------------------

async def main():
    print("🚀 Iniciando Demo TruthGPT Async (Empresarial)...\n")
    
    # Inicializar agente
    agente = MultiUserReActAgent(llm_engine=mock_truthgpt_inference)
    
    # Registrar herramientas (Instancias de clase)
    agente.register_tool(WebSearchTool())
    agente.register_tool(SystemBashTool())

    # --- SIMULACIÓN MULTI-USUARIO ---
    
    # Usuario A
    await agente.memory.add_message("User_A", "user", "I hate meetings.")
    print("--- User A (Telegram) ---")
    res_a = await agente.process_message("User_A", "What did I say about meetings?")
    print(f"TruthGPT: {res_a}\n")

    # Usuario B
    print("--- User B (Discord) ---")
    res_b = await agente.process_message("User_B", "Check the bitcoin price.")
    print(f"TruthGPT: {res_b}\n")

if __name__ == "__main__":
    asyncio.run(main())

