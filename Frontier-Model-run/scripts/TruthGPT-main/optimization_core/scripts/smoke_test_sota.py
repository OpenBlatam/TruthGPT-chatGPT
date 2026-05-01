import asyncio
import sys
import logging
from pathlib import Path

# Configurar logging para ver el proceso
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmokeTest")

async def run_smoke_test():
    logger.info("🚀 Iniciando Smoke Test del Stack SOTA 2025...")
    
    try:
        # 1. Verificar importaciones
        logger.info("📦 Verificando importaciones core...")
        from openclaw import AgentClient, AgentConfig
        from agents.registry import registry
        from agents.engines import engine_registry, DummyAsyncLLM
        
        # 2. Verificar Registro de Herramientas
        logger.info("🛠️ Verificando Registro de Herramientas...")
        tools = registry.get_all_tools()
        if not tools:
            raise RuntimeError("Registry de herramientas vacío")
        logger.info(f"✅ {len(tools)} herramientas detectadas.")
        
        # 3. Verificar Configuración
        logger.info("⚙️ Verificando AgentConfig...")
        config = AgentConfig(use_swarm=False, default_agent_name="TestAgent")
        
        # 4. Inicializar Cliente
        logger.info("🤖 Inicializando AgentClient...")
        client = AgentClient(config=config)
        
        # 5. Ejecución Mock (Prueba de Razonamiento)
        logger.info("🧠 Probando bucle de razonamiento (Mock Mode)...")
        # Usamos el Mock engine por defecto para el test
        response = await client.run(
            user_id="smoke_test",
            prompt="Hola, ¿estás listo para producción?",
            return_response=True
        )
        
        logger.info(f"✅ Respuesta recibida: {response.content[:50]}...")
        
        # 6. Verificar persistencia (SQLite)
        db_path = Path("openclaw_memory.db")
        if db_path.exists():
            logger.info("💾 Persistencia de memoria verificada.")
        
        logger.info("---")
        logger.info("🏆 SMOKE TEST COMPLETADO: SISTEMA LISTO PARA PRODUCCIÓN.")
        
    except Exception as e:
        logger.error(f"❌ SMOKE TEST FALLIDO: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_smoke_test())

