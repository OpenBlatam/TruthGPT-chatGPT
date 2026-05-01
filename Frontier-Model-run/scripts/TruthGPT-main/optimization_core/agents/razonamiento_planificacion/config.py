from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

class AgentSettings(BaseSettings):
    """Configuración central para el Agente TruthGPT."""
    
    # Inferencia
    MAX_ITERATIONS: int = Field(default=5, description="Máximo de bucles ReAct por mensaje")
    MODEL_TEMPERATURE: float = 0.7
    
    # Persistencia
    DATABASE_PATH: str = "data/agent_memory.db"
    
    # Seguridad
    FORBIDDEN_BASH_COMMANDS: List[str] = ["rm", "chmod", "format", "del", "mkfs"]
    
    # Prompting
    AGENT_NAME: str = "TruthGPT"
    SYSTEM_PROMPT_TEMPLATE: str = (
        "You are {name}, an elite personal and autonomous AI assistant.\n"
        "Analyze the context and use the tools only if absolutely necessary.\n"
    )

    class Config:
        env_prefix = "TRUTHGPT_" # Permite TRUTHGPT_MAX_ITERATIONS como var de entorno

settings = AgentSettings()

