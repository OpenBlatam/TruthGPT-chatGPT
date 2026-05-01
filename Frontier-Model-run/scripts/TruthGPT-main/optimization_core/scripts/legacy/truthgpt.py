"""
TruthGPT Researcher SDK
A high-level Python entry point for researchers to interact with the swarm.
"""

from typing import Optional, List
from agents.client import AgentClient
from modules.base.core_system.core.papers import PaperRegistry, PaperAdapter

class TruthGPT:
    """Interface unificada para interactuar con agentes y papers."""
    
    def __init__(self, use_swarm: bool = True):
        self.client = AgentClient(use_swarm=use_swarm)
        self.registry = PaperRegistry()
        self.adapter = PaperAdapter()
        
    async def ask(self, prompt: str, user_id: str = "researcher"):
        """Realiza una consulta al enjambre de agentes."""
        return await self.client.run(user_id=user_id, prompt=prompt)
        
    def list_papers(self, category: Optional[str] = None) -> List[str]:
        """Lista los papers de investigación disponibles."""
        return self.registry.list_available_papers(category=category)
        
    def get_paper_info(self, paper_id: str):
        """Obtiene metadatos de un paper específico."""
        return self.registry.get_paper_metadata(paper_id)
        
    def apply_paper(self, model, paper_id: str, config: Optional[dict] = None):
        """Aplica una técnica de un paper a un modelo PyTorch."""
        return self.adapter.apply_paper(model, paper_id, config=config)

# Singleton instance for easy import
api = TruthGPT()

