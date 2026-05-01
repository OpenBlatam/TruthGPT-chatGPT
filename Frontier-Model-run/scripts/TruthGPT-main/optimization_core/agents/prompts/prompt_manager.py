"""
Centralized Prompt Management for TruthGPT Agents.
Decouples system instructions from agent logic for easier tuning.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class PromptTemplate(BaseModel):
    template: str
    description: Optional[str] = None

class PromptManager:
    """Manages system prompts and dynamic instruction templates."""
    
    _instance = None
    _templates: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._instance._load_defaults()
        return cls._instance

    def _load_defaults(self):
        """Initializes default system prompts."""
        self._templates["base_agent"] = (
            "You are {name}, an autonomous agent powered by TruthGPT.\n"
            "Your role: {role}\n"
            "Current date: 2025 Standard Stack.\n"
        )
        
        self._templates["react_core"] = (
            "You operate using a ReAct (Reasoning and Action) loop.\n"
            "Analyse the user request and decide if you need tools.\n"
            "Always maintain a high standard of reasoning."
        )

        self._templates["json_output"] = (
            "IMPORTANTE: Debes responder ÚNICA y EXCLUSIVAMENTE con un JSON puro que cumpla estrictamente este esquema:\n"
            "{schema}\n"
            "No incluyas NADA de texto fuera del JSON."
        )

    def get_prompt(self, key: str, **kwargs) -> str:
        """Retrieves a prompt and injects variables."""
        template = self._templates.get(key, "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing key for prompt template '{key}': {e}")
            return template

    def register_template(self, key: str, template: str):
        """Adds or updates a prompt template."""
        self._templates[key] = template

# Global singleton
prompt_manager = PromptManager()

