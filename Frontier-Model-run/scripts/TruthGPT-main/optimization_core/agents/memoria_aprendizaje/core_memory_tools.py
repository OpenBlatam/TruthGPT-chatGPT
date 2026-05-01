"""
Agent Tools for Core Memory Manipulation.
"""

try:
    from agents.razonamiento_planificacion.tools import BaseTool, ToolResult
except ImportError:
    from ..razonamiento_planificacion.tools import BaseTool, ToolResult

class CoreMemoryAppendTool(BaseTool):
    """Tool to append new information to the agent's core memory."""
    
    name = "core_memory_append"
    description = (
        "Añade información nueva a la memoria CORE (persona o human). "
        "Usa 'human' para datos del usuario y 'persona' para reglas del agente. "
        "Formato: block:content (ej: human:Le gusta el café solo)"
    )

    def __init__(self, core_memory: CoreMemory):
        super().__init__()
        self.core_memory = core_memory

    async def run(self, command: str) -> ToolResult:
        if ":" not in command:
            return ToolResult(output="Error: Formato inválido. Usa 'block:content'.")
        
        block, content = command.split(":", 1)
        block = block.strip().lower()
        content = content.strip()
        
        return ToolResult(
            output=f"Preparando actualización de memoria CORE ({block})...",
            signal="core_memory_append",
            metadata={"block": block, "content": content}
        )

class CoreMemoryReplaceTool(BaseTool):
    """Tool to replace an entire block of core memory."""
    
    name = "core_memory_replace"
    description = (
        "Reemplaza un bloque entero de la memoria CORE. "
        "Formato: block:new_full_content"
    )

    def __init__(self, core_memory: CoreMemory):
        super().__init__()
        self.core_memory = core_memory

    async def run(self, command: str) -> ToolResult:
        if ":" not in command:
            return ToolResult(output="Error: Formato inválido.")
        
        block, content = command.split(":", 1)
        block = block.strip().lower()
        content = content.strip()

        return ToolResult(
            output=f"Preparando reemplazo total de memoria CORE ({block})...",
            signal="core_memory_replace",
            metadata={"block": block, "content": content}
        )

