"""
Base interface for agent memory systems.
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """Interfaz base para sistemas de memoria de agentes."""

    @abstractmethod
    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persiste un mensaje en la memoria."""

    @abstractmethod
    async def get_history(
        self, user_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Recupera el historial de mensajes."""

    @abstractmethod
    async def clear_memory(self, user_id: str) -> None:
        """Limpia la memoria de un usuario específico."""

