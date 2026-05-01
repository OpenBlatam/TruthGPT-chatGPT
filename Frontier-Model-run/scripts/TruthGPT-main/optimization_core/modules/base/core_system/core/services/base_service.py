"""
Base service class with common functionality.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..service_registry import ServiceRegistry
from ..event_system import EventEmitter, EventType, emit_event

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Base class for all services.
    Provides common functionality like event emission and service access.
    """
    
    def __init__(
        self,
        registry: Optional[ServiceRegistry] = None,
        event_emitter: Optional[EventEmitter] = None,
        name: Optional[str] = None
    ):
        """
        Initialize base service.
        
        Args:
            registry: Service registry
            event_emitter: Event emitter
            name: Service name
        """
        self.registry = registry or ServiceRegistry()
        self.event_emitter = event_emitter or EventEmitter()
        self.name = name or self.__class__.__name__
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return
        
        self._do_initialize()
        self._initialized = True
        logger.info(f"Service '{self.name}' initialized")
    
    def _do_initialize(self) -> None:
        """Override for custom initialization."""
        pass
    
    def get_service(self, name: str, **kwargs) -> Any:
        """
        Get a service from the registry.
        
        Args:
            name: Service name
            **kwargs: Service arguments
        
        Returns:
            Service instance
        """
        return self.registry.get(name, **kwargs)
    
    def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit an event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        self.event_emitter.emit(event_type, data, self.name)
    
    def validate_config(self, config: Dict[str, Any], required_keys: list[str]) -> None:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            required_keys: Required keys
        
        Raises:
            ValueError: If validation fails
        """
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required configuration keys: {missing}")



