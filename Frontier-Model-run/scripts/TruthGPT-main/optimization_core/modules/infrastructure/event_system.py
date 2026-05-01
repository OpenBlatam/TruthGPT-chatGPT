"""
Event system for optimization_core.

Provides an event-driven architecture for component communication.
"""
import logging
import time
from typing import Dict, List, Callable, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types."""
    ENGINE_INITIALIZED = "engine_initialized"
    ENGINE_ERROR = "engine_error"
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    METRICS_UPDATED = "metrics_updated"
    CONFIG_CHANGED = "config_changed"
    CUSTOM = "custom"

class Event(BaseModel):
    """Event data structure."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    event_type: EventType
    source: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (Pydantic model_dump)."""
        return self.model_dump()

class EventEmitter:
    """Event emitter for publishing events."""
    
    def __init__(self, name: str = "EventEmitter"):
        """
        Initialize event emitter.
        
        Args:
            name: Name of emitter
        """
        self.name = name
        self._listeners: Dict[EventType, List[Callable]] = defaultdict(list)
        self._global_listeners: List[Callable] = []
    
    def on(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ):
        """
        Register event listener.
        
        Args:
            event_type: Type of event
            callback: Callback function
        """
        self._listeners[event_type].append(callback)
        logger.debug(f"Registered listener for {event_type.value} in {self.name}")
    
    def on_any(self, callback: Callable[[Event], None]):
        """
        Register listener for all events.
        
        Args:
            callback: Callback function
        """
        self._global_listeners.append(callback)
    
    def off(
        self,
        event_type: EventType,
        callback: Optional[Callable] = None
    ):
        """
        Unregister event listener.
        
        Args:
            event_type: Type of event
            callback: Specific callback to remove (removes all if None)
        """
        if callback is None:
            self._listeners[event_type].clear()
        else:
            if callback in self._listeners[event_type]:
                self._listeners[event_type].remove(callback)
    
    def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ):
        """
        Emit an event.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Event source (defaults to emitter name)
        """
        event = Event(
            event_type=event_type,
            source=source or self.name,
            data=data or {}
        )
        
        # Call specific listeners
        for callback in self._listeners[event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    f"Error in event listener for {event_type.value}: {e}",
                    exc_info=True
                )
        
        # Call global listeners
        for callback in self._global_listeners:
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    f"Error in global event listener: {e}",
                    exc_info=True
                )
    
    def clear(self):
        """Clear all listeners."""
        self._listeners.clear()
        self._global_listeners.clear()

class EventBus:
    """Global event bus for cross-component communication."""
    
    def __init__(self):
        """Initialize event bus."""
        self._emitters: Dict[str, EventEmitter] = {}
        self._global_listeners: List[Tuple[EventType, Callable]] = []
    
    def get_emitter(self, name: str) -> EventEmitter:
        """
        Get or create event emitter.
        
        Args:
            name: Emitter name
        
        Returns:
            EventEmitter instance
        """
        if name not in self._emitters:
            self._emitters[name] = EventEmitter(name=name)
        return self._emitters[name]
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
        emitter_name: Optional[str] = None
    ):
        """
        Subscribe to events.
        
        Args:
            event_type: Type of event
            callback: Callback function
            emitter_name: Specific emitter (all if None)
        """
        if emitter_name:
            emitter = self.get_emitter(emitter_name)
            emitter.on(event_type, callback)
        else:
            # Subscribe to all emitters
            self._global_listeners.append((event_type, callback))
            for emitter in self._emitters.values():
                emitter.on(event_type, callback)
    
    def publish(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: str = "global"
    ):
        """
        Publish event to all subscribers.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Event source
        """
        event = Event(
            event_type=event_type,
            source=source,
            data=data or {}
        )
        
        # Publish to all emitters
        for emitter in self._emitters.values():
            emitter.emit(event_type, data, source)
        
        # Call global listeners
        for event_type_filter, callback in self._global_listeners:
            if event_type_filter == event_type:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in global listener: {e}", exc_info=True)

# Global event bus instance
_global_event_bus = EventBus()

def get_event_bus() -> EventBus:
    """Get global event bus."""
    return _global_event_bus

def get_emitter(name: str) -> EventEmitter:
    """Get or create event emitter from global bus."""
    return _global_event_bus.get_emitter(name)
