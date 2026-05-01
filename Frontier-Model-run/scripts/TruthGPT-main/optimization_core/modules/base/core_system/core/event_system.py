"""
Event system for decoupled communication between modules.
Implements Observer pattern for maximum modularity.
"""
import logging
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types."""
    TRAINING_STARTED = "training.started"
    TRAINING_STEP = "training.step"
    TRAINING_EPOCH = "training.epoch"
    TRAINING_FINISHED = "training.finished"
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_FINISHED = "evaluation.finished"
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_LOADED = "checkpoint.loaded"
    MODEL_LOADED = "model.loaded"
    MODEL_SAVED = "model.saved"
    ERROR_OCCURRED = "error.occurred"
    METRIC_LOGGED = "metric.logged"


@dataclass
class Event:
    """Event data structure."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventEmitter:
    """
    Event emitter for publishing events.
    Implements Observer pattern.
    """
    
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._global_listeners: List[Callable] = []
        self._lock = threading.Lock()
    
    def on(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        """
        Register an event handler.
        
        Args:
            event_type: Type of event to listen to
            handler: Handler function
        """
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(handler)
            logger.debug(f"Handler registered for {event_type.value}")
    
    def once(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        """
        Register a one-time event handler.
        
        Args:
            event_type: Type of event
            handler: Handler function
        """
        def wrapper(event: Event):
            handler(event)
            self.off(event_type, wrapper)
        
        self.on(event_type, wrapper)
    
    def off(
        self,
        event_type: EventType,
        handler: Optional[Callable] = None
    ) -> None:
        """
        Unregister an event handler.
        
        Args:
            event_type: Type of event
            handler: Specific handler to remove (all if None)
        """
        with self._lock:
            if event_type in self._listeners:
                if handler:
                    self._listeners[event_type].remove(handler)
                else:
                    self._listeners[event_type].clear()
                logger.debug(f"Handler unregistered for {event_type.value}")
    
    def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> None:
        """
        Emit an event.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Event source identifier
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source
        )
        
        # Call specific handlers
        with self._lock:
            handlers = self._listeners.get(event_type, [])
            all_handlers = handlers + self._global_listeners
        
        for handler in all_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler for {event_type.value}: {e}",
                    exc_info=True
                )
        
        logger.debug(f"Event emitted: {event_type.value} from {source}")
    
    def on_any(self, handler: Callable[[Event], None]) -> None:
        """
        Register a handler for all events.
        
        Args:
            handler: Handler function
        """
        with self._lock:
            self._global_listeners.append(handler)
    
    def remove_all_listeners(self, event_type: Optional[EventType] = None) -> None:
        """
        Remove all listeners for an event type or all events.
        
        Args:
            event_type: Event type (None for all)
        """
        with self._lock:
            if event_type:
                self._listeners.pop(event_type, None)
            else:
                self._listeners.clear()
                self._global_listeners.clear()


# Global event emitter
_event_emitter = EventEmitter()


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter."""
    return _event_emitter


def emit_event(
    event_type: EventType,
    data: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None
) -> None:
    """Emit an event using the global emitter."""
    _event_emitter.emit(event_type, data, source)


def on_event(
    event_type: EventType,
    handler: Callable[[Event], None]
) -> None:
    """Register an event handler."""
    _event_emitter.on(event_type, handler)



