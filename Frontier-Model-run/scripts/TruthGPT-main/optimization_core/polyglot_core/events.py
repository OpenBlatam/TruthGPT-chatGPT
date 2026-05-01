"""
Event system for polyglot_core.

Provides event-driven architecture for monitoring and reacting to operations.
"""

from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading


class EventType(str, Enum):
    """Event types."""
    BACKEND_SELECTED = "backend_selected"
    OPERATION_STARTED = "operation_started"
    OPERATION_COMPLETED = "operation_completed"
    OPERATION_FAILED = "operation_failed"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    ERROR_OCCURRED = "error_occurred"
    CONFIG_CHANGED = "config_changed"


@dataclass
class Event:
    """Event data."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None


class EventEmitter:
    """
    Event emitter for polyglot_core.
    
    Allows components to emit events and listeners to react to them.
    """
    
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._lock = threading.Lock() if threading else None
    
    def on(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Register event listener.
        
        Args:
            event_type: Event type to listen for
            callback: Callback function (takes Event as argument)
        """
        if self._lock:
            with self._lock:
                if event_type not in self._listeners:
                    self._listeners[event_type] = []
                self._listeners[event_type].append(callback)
        else:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(callback)
    
    def off(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Unregister event listener.
        
        Args:
            event_type: Event type
            callback: Callback function to remove
        """
        if self._lock:
            with self._lock:
                if event_type in self._listeners:
                    try:
                        self._listeners[event_type].remove(callback)
                    except ValueError:
                        pass
        else:
            if event_type in self._listeners:
                try:
                    self._listeners[event_type].remove(callback)
                except ValueError:
                    pass
    
    def emit(self, event_type: EventType, data: Optional[Dict[str, Any]] = None, source: Optional[str] = None):
        """
        Emit an event.
        
        Args:
            event_type: Event type
            data: Event data
            source: Event source
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source
        )
        
        listeners = []
        if self._lock:
            with self._lock:
                listeners = self._listeners.get(event_type, []).copy()
        else:
            listeners = self._listeners.get(event_type, []).copy()
        
        for callback in listeners:
            try:
                callback(event)
            except Exception as e:
                # Log error but don't stop other listeners
                try:
                    from .logging import get_logger
                    logger = get_logger()
                    logger.error(f"Error in event listener: {e}", exc_info=True)
                except Exception:
                    pass
    
    def once(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Register one-time event listener.
        
        Args:
            event_type: Event type
            callback: Callback function
        """
        def wrapper(event: Event):
            callback(event)
            self.off(event_type, wrapper)
        
        self.on(event_type, wrapper)
    
    def remove_all_listeners(self, event_type: Optional[EventType] = None):
        """
        Remove all listeners for an event type.
        
        Args:
            event_type: Event type (None for all)
        """
        if self._lock:
            with self._lock:
                if event_type:
                    self._listeners.pop(event_type, None)
                else:
                    self._listeners.clear()
        else:
            if event_type:
                self._listeners.pop(event_type, None)
            else:
                self._listeners.clear()


# Global event emitter
_global_emitter = EventEmitter()


def get_event_emitter() -> EventEmitter:
    """Get global event emitter."""
    return _global_emitter


def emit_event(event_type: EventType, data: Optional[Dict[str, Any]] = None, source: Optional[str] = None):
    """Emit an event."""
    _global_emitter.emit(event_type, data, source)


def on_event(event_type: EventType, callback: Callable[[Event], None]):
    """Register event listener."""
    _global_emitter.on(event_type, callback)


def off_event(event_type: EventType, callback: Callable[[Event], None]):
    """Unregister event listener."""
    _global_emitter.off(event_type, callback)













