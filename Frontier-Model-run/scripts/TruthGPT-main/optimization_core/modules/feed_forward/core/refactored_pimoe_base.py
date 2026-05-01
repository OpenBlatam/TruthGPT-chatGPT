"""
Refactored PiMoE Base Components
================================

Base classes and utilities for the refactored production system.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
import logging
import time

from .interfaces import (
    LoggerProtocol, MonitorProtocol, ErrorHandlerProtocol, RequestQueueProtocol,
    SystemConfig, ProductionConfig, LogLevel, ProductionMode, RequestData, ResponseData,
    OptimizationLevel
)

T = TypeVar('T')

@dataclass
class BaseConfig:
    """Base configuration."""
    pass

class BaseService(ABC):
    """Abstract base service."""
    def __init__(self, config: ProductionConfig, logger: Optional[LoggerProtocol] = None):
        self.config = config
        self.logger = logger
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the service."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the service."""
        pass

class DIContainer:
    """Simple Dependency Injection Container."""
    def __init__(self):
        self._services: Dict[Any, Any] = {}

    def register_instance(self, protocol: Any, instance: Any) -> None:
        self._services[protocol] = instance

    def get(self, protocol: Any) -> Any:
        if protocol not in self._services:
            raise KeyError(f"Service {protocol} not registered")
        return self._services[protocol]

class Event:
    """System event."""
    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        self.data = data
        self.timestamp = time.time()

class EventBus:
    """Simple synchronous Event Bus."""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_name: str, handler: Callable[[Event], None]) -> None:
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    def publish(self, event: Event) -> None:
        if event.name in self._subscribers:
            for handler in self._subscribers[event.name]:
                try:
                    handler(event)
                except Exception as e:
                    logging.error(f"Error handling event {event.name}: {e}")

class ResourceManager:
    """Manages lifecycle of resources."""
    def __init__(self, config: ProductionConfig):
        self.config = config
        self._resources: List[Tuple[str, Any, Callable[[], None]]] = []

    def register_resource(self, name: str, resource: Any, cleanup_fn: Callable[[], None]) -> None:
        self._resources.append((name, resource, cleanup_fn))

    def cleanup_all(self) -> None:
        for name, _, cleanup in reversed(self._resources):
            try:
                cleanup()
            except Exception as e:
                logging.error(f"Error cleaning up resource {name}: {e}")
        self._resources.clear()

class MetricsCollector:
    """Collects and aggregates metrics."""
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

    def increment_counter(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def record_histogram(self, name: str, value: float) -> None:
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
        # Keep only recent history to avoid memory issues
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'counters': self._counters.copy(),
            'gauges': self._gauges.copy(),
            'histograms': {k: {'count': len(v), 'avg': sum(v)/len(v) if v else 0} for k, v in self._histograms.items()}
        }

class HealthChecker:
    """Registry for health checks."""
    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        self._checks[name] = check_fn

    def run_checks(self) -> Dict[str, Any]:
        results = {}
        all_healthy = True
        for name, check in self._checks.items():
            try:
                is_healthy = check()
                results[name] = 'healthy' if is_healthy else 'unhealthy'
                if not is_healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = f'error: {e}'
                all_healthy = False
        
        return {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'checks': results,
            'timestamp': time.time()
        }

class BasePiMoESystem(ABC):
    """Abstract base class for PiMoE systems."""
    def __init__(self, config: ProductionConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
        
    @abstractmethod
    def process_request(self, request_data: RequestData) -> ResponseData:
        pass

class ServiceFactory:
    """Helper for creating services."""
    @staticmethod
    def create_service(service_cls: Type[T], config: ProductionConfig, **kwargs) -> T:
        return service_cls(config, **kwargs)

