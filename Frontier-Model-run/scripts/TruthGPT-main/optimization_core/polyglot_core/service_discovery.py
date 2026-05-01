"""
Service discovery utilities for polyglot_core.

Provides service registration, discovery, and health checking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time


class ServiceStatus(str, Enum):
    """Service status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


@dataclass
class ServiceInfo:
    """Service information."""
    name: str
    address: str
    port: int
    version: str = "1.0.0"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None


class ServiceRegistry:
    """
    Service registry for polyglot_core.
    
    Manages service registration and discovery.
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._heartbeat_timeout: float = 30.0  # seconds
    
    def register(
        self,
        name: str,
        address: str,
        port: int,
        version: str = "1.0.0",
        **metadata
    ) -> str:
        """
        Register a service.
        
        Args:
            name: Service name
            address: Service address
            port: Service port
            version: Service version
            **metadata: Additional metadata
            
        Returns:
            Service ID
        """
        service_id = f"{name}_{address}_{port}"
        
        service = ServiceInfo(
            name=name,
            address=address,
            port=port,
            version=version,
            metadata=metadata
        )
        
        self._services[service_id] = service
        return service_id
    
    def deregister(self, service_id: str):
        """Deregister a service."""
        self._services.pop(service_id, None)
    
    def heartbeat(self, service_id: str):
        """Update service heartbeat."""
        if service_id in self._services:
            self._services[service_id].last_heartbeat = datetime.now()
            self._services[service_id].status = ServiceStatus.HEALTHY
    
    def discover(self, name: Optional[str] = None) -> List[ServiceInfo]:
        """
        Discover services.
        
        Args:
            name: Optional service name filter
            
        Returns:
            List of matching services
        """
        services = list(self._services.values())
        
        # Filter by name if provided
        if name:
            services = [s for s in services if s.name == name]
        
        # Filter out stale services
        now = datetime.now()
        healthy_services = []
        
        for service in services:
            if service.last_heartbeat:
                elapsed = (now - service.last_heartbeat).total_seconds()
                if elapsed > self._heartbeat_timeout:
                    service.status = ServiceStatus.UNHEALTHY
                else:
                    healthy_services.append(service)
            else:
                healthy_services.append(service)
        
        return healthy_services
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service by ID."""
        return self._services.get(service_id)
    
    def list_services(self) -> List[ServiceInfo]:
        """List all registered services."""
        return list(self._services.values())
    
    def update_status(self, service_id: str, status: ServiceStatus):
        """Update service status."""
        if service_id in self._services:
            self._services[service_id].status = status


# Global service registry
_global_service_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """Get global service registry."""
    return _global_service_registry


def register_service(name: str, address: str, port: int, **kwargs) -> str:
    """Convenience function to register service."""
    return _global_service_registry.register(name, address, port, **kwargs)













