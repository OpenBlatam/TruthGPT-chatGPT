"""
Health check system for polyglot_core.

Provides health monitoring, backend availability checks, and system status.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import time


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'details': self.details
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    components: List[ComponentHealth] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'components': [c.to_dict() for c in self.components],
            'summary': self.summary
        }


class HealthChecker:
    """
    Health checker for polyglot_core components.
    
    Monitors backend availability, performance, and system resources.
    """
    
    def __init__(self):
        self._components: Dict[str, ComponentHealth] = {}
        self._last_check: Optional[datetime] = None
    
    def check_backend(self, backend_name: str) -> ComponentHealth:
        """
        Check backend health.
        
        Args:
            backend_name: Backend name
            
        Returns:
            ComponentHealth
        """
        from .backend import is_backend_available, get_backend_info
        
        try:
            available = is_backend_available(backend_name)
            info = get_backend_info(backend_name)
            
            if available:
                status = HealthStatus.HEALTHY
                message = f"Backend {backend_name} is available"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Backend {backend_name} is not available"
            
            details = {
                'available': available,
                'version': info.version if info else None,
                'features': info.features if info else []
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking backend {backend_name}: {str(e)}"
            details = {'error': str(e)}
        
        health = ComponentHealth(
            name=f"backend_{backend_name}",
            status=status,
            message=message,
            last_check=datetime.now(),
            details=details
        )
        
        self._components[f"backend_{backend_name}"] = health
        return health
    
    def check_module(self, module_name: str) -> ComponentHealth:
        """
        Check module health.
        
        Args:
            module_name: Module name
            
        Returns:
            ComponentHealth
        """
        from .integration import check_polyglot_availability
        
        try:
            availability = check_polyglot_availability()
            available = availability.get(module_name, False)
            
            if available:
                status = HealthStatus.HEALTHY
                message = f"Module {module_name} is available"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Module {module_name} is not available"
            
            details = {'available': available}
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking module {module_name}: {str(e)}"
            details = {'error': str(e)}
        
        health = ComponentHealth(
            name=f"module_{module_name}",
            status=status,
            message=message,
            last_check=datetime.now(),
            details=details
        )
        
        self._components[f"module_{module_name}"] = health
        return health
    
    def check_performance(self) -> ComponentHealth:
        """
        Check performance health.
        
        Returns:
            ComponentHealth
        """
        try:
            from .metrics import get_metrics_collector
            
            collector = get_metrics_collector()
            summaries = collector.get_all_summaries()
            
            # Check for high latency
            high_latency = False
            for name, summary in summaries.items():
                if 'latency' in name and summary.p95 > 1000:  # > 1 second
                    high_latency = True
                    break
            
            if high_latency:
                status = HealthStatus.DEGRADED
                message = "High latency detected in some operations"
            else:
                status = HealthStatus.HEALTHY
                message = "Performance is healthy"
            
            details = {
                'metrics_count': len(summaries),
                'high_latency': high_latency
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking performance: {str(e)}"
            details = {'error': str(e)}
        
        health = ComponentHealth(
            name="performance",
            status=status,
            message=message,
            last_check=datetime.now(),
            details=details
        )
        
        self._components["performance"] = health
        return health
    
    def check_all(self) -> SystemHealth:
        """
        Check all components.
        
        Returns:
            SystemHealth
        """
        components = []
        
        # Check backends
        from .backend import get_available_backends
        backends = get_available_backends()
        for backend_info in backends:
            components.append(self.check_backend(backend_info.name))
        
        # Check key modules
        for module in ['cache', 'attention', 'compression', 'inference']:
            components.append(self.check_module(module))
        
        # Check performance
        components.append(self.check_performance())
        
        # Determine overall status
        statuses = [c.status for c in components]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Summary
        summary = {
            'total_components': len(components),
            'healthy': sum(1 for c in components if c.status == HealthStatus.HEALTHY),
            'degraded': sum(1 for c in components if c.status == HealthStatus.DEGRADED),
            'unhealthy': sum(1 for c in components if c.status == HealthStatus.UNHEALTHY),
            'unknown': sum(1 for c in components if c.status == HealthStatus.UNKNOWN)
        }
        
        self._last_check = datetime.now()
        
        return SystemHealth(
            status=overall_status,
            components=components,
            summary=summary
        )
    
    def get_component(self, name: str) -> Optional[ComponentHealth]:
        """Get component health by name."""
        return self._components.get(name)
    
    def print_status(self):
        """Print formatted health status."""
        health = self.check_all()
        
        print("\n" + "=" * 80)
        print("Polyglot Core - Health Status")
        print("=" * 80)
        print(f"Overall Status: {health.status.value.upper()}")
        print(f"Timestamp: {health.timestamp.isoformat()}")
        print()
        print("Components:")
        for component in health.components:
            status_icon = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.DEGRADED: "⚠",
                HealthStatus.UNHEALTHY: "✗",
                HealthStatus.UNKNOWN: "?"
            }.get(component.status, "?")
            
            print(f"  {status_icon} {component.name}: {component.status.value}")
            if component.message:
                print(f"    {component.message}")
        
        print()
        print("Summary:")
        for key, value in health.summary.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")


# Global health checker
_global_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    return _global_health_checker


def check_health() -> SystemHealth:
    """Check system health."""
    return _global_health_checker.check_all()


def print_health_status():
    """Print health status."""
    _global_health_checker.print_status()













