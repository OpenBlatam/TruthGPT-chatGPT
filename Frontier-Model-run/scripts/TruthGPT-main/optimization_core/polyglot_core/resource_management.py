"""
Resource management utilities for polyglot_core.

Provides resource allocation, monitoring, and optimization.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading


class ResourceType(str, Enum):
    """Resource type."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceUsage:
    """Resource usage information."""
    resource_type: ResourceType
    used: float
    total: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def percentage(self) -> float:
        """Get usage percentage."""
        return (self.used / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def available(self) -> float:
        """Get available amount."""
        return self.total - self.used


@dataclass
class ResourceQuota:
    """Resource quota definition."""
    resource_type: ResourceType
    limit: float
    unit: str = ""
    soft_limit: Optional[float] = None
    hard_limit: Optional[float] = None


class ResourceManager:
    """
    Resource manager for polyglot_core.
    
    Manages resource allocation and monitoring.
    """
    
    def __init__(self):
        self._quotas: Dict[ResourceType, ResourceQuota] = {}
        self._usage_history: Dict[ResourceType, List[ResourceUsage]] = {}
        self._allocations: Dict[str, Dict[ResourceType, float]] = {}
        self._lock = threading.Lock() if threading else None
    
    def set_quota(self, quota: ResourceQuota):
        """Set resource quota."""
        if self._lock:
            with self._lock:
                self._quotas[quota.resource_type] = quota
        else:
            self._quotas[quota.resource_type] = quota
    
    def get_quota(self, resource_type: ResourceType) -> Optional[ResourceQuota]:
        """Get resource quota."""
        return self._quotas.get(resource_type)
    
    def record_usage(self, usage: ResourceUsage):
        """
        Record resource usage.
        
        Args:
            usage: Resource usage information
        """
        if self._lock:
            with self._lock:
                if usage.resource_type not in self._usage_history:
                    self._usage_history[usage.resource_type] = []
                self._usage_history[usage.resource_type].append(usage)
                
                # Keep only recent history
                if len(self._usage_history[usage.resource_type]) > 1000:
                    self._usage_history[usage.resource_type] = self._usage_history[usage.resource_type][-1000:]
        else:
            if usage.resource_type not in self._usage_history:
                self._usage_history[usage.resource_type] = []
            self._usage_history[usage.resource_type].append(usage)
            
            if len(self._usage_history[usage.resource_type]) > 1000:
                self._usage_history[usage.resource_type] = self._usage_history[usage.resource_type][-1000:]
    
    def allocate(
        self,
        allocation_id: str,
        resources: Dict[ResourceType, float]
    ) -> bool:
        """
        Allocate resources.
        
        Args:
            allocation_id: Allocation identifier
            resources: Dictionary of resource types to amounts
            
        Returns:
            True if allocation successful
        """
        if self._lock:
            with self._lock:
                # Check quotas
                for resource_type, amount in resources.items():
                    quota = self._quotas.get(resource_type)
                    if quota:
                        # Check current usage
                        current_usage = self._get_current_usage(resource_type)
                        if current_usage + amount > quota.limit:
                            return False
                
                # Allocate
                self._allocations[allocation_id] = resources
                return True
        else:
            # Check quotas
            for resource_type, amount in resources.items():
                quota = self._quotas.get(resource_type)
                if quota:
                    current_usage = self._get_current_usage(resource_type)
                    if current_usage + amount > quota.limit:
                        return False
            
            self._allocations[allocation_id] = resources
            return True
    
    def deallocate(self, allocation_id: str):
        """Deallocate resources."""
        if self._lock:
            with self._lock:
                self._allocations.pop(allocation_id, None)
        else:
            self._allocations.pop(allocation_id, None)
    
    def _get_current_usage(self, resource_type: ResourceType) -> float:
        """Get current usage for resource type."""
        total = 0.0
        for allocation in self._allocations.values():
            total += allocation.get(resource_type, 0.0)
        return total
    
    def get_current_usage(self, resource_type: ResourceType) -> ResourceUsage:
        """
        Get current resource usage.
        
        Args:
            resource_type: Resource type
            
        Returns:
            Resource usage information
        """
        quota = self._quotas.get(resource_type)
        total = quota.limit if quota else 0.0
        used = self._get_current_usage(resource_type)
        
        return ResourceUsage(
            resource_type=resource_type,
            used=used,
            total=total
        )
    
    def get_usage_history(
        self,
        resource_type: ResourceType,
        limit: int = 100
    ) -> List[ResourceUsage]:
        """Get usage history."""
        history = self._usage_history.get(resource_type, [])
        return history[-limit:] if len(history) > limit else history
    
    def check_quota_exceeded(self) -> List[ResourceType]:
        """Check which quotas are exceeded."""
        exceeded = []
        
        for resource_type in ResourceType:
            usage = self.get_current_usage(resource_type)
            if usage.percentage > 100:
                exceeded.append(resource_type)
        
        return exceeded


# Global resource manager
_global_resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get global resource manager."""
    return _global_resource_manager


def allocate_resources(allocation_id: str, resources: Dict[ResourceType, float]) -> bool:
    """Convenience function to allocate resources."""
    return _global_resource_manager.allocate(allocation_id, resources)


