"""
Load balancing utilities for polyglot_core.

Provides load balancing across multiple backends and services.
"""

from typing import List, Optional, Callable, Any, Dict
from dataclasses import dataclass
from enum import Enum
import random
import statistics


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategy."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LATENCY = "least_latency"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class BackendInstance:
    """Backend instance for load balancing."""
    id: str
    backend: Any
    weight: float = 1.0
    active_connections: int = 0
    avg_latency_ms: float = 0.0
    is_healthy: bool = True


class LoadBalancer:
    """
    Load balancer for polyglot_core.
    
    Distributes requests across multiple backend instances.
    """
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self._instances: List[BackendInstance] = []
        self._round_robin_index: int = 0
    
    def add_instance(self, instance_id: str, backend: Any, weight: float = 1.0) -> str:
        """
        Add backend instance.
        
        Args:
            instance_id: Instance identifier
            backend: Backend instance
            weight: Instance weight (for weighted strategies)
            
        Returns:
            Instance ID
        """
        instance = BackendInstance(
            id=instance_id,
            backend=backend,
            weight=weight
        )
        
        self._instances.append(instance)
        return instance_id
    
    def remove_instance(self, instance_id: str):
        """Remove backend instance."""
        self._instances = [i for i in self._instances if i.id != instance_id]
    
    def select_instance(self, key: Optional[str] = None) -> Optional[BackendInstance]:
        """
        Select backend instance based on strategy.
        
        Args:
            key: Optional key for consistent hashing
            
        Returns:
            Selected backend instance
        """
        healthy_instances = [i for i in self._instances if i.is_healthy]
        
        if not healthy_instances:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            instance = healthy_instances[self._round_robin_index % len(healthy_instances)]
            self._round_robin_index += 1
            return instance
        
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(healthy_instances)
        
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            # Select based on weights
            total_weight = sum(i.weight for i in healthy_instances)
            if total_weight == 0:
                return random.choice(healthy_instances)
            
            r = random.random() * total_weight
            cumulative = 0.0
            for instance in healthy_instances:
                cumulative += instance.weight
                if r <= cumulative:
                    return instance
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(healthy_instances, key=lambda x: x.active_connections)
        
        elif self.strategy == LoadBalanceStrategy.LEAST_LATENCY:
            return min(healthy_instances, key=lambda x: x.avg_latency_ms)
        
        elif self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            if key:
                # Simple hash-based selection
                hash_value = hash(key)
                index = hash_value % len(healthy_instances)
                return healthy_instances[index]
            else:
                return random.choice(healthy_instances)
        
        return healthy_instances[0]
    
    def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation through load balancer.
        
        Args:
            operation: Operation function (takes backend as first arg)
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Operation result
        """
        instance = self.select_instance()
        
        if not instance:
            raise RuntimeError("No healthy backend instances available")
        
        instance.active_connections += 1
        
        try:
            import time
            start = time.perf_counter()
            result = operation(instance.backend, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            
            # Update latency (exponential moving average)
            instance.avg_latency_ms = 0.9 * instance.avg_latency_ms + 0.1 * latency_ms
            
            return result
        finally:
            instance.active_connections -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'total_instances': len(self._instances),
            'healthy_instances': sum(1 for i in self._instances if i.is_healthy),
            'strategy': self.strategy.value,
            'instances': [
                {
                    'id': i.id,
                    'weight': i.weight,
                    'active_connections': i.active_connections,
                    'avg_latency_ms': i.avg_latency_ms,
                    'is_healthy': i.is_healthy
                }
                for i in self._instances
            ]
        }


# Global load balancer
_global_load_balancer = LoadBalancer()


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer."""
    return _global_load_balancer


def create_load_balancer(strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN) -> LoadBalancer:
    """Create a new load balancer."""
    return LoadBalancer(strategy)


