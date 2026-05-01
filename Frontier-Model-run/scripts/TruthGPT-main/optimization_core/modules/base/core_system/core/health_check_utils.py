"""
Health check utilities for optimization_core.

Provides reusable health check implementations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, List

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# HEALTH STATUS
# ════════════════════════════════════════════════════════════════════════════════

class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ════════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK RESULT
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def is_healthy(self) -> bool:
        """Check if health check passed."""
        return self.status == HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# ════════════════════════════════════════════════════════════════════════════════
# HEALTH CHECKER (SYNC)
# ════════════════════════════════════════════════════════════════════════════════

class HealthChecker:
    """
    Health checker for services/components (sync).
    
    Example:
        >>> checker = HealthChecker()
        >>> checker.register("database", lambda: check_db_connection())
        >>> result = checker.check("database")
        >>> if result.is_healthy():
        ...     print("Database is healthy")
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
    
    def register(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult]
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Function that returns HealthCheckResult
        """
        self.checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    def check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Check name
        
        Returns:
            HealthCheckResult
        
        Raises:
            KeyError: If check not registered
        """
        if name not in self.checks:
            raise KeyError(f"Health check '{name}' not registered")
        
        try:
            return self.checks[name]()
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}", exc_info=True)
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        for name in self.checks:
            results[name] = self.check(name)
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status.
        
        Returns:
            HealthStatus (UNHEALTHY if any check fails, HEALTHY otherwise)
        """
        results = self.check_all()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        for result in results.values():
            if not result.is_healthy():
                return HealthStatus.UNHEALTHY
        
        return HealthStatus.HEALTHY


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC HEALTH CHECKER
# ════════════════════════════════════════════════════════════════════════════════

class AsyncHealthChecker:
    """
    Health checker for services/components (async).
    
    Example:
        >>> checker = AsyncHealthChecker()
        >>> checker.register("database", lambda: async_check_db())
        >>> result = await checker.check("database")
    """
    
    def __init__(self):
        """Initialize async health checker."""
        self.checks: Dict[str, Callable[[], Any]] = {}
    
    def register(
        self,
        name: str,
        check_func: Callable[[], Any]
    ) -> None:
        """
        Register an async health check.
        
        Args:
            name: Check name
            check_func: Async function that returns HealthCheckResult
        """
        self.checks[name] = check_func
        logger.debug(f"Registered async health check: {name}")
    
    async def check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check (async).
        
        Args:
            name: Check name
        
        Returns:
            HealthCheckResult
        
        Raises:
            KeyError: If check not registered
        """
        if name not in self.checks:
            raise KeyError(f"Health check '{name}' not registered")
        
        try:
            result = await self.checks[name]()
            if not isinstance(result, HealthCheckResult):
                # Wrap result if needed
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Check completed"
                )
            return result
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}", exc_info=True)
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks (async).
        
        Returns:
            Dictionary of check results
        """
        results = {}
        for name in self.checks:
            results[name] = await self.check(name)
        return results
    
    async def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status (async).
        
        Returns:
            HealthStatus (UNHEALTHY if any check fails, HEALTHY otherwise)
        """
        results = await self.check_all()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        for result in results.values():
            if not result.is_healthy():
                return HealthStatus.UNHEALTHY
        
        return HealthStatus.HEALTHY


# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_simple_check(
    name: str,
    check_func: Callable[[], bool],
    message: str = ""
) -> Callable[[], HealthCheckResult]:
    """
    Create a simple health check function.
    
    Args:
        name: Check name
        check_func: Function that returns True if healthy
        message: Optional message
    
    Returns:
        Health check function
    
    Example:
        >>> check = create_simple_check(
        ...     "database",
        ...     lambda: db.is_connected()
        ... )
        >>> result = check()
    """
    def health_check() -> HealthCheckResult:
        try:
            is_healthy = check_func()
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message=message or ("Healthy" if is_healthy else "Unhealthy")
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    return health_check


async def create_async_simple_check(
    name: str,
    check_func: Callable[[], Any],
    message: str = ""
) -> Callable[[], Any]:
    """
    Create a simple async health check function.
    
    Args:
        name: Check name
        check_func: Async function that returns True if healthy
        message: Optional message
    
    Returns:
        Async health check function
    """
    async def health_check() -> HealthCheckResult:
        try:
            is_healthy = await check_func()
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message=message or ("Healthy" if is_healthy else "Unhealthy")
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    return health_check


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "HealthStatus",
    # Result
    "HealthCheckResult",
    # Checkers
    "HealthChecker",
    "AsyncHealthChecker",
    # Helpers
    "create_simple_check",
    "create_async_simple_check",
]













