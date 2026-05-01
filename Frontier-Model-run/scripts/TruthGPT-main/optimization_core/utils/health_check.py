"""
Health check utilities for optimization_core.

Provides utilities for checking system health and component availability.
"""
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckResult(BaseModel):
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


class HealthChecker:
    """Health checker for components."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, callable] = {}
    
    def register_check(
        self,
        component: str,
        check_func: callable
    ):
        """
        Register a health check.
        
        Args:
            component: Component name
            check_func: Check function that returns HealthCheckResult
        """
        self.checks[component] = check_func
        logger.debug(f"Registered health check for {component}")
    
    def check(
        self,
        component: Optional[str] = None
    ) -> Dict[str, HealthCheckResult]:
        """
        Run health checks.
        
        Args:
            component: Specific component to check (all if None)
        
        Returns:
            Dictionary of health check results
        """
        if component:
            if component not in self.checks:
                return {
                    component: HealthCheckResult(
                        component=component,
                        status=HealthStatus.UNKNOWN,
                        message=f"Component {component} not registered"
                    )
                }
            
            try:
                result = self.checks[component]()
                return {component: result}
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}", exc_info=True)
                return {
                    component: HealthCheckResult(
                        component=component,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {e}"
                    )
                }
        
        # Check all components
        results = {}
        for comp_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[comp_name] = result
            except Exception as e:
                logger.error(f"Health check failed for {comp_name}: {e}", exc_info=True)
                results[comp_name] = HealthCheckResult(
                    component=comp_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}"
                )
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status.
        
        Returns:
            Overall health status
        """
        results = self.check()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


def check_vllm_available() -> HealthCheckResult:
    """Check if vLLM is available."""
    try:
        import vllm
        return HealthCheckResult(
            component="vllm",
            status=HealthStatus.HEALTHY,
            message="vLLM is available",
            details={"version": getattr(vllm, "__version__", "unknown")}
        )
    except ImportError:
        return HealthCheckResult(
            component="vllm",
            status=HealthStatus.DEGRADED,
            message="vLLM is not installed",
            details={"install": "pip install vllm>=0.2.0"}
        )


def check_tensorrt_llm_available() -> HealthCheckResult:
    """Check if TensorRT-LLM is available."""
    try:
        import tensorrt_llm
        return HealthCheckResult(
            component="tensorrt_llm",
            status=HealthStatus.HEALTHY,
            message="TensorRT-LLM is available"
        )
    except ImportError:
        return HealthCheckResult(
            component="tensorrt_llm",
            status=HealthStatus.DEGRADED,
            message="TensorRT-LLM is not installed",
            details={"install": "pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"}
        )


def check_polars_available() -> HealthCheckResult:
    """Check if Polars is available."""
    try:
        import polars as pl
        return HealthCheckResult(
            component="polars",
            status=HealthStatus.HEALTHY,
            message="Polars is available",
            details={"version": pl.__version__}
        )
    except ImportError:
        return HealthCheckResult(
            component="polars",
            status=HealthStatus.DEGRADED,
            message="Polars is not installed",
            details={"install": "pip install polars"}
        )


def check_gpu_available() -> HealthCheckResult:
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return HealthCheckResult(
                component="gpu",
                status=HealthStatus.HEALTHY,
                message="GPU is available",
                details={
                    "device_count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
                }
            )
        else:
            return HealthCheckResult(
                component="gpu",
                status=HealthStatus.DEGRADED,
                message="GPU is not available (CPU only)"
            )
    except ImportError:
        return HealthCheckResult(
            component="gpu",
            status=HealthStatus.UNKNOWN,
            message="PyTorch not available for GPU check"
        )


def create_default_health_checker() -> HealthChecker:
    """
    Create health checker with default checks.
    
    Returns:
        HealthChecker instance
    """
    checker = HealthChecker()
    checker.register_check("vllm", check_vllm_available)
    checker.register_check("tensorrt_llm", check_tensorrt_llm_available)
    checker.register_check("polars", check_polars_available)
    checker.register_check("gpu", check_gpu_available)
    return checker

