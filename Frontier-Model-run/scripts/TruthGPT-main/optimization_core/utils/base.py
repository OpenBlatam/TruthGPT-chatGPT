"""
Base utility classes and helper functions for TruthGPT optimization core.
Provides common abstractions to reduce boilerplate and improve maintainability.
"""

import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict
import time

logger = logging.getLogger(__name__)

class BaseOptimizationModel(BaseModel):
    """
    Base Pydantic model for all optimization configurations and results.
    Standardizes settings like arbitrary type allowance and provides utility methods.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='ignore'
    )

    def to_summary(self) -> Dict[str, Any]:
        """Returns a simplified dictionary summary of the model."""
        return self.model_dump(exclude_none=True)

class CudaResourceManager:
    """
    Helper for standardized CUDA resource management (streams, memory, etc.).
    Reduces boilerplate in classes that require parallel GPU processing.
    """
    
    @staticmethod
    def get_streams(num_streams: int, enabled: bool = True) -> Optional[List[torch.cuda.Stream]]:
        """
        Safely initialize a list of CUDA streams.
        
        Args:
            num_streams: Number of streams to create.
            enabled: Whether streams are actually requested.
            
        Returns:
            List of torch.cuda.Stream objects or None if not available/requested.
        """
        if enabled and torch.cuda.is_available():
            try:
                return [torch.cuda.Stream() for _ in range(num_streams)]
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA streams: {e}")
                return None
        return None

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Returns a standardized dictionary of current device information."""
        info = {"device": "cpu", "available": False}
        if torch.cuda.is_available():
            info.update({
                "device": "cuda",
                "available": True,
                "count": torch.cuda.device_count(),
                "name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024)
            })
        return info

def system_metrics_collector() -> Dict[str, float]:
    """
    Helper function to collect standardized system metrics.
    Abstracts psutil and torch.cuda calls.
    """
    metrics = {
        "timestamp": time.time(),
        "cpu_percent": 0.0,
        "memory_used_gb": 0.0,
        "gpu_used_mb": 0.0
    }
    
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent()
        metrics["memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
    except ImportError:
        pass
        
    if torch.cuda.is_available():
        metrics["gpu_used_mb"] = torch.cuda.memory_allocated() / (1024**2)
        
    return metrics
