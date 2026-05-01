"""
Results definitions for VR/AR Optimization Engine.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

@dataclass
class VROptimizationResult:
    """VR/AR optimization result."""
    success: bool
    optimization_time: float
    optimized_system: Any
    performance_metrics: Dict[str, float]
    immersion_metrics: Dict[str, float]
    spatial_metrics: Dict[str, float]
    haptic_metrics: Dict[str, float]
    neural_metrics: Dict[str, float]
    consciousness_metrics: Dict[str, float]
    reality_metrics: Dict[str, float]
    immersive_technologies_used: List[str]
    optimization_applied: List[str]
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

