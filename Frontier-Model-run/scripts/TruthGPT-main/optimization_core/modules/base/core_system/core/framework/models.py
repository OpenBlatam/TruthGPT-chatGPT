"""
Data Models for AI Optimization
"""

import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List
from .metrics_calculator import AIOptimizationLevel


class AIOptimizationResult(BaseModel):
    """Result of AI optimization."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    optimized_model: nn.Module = Field(..., description="The resulting optimized neural network model.")
    speed_improvement: float = Field(..., description="Relative speed improvement metric.")
    memory_reduction: float = Field(..., description="Relative memory footprint reduction metric.")
    accuracy_preservation: float = Field(..., description="Percentage of accuracy preserved.")
    intelligence_score: float = Field(..., description="Aggregate intelligence improvement score.")
    learning_efficiency: float = Field(..., description="Efficiency marker of the optimization.")
    optimization_time: float = Field(..., description="Latency/Time consumed by the optimization process.")
    level: AIOptimizationLevel = Field(..., description="Optimization tier applied.")
    techniques_applied: List[str] = Field(..., description="List of string names for techniques applied.")
    performance_metrics: Dict[str, float] = Field(..., description="Structured metrics pre and post optimization.")
    ai_insights: Dict[str, Any] = Field(default_factory=dict, description="Autonomous AI observations/insights.")
    neural_adaptation: float = Field(default=0.0, description="Internal model structural change factor.")
    cognitive_enhancement: float = Field(default=0.0, description="Internal algorithmic complexity improvement.")
    artificial_wisdom: float = Field(default=0.0, description="General robustness score improvement.")

