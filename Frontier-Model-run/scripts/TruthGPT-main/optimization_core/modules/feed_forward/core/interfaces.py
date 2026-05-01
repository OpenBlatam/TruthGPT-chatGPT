"""
PiMoE Core Interfaces and Protocols
===================================

Defines the core abstractions for the Progressive Mixture of Experts system.
Single source of truth for types, enums, and protocols.
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TypeVar, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn

# Type definitions
T = TypeVar('T')
RequestData = Dict[str, Any]
ResponseData = Dict[str, Any]

class ExpertType(Enum):
    """Expert type classifications."""
    REASONING = "reasoning"
    COMPUTATION = "computation"
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    LANGUAGE = "language"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    SPECIALIZED = "specialized"

class ExpertStatus(Enum):
    """Expert status states."""
    IDLE = "idle"
    PROCESSING = "processing"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"

class ProductionMode(Enum):
    """Production deployment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_PERFORMANCE = "high_performance"
    COST_OPTIMIZED = "cost_optimized"

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class OptimizationLevel(Enum):
    """Optimization levels."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    EXTREME = "extreme"

@dataclass
class RoutingDecision:
    """Represents a routing decision for a single token."""
    token_id: int
    expert_id: int
    expert_type: ExpertType
    confidence: float
    routing_score: float
    load_balance_weight: float

@dataclass
class ExpertResult:
    """Result of expert processing."""
    output: torch.Tensor
    processing_time: float
    expert_id: str
    expert_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class ExpertConfig:
    """Base expert configuration."""
    expert_id: str
    expert_type: ExpertType
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    use_bias: bool = True
    layer_norm_eps: float = 1e-5
    max_sequence_length: int = 2048
    enable_gradient_checkpointing: bool = False
    enable_quantization: bool = False
    quantization_bits: int = 8
    enable_pruning: bool = False
    pruning_ratio: float = 0.1
    enable_caching: bool = True
    cache_size: int = 100
    enable_metrics: bool = True
    enable_logging: bool = True

@runtime_checkable
class LoggerProtocol(Protocol):
    def log_info(self, message: str, **kwargs: Any) -> None: ...
    def log_warning(self, message: str, **kwargs: Any) -> None: ...
    def log_error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None: ...
    def log_metrics(self, metrics: Dict[str, Any]) -> None: ...

@runtime_checkable
class MonitorProtocol(Protocol):
    def record_request(self, success: bool = True) -> None: ...
    def get_health_status(self) -> Dict[str, Any]: ...

@runtime_checkable
class PiMoEProcessorProtocol(Protocol):
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]: ...
    def get_system_stats(self) -> Dict[str, Any]: ...
    def health_check(self) -> Dict[str, Any]: ...

@runtime_checkable
class ErrorHandlerProtocol(Protocol):
    def handle_error(self, error: Exception, context: str = "") -> bool: ...
    def should_circuit_break(self) -> bool: ...

@runtime_checkable
class RequestQueueProtocol(Protocol):
    def submit_request(self, request_data: RequestData, callback: Any) -> str: ...
    def get_queue_stats(self) -> Dict[str, Any]: ...

@dataclass
class SystemConfig:
    hidden_size: int = 512
    num_experts: int = 8
    max_batch_size: int = 32
    max_sequence_length: int = 2048

@dataclass
class ProductionConfig:
    system_config: SystemConfig = field(default_factory=SystemConfig)
    production_mode: ProductionMode = ProductionMode.PRODUCTION
    log_level: LogLevel = LogLevel.INFO
    enable_monitoring: bool = True
    enable_metrics: bool = True
    enable_quantization: bool = True
    enable_pruning: bool = True
    mixed_precision: bool = True
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0

