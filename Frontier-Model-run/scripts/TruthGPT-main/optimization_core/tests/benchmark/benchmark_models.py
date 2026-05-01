"""
Benchmark Data Models

Data classes for benchmark results and reports.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    name: str
    module: str
    backend: str
    latency_ms: float
    throughput_tokens_per_sec: float = 0.0
    memory_mb: float = 0.0
    tokens_generated: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedSourceResult:
    """Resultado de un modelo closed source."""
    model_name: str
    latency_ms: float
    tokens_generated: int
    cost_per_1k_tokens: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    
    @property
    def throughput_tokens_per_sec(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.latency_ms / 1000.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'throughput_tokens_per_sec': self.throughput_tokens_per_sec
        }


@dataclass
class BenchmarkReport:
    """Reporte completo de benchmarks."""
    timestamp: str
    polyglot_results: List[BenchmarkResult]
    closed_source_results: List[ClosedSourceResult]
    module_status: Dict[str, bool]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'polyglot_results': [r.to_dict() for r in self.polyglot_results],
            'closed_source_results': [r.to_dict() for r in self.closed_source_results],
            'module_status': self.module_status,
            'summary': self.summary
        }
    
    def save_json(self, path: Path):
        """Guardar reporte en JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Reporte guardado en: {path}")













