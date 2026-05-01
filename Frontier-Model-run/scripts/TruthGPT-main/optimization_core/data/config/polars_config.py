"""
Polars Processor Configuration
==============================

Configuration classes for Polars data processor.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from optimization_core.core.config_base import ValidatedConfig


@dataclass
class PolarsProcessorConfig(ValidatedConfig):
    """
    Configuration for Polars processor.
    
    Attributes:
        lazy: Use lazy evaluation (recommended for performance)
        streaming: Use streaming for large files
        num_threads: Number of threads for parallel processing
        memory_limit: Memory limit in bytes (None for unlimited)
        read_options: Default options for read operations
        write_options: Default options for write operations
    """
    lazy: bool = True
    streaming: bool = False
    num_threads: Optional[int] = None
    memory_limit: Optional[int] = None
    read_options: Dict[str, Any] = field(default_factory=dict)
    write_options: Dict[str, Any] = field(default_factory=dict)
    
    def _validate(self) -> None:
        """Validate configuration after initialization."""
        super()._validate()
        
        if self.num_threads is not None:
            self.validate_positive_int(self.num_threads, "num_threads")
        
        if self.memory_limit is not None:
            self.validate_positive_int(self.memory_limit, "memory_limit")


