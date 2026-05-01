"""
Data Processor Factory - Unified Factory for Data Processors.

Provides a unified factory for creating data processors with automatic
selection based on availability and requirements.

Features:
- Automatic processor selection
- Fallback mechanisms
- Performance optimization
- Type safety
"""
import logging
from typing import Optional, Union, Dict, Any
from enum import Enum

from .polars_processor import PolarsProcessor, POLARS_AVAILABLE, create_polars_processor

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# PROCESSOR TYPE ENUM
# ════════════════════════════════════════════════════════════════════════════════

class ProcessorType(str, Enum):
    """Enum for processor types."""
    POLARS = "polars"
    PANDAS = "pandas"  # Fallback
    DASK = "dask"      # For very large datasets
    AUTO = "auto"


# ════════════════════════════════════════════════════════════════════════════════
# PROCESSOR FACTORY
# ════════════════════════════════════════════════════════════════════════════════

def create_data_processor(
    processor_type: Union[str, ProcessorType] = ProcessorType.AUTO,
    lazy: bool = True,
    streaming: bool = False,
    num_threads: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Factory function to create data processor.
    
    Automatically selects the best available processor based on:
    - Processor availability
    - Dataset size
    - Performance requirements
    - User preferences
    
    Args:
        processor_type: Processor type (polars, pandas, dask, auto)
        lazy: Use lazy evaluation (for Polars)
        streaming: Enable streaming (for large datasets)
        num_threads: Number of threads (None = auto)
        **kwargs: Processor-specific arguments
    
    Returns:
        Data processor instance
    
    Raises:
        ValueError: If processor type is invalid
        RuntimeError: If processor is unavailable
    
    Examples:
        >>> # Auto-select best processor
        >>> processor = create_data_processor()
        
        >>> # Explicit Polars processor
        >>> processor = create_data_processor(ProcessorType.POLARS, lazy=True)
        
        >>> # Streaming processor for large datasets
        >>> processor = create_data_processor(
        ...     processor_type=ProcessorType.POLARS,
        ...     streaming=True,
        ...     num_threads=8
        ... )
    """
    # Normalize processor type
    if isinstance(processor_type, str):
        try:
            processor_type = ProcessorType(processor_type.lower())
        except ValueError:
            raise ValueError(
                f"Unknown processor type: {processor_type}. "
                f"Available: {[t.value for t in ProcessorType]}"
            )
    
    # Auto-select processor if requested
    if processor_type == ProcessorType.AUTO:
        processor_type = _select_best_processor(streaming=streaming)
        logger.info(f"Auto-selected processor: {processor_type.value}")
    
    # Create processor based on type
    if processor_type == ProcessorType.POLARS:
        if not POLARS_AVAILABLE:
            raise RuntimeError(
                "Polars is not available. Install with: pip install polars"
            )
        return create_polars_processor(
            lazy=lazy,
            streaming=streaming,
            num_threads=num_threads,
            **kwargs
        )
    
    elif processor_type == ProcessorType.PANDAS:
        # Fallback to pandas
        try:
            import pandas as pd
            logger.warning(
                "Using pandas as fallback. Consider installing Polars for "
                "10-100x better performance: pip install polars"
            )
            # Return pandas directly (could wrap in adapter if needed)
            return pd
        except ImportError:
            raise RuntimeError(
                "Neither Polars nor pandas is available. "
                "Install at least one: pip install polars or pip install pandas"
            )
    
    elif processor_type == ProcessorType.DASK:
        # Dask for very large datasets
        try:
            import dask.dataframe as dd
            logger.info("Using Dask for distributed processing")
            return dd
        except ImportError:
            logger.warning(
                "Dask not available, falling back to Polars or pandas"
            )
            return create_data_processor(
                processor_type=ProcessorType.AUTO,
                lazy=lazy,
                streaming=streaming,
                num_threads=num_threads,
                **kwargs
            )
    
    else:
        raise ValueError(
            f"Unknown processor type: {processor_type}. "
            f"Available: {[t.value for t in ProcessorType]}"
        )


def _select_best_processor(
    streaming: bool = False,
    dataset_size_gb: Optional[float] = None
) -> ProcessorType:
    """
    Select best available processor based on requirements.
    
    Selection priority:
    1. Polars (best general purpose, 10-100x faster than pandas)
    2. Dask (for very large distributed datasets)
    3. Pandas (fallback)
    
    Args:
        streaming: Whether streaming is required
        dataset_size_gb: Estimated dataset size in GB
    
    Returns:
        Selected processor type
    """
    # For very large datasets (>100GB), prefer Dask if available
    if dataset_size_gb and dataset_size_gb > 100:
        try:
            import dask
            return ProcessorType.DASK
        except ImportError:
            pass
    
    # Polars is preferred for most use cases
    if POLARS_AVAILABLE:
        return ProcessorType.POLARS
    
    # Fallback to pandas
    try:
        import pandas
        return ProcessorType.PANDAS
    except ImportError:
        raise RuntimeError(
            "No data processors available. Install at least one of: "
            "Polars (pip install polars), "
            "Dask (pip install dask[dataframe]), or "
            "pandas (pip install pandas)"
        )


def list_available_processors() -> Dict[str, bool]:
    """
    List all available data processors with their availability status.
    
    Returns:
        Dictionary mapping processor types to availability
    
    Examples:
        >>> processors = list_available_processors()
        >>> print(processors)
        {'polars': True, 'pandas': True, 'dask': False}
    """
    pandas_available = False
    dask_available = False
    
    try:
        import pandas
        pandas_available = True
    except ImportError:
        pass
    
    try:
        import dask
        dask_available = True
    except ImportError:
        pass
    
    return {
        ProcessorType.POLARS.value: POLARS_AVAILABLE,
        ProcessorType.PANDAS.value: pandas_available,
        ProcessorType.DASK.value: dask_available,
    }


def get_processor_recommendation(
    dataset_size_gb: Optional[float] = None,
    requires_streaming: bool = False,
    requires_distributed: bool = False
) -> ProcessorType:
    """
    Get processor recommendation based on requirements.
    
    Args:
        dataset_size_gb: Dataset size in GB
        requires_streaming: Whether streaming is required
        requires_distributed: Whether distributed processing is required
    
    Returns:
        Recommended processor type
    
    Examples:
        >>> # Small dataset
        >>> rec = get_processor_recommendation(dataset_size_gb=1.0)
        >>> print(rec)  # ProcessorType.POLARS
        
        >>> # Very large dataset
        >>> rec = get_processor_recommendation(dataset_size_gb=500.0)
        >>> print(rec)  # ProcessorType.DASK
    """
    if requires_distributed or (dataset_size_gb and dataset_size_gb > 100):
        try:
            import dask
            return ProcessorType.DASK
        except ImportError:
            logger.warning("Dask not available, recommending Polars")
    
    if POLARS_AVAILABLE:
        return ProcessorType.POLARS
    
    return ProcessorType.PANDAS


def get_processor_info(processor_type: Union[str, ProcessorType]) -> Dict[str, Any]:
    """
    Get information about a processor type.
    
    Args:
        processor_type: Processor type
    
    Returns:
        Dictionary with processor information
    
    Examples:
        >>> info = get_processor_info(ProcessorType.POLARS)
        >>> print(info['performance'])
        '10-100x faster than pandas'
    """
    if isinstance(processor_type, str):
        processor_type = ProcessorType(processor_type.lower())
    
    info = {
        ProcessorType.POLARS: {
            "name": "Polars",
            "performance": "10-100x faster than pandas",
            "memory": "2-3x less memory than pandas",
            "features": [
                "Lazy evaluation",
                "Query optimization",
                "Multi-threaded",
                "Native Parquet/CSV/JSONL",
                "Streaming support"
            ],
            "best_for": "General purpose data processing",
            "install": "pip install polars",
        },
        ProcessorType.PANDAS: {
            "name": "Pandas",
            "performance": "Baseline",
            "memory": "Standard",
            "features": [
                "Mature ecosystem",
                "Rich functionality",
                "Easy to use"
            ],
            "best_for": "Small to medium datasets, compatibility",
            "install": "pip install pandas",
        },
        ProcessorType.DASK: {
            "name": "Dask",
            "performance": "Distributed processing",
            "memory": "Distributed across cluster",
            "features": [
                "Distributed computing",
                "Out-of-core processing",
                "Cluster support",
                "Pandas-like API"
            ],
            "best_for": "Very large datasets (>100GB), distributed processing",
            "install": "pip install dask[dataframe]",
        },
    }
    
    return info.get(processor_type, {})

